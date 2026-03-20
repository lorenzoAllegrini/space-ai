"""SML Server — Runs on the Leopard DPU.

Simplified single-process architecture:
  - Receives an experience (block of data + optional labels).
  - Performs learning (fit) on the experience.
  - Performs inference (predict) on the same experience.
"""
import pandas as pd

import argparse
import glob
import json
import logging
import os
import warnings

import numpy as np
import torch
import zmq
import time
from spaceai.models.anomaly_classifier import AnomalyClassifier, NDPMDetector
from examples.utils.model_creators import create_classifier

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")

def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="SML Inference Server")
    parser.add_argument("--classifier_dir", required=True, help="Path to the directory containing .pt files or for saving logs")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--channel", type=str, default=None, help="Specific channel to serve")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    return parser.parse_known_args(str_args)


def load_models(run_dir, target_channel=None):
    """Load classifiers from the run directory."""
    trained_classifiers = {}

    classifier_files = sorted(glob.glob(os.path.join(run_dir, "classifier-*.pt")))
    for path in classifier_files:
        channel_id = os.path.basename(path).replace("classifier-", "").replace(".pt", "")
        if target_channel is not None and channel_id != target_channel:
            continue
        classifier = AnomalyClassifier.load(path)
        trained_classifiers[channel_id] = classifier
        logging.info("Loaded classifier for channel %s", channel_id)

    logging.info("Loaded %d classifiers.", len(trained_classifiers))
    return trained_classifiers


def main():
    args, _other_args = parse_exp_args()
    os.makedirs(args.run_dir, exist_ok=True)

    trained_classifiers = load_models(args.classifier_dir, args.channel)

    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.setsockopt(zmq.LINGER, 0)
    data_socket.bind(f"tcp://0.0.0.0:{args.port}")

    logging.info("Server started on port %d. Persistent multi-channel learning (NDPM: %s)", args.port, args.ndpm)
    all_experiences_dict = {}
    
    try:
        while True:
            frames = data_socket.recv_multipart()
            if len(frames) != 2:
                continue

            channel_id = frames[0].decode("utf-8")
            
            if channel_id not in trained_classifiers:
                data_socket.send_json({"error": f"Channel {channel_id} not loaded "})
                continue

            try:
                payload = json.loads(frames[1].decode("utf-8"))
            except json.JSONDecodeError:
                data_socket.send_json({"error": "Invalid JSON mapping"})
                continue

            action = payload.get("action", "fit_predict")
            experience_data = payload.get("experience_data")
            labels = payload.get("labels") 
            
            if experience_data is None:
                data_socket.send_json({"error": "Missing experience_data"})
                continue

            exp_np = np.array(experience_data)
            y_train = np.array(labels) if labels is not None and len(labels) > 0 else None

            classifier = trained_classifiers[channel_id]
            all_experiences = all_experiences_dict.get(channel_id)
            
            logging.info("Processing experience for channel %s (action: %s)...", channel_id, action)
            
            response = {"forward_predictions": [], "backward_predictions": [], "labels": []}
            
            if action in ("predict", "fit_predict"):
                forward_preds, forward_metrics = classifier.predict(exp_np)
                response["forward_predictions"] = forward_preds.tolist() if isinstance(forward_preds, np.ndarray) else list(forward_preds)
                # optionally include forward_metrics? Note: original implementation didn't, but can be added.

                if all_experiences is not None and len(all_experiences) > 0:
                    backward_preds, backward_metrics = classifier.predict(all_experiences)
                else:
                    backward_preds = np.array([], dtype=int)
                response["backward_predictions"] = backward_preds.tolist() if isinstance(backward_preds, np.ndarray) else list(backward_preds)

            # 2. Fit Phase
            if action in ("fit", "fit_predict"):
                metrics = classifier.fit(exp_np, y_train)
                response["metrics"] = metrics
                
                if all_experiences is not None and len(all_experiences) > 0:
                    all_experiences_dict[channel_id] = np.concatenate((all_experiences, exp_np), axis=0)
                else:
                    all_experiences_dict[channel_id] = exp_np

            data_socket.send_multipart([
                channel_id.encode("utf-8"),
                json.dumps(response).encode("utf-8")
            ])

    except KeyboardInterrupt:
        logging.info("Shutting down.")
    finally:
        data_socket.close()
        context.term()


if __name__ == "__main__":
    main()
