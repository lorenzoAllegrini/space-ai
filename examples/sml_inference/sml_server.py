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
from spaceai.models.anomaly_classifier.ndpm_internal import Config
from spaceai.preprocessing.ts_splitter import TSSplitter
from examples.utils.model_creators import create_classifier

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")

def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="SML Inference Server")
    parser.add_argument("--run_dir", required=True, help="Path to the directory containing .pt files or for saving logs")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--channel", type=str, default=None, help="Specific channel to serve")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--ndpm", action="store_true", default=False,
                        help="Use NDPMDetector as the default model if none exists")
    parser.add_argument("--ndpm_config", type=str, default=None,
                        help="Path to a YAML configuration file for NDPM")
    parser.add_argument("--ndpm_threshold", type=float, default=None,
                        help="Override NDPM anomaly threshold (e.g., -500.0, lower is less sensitive)")
    return parser.parse_known_args(str_args)


def load_models(run_dir, target_channel=None):
    """Load classifiers and feature extractor from the run directory."""
    trained_classifiers = {}
    feature_extractor = None

    classifier_files = sorted(glob.glob(os.path.join(run_dir, "classifier-*.pt")))
    for path in classifier_files:
        channel_id = os.path.basename(path).replace("classifier-", "").replace(".pt", "")
        if target_channel is not None and channel_id != target_channel:
            continue
        classifier = AnomalyClassifier.load(path)
        trained_classifiers[channel_id] = classifier
        logging.info("Loaded classifier for channel %s", channel_id)

    fe_path = os.path.join(run_dir, "feature_extractor.pt")
    if os.path.exists(fe_path):
        feature_extractor = torch.load(fe_path, weights_only=False)
        logging.info("Loaded feature extractor from %s", fe_path)

    logging.info("Loaded %d classifiers.", len(trained_classifiers))
    return trained_classifiers, feature_extractor





def main():
    args, _other_args = parse_exp_args()
    os.makedirs(args.run_dir, exist_ok=True)

    trained_classifiers, feature_extractor = load_models(args.run_dir, args.channel)
    
    if feature_extractor is None:
        from spaceai.preprocessing.feature_extractors.utils import get_feature_extractor
        fe_type = getattr(args, "feature_extractor", "base_statistics")
        feature_extractor = get_feature_extractor(
            fe_type,
            window_size=args.window_size,
            stride=args.stride
        )
        logging.info("Cold Start: Created new %s feature extractor", fe_type)

    # ts_splitter is generic, based on internal window/stride or args
    ts_splitter = TSSplitter(window_size=feature_extractor.window_size, step_size=feature_extractor.stride)

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
                if args.ndpm:
                    logging.info("Lazy-loading: Initializing NEW persistent NDPM model for channel %s", channel_id)
                    args.model = "ndpm"
                    args.channel = channel_id 
                    factory, _ = create_classifier(args, [], input_dim=feature_extractor.output_dim)
                    trained_classifiers[channel_id] = factory()
                else:
                    data_socket.send_json({"error": f"Channel {channel_id} not loaded and --ndpm fallback is OFF"})
                    continue

            try:
                payload = json.loads(frames[1].decode("utf-8"))
            except json.JSONDecodeError:
                data_socket.send_json({"error": "Invalid JSON mapping"})
                continue

            experience_data = payload.get("experience_data")
            labels = payload.get("labels") 
            sampling_period = payload.get("sampling_period") # Optional
            
            if experience_data is None:
                data_socket.send_json({"error": "Missing experience_data"})
                continue

            exp_np = np.array(experience_data)
            windows = ts_splitter.split(exp_np, sampling_period=sampling_period)
            
            y_train = None
            window_labels = None
            if labels is not None:
                labels_np = np.array(labels)
                window_labels = ts_splitter.split_labels(labels_np, sampling_period=sampling_period)
                y_train = window_labels

            if windows.size == 0:
                data_socket.send_json({"prediction": 0, "score": 0.0, "info": "Not enough data for a window"})
                continue

            if feature_extractor is not None:
                data = feature_extractor.transform(windows)

            else:
                data = windows

            classifier = trained_classifiers[channel_id]
            
            logging.info("Fitting model on experience for channel %s...", channel_id)
            
            all_experiences = all_experiences_dict.get(channel_id)
            
            if all_experiences is not None and len(all_experiences) > 0:
                forward_anomaly_score = classifier.predict(data)
                backward_anomaly_score = classifier.predict(all_experiences)
                forward_preds = np.array(forward_anomaly_score)
                backward_preds = np.array(backward_anomaly_score)
                
                new_data = np.array(data.values) if isinstance(data, pd.DataFrame) else np.array(data)
                all_experiences_dict[channel_id] = np.concatenate((all_experiences, new_data), axis=0)
            else:
                forward_preds = np.array([], dtype=int)
                backward_preds = np.array([], dtype=int)
                all_experiences_dict[channel_id] = np.array(data.values) if isinstance(data, pd.DataFrame) else np.array(data)

            classifier.fit(data, y_train)

            response = {
                "forward_predictions": forward_preds.tolist() if isinstance(forward_preds, np.ndarray) else [],
                "backward_predictions": backward_preds.tolist() if isinstance(backward_preds, np.ndarray) else [],
                "labels": window_labels.tolist() if isinstance(window_labels, np.ndarray) else []
            }
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
