"""SML Server — Runs on the Leopard DPU."""

import argparse
import glob
import logging
import os
import time
import warnings

import numpy as np
import torch
import zmq

from spaceai.models.anomaly_classifier import AnomalyClassifier

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")

DATASET_LIST = ["ops-sat", "nasa", "esa_m1", "esa_m2"]
MODEL_LIST = [
    "ocsvm",
    "xgboost",
    "ridge_regression",
    "dpmm",
    "iforest",
    "pca",
    "knn",
    "lof",
    "pyod_ocsvm",
    "ecod",
    "copod",
    "cblof",
    "hbos",
]
DPMM_MODEL_TYPE = ["full", "diagonal", "single", "unit"]
DPMM_MODE = ["likelihood_threshold", "cluster_labels"]
FEATURE_EXTRACTOR_LIST = ["none", "base_statistics", "rocket"]


def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="SML Inference Server")
    parser.add_argument("--run-dir", required=True, help="Path to the training directory containing .pt files")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--channel", type=str, default=None, help="Specific channel to serve. If not set, binds to first received.")
    parser.add_argument("--window-size", type=int, default=100)
    return parser.parse_known_args(str_args)


def load_models(run_dir, target_channel=None):
    """Load classifiers and feature extractor from the run directory."""
    trained_classifiers = {}
    feature_extractor = None

    # Load classifiers
    classifier_files = sorted(glob.glob(os.path.join(run_dir, "classifier-*.pt")))
    for path in classifier_files:
        channel_id = os.path.basename(path).replace("classifier-", "").replace(".pt", "")
        if target_channel is not None and channel_id != target_channel:
            continue
        
        classifier = AnomalyClassifier.load(path)
        trained_classifiers[channel_id] = classifier
        logging.info("Loaded classifier for channel %s", channel_id)

    # Load feature extractor
    fe_path = os.path.join(run_dir, "feature_extractor.pt")
    if os.path.exists(fe_path):
        feature_extractor = torch.load(fe_path, weights_only=False)
        logging.info("Loaded feature extractor from %s", fe_path)

    logging.info("Loaded %d classifiers.", len(trained_classifiers))
    return trained_classifiers, feature_extractor


def main():
    from collections import deque
    args, _other_args = parse_exp_args()

    trained_classifiers, feature_extractor = load_models(args.run_dir, args.channel)

    if not trained_classifiers:
        logging.error("No classifiers found matching criteria in %s. Exiting.", args.run_dir)
        return

    context = zmq.Context()
    
    # PULL socket for incoming stream
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    
    # PUB socket for outgoing alerts
    pub_port = args.port + 1
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://0.0.0.0:{pub_port}")

    logging.info("Server PULL listening on %d, PUB signaling on %d", args.port, pub_port)

    active_channel = args.channel
    buffers = {ch: deque(maxlen=args.window_size) for ch in trained_classifiers.keys()}

    try:
        import json
        while True:
            # High-performance multipart receiving
            frames = socket.recv_multipart()
            if len(frames) != 2:
                continue
                
            channel_id = frames[0].decode("utf-8")

            # Lock onto the first channel seen if not explicitly specified
            if active_channel is None:
                active_channel = channel_id
                logging.info("Server locked onto channel: %s", active_channel)

            # ZERO-OVERHEAD DROPPING: Drop immediately if channel mismatch, without parsing JSON
            if channel_id != active_channel or channel_id not in trained_classifiers:
                continue

            # Parse JSON payload only for the active channel
            try:
                payload = json.loads(frames[1].decode("utf-8"))
            except json.JSONDecodeError:
                continue
                
            timestep = payload.get("timestep")
            value = payload.get("value")
            
            if timestep is None or value is None:
                continue

            # Append the single value to our sliding window state buffer
            buffers[channel_id].append(value)
            
            # Predict only if the window is fully populated
            if len(buffers[channel_id]) == args.window_size:
                classifier = trained_classifiers[channel_id]
                window = list(buffers[channel_id])
                channel_data_np = np.array(window, dtype=np.float32).reshape(1, -1)

                if feature_extractor is not None:
                    channel_data_np = feature_extractor.transform(channel_data_np)

                y_pred = classifier.predict(channel_data_np)
                pred_val = int(y_pred[0])
                
                if pred_val == 1:
                    logging.warning("⚠️ ANOMALY DETECTED on channel %s at timestep %s!", channel_id, timestep)
                    alert_json = json.dumps({
                        "channel": channel_id,
                        "timestep": timestep,
                        "message": "Anomaly Detected"
                    }).encode("utf-8")
                    
                    # High-performance multipart PUB. Native ZMQ subscriber filtering.
                    pub_socket.send_multipart([frames[0], alert_json])

    except KeyboardInterrupt:
        logging.info("Server shutting down.")
    finally:
        socket.close()
        pub_socket.close()
        context.term()


if __name__ == "__main__":
    main()
