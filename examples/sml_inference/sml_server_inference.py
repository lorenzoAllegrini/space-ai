"""SML Batch Inference Server — Runs on the Leopard DPU.

Designed to offload Benchmark training and testing to the DPU.
Handles 'train' and 'test' actions via ZMQ REP/REQ.
"""

import argparse
import glob
import json
import logging
import os
import warnings

import numpy as np
import torch
import zmq

from spaceai.models.anomaly_classifier import AnomalyClassifier
from spaceai.benchmark.callbacks import CallbackHandler, SystemMonitorCallback

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFERENCE-SERVER] %(message)s")

def parse_args():
    # --- Warm-up / Dependency Check ---
    logging.info("Checking dependencies...")
    try:
        import sklearn.pipeline
        import sklearn.preprocessing
        import sklearn.ensemble
        import xgboost
        import torch_dpmm
        import pyod
        from spaceai.models.anomaly_classifier import ndpm_detector
        logging.info("All critical dependencies (sklearn, xgboost, torch_dpmm, pyod, spaceai) loaded successfully.")
    except ImportError as e:
        logging.error("CRITICAL: Dependency check failed: %s", e)
        # We don't exit here to allow manual debugging if needed, but the server will likely fail later anyway.
    
    parser = argparse.ArgumentParser(description="SML Batch Inference Server")
    parser.add_argument("--run_dir", required=True, help="Path for saving/loading models")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--channel", type=str, default=None, help="Specific channel to serve")
    parser.add_argument("--ndpm", action="store_true", default=True, help="Allow cold start for new channels")
    return parser.parse_args()

def load_models(run_dir, target_channel=None):
    trained_classifiers = {}
    classifier_files = sorted(glob.glob(os.path.join(run_dir, "classifier-*.pt")))
    for path in classifier_files:
        channel_id = os.path.basename(path).replace("classifier-", "").replace(".pt", "")
        if target_channel is not None and channel_id != target_channel:
            continue
        try:
            classifier = AnomalyClassifier.load(path)
            trained_classifiers[channel_id] = classifier
            logging.info("Loaded classifier for channel %s", channel_id)
        except Exception as e:
            logging.error("Failed to load classifier %s: %s", path, e)
    return trained_classifiers

def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    
    callback_handler = CallbackHandler(callbacks=[SystemMonitorCallback()], call_every_ms=100)
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    
    logging.info("Batch Inference Server started on port %d", args.port)
    while True:
        results = {}
        try:
            request = socket.recv_pyobj()
            
            channel_id = request.get("channel_id")
            
            train_channel, test_channel = request.get("train_data"), request.get("test_data")
            train_labels, test_labels = request.get("train_labels"), request.get("test_labels")
            
            classifier = request.get("classifier")
            feature_extractor = request.get("feature_extractor")

            if train_channel is None or test_channel is None:
                socket.send_pyobj({"error": "Missing data field (train_data or test_data)"})
                continue

            if channel_id is None:
                socket.send_pyobj({"error": "Missing channel_id"})
                continue

            if classifier is None:
                raise ValueError("Classifier not found. Please provide a classifier.")
            
            if feature_extractor is not None:
                callback_handler.start()
                logging.info("Applying transform to train set...")
                train_channel = feature_extractor.transform(train_channel)
                callback_handler.stop()
                results.update({f"train_set_feature_extraction_{k}": v for k, v in callback_handler.collect(reset=True).items()})
                
                callback_handler.start()
                logging.info("Applying transform to test set...")
                test_channel = feature_extractor.transform(test_channel)
                callback_handler.stop()
                results.update({f"test_set_feature_extraction_{k}": v for k, v in callback_handler.collect(reset=True).items()})
                    
            if train_labels is not None:
                logging.info("Fitting model...")
                callback_handler.start()
                classifier.fit(X=train_channel, y=train_labels)
                callback_handler.stop()
                results.update({f"train_{k}": v for k, v in callback_handler.collect(reset=True).items()})
                save_path = os.path.join(args.run_dir, f"classifier-{channel_id}.pt")
                if hasattr(classifier, "save"):
                    classifier.save(save_path)
                else:
                    logging.info("Classifier has no 'save' method, using torch.save as fallback")
                    torch.save(classifier, save_path)

            logging.info("Predicting...")
            callback_handler.start()
            y_pred = classifier.predict(X=test_channel)
            callback_handler.stop()
            results.update({f"test_{k}": v for k, v in callback_handler.collect(reset=True).items()})
            
            socket.send_pyobj({
                "status": "success",
                "y_pred": y_pred,
                "metrics": results,
                "classifier": classifier,
                "feature_extractor": feature_extractor
            })

        except Exception as e:
            logging.error("Error processing request: %s", e)
            try:
                socket.send_pyobj({"error": str(e)})
            except:
                pass


if __name__ == "__main__":
    main()
