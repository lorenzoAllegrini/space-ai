"""SML Server — Runs on the Leopard DPU.

Inference server for Benchmark offloading.
Uses the zmq.pyobj protocol for complex object transfer.
"""

import argparse
import glob
import logging
import os
import warnings

import torch
import zmq
from spaceai.models.anomaly_classifier import AnomalyClassifier, AnomalyDetectionPipeline, PipelineMessage
from spaceai.benchmark.callbacks import CallbackHandler, SystemMonitorCallback
import pickle

from typing import Optional, List, Dict, Any
warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER-INF] %(message)s")

def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="SML Inference/Benchmark Server")
    parser.add_argument("--run_dir", required=True, help="Path for saving models and logs")
    parser.add_argument("--port", type=int, default=5556)
    return parser.parse_known_args(str_args)


def main():
    args, _other_args = parse_exp_args()
    os.makedirs(args.run_dir, exist_ok=True)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")

    callback_handler = CallbackHandler([SystemMonitorCallback()])

    logging.info("Inference Server listening on port %d...", args.port)

    try:
        while True:
            task = socket.recv_multipart()
            
            if len(task) != 2:
                safe_channel = task[0] if len(task) > 0 else b"unknown"
                socket.send_multipart([safe_channel, pickle.dumps({"error": "Invalid task"})])
                continue
            
            channel_id = task[0].decode("utf-8")
            try:
                payload = pickle.loads(task[1])
            except Exception as e:
                socket.send_multipart([task[0], pickle.dumps({"error": "Invalid task"})])
                continue
            
            action = payload.get("action")
            pipeline = payload.get("pipeline")
            data = payload.get("channel_data")
            labels = payload.get("channel_labels")

            logging.info("Task received: %s for channel %s", action, channel_id)

            if pipeline is None:
                socket.send_multipart([task[0], pickle.dumps({"error": "No pipeline provided in task"})])
                continue
            
            results = {}
            y_pred = None

            try:
                if action == "fit":
                    logging.info("Fitting...")
                    msgs = pipeline.fit(channel_data=data, channel_labels=labels, return_message=True)
                    response_payload = {
                        "classifier": pipeline,
                        "msgs": msgs
                    }
                elif action == "predict":
                    logging.info("Predicting...")
                    msg = pipeline.predict(channel_data=data, return_message=True)
                    response_payload = {
                        "classifier": pipeline,
                        "msgs": [msg]
                    }
                
                socket.send_multipart([task[0], pickle.dumps(response_payload)])

            except Exception as e:
                logging.error("Error processing task: %s", e)
                socket.send_multipart([task[0], pickle.dumps({"error": str(e)})])

    except KeyboardInterrupt:
        logging.info("Shutting down.")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
