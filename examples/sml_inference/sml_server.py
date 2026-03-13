"""SML Server — Runs on the Leopard DPU.

Two-process architecture:
  - inference_process: classifies incoming telemetry, publishes anomaly alerts.
  - update_process:    buffers data + labels, periodically retrains the classifier,
                       sends the updated .pt path to inference via IPC for hot-swap.
"""

import argparse
import glob
import json
import logging
import os
import time
import warnings

import numpy as np
import torch
import zmq
import multiprocessing

from spaceai.models.anomaly_classifier import AnomalyClassifier

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")

IPC_MODEL_UPDATE = "ipc:///tmp/model_update"


def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="SML Inference Server")
    parser.add_argument("--run-dir", required=True, help="Path to the training directory containing .pt files")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--channel", type=str, default=None, help="Specific channel to serve")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--model_update", action="store_true", default=False)
    parser.add_argument("--model_update_interval", type=int, default=60,
                        help="Seconds between model retraining cycles")
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


# ---------------------------------------------------------------------------
#  Inference Process
# ---------------------------------------------------------------------------

def inference_process(args, trained_classifiers, feature_extractor):
    """Classify incoming telemetry windows and publish anomaly alerts."""

    window_size = feature_extractor.window_size if feature_extractor is not None else args.window_size
    stride = feature_extractor.stride if feature_extractor is not None else args.stride

    context = zmq.Context()

    # SUB socket for data coming from client
    data_socket = context.socket(zmq.SUB)
    data_socket.bind(f"tcp://0.0.0.0:{args.port}")
    data_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # PUB socket for publishing anomalies for each channel to clients
    pub_port = args.port + 1
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://0.0.0.0:{pub_port}")

    # PAIR socket to recieve model updates (only if enabled)
    update_socket = None
    if args.model_update:
        update_socket = context.socket(zmq.PAIR)
        update_socket.connect(IPC_MODEL_UPDATE)

    logging.info("[INFERENCE] SUB on %d, PUB on %d (window=%d, stride=%d)",
                 args.port, pub_port, window_size, stride)

    buffers = {ch: [] for ch in trained_classifiers.keys()}
    global_idx = {ch: 0 for ch in trained_classifiers.keys()}

    try:
        while True:
            frames = data_socket.recv_multipart()
            if len(frames) != 2:
                continue

            channel_id = frames[0].decode("utf-8")
            if channel_id not in trained_classifiers:
                continue

            try:
                payload = json.loads(frames[1].decode("utf-8"))
            except json.JSONDecodeError:
                continue

            values = payload.get("values")
            if values is None:
                continue

            buffers[channel_id].extend(values)

            while len(buffers[channel_id]) >= window_size:
                window = buffers[channel_id][:window_size]
                channel_data_np = np.array(window, dtype=np.float32).reshape(1, -1)

                if feature_extractor is not None:
                    channel_data_np = feature_extractor.transform(channel_data_np)

                y_pred = trained_classifiers[channel_id].predict(channel_data_np)

                buffers[channel_id] = buffers[channel_id][stride:]
                global_idx[channel_id] += stride

                if int(y_pred[0]) == 1:
                    alert_json = json.dumps({
                        "channel": channel_id,
                        "timestep_start": global_idx[channel_id] - stride,
                        "timestep_end": global_idx[channel_id] - stride + window_size,
                        "message": "Anomaly Detected"
                    }).encode("utf-8")
                    pub_socket.send_multipart([frames[0], alert_json])
                    logging.info("⚠️ ANOMALY on %s [%d:%d]",
                                 channel_id,
                                 global_idx[channel_id] - stride,
                                 global_idx[channel_id] - stride + window_size)

            # Check for model hot-swap via IPC (non-blocking)
            if update_socket:
                try:
                    msg = update_socket.recv(zmq.NOBLOCK)
                    new_model_path = msg.decode("utf-8")
                    ch = os.path.basename(new_model_path).replace("classifier-", "").replace(".pt", "")
                    if ch in trained_classifiers:
                        trained_classifiers[ch] = AnomalyClassifier.load(new_model_path)
                except zmq.Again:
                    pass

    except KeyboardInterrupt:
        logging.info("[INFERENCE] Shutting down.")
    finally:
        data_socket.close()
        pub_socket.close()
        update_socket.close()
        context.term()


def update_process(args, trained_classifiers, feature_extractor):
    """Buffer data + labels, periodically retrain classifiers, notify inference via IPC."""

    window_size = feature_extractor.window_size if feature_extractor is not None else args.window_size
    stride = feature_extractor.stride if feature_extractor is not None else args.stride

    context = zmq.Context()

    # SUB socket for data coming from client
    # Small delay to ensure inference bind happens first
    time.sleep(1)
    data_socket = context.socket(zmq.SUB)
    data_socket.connect(f"tcp://127.0.0.1:{args.port}")
    data_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # PAIR socket for model updates
    update_socket = context.socket(zmq.PAIR)
    update_socket.bind(IPC_MODEL_UPDATE)

    logging.info("[UPDATE] Listening on %d, retrain interval=%ds",
                 args.port, args.model_update_interval)

    # Per-channel training buffers: accumulate (X, y) pairs
    train_X = {ch: [] for ch in trained_classifiers.keys()}
    train_y = {ch: [] for ch in trained_classifiers.keys()}
    buffers = {ch: [] for ch in trained_classifiers.keys()}
    label_buffers = {ch: [] for ch in trained_classifiers.keys()}
    last_retrain_time = time.time()

    try:
        while True:
            # Non-blocking receive so we can check retrain interval
            try:
                frames = data_socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                frames = None

            if frames is not None and len(frames) == 2:
                channel_id = frames[0].decode("utf-8")
                if channel_id not in trained_classifiers:
                    pass
                else:
                    try:
                        payload = json.loads(frames[1].decode("utf-8"))
                    except json.JSONDecodeError:
                        payload = None

                    if payload is not None:
                        values = payload.get("values")
                        labels = payload.get("labels")

                        if values is not None and labels is not None:
                            buffers[channel_id].extend(values)
                            label_buffers[channel_id].extend(labels)

                            # Segment into windows for training
                            while len(buffers[channel_id]) >= window_size:
                                window = buffers[channel_id][:window_size]
                                window_labels = label_buffers[channel_id][:window_size]

                                if feature_extractor is not None:
                                    x = feature_extractor.transform(
                                        np.array(window, dtype=np.float32).reshape(1, -1)
                                    )
                                else:
                                    x = np.array(window, dtype=np.float32).reshape(1, -1)

                                # Label for the window: 1 if any point is anomalous
                                y = 1 if any(l == 1 for l in window_labels) else 0

                                train_X[channel_id].append(x)
                                train_y[channel_id].append(y)

                                buffers[channel_id] = buffers[channel_id][stride:]
                                label_buffers[channel_id] = label_buffers[channel_id][stride:]

            # Periodically retrain
            elapsed = time.time() - last_retrain_time
            if elapsed >= args.model_update_interval:
                for channel_id in trained_classifiers.keys():
                    if not train_X[channel_id]:
                        continue

                    X_train = np.vstack(train_X[channel_id])
                    y_train = np.array(train_y[channel_id])

                    logging.info("[UPDATE] Retraining %s with %d samples...",
                                 channel_id, len(y_train))

                    trained_classifiers[channel_id].fit(X_train, y_train)

                    # Save updated model
                    updated_path = os.path.join(args.run_dir, f"classifier-{channel_id}.pt")
                    trained_classifiers[channel_id].save(updated_path)

                    update_socket.send_string(updated_path)

                    train_X[channel_id].clear()
                    train_y[channel_id].clear()

                last_retrain_time = time.time()

    except KeyboardInterrupt:
        logging.info("[UPDATE] Shutting down.")
    finally:
        data_socket.close()
        update_socket.close()
        context.term()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args, _other_args = parse_exp_args()

    trained_classifiers, feature_extractor = load_models(args.run_dir, args.channel)
    if not trained_classifiers:
        logging.error("No classifiers found in %s. Exiting.", args.run_dir)
        return

    inference_p = multiprocessing.Process(
        target=inference_process,
        args=(args, trained_classifiers, feature_extractor)
    )
    inference_p.start()

    if args.model_update:
        update_p = multiprocessing.Process(
            target=update_process,
            args=(args, trained_classifiers, feature_extractor)
        )
        update_p.start()

    inference_p.join()
    if args.model_update:
        update_p.join()


if __name__ == "__main__":
    main()
