"""SML Server — Runs on the Leopard DPU.

Simplified single-process architecture:
  - Receives an experience (block of data + optional labels).
  - Performs learning (fit) on the experience.
  - Performs inference (predict) on the same experience.
"""

import argparse
import glob
import json
import logging
import os
import sys
import types
import warnings

# Hack to handle unpickling of objects that refer to 'utils' from the examples dir
# when running as a standalone script or in a bundled environment.
if 'utils' not in sys.modules:
    try:
        import examples.utils as ex_utils
        import examples.utils.dataset_exp as ex_dataset_exp
        import examples.utils.model_creators as ex_model_creators
        import examples.utils.reproducibility as ex_reproducibility
        
        sys.modules['utils'] = ex_utils
        sys.modules['utils.dataset_exp'] = ex_dataset_exp
        sys.modules['utils.model_creators'] = ex_model_creators
        sys.modules['utils.reproducibility'] = ex_reproducibility
    except ImportError:
        # Fallback to stubs if examples.utils is not found
        utils = types.ModuleType('utils')
        sys.modules['utils'] = utils
        sys.modules['utils.dataset_exp'] = types.ModuleType('utils.dataset_exp')
        sys.modules['utils.model_creators'] = types.ModuleType('utils.model_creators')
        sys.modules['utils.reproducibility'] = types.ModuleType('utils.reproducibility')

sys.setrecursionlimit(10000)

import gc
import psutil
import numpy as np
import torch
import zmq
from spaceai.models.anomaly_classifier import AnomalyClassifier, AnomalyDetectionPipeline

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")

def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="SML Inference Server")
    parser.add_argument("--classifier_dir", default=None, help="Path to the directory containing .pt files (optional)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--channel", type=str, default=None, help="Specific channel to serve")
    return parser.parse_known_args(str_args)


def load_models(run_dir, target_channel=None):
    """Load classifiers (which can be Pipelines) from the run directory."""
    trained_classifiers = {}

    classifier_files = sorted(glob.glob(os.path.join(run_dir, "classifier-*.pt")))
    for path in classifier_files:
        channel_id = os.path.basename(path).replace("classifier-", "").replace(".pt", "")
        if target_channel is not None and channel_id != target_channel:
            continue
        
        try:
            # AnomalyClassifier.load uses torch.load internally
            try:
                classifier = AnomalyClassifier.load(path)
            except Exception as e:
                if "magic number" in str(e).lower() or "pickle" in str(e).lower():
                    import pickle
                    logging.info("torch.load failed (possibly raw pickle), trying pickle.load...")
                    with open(path, 'rb') as f:
                        classifier = pickle.load(f)
                else:
                    raise e
            trained_classifiers[channel_id] = classifier
            logging.info("Loaded classifier/pipeline for channel %s", channel_id)
        except Exception as e:
            logging.error("Failed to load classifier for channel %s from %s: %s", channel_id, path, e)

    logging.info("Loaded %d classifiers.", len(trained_classifiers))
    return trained_classifiers


def convert_numpy(obj):
    """Recursively convert numpy objects to serializable python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(i) for i in obj]
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16)):
        return int(obj)
    return obj

def get_memory_usage():
    """Returns the current RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    args, _other_args = parse_exp_args()

    if args.classifier_dir:
        trained_classifiers = load_models(args.classifier_dir, args.channel)
    else:
        trained_classifiers = {}
        logging.info("No classifier_dir provided. Starting in dynamic-only mode.")

    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.setsockopt(zmq.LINGER, 0)
    data_socket.bind(f"tcp://0.0.0.0:{args.port}")

    print(f"Server started on port {args.port}. Waiting for experiences...", flush=True)
    
    try:
        while True:
            frames = data_socket.recv_multipart()
            initial_mem = get_memory_usage()
            print(f"DEBUG: Received {len(frames)} frames. [RAM: {initial_mem:.1f}MB]", flush=True)
            
            if len(frames) != 2:
                print(f"ERROR: Expected 2 frames, got {len(frames)}", flush=True)
                continue
            
            channel_id = frames[0].decode("utf-8")

            use_pickle = False
            try:
                # 1. Try JSON first
                payload = json.loads(frames[1].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 2. Fallback to Pickle
                try:
                    import pickle
                    payload = pickle.loads(frames[1])
                    use_pickle = True
                except Exception as e:
                    logging.error("Failed to decode payload as JSON or Pickle: %s", e)
                    continue

            action = payload.get("action", "fit_predict")
            
            # In Pickle mode, the payload might have different keys (from SMLClientPipeline)
            experience_data = payload.get("experience_data")
            if experience_data is None:
                experience_data = payload.get("channel_data")
            
            labels = payload.get("labels")
            if labels is None:
                labels = payload.get("channel_labels")
            
            if experience_data is None:
                err_msg = {"error": "Missing experience_data"}
                if use_pickle:
                    data_socket.send_multipart([frames[0], pickle.dumps(err_msg)])
                else:
                    data_socket.send_multipart([frames[0], json.dumps(err_msg).encode("utf-8")])
                continue

            exp_np = experience_data
            y_train = np.array(labels) if labels is not None and len(labels) > 0 else None
            
            classifier = trained_classifiers.get(channel_id)
            if classifier is None:
                 # In Pipeline mode, the client might send the whole pipeline to fit/predict
                 classifier = payload.get("pipeline")
                 if classifier is None:
                      err_resp = {"error": f"Channel {channel_id} not loaded and no pipeline provided"}
                      if use_pickle:
                          import pickle
                          data_socket.send_multipart([frames[0], pickle.dumps(err_resp)])
                      else:
                          data_socket.send_multipart([frames[0], json.dumps(err_resp).encode("utf-8")])
                      continue
            
            n_samples = len(exp_np) if hasattr(exp_np, '__len__') else 0
            logging.info("[SERVER] Processing channel %s [Action: %s, Samples: %d, RAM: %.1fMB]", 
                         channel_id, action, n_samples, get_memory_usage())
            
            response = {"forward_predictions": [], "backward_predictions": [], "metrics": {}}
            
            try:
                # 1. Predict Phase
                if action in ("predict", "fit_predict"):
                    print(f"DEBUG: Starting prediction phase for channel {channel_id}...", flush=True)
                    if hasattr(classifier, 'predict'):
                        # Using return_message=True to get the fully populated PipelineMessage
                        res_msg = classifier.predict(exp_np, return_message=True)
                        print(f"DEBUG: Prediction successful for channel {channel_id}. [RAM: {get_memory_usage():.1f}MB]", flush=True)
                        if isinstance(res_msg, PipelineMessage):
                            response["msgs"] = [res_msg]
                        else:
                            # Fallback if not a message
                            response["forward_predictions"] = res_msg[0].tolist() if hasattr(res_msg[0], 'tolist') else res_msg[0]
                            response.update(res_msg[1])

                # 2. Fit Phase
                if action in ("fit", "fit_predict"):
                    print(f"DEBUG: Starting fit phase for channel {channel_id}...", flush=True)
                    if hasattr(classifier, 'fit'):
                        # Using return_message=True to catch validation results in messages
                        res_fit = classifier.fit(exp_np, channel_labels=y_train, return_message=True)
                        print(f"DEBUG: Fit successful for channel {channel_id}. [RAM: {get_memory_usage():.1f}MB]", flush=True)
                        
                        if isinstance(res_fit, list): # Multiple messages (train/val)
                            response["msgs"] = res_fit
                            response["metrics"] = res_fit[0].results if res_fit else {}
                        elif isinstance(res_fit, PipelineMessage):
                            response["msgs"] = [res_fit]
                            response["metrics"] = res_fit.results
                        else:
                            response["metrics"] = res_fit
                        
                print(f"DEBUG: Processing completed successfully for channel {channel_id}.", flush=True)

            except Exception as e:
                import traceback
                err_trace = traceback.format_exc()
                print(f"ERROR: Exception during processing for channel {channel_id}: {e}\n{err_trace}", flush=True)
                response["error"] = str(e)
                response["traceback"] = err_trace

            # Final check: we need 'classifier' and 'msgs' for SMLClientPipeline
            if use_pickle:
                if "msgs" not in response or not response["msgs"]:
                    # Create a blank message if missing
                    from spaceai.models.anomaly_classifier import PipelineMessage
                    msg = PipelineMessage(data=response.get("forward_predictions", []), 
                                          results=response.get("metrics", {}),
                                          metadata={})
                    response["msgs"] = [msg]
                
                # Debug log for final verification
                final_msg = response["msgs"][0]
                data_len = final_msg.data.shape if hasattr(final_msg.data, 'shape') else len(final_msg.data) if hasattr(final_msg.data, '__len__') else 0
                print(f"DEBUG: Final message data length: {data_len}", flush=True)
                
                response = {
                    "classifier": classifier,
                    "msgs": response["msgs"]
                }

            # Finalizing the response based on protocol
            try:
                if use_pickle:
                    print(f"DEBUG: Preparing Pickle response for channel {channel_id}...", flush=True)
                    payload_bytes = pickle.dumps(response)
                    print(f"DEBUG: Response pickled successful ({len(payload_bytes)} bytes). Sending...", flush=True)
                    data_socket.send_multipart([frames[0], payload_bytes])
                else:
                    print(f"DEBUG: Preparing JSON response for channel {channel_id}...", flush=True)
                    serializable_response = convert_numpy(response)
                    payload_bytes = json.dumps(serializable_response).encode("utf-8")
                    print(f"DEBUG: Response JSONed successful ({len(payload_bytes)} bytes). Sending...", flush=True)
                    data_socket.send_multipart([frames[0], payload_bytes])
                
                print(f"DEBUG: Response sent successfully. [Final RAM: {get_memory_usage():.1f}MB]", flush=True)
            except Exception as e:
                import traceback
                print(f"ERROR: Failed to send response: {e}\n{traceback.format_exc()}", flush=True)
            
            # Explicit cleanup
            del exp_np
            del labels
            del payload
            del response
            gc.collect()

    except KeyboardInterrupt:
        logging.info("Shutting down.")
    finally:
        data_socket.close()
        context.term()


if __name__ == "__main__":
    main()
