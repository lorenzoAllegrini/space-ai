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
import gc
import psutil
import time
import numpy as np
import torch
import zmq

# Ensure paths are correct for local imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
from spaceai.preprocessing import get_feature_extractor
from spaceai.benchmark.callbacks import SystemMonitorCallback, CallbackHandler
from spaceai.models.anomaly import ThresholdDetector, MoLooKDEDetector
from spaceai.models.anomaly_classifier.rolling_window_classifier import RollingWindowClassifier
from utils.model_creators import create_classifier
from utils.reproducibility import set_seed

sys.setrecursionlimit(10000)
warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")

def parse_exp_args(str_args=None):
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="SML Inference Server")
    parser.add_argument("--classifier_dir", default=None, help="Path to the directory containing .pt files (optional)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--channel", type=str, default=None, help="Specific channel to serve")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"], help="Device to use for torch models")
    return parser.parse_known_args(str_args)

def initialize_pipeline_from_args(payload_args):
    """Initialize a RollingWindowClassifier and its components from configuration arguments."""
    if isinstance(payload_args, (list, tuple)) and len(payload_args) == 2:
        args, other_args = payload_args
    else:
        args, other_args = payload_args, None

    set_seed(getattr(args, 'seed', 42))
    
    # === DEBUG: Dump all critical params for parity check ===
    window_size = args.window_size
    step_size = args.step_size
    min_window = getattr(args, 'min_window', None) or 10
    max_window = getattr(args, 'max_window', None) or 300
    perc_step_size = getattr(args, 'perc_step_size', None) or 1.0
    eval_perc = getattr(args, 'eval_perc', None)
    detector_type = getattr(args, 'detector', 'threshold')
    fe_params = getattr(args, 'feature_extraction_params', getattr(args, 'fe_params', {}))
    
    print(f"\n[SERVER-FACTORY] === Pipeline Construction Parameters ===", flush=True)
    print(f"[SERVER-FACTORY] window_size={window_size}, step_size={step_size}", flush=True)
    print(f"[SERVER-FACTORY] min_window={min_window}, max_window={max_window}", flush=True)
    print(f"[SERVER-FACTORY] perc_step_size={perc_step_size}", flush=True)
    print(f"[SERVER-FACTORY] eval_perc={eval_perc}", flush=True)
    print(f"[SERVER-FACTORY] detector={detector_type}", flush=True)
    print(f"[SERVER-FACTORY] feature_extractor={args.feature_extractor}", flush=True)
    print(f"[SERVER-FACTORY] fe_params={fe_params}", flush=True)
    print(f"[SERVER-FACTORY] model={getattr(args, 'model', '?')}", flush=True)
    print(f"[SERVER-FACTORY] base_classifier_params={getattr(args, 'base_classifier_params', {})}", flush=True)
    print(f"[SERVER-FACTORY] seed={getattr(args, 'seed', 42)}", flush=True)
    print(f"[SERVER-FACTORY] ==========================================\n", flush=True)

    # 1. Splitter
    ts_splitter = TimeSeriesSplitter(
        window_size=window_size,
        step_size=step_size,
        min_window=min_window,
        max_window=max_window,
        perc_step_size=perc_step_size,
    )
    
    # 2. Feature Extractor
    feature_extractor = get_feature_extractor(
        args.feature_extractor,
        window_size=window_size,
        stride=step_size,
        n_kernel=getattr(args, 'n_kernel', None),
        **fe_params
    )
    
    # 3. Base Classifier
    base_classifier, is_supervised = create_classifier(args, other_args)
    
    # Enable performance monitoring on the server/DPU
    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)
    
    # Check if the returned classifier is a self-contained sequence model (like Telemanom)
    # that shouldn't be wrapped in RollingWindowClassifier
    from spaceai.models.anomaly_classifier.telemanom_classifier import SequenceModelClassifier
    if isinstance(base_classifier, SequenceModelClassifier):
        logging.info("[SERVER-FACTORY] SequenceModelClassifier detected. Skipping RollingWindow wrapping.")
        base_classifier.callback_handler = handler
        return base_classifier

    # 4. Detector
    detector_params = getattr(args, 'detector_params', {})
    detector = None
    
    if detector_type == "threshold":
        detector = ThresholdDetector(**{**dict(threshold=0.9), **detector_params})
    elif detector_type == "molookde":
        detector = MoLooKDEDetector(**{**dict(alpha=0.001), **detector_params})
        
    # 5. Assembly
    pipeline = RollingWindowClassifier(
        base_classifier=base_classifier,
        supervised_classifier=is_supervised,
        ts_splitter=ts_splitter,
        feature_extractor=feature_extractor,
        callback_handler=handler,
        detector=detector,
        eval_perc=eval_perc,
    )
    
    logging.info("[SERVER-FACTORY] Pipeline initialized successfully.")
    return pipeline

def patch_sklearn_fitted(obj):
    """Hack to ensure unpickled scikit-learn models are recognized as fitted."""
    if obj is None: return
    for attr in ['components_', 'labels_', 'centers_', 'mu_prior_strength', 'likelihood_threshold', 'is_fitted_']:
        if hasattr(obj, attr):
            if not hasattr(obj, 'fitted_'): setattr(obj, 'fitted_', True)
            if not hasattr(obj, 'is_fitted_'): setattr(obj, 'is_fitted_', True)
    if hasattr(obj, 'steps'): # Pipeline
        for _, step in obj.steps: patch_sklearn_fitted(step)
    elif hasattr(obj, 'base_classifier'): # RWC
        patch_sklearn_fitted(obj.base_classifier)

def convert_numpy(obj):
    """Recursively convert numpy objects to serializable python types."""
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [convert_numpy(i) for i in obj]
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)): return int(obj)
    return obj

def get_memory_usage():
    """Returns the current RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def inspect_sml_object(obj, name="classifier"):
    """Recursively dumps interesting attributes of the SML pipeline."""
    if obj is None: return
    print(f"\n[DIAGNOSTIC-INSPECT] === Deep Inspection of {name} ({type(obj).__name__}) [ID: {id(obj)}] ===", flush=True)
    attrs = ['window_size', 'step_size', 'alpha', 'p', 'unitize', 'smoothing_alpha', 'min_allowed_ll', 'pot_threshold', 
             'alpha_dp', 'mu_prior_strength', 'var_prior_strength', 'num_iterations', 'lr']
    for attr in attrs:
        if hasattr(obj, attr):
            print(f"[DIAGNOSTIC-INSPECT] -> {attr}: {getattr(obj, attr)}", flush=True)
    if hasattr(obj, 'steps'): # Pipeline
        for s_name, s_obj in obj.steps: inspect_sml_object(s_obj, name=f"{name}.{s_name}")
    elif hasattr(obj, 'base_classifier'):
        inspect_sml_object(obj.base_classifier, name=f"{name}.base")
    elif hasattr(obj, 'detector'):
        inspect_sml_object(obj.detector, name=f"{name}.detector")

def main():
    args, _other_args = parse_exp_args()
    trained_classifiers = {}
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.setsockopt(zmq.LINGER, 0)
    data_socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"Server started on port {args.port}. Waiting for experiences...", flush=True)
    
    try:
        while True:
            frames = data_socket.recv_multipart()
            channel_id = frames[0].decode("utf-8")
            
            # --- RAM MANAGEMENT ---
            if channel_id not in trained_classifiers and len(trained_classifiers) > 0:
                trained_classifiers.clear()
                gc.collect()
            
            try:
                payload = json.loads(frames[1].decode("utf-8"))
                use_pickle = False
            except:
                import pickle
                payload = pickle.loads(frames[1])
                use_pickle = True
            
            action = payload.get("action", "predict")
            experience_data = payload.get("experience_data", payload.get("channel_data"))
            labels = payload.get("labels", payload.get("channel_labels"))
            exp_np = experience_data
            y_train = np.array(labels) if labels is not None and len(labels) > 0 else None
            
            classifier = trained_classifiers.get(channel_id)
            
            # Lazy init from Recipe
            if classifier is None and "args" in payload:
                logging.info("[SERVER] Initializing from recipe for channel %s...", channel_id)
                classifier = initialize_pipeline_from_args(payload["args"])
            
            if classifier is None:
                classifier = payload.get("pipeline")
                if classifier is not None: patch_sklearn_fitted(classifier)
            
            if classifier is None:
                err_resp = {"error": f"Channel {channel_id} not loaded and no recipe/pipeline provided"}
                data_socket.send_multipart([frames[0], json.dumps(err_resp).encode("utf-8")])
                continue
            
            logging.info("[SERVER] Processing %s [Action: %s, RAM: %.1fMB]", channel_id, action, get_memory_usage())
            # Se è un oggetto Dataset, accediamo a .data per la shape
            d_shape = exp_np.data.shape if hasattr(exp_np, 'data') else "N/A"
            logging.info("[SERVER] Data SHAPE: %s", str(d_shape))
            if hasattr(exp_np, 'data') and len(exp_np.data) > 0:
                logging.info("[SERVER] First row: %s", str(exp_np.data[0]))
            inspect_sml_object(classifier, name="active_server_classifier_START")
            response = {"preds": [], "metrics": {}}
            
            try:
                if action == "fit":
                    if hasattr(classifier, 'fit'):
                        metrics = classifier.fit(exp_np, channel_labels=y_train)
                        trained_classifiers[channel_id] = classifier
                        # Include fitted ts_splitter state for client sync
                        if hasattr(classifier, 'ts_splitter'):
                            metrics['_fitted_window_size'] = classifier.ts_splitter.window_size
                            metrics['_fitted_step_size'] = classifier.ts_splitter.step_size
                        response["metrics"] = metrics
                        inspect_sml_object(classifier, name="post_fit_classifier")
                
                if action == "predict":
                    if hasattr(classifier, 'predict'):
                        predictions, metrics = classifier.predict(exp_np)
                        response["preds"] = predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                        response.update(metrics)
            except Exception as e:
                import traceback
                err_trace = traceback.format_exc()
                print(f"ERROR: {e}\n{err_trace}", flush=True)
                response["error"] = str(e)
            
            data_socket.send_multipart([frames[0], json.dumps(convert_numpy(response)).encode("utf-8")])
    finally:
        data_socket.close()
        context.term()

if __name__ == "__main__":
    main()
