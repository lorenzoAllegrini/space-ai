import pickle
import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zmq
import pandas as pd

from .anomaly_classifier import AnomalyClassifier

def inspect_sml_object(obj, name="classifier"):
    """Recursively dumps interesting attributes of the SML pipeline."""
    if obj is None: return
    print(f"\n[DIAGNOSTIC-INSPECT-CLIENT] === Deep Inspection of {name} ({type(obj).__name__}) ===", flush=True)
    
    # DPMM / Selection / Detection parameters
    interesting = [
        'max_features', 'window_size', 'early_stop', 'patience', 
        'alpha', 'pot_percentile', 'bandwidth_', 'prior_iqr_',
        'K', 'alphaDP', 'alpha_dp', 'mu_prior_strength', 'var_prior_strength',
        'num_iterations', 'lr', 'min_delta', 'p', 'unitize', 'smoothing_alpha',
        'min_allowed_ll', 'pot_threshold'
    ]
    
    for attr in interesting:
        if hasattr(obj, attr):
            print(f"[DIAGNOSTIC-INSPECT-CLIENT] -> {attr}: {getattr(obj, attr)}", flush=True)
            
    # Specific for DPMM or PyTorch models (Weights Signature)
    # We try different ways to find parameters
    params_to_check = []
    
    # 1. Standard parameters()
    if hasattr(obj, 'parameters'):
        try: params_to_check.extend(list(obj.parameters()))
        except: pass
        
    # 2. Internal PyTorch dicts (for dynamically registered params)
    if hasattr(obj, '_parameters'):
        params_to_check.extend(obj._parameters.values())
    if hasattr(obj, '_buffers'):
        params_to_check.extend(obj._buffers.values())
    
    # 3. Fallback for DPMM specific internal lists
    if hasattr(obj, 'mix_weights_var_eta'):
        params_to_check.extend(obj.mix_weights_var_eta)
    if hasattr(obj, 'emission_var_eta'):
        params_to_check.extend(obj.emission_var_eta)

    if params_to_check:
        try:
            import torch
            valid_vals = []
            for p in params_to_check:
                if p is None: continue
                # Handle both Tensors and Parameters
                tensor_data = p.data if hasattr(p, 'data') else p
                if isinstance(tensor_data, torch.Tensor):
                    valid_vals.append(tensor_data.detach().cpu())
            
            if valid_vals:
                total_sum = sum(t.sum().item() for t in valid_vals)
                total_abs_mean = sum(t.abs().mean().item() for t in valid_vals) / len(valid_vals)
                print(f"[DIAGNOSTIC-WEIGHTS-CLIENT] -> {name} Signature: Sum={total_sum:.8f}, AbsMean={total_abs_mean:.8f}", flush=True)
        except Exception as e:
            # print(f"[DIAGNOSTIC-DEBUG] Weights error: {e}")
            pass

    # Recursive inspection
    if hasattr(obj, 'steps'): # scikit-learn Pipeline
        for step_name, step_obj in obj.steps:
            inspect_sml_object(step_obj, name=f"{name}.{step_name}")
    elif hasattr(obj, 'transformer_list'): # scikit-learn FeatureUnion
        for step_name, step_obj in obj.transformer_list:
            inspect_sml_object(step_obj, name=f"{name}.{step_name}")
    elif hasattr(obj, 'base_classifier'): # SML Wrapper / RollingWindow
        inspect_sml_object(obj.base_classifier, name=f"{name}.base")
        if hasattr(obj, 'feature_extractor'):
            inspect_sml_object(obj.feature_extractor, name=f"{name}.extractor")
        if hasattr(obj, 'detector'):
            inspect_sml_object(obj.detector, name=f"{name}.detector")

class SMLClientClassifier(AnomalyClassifier):
    """
    Client wrapper for Continual Learning (SML) backend.
    Delegates streaming prediction and learning tasks to a remote server via ZeroMQ.
    """

    def __init__(self, 
                 server_ip: str = "localhost", 
                 port: int = 5555, 
                 channel_id: str = "default",
                 base_classifier: Optional[AnomalyClassifier] = None,
                 args: Optional[Any] = None) -> None:
        self.server_ip = server_ip
        self.port = port
        self.channel_id = channel_id
        self.base_classifier = base_classifier
        self.args = args # Memorandum della ricetta
        
        self.context = zmq.Context.instance()
        self.socket = None
        self._reconnect()
        
    def _reconnect(self) -> None:
        """Destroy and recreate the ZMQ socket to recover from invalid REQ-REP state."""
        if self.socket is not None:
            self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 3000000)  # 15 min timeout for slow DPU
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
        
    def _send_request(self, action: str, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        payload = {
            "action": action,
            "experience_data": X,
            "labels": y if y is not None else [],
        }
        
        saved_cb = None # Initialize to avoid UnboundLocalError in except block

        # NUOVO PROTOCOLLO: Inviamo gli args invece dell'oggetto se disponibili
        if self.args is not None:
            payload["args"] = self.args
        elif self.base_classifier is not None:
            # Vecchio protocollo (supporto legacy / fallback)
            saved_cb = getattr(self.base_classifier, "callback_handler", None)
            if saved_cb is not None:
                self.base_classifier.callback_handler = None
            payload["pipeline"] = self.base_classifier
        
        channel_bytes = self.channel_id.encode("utf-8")
        try:
            # INSPECT LOCAL PIPELINE BEFORE SENDING (Solo se siamo in modalità legacy)
            if "pipeline" in payload:
                inspect_sml_object(payload["pipeline"], name="local_pipeline_PRE_PICKLE")
            elif "args" in payload:
                print(f"[DIAGNOSTIC-CLIENT] Sending ARGS instead of object for channel {self.channel_id}", flush=True)

            self.socket.send_multipart([channel_bytes, pickle.dumps(payload)])
            
            # Ripristiniamo immediatamente i callback locali (se eravamo in modalità legacy)
            if "pipeline" in payload and self.base_classifier is not None and saved_cb is not None:
                self.base_classifier.callback_handler = saved_cb
                
            resp_frames = self.socket.recv_multipart()
        except (zmq.Again, zmq.ZMQError):
            if saved_cb is not None:
                self.base_classifier.callback_handler = saved_cb
            logging.error("ZMQ error or timeout at %s:%s. Reconnecting...", self.server_ip, self.port)
            self._reconnect()
            return {"error": "timeout_or_zmq_error"}
        
        if len(resp_frames) != 2:
            return {"error": "invalid_response_format"}
            
        try:
            # Prova a decodificare come JSON (standard per le risposte del server)
            return json.loads(resp_frames[1].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Fallback su Pickle se il server ha risposto con oggetti complessi
                return pickle.loads(resp_frames[1])
            except Exception as e:
                return {"error": f"decode_failed: {str(e)}"}

    def fit(self, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Fit is handled server-side independently."""
        response = self._send_request("fit", X, y, **kwargs)
        if "error" in response:
            logging.error("[SML-CLIENT] Fit error: %s", response["error"])
        
        metrics = response.get("metrics", {})
        
        # Sync local ts_splitter with server-fitted values
        if self.base_classifier is not None and hasattr(self.base_classifier, 'ts_splitter'):
            ws = metrics.pop('_fitted_window_size', None)
            ss = metrics.pop('_fitted_step_size', None)
            if ws is not None:
                self.base_classifier.ts_splitter.window_size = ws
                print(f"[SML-CLIENT] Synced ts_splitter.window_size = {ws}", flush=True)
            if ss is not None:
                self.base_classifier.ts_splitter.step_size = ss
                print(f"[SML-CLIENT] Synced ts_splitter.step_size = {ss}", flush=True)
        
        return metrics

    def predict(self, X: Any, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predicts data using the server-side model."""
        response = self._send_request("predict", X, None, **kwargs)
        if "error" in response:
            logging.error("[SML-CLIENT] Predict error: %s", response["error"])
            return np.zeros(len(X)), response

        predictions = response.get("preds", [])
        metrics = response.get("metrics", {})
        return np.atleast_1d(predictions), metrics

    def fit_predict(self, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sends experience data to server for BOTH prediction and training (streaming)."""
        response = self._send_request("fit_predict", X, y, **kwargs)
        if "error" in response:
            logging.error("[SML-CLIENT] FitPredict error: %s", response["error"])
            return np.zeros(len(X)), response

        predictions = response.get("preds", [])
        metrics = response.get("metrics", {})
        return np.array(predictions), metrics

    def map_to_timestamps(
        self, channel_data: Any, anomalies: List[Tuple[int, int]]
    ) -> List[Tuple[Any, Any]]:
        """Delegates mapping to the internal base_classifier if present."""
        if self.base_classifier is not None and hasattr(self.base_classifier, "map_to_timestamps"):
            return self.base_classifier.map_to_timestamps(channel_data, anomalies)
            
        has_timestamps = hasattr(channel_data, "timestamps") and channel_data.timestamps is not None and len(channel_data.timestamps) > 0
        
        limit = float('inf')
        if has_timestamps:
            limit = len(channel_data.timestamps)
        elif hasattr(channel_data, "data") and channel_data.data is not None:
            limit = len(channel_data.data)
        elif isinstance(channel_data, np.ndarray):
            limit = len(channel_data)
            
        offset = getattr(channel_data, "start_idx", 0)
        
        intervals = []
        for s, e in anomalies:
            if s >= limit: continue
            if e >= limit: e = limit - 1
            if has_timestamps:
                intervals.append((channel_data.timestamps[s], channel_data.timestamps[e]))
            else:
                intervals.append((s + offset, e + offset))
        return intervals

    def prepare_labels(self, channel_labels: Any) -> List[Tuple[int, int]]:
        """Delegates label preparation to the internal base_classifier if present."""
        if self.base_classifier is not None and hasattr(self.base_classifier, "prepare_labels"):
            return self.base_classifier.prepare_labels(channel_labels)
        return []

    def save(self, path: str) -> None:
        pass

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        raise NotImplementedError("Cannot load SMLClientClassifier directly.")
