import pickle
import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zmq
import pandas as pd

from spaceai.models.anomaly_classifier.anomaly_classifier import AnomalyClassifier

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
            ws = metrics.get('_fitted_window_size', None)
            ss = metrics.get('_fitted_step_size', None)
            if ws is not None:
                self.base_classifier.ts_splitter.window_size = ws
            if ss is not None:
                self.base_classifier.ts_splitter.step_size = ss
        
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

    def step(self, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sends experience data to server for adaptive STEP (predict + update)."""
        response = self._send_request("step", X, y, **kwargs)
        if "error" in response:
            logging.error("[SML-CLIENT] Step error: %s", response["error"])
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
