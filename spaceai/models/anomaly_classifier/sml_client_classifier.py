import json
import logging
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

    def __init__(self, server_ip: str = "localhost", port: int = 5555, channel_id: str = "default") -> None:
        self.server_ip = server_ip
        self.port = port
        self.channel_id = channel_id
        
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
        
    def _send_request(self, action: str, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        start_idx = kwargs.get("start_idx", 0)
        end_idx = kwargs.get("end_idx", len(X))
        
        payload = {
            "action": action,
            "timesteps": (int(start_idx), int(end_idx)),
            "experience_data": X.tolist() if isinstance(X, np.ndarray) else list(X),
            "labels": y.tolist() if y is not None else [],
        }
        
        channel_bytes = self.channel_id.encode("utf-8")
        self.socket.send_multipart([channel_bytes, json.dumps(payload).encode("utf-8")])
        resp_frames = self.socket.recv_multipart()
        
        if len(resp_frames) != 2:
            logging.warning("Invalid response from server.")
            return {}
            
        resp_channel = resp_frames[0].decode("utf-8")
        if resp_channel != self.channel_id:
            logging.warning("Response channel mismatch: %s != %s", resp_channel, self.channel_id)
            return {}

        return json.loads(resp_frames[1].decode("utf-8"))

    def fit(self, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Fit is handled server-side independently."""
        response = self._send_request("fit", X, y, **kwargs)
        return response.get("metrics", {})

    def predict(self, X: Any, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predicts data using the server-side model."""
        response = self._send_request("predict", X, None, **kwargs)
        forward_predictions = response.get("forward_predictions", [])
        
        metrics = {"backward_predictions": response.get("backward_predictions", [])}
        if "metrics" in response:
            metrics.update(response["metrics"])
            
        return np.array(forward_predictions), metrics

    def fit_predict(self, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sends experience data to server for BOTH prediction and training (streaming)."""
        response = self._send_request("fit_predict", X, y, **kwargs)
        forward_predictions = response.get("forward_predictions", [])
        
        metrics = {"backward_predictions": response.get("backward_predictions", [])}
        if "metrics" in response:
            metrics.update(response["metrics"])
            
        return np.array(forward_predictions), metrics

    def map_to_timestamps(
        self, channel_data: Any, anomalies: List[Tuple[int, int]]
    ) -> List[Tuple[Any, Any]]:
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

    def save(self, path: str) -> None:
        pass

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        raise NotImplementedError("Cannot load SMLClientClassifier directly.")
