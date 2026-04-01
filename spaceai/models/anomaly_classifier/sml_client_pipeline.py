import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zmq

from spaceai.models.anomaly_classifier.anomaly_classifier import AnomalyDetectionPipeline
from spaceai.benchmark.callbacks import CallbackHandler

import pickle 

class SMLClientPipeline(AnomalyDetectionPipeline):
    """
    Client proxy for a remote AnomalyDetectionPipeline.
    Delegates all processing (splitting, FE, fit, predict) to the Leopard DPU.
    """

    def __init__(
        self,
        local_pipeline: AnomalyDetectionPipeline,
        server_ip: str = "127.0.0.1",
        port: int = 5556,
        channel_id: str = "default",
        callback_handler: Optional[CallbackHandler] = None,
        eval_perc: Optional[float] = None
    ) -> None:
        super().__init__(steps=[], callback_handler=callback_handler, eval_perc=eval_perc)
        self.local_pipeline = local_pipeline
        self.server_ip = server_ip
        self.port = port
        self.channel_id = channel_id
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        
        # 1. Aggiunto il LINGER per evitare freeze del PC
        self.socket.setsockopt(zmq.LINGER, 0) 
        # 2. Timeout Ricezione: aumentato a 1 ORA (3600 secondi) per test pesanti sul Leopard
        self.socket.setsockopt(zmq.RCVTIMEO, 3600000)
        self.socket.connect(f"tcp://{self.server_ip}:{self.port}")

    def _send_task(
        self, action: str, channel_data, channel_labels=None
    ) -> List[Any]:
        """Sends a task to the server and returns the result."""
        try:
            channel_bytes = self.channel_id.encode("utf-8")
            
            # 1. Temporarily strip unpickleable components from local_pipeline
            original_handler = self.local_pipeline.callback_handler
            self.local_pipeline.callback_handler = None
            # 1. Prepare Task Payload
            # Clear callback_handler to avoid pickling RLock
            for _, step in self.local_pipeline.steps:
                if hasattr(step, 'callback_handler'):
                    step.callback_handler = None
            
            # Ensure NASA objects have window_size set to avoid unpickling error if it calls __getitem__
            # though in SML mode we typically use raw data on the server, 
            # if we send the object we must ensure it's in a safe state.
            if hasattr(channel_data, 'window_size') and channel_data.window_size is None:
                # Use a safe default if not set
                channel_data.window_size = getattr(self.local_pipeline, 'window_size', 100)

            task = {
                "action": action,
                "channel_id": self.channel_id,
                "pipeline": self.local_pipeline,
                "experience_data": channel_data,
                "channel_labels": channel_labels,
            }
            
            payload_bytes = pickle.dumps(task)
            
            # 2. Restore local state
            self.local_pipeline.callback_handler = original_handler
            
            self.socket.send_multipart([channel_bytes, payload_bytes])
            
            resp_frames = self.socket.recv_multipart()
            
            # 2. Corretto frames -> resp_frames
            if len(resp_frames) != 2:
                print(f"DEBUG: Server returned malformed multipart message. Expected 2 frames, got {len(resp_frames)}.")
                logging.error("Server returned malformed multipart message.")
                return []

            # 3. Decodifichiamo i bytes in stringa PRIMA di confrontarli
            resp_channel = resp_frames[0].decode("utf-8")
            if resp_channel != self.channel_id:
                print(f"DEBUG: Response channel mismatch: {resp_channel} != {self.channel_id}")
                logging.warning("Response channel mismatch: %s != %s", resp_channel, self.channel_id)
                return []
            
            # 4. Spacchettiamo il dizionario con Pickle PRIMA di cercare gli errori
            payload = pickle.loads(resp_frames[1])
            print(f"payload: {payload}")
            # 5. Ora usiamo SOLO la variabile 'payload' (response non esiste!)
            if "error" in payload:
                print(f"DEBUG: Server returned error: {payload['error']}")
                logging.error("Server error: %s", payload["error"])
                return []
            
            if "classifier" in payload:
                self.local_pipeline = payload["classifier"]
            

            if "msgs" in payload:
                print(payload["msgs"])
                return payload["msgs"]

            print("DEBUG: Server response payload missing 'msgs' key.")
            return []
            
        except Exception as e:
            print(f"DEBUG: SML Client Communication Error: {e}")
            logging.error("Failed to communicate with SML server: %s", e)
            return []

    def fit(
        self,
        channel_data: Any,
        channel_labels: Optional[np.ndarray] = None,
        results_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Offload fit to the server."""
        msgs = self._send_task("fit", channel_data, channel_labels)
        if not msgs:
            logging.error("SML Fit Failed: Empty message list returned from server.")
            return {"error": "Server error or timeout during fit"}
            
        return {**msgs[0].results, **msgs[0].metadata}

    def predict(
        self, channel_data: Any, results_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Offload predict to the server."""
        msgs = self._send_task("predict", channel_data, None) 
        if not msgs:
            logging.error("SML Predict Failed: Empty message list returned from server.")
            return np.array([]), {"error": "Server error or timeout during predict"}

        return np.array(msgs[0].data), msgs[0].results

    def prepare_labels(self, channel_data: Any, results: Optional[Dict[str, Any]] = None) -> List[Tuple[int, int]]:
        """
        Usually prepare_labels is called before test.
        In SMLClient mode, we can either do it locally (if we have the local_pipeline)
        or offload. Doing it locally is faster as it doesn't involve the network.
        """
        return self.local_pipeline.prepare_labels(channel_data)

    def map_to_timestamps(
        self, channel_data: Any, anomalies: List[Tuple[int, int]], results: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, Any]]:
        """Map anomalies back to timestamps locally."""
        return self.local_pipeline.map_to_timestamps(channel_data, anomalies, results=results)

    def save(self, path: str) -> None:
        """Save the underlying local pipeline state."""
        self.local_pipeline.save(path)
