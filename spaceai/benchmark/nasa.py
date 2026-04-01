"""NASA benchmark module for anomaly detection on NASA telemetry data."""

from __future__ import annotations

from typing import Tuple, List, Optional
import pandas as pd

from spaceai.data import NASA

from .benchmark import Benchmark


class NASABenchmark(Benchmark):
    """Benchmark for NASA telemetry anomaly detection dataset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def channels(self) -> List[str]:
        """Get the default list of channels for the benchmark."""
        return NASA.channel_ids

    def get_global_temporal_params(self, channels: List[str]) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """Get global start time and period for event-level aggregation."""
        return None, None

    def compute_global_event_metrics(
        self,
        channels: Optional[List[str]] = None,
        time_aware: bool = True,
        recover_state: bool = True,
        **extra_metrics,
    ) -> dict:
        """Override to compute global NASA metrics by aggregating TP, FP, FN, and TN across channels."""
        import os
        import logging
        import pandas as pd
        
        if self.all_metadata:
             pd.DataFrame.from_records(self.all_metadata).to_csv(
                os.path.join(self.run_dir, "metadata.csv"), index=False
            )

        if self.challenge and self.challenge_pointwise_predictions:
            logging.info("Generating aggregate challenge prediction files...")
            agg_df = pd.DataFrame(self.challenge_pointwise_predictions)
            if self.challenge_timestamps is not None:
                agg_df.insert(0, "UT", self.challenge_timestamps)
            
            agg_channels_path = os.path.join(self.run_dir, "challenge_channels_predictions.csv")
            agg_df.to_csv(agg_channels_path, index=False)
            
            channel_cols = [c for c in agg_df.columns if c != "UT"]
            global_prediction = (agg_df[channel_cols].sum(axis=1) > 0).astype(int)
            
            submission_df = pd.DataFrame({"prediction": global_prediction})
            if self.challenge_timestamps is not None:
                submission_df.insert(0, "UT", self.challenge_timestamps)
            
            submission_path = os.path.join(self.run_dir, "challenge_submission.csv")
            submission_df.to_csv(submission_path, index=False)
            
        # Aggregate logic
        if not self.all_results:
            return {}
            
        df = pd.DataFrame.from_records(self.all_results)
        df = df[df["channel_id"] != "GLOBAL_EVENT_LEVEL"]
        
        tp = df["true_positives"].sum()
        fp = df["false_positives"].sum()
        fn = df["false_negatives"].sum()
        tot_neg = df["test_negatives"].sum() if "test_negatives" in df.columns else 0
        det_neg = df["detected_negatives"].sum() if "detected_negatives" in df.columns else 0
        
        tnr = (det_neg / tot_neg) if tot_neg > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec_corr = precision * tnr
        
        f1 = (2 * prec_corr * recall / (prec_corr + recall)) if (prec_corr + recall) > 0 else 0.0
        f05 = ((1 + 0.5**2) * prec_corr * recall / (0.5**2 * prec_corr + recall)) if (0.5**2 * prec_corr + recall) > 0 else 0.0
        
        global_results = {
            "channel_id": "GLOBAL_EVENT_LEVEL",
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "detected_negatives": det_neg,
            "test_negatives": tot_neg,
            "tnr": tnr,
            "precision": precision,
            "precision_corrected": prec_corr,
            "recall": recall,
            "f1": f1,
            "corrected_f1": f1,
            "f0.5": f05,
            "corrected_f0.5": f05,
            "n_anomalies": df["n_anomalies"].sum() if "n_anomalies" in df.columns else 0,
            "n_detected": df["n_detected"].sum() if "n_detected" in df.columns else 0,
            "train_time": df.get("train_time", pd.Series([0.0])).mean(),
            "predict_time": df.get("predict_time", pd.Series([0.0])).mean()
        }
        
        for prefix in ["val_", "train_"]:
            if f"{prefix}true_positives" in df.columns:
                p_tp = df[f"{prefix}true_positives"].sum()
                p_fp = df[f"{prefix}false_positives"].sum()
                p_fn = df[f"{prefix}false_negatives"].sum()
                p_tot_neg = df[f"{prefix}test_negatives"].sum() if f"{prefix}test_negatives" in df.columns else 0
                p_det_neg = df[f"{prefix}detected_negatives"].sum() if f"{prefix}detected_negatives" in df.columns else 0
                
                p_tnr = (p_det_neg / p_tot_neg) if p_tot_neg > 0 else 0.0
                p_precision = (p_tp / (p_tp + p_fp)) if (p_tp + p_fp) > 0 else 0.0
                p_recall = (p_tp / (p_tp + p_fn)) if (p_tp + p_fn) > 0 else 0.0
                p_prec_corr = p_precision * p_tnr
                
                p_f1 = (2 * p_prec_corr * p_recall / (p_prec_corr + p_recall)) if (p_prec_corr + p_recall) > 0 else 0.0
                p_f05 = ((1 + 0.5**2) * p_prec_corr * p_recall / (0.5**2 * p_prec_corr + p_recall)) if (0.5**2 * p_prec_corr + p_recall) > 0 else 0.0
                
                global_results.update({
                    f"{prefix}true_positives": p_tp,
                    f"{prefix}false_positives": p_fp,
                    f"{prefix}false_negatives": p_fn,
                    f"{prefix}detected_negatives": p_det_neg,
                    f"{prefix}test_negatives": p_tot_neg,
                    f"{prefix}tnr": p_tnr,
                    f"{prefix}precision": p_precision,
                    f"{prefix}precision_corrected": p_prec_corr,
                    f"{prefix}recall": p_recall,
                    f"{prefix}f1": p_f1,
                    f"{prefix}corrected_f1": p_f1,
                    f"{prefix}f0.5": p_f05,
                    f"{prefix}corrected_f0.5": p_f05,
                    f"{prefix}n_anomalies": df[f"{prefix}n_anomalies"].sum() if f"{prefix}n_anomalies" in df.columns else 0,
                    f"{prefix}n_detected": df[f"{prefix}n_detected"].sum() if f"{prefix}n_detected" in df.columns else 0,
                })
        
        self.all_results.append(global_results)
        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )
            
        return global_results

    def load_channel(
        self, channel_id: str, train: bool = True, overlapping_train: bool = True, **kwargs
    ) -> NASA:
        """Load the training or testing dataset for a given channel."""
        if train:
            return NASA(
                root=self.data_root,
                channel_id=channel_id,
                mode="prediction",
                overlapping=overlapping_train,
                **kwargs
            )
        else:
            return NASA(
                root=self.data_root,
                channel_id=channel_id,
                mode="anomaly",
                overlapping=False,
                train=False,
                drop_last=False,
                **kwargs
            )
