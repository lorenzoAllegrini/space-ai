"""Run SML E2E experiment module."""

import argparse
import warnings
from datetime import datetime
import logging
import yaml
import numpy as np

from spaceai.preprocessing import (
    TimeSeriesSplitter,
    get_feature_extractor,
)

from utils.dataset_exp import (
    get_dataset_benchmark,
)
from utils.model_creators import (
    create_classifier,
)
from utils.reproducibility import set_seed
from spaceai.benchmark.callbacks import SystemMonitorCallback, CallbackHandler
from spaceai.models.anomaly_classifier.rolling_window_classifier import RollingWindowClassifier
from spaceai.models.anomaly_classifier.sml_client_classifier import SMLClientClassifier
from spaceai.benchmark import ESABenchmark
warnings.simplefilter("ignore", FutureWarning)

DATASET_LIST = ["ops", "nasa", "esa"]
MODEL_LIST = ["ocsvm", "xgboost", "ridge_regression", "dpmm", "iforest", "pca", "knn", "lof", "pyod_ocsvm", "ecod", "copod", "cblof", "hbos", "ndpm"]
FEATURE_EXTRACTOR_LIST = ["none", "base_statistics", "rocket"]

def parse_sml_args(str_args=None):
    """Parse args for SML experiment."""
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None)
    args, remaining_argv = conf_parser.parse_known_args(str_args)

    defaults = {
        "exp_dir": "experiments_sml",
        "mission_id": 1,
        "feature_extractor": "none",
        "window_size": 50,
        "step_size": 50,
        "seed": 42,
        "server_ip": "localhost",
        "server_port": 5555
    }

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config: defaults.update(yaml_config)

    parser = argparse.ArgumentParser(description="SML Bridge Experiment", parents=[conf_parser])
    parser.add_argument("--base_dir", help="Base directory for the dataset")
    parser.add_argument("--exp-dir", help="Experiments output directory")
    parser.add_argument("--dataset", choices=DATASET_LIST)
    parser.add_argument("--mission-id", type=int)
    parser.add_argument("--model", choices=MODEL_LIST)
    parser.add_argument("--feature-extractor", choices=FEATURE_EXTRACTOR_LIST)
    parser.add_argument("--channels", type=str, nargs="+")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--detector", choices=["threshold", "molookde", "none"], default="threshold")
    parser.add_argument("--server-ip", type=str)
    parser.add_argument("--server-port", "--port", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id", type=str, help="Override run_id")
    
    # DPMM specific
    parser.add_argument("--dpmm-type", choices=["full", "unit", "diagonal"], help="DPMM covariance type")
    parser.add_argument("--dpmm-mode", choices=["likelihood_threshold", "score_threshold"], help="DPMM anomaly detection mode")

    parser.set_defaults(**defaults)
    return parser.parse_known_args(remaining_argv)

def run_sml_exp():
    args, other_args = parse_sml_args()
    set_seed(args.seed)

    from spaceai.models.anomaly import ThresholdDetector, MoLooKDEDetector
    detector_params = getattr(args, 'detector_params', {})
    if args.detector == "threshold":
        detector = ThresholdDetector(**{**dict(threshold=0.9), **detector_params})
    elif args.detector == "molookde":
        detector = MoLooKDEDetector(**{**dict(alpha=0.001), **detector_params})
    else:
        detector = None
    
    # Inject DPMM params into base_classifier_params if needed
    if args.model == "dpmm":
        base_params = getattr(args, 'base_classifier_params', {})
        if args.dpmm_type: base_params['dpmm_type'] = args.dpmm_type
        if args.dpmm_mode: base_params['dpmm_mode'] = args.dpmm_mode
        args.base_classifier_params = base_params

    handler = CallbackHandler([SystemMonitorCallback()], call_every_ms=100)

    # Generate a more descriptive run_id including model type/mode
    if getattr(args, 'run_id', None):
        run_id = args.run_id
    else:
        run_id = f"SML_{args.model}"
        
        # Add model-specific type/mode if present
        for attr in ['type', 'mode']:
            val = getattr(args, f"{args.model}_{attr}", getattr(args, attr, None))
            if val: run_id += f"_{val}"
                
        run_id += f"_{args.detector}_w{args.window_size}_s{args.step_size}"
        
        # Check for dynamic scaling in classifier params
        if getattr(args, 'base_classifier_params', {}).get('dynamic_scaling', False):
            run_id += "_ds"
    benchmark = get_dataset_benchmark(
        dataset_name=args.dataset,
        data_path=args.base_dir,
        exp_dir=args.exp_dir,
        run_id=run_id,
        mission_id=args.mission_id,
    )
    
    channels = benchmark.channels if args.channels is None else args.channels
    wrapper_params = getattr(args, 'wrapper_params', {})
    eval_perc = getattr(args, 'eval_perc', wrapper_params.get('eval_perc', None))

    for channel_name in channels:
        logging.info("--- Starting SML Pipeline for Channel %s ---", channel_name)
        
        _minw = getattr(args, 'min_window', None) or 10
        _maxw = getattr(args, 'max_window', None) or 300
        _pss = getattr(args, 'perc_step_size', None) or 1.0
        _fe_params = getattr(args, 'feature_extraction_params', getattr(args, 'fe_params', {}))
        
        print(f"\n[SML-CLIENT-FACTORY] === Pipeline Construction Parameters ===", flush=True)
        print(f"[SML-CLIENT-FACTORY] window_size={args.window_size}, step_size={args.step_size}", flush=True)
        print(f"[SML-CLIENT-FACTORY] min_window={_minw}, max_window={_maxw}", flush=True)
        print(f"[SML-CLIENT-FACTORY] perc_step_size={_pss}", flush=True)
        print(f"[SML-CLIENT-FACTORY] eval_perc={eval_perc}", flush=True)
        print(f"[SML-CLIENT-FACTORY] detector={args.detector}", flush=True)
        print(f"[SML-CLIENT-FACTORY] feature_extractor={args.feature_extractor}", flush=True)
        print(f"[SML-CLIENT-FACTORY] fe_params={_fe_params}", flush=True)
        print(f"[SML-CLIENT-FACTORY] model={args.model}", flush=True)
        print(f"[SML-CLIENT-FACTORY] base_classifier_params={getattr(args, 'base_classifier_params', {})}", flush=True)
        print(f"[SML-CLIENT-FACTORY] seed={getattr(args, 'seed', 42)}", flush=True)
        print(f"[SML-CLIENT-FACTORY] ==========================================\n", flush=True)

        ts_splitter = TimeSeriesSplitter(
            window_size=args.window_size,
            step_size=args.step_size,
            min_window=_minw,
            max_window=_maxw,
            perc_step_size=_pss,
        )
        
        feature_extractor = get_feature_extractor(
            args.feature_extractor,
            window_size=args.window_size,
            stride=args.step_size,
            n_kernel=getattr(args, 'n_kernel', None),
            **_fe_params
        )

        base_classifier, is_supervised = create_classifier(args, other_args)

        # Check if the returned classifier is a self-contained sequence model (like Telemanom)
        from spaceai.models.anomaly_classifier.telemanom_classifier import SequenceModelClassifier
        if isinstance(base_classifier, SequenceModelClassifier):
            rolling_window_pipeline = base_classifier
            logging.info("[CLIENT-FACTORY] SequenceModelClassifier detected. Bypassing RollingWindow wrapping.")
        else:
            # 1. Creiamo la pipeline locale (RollingWindowClassifier)
            rolling_window_pipeline = RollingWindowClassifier(
                base_classifier=base_classifier,
                supervised_classifier=is_supervised,
                ts_splitter=ts_splitter,
                feature_extractor=feature_extractor,
                callback_handler=handler,
                detector=detector,
                eval_perc=eval_perc,
            )

        # 2. Avvolgiamo tutto nello SMLClientClassifier
        # Passiamo anche gli args per permettere al server di inizializzare la pipeline localmente
        sml_client = SMLClientClassifier(
            server_ip=args.server_ip,
            port=args.server_port,
            channel_id=channel_name,
            base_classifier=rolling_window_pipeline,
            args=(args, other_args) # Passiamo la "ricetta" completa
        )

        # Check for dataset-specific flags like use_telecommands
        dataset_kwargs = {}
        if hasattr(args, 'use_telecommands'):
            dataset_kwargs['use_telecommands'] = args.use_telecommands

        logging.info("[CLIENT] Requesting remote FIT for channel %s (via ARGS)...", channel_name)
        start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        fitted_client, fitting_metrics = benchmark.fit_channel(
            channel_id=channel_name,
            classifier=sml_client,
            **dataset_kwargs
        )
        
        logging.info("[CLIENT] Requesting remote TEST for channel %s...", channel_name)
        test_results = benchmark.test_channel(
            channel_id=channel_name,
            classifier=fitted_client,
            **dataset_kwargs
        )
        
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Merge all metrics for the final results.csv
        final_results = {
            "start_date": start_date,
            "end_date": end_date,
            **fitting_metrics,
            **test_results
        }
        
        # Ensure fit_time is explicitly set if fitting_metrics has it
        if "fit_time" not in final_results and "fitting_time" in final_results:
            final_results["fit_time"] = final_results["fitting_time"]

        print(f"Final Channel Results: {final_results}")
        
        # If the benchmark has a results list, update it (specific to some Benchmark implementations)
        if hasattr(benchmark, 'results'):
            # Find and update the entry for this channel
            for i, res in enumerate(benchmark.results):
                if res.get('channel_id') == channel_name:
                    benchmark.results[i].update(final_results)
                    break

    if isinstance(benchmark, ESABenchmark):
        results = benchmark.compute_global_event_metrics(channels=channels)
        print(f"Global ESA Metrics: {results}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [SML-E2E] %(message)s")
    run_sml_exp()
