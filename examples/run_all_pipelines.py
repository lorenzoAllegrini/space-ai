"""Run all pipeline combinations experiment module.

Reads a config file (same format as run_pipeline.py) where any parameter
value can be a list of alternatives. This script generates the cartesian
product of all such lists and invokes run_pipeline.py once per combination.

Example config:
  base_classifier_params:
    num_iterations: [100, 500]
    dynamic_scaling: [true, false]

This will call run_pipeline.py for all 4 combinations.
"""

import argparse
import itertools
import logging
import subprocess
import sys
import yaml
import copy

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Path to run_pipeline.py relative to this script
PIPELINE_SCRIPT = Path(__file__).parent / "run_pipeline.py"


def _flatten_lists(config: dict, parent_key: str = "") -> dict:
    """Recursively collect all keys whose value is a list of scalars (sweep params)."""
    sweep_params = {}
    EXCLUDED_KEYS = {"channels"}
    
    for k, v in config.items():
        if k in EXCLUDED_KEYS:
            continue
            
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, list) and len(v) > 0 and not isinstance(v[0], dict):
            sweep_params[full_key] = v
        elif isinstance(v, dict):
            sweep_params.update(_flatten_lists(v, parent_key=full_key))
    return sweep_params



def _set_nested(config: dict, dotted_key: str, value):
    """Set a value in a nested dict using a dot-separated key path."""
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _get_nested(config: dict, dotted_key: str):
    """Get a value from a nested dict using a dot-separated key path."""
    keys = dotted_key.split(".")
    d = config
    for k in keys:
        d = d[k]
    return d


def build_combinations(config: dict):
    """
    Find all list-valued scalar parameters and return:
      - A list of (key, values) pairs for the swept params.
      - An iterable of dicts, one per combination, with single values substituted.
    """
    sweep_params = _flatten_lists(config)

    if not sweep_params:
        # No sweep params; single combination equal to config itself
        logger.info("No sweep parameters found. Running a single experiment.")
        yield config
        return

    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    logger.info(f"Sweep parameters detected ({len(keys)}):")
    for k, vals in zip(keys, value_lists):
        logger.info(f"  {k}: {vals}")

    total = 1
    for v in value_lists:
        total *= len(v)
    logger.info(f"Total combinations: {total}")

    for combo in itertools.product(*value_lists):
        cfg_copy = copy.deepcopy(config)
        for k, v in zip(keys, combo):
            _set_nested(cfg_copy, k, v)
        yield cfg_copy


def write_temp_config(config: dict, tmp_path: Path) -> Path:
    """Write a config dict to a temporary YAML file and return its path."""
    with open(tmp_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return tmp_path


def run_pipeline_for_config(config: dict, combo_index: int, total: int, extra_args: list):
    """Write a temporary config and invoke run_pipeline.py as a subprocess."""
    tmp_config_path = Path("/tmp") / f"_run_all_pipelines_combo_{combo_index}.yaml"
    write_temp_config(config, tmp_config_path)

    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--config", str(tmp_config_path),
    ] + extra_args

    logger.info(f"\n{'='*60}")
    logger.info(f"Combination {combo_index}/{total}")
    logger.info(f"Config: {tmp_config_path}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.warning(
            f"run_pipeline.py exited with code {result.returncode} "
            f"for combination {combo_index}/{total}. Continuing to next combination."
        )
    else:
        logger.info(f"Combination {combo_index}/{total} completed successfully.")

    return result.returncode


def get_predicted_run_id(config: dict) -> str:
    """
    Replicates the run_id generation logic from run_pipeline.py.
    This allows the script to check for existing results before running.
    """
    base_params = config.get('base_classifier_params', {})
    ds = "T" if base_params.get('dynamic_scaling', False) else "F"
    ch = "T" if config.get('challenge', False) else "F"
    
    # Extract sweep hyperparameters
    lr = base_params.get('lr', 0.1)
    n_cl = base_params.get('n_clusters', 100)
    adp = base_params.get('alpha_dp', 3.0)
    vp = base_params.get('var_prior', 3.0)
    vps = base_params.get('var_prior_strength', 1.0)
    mps = base_params.get('mu_prior_strength', 0.001)
    q = base_params.get('quantile', 0.001)
    
    dp = config.get('detector_params', {})
    alpha = dp.get('alpha', 0.05)
    pot = dp.get('pot_percentile', 0.0)
    p = dp.get('p', 0.0)
    
    wrapper_params = config.get('wrapper_params', {})
    eval_perc = config.get('eval_perc', config.get('perc_eval', wrapper_params.get('eval_perc', 0.0)))
    
    # Core identifying components
    fe = config.get('feature_extractor', 'none')
    ds_name = config.get('dataset', 'esa')
    model = config.get('model', 'dpmm')
    detector = config.get('detector', 'threshold')
    
    # Construct ID exactly as run_pipeline.py does
    run_id = f"pipeline_{fe}_{ds_name}_{model}_{detector}_lr{lr}_nc{n_cl}_adp{adp}_vp{vp}_vps{vps}_mps{mps}_q{q}_al{alpha}_po{pot}_p{p}_ds{ds}_ch{ch}_ep{eval_perc}"
    return run_id



def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all combinations of hyperparameters defined as lists in a "
            "YAML config file, calling run_pipeline.py for each combination."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (same format as run_pipeline.py, "
             "but list values are treated as sweep axes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If set, print all combinations but do NOT actually run run_pipeline.py.",
    )
    parser.add_argument(
        "--skip-existent",
        action="store_true",
        default=False,
        help="If set, skip combinations that already have a results.csv file.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=False,
        help="If set, stop immediately when a combination returns a non-zero exit code.",
    )

    # Capture any extra args to forward to run_pipeline.py
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def main():
    args, extra_args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f) or {}

    # Collect all combinations
    combinations = list(build_combinations(base_config))
    total = len(combinations)

    if args.dry_run:
        logger.info(f"\nDry run: {total} combination(s) would be run.")
        for i, cfg in enumerate(combinations, 1):
            logger.info(f"\n--- Combination {i}/{total} ---")
            logger.info(yaml.dump(cfg, default_flow_style=False))
        return

    failed = []
    exp_dir = Path(base_config.get("exp_dir", "experiments"))
    
    sweep_params = _flatten_lists(base_config)
    swept_keys = list(sweep_params.keys())
    only_features_swept = (len(swept_keys) == 1 and swept_keys[0] == "feature_extraction_params.selected_features")

    base_extra_args = list(extra_args)
    if args.skip_existent:
        base_extra_args.append("--skip-existent")

    for i, cfg in enumerate(combinations, 1):
        # 1. Determine run_id for this combination
        if only_features_swept:
            fe = cfg.get('feature_extractor', 'none')
            ds_name = cfg.get('dataset', 'esa')
            model = cfg.get('model', 'dpmm')
            detector = cfg.get('detector', 'threshold')
            features = _get_nested(cfg, "feature_extraction_params.selected_features")
            short_features = "_".join(features) if features else "all"
            combo_run_id = f"pipeline_{fe}_{ds_name}_{model}_{detector}_{short_features}"
            
            # Send this run_id explicitly to the subprocess
            current_extra_args = base_extra_args + ["--run-id", combo_run_id]
        else:
            combo_run_id = get_predicted_run_id(cfg)
            current_extra_args = base_extra_args

        # 2. Skip existent
        if args.skip_existent:
            results_path = exp_dir / combo_run_id / "results.csv"
            if results_path.exists():
                logger.info(f"Skipping combination {i}/{total} as results already exist at {results_path}")
                continue

        # 3. Execute
        rc = run_pipeline_for_config(cfg, combo_index=i, total=total, extra_args=current_extra_args)

        if rc != 0:
            failed.append(i)
            if args.stop_on_error:
                logger.error(f"Stopping due to error in combination {i}/{total}.")
                sys.exit(rc)

    logger.info(f"\n{'='*60}")
    logger.info(f"All combinations finished. {total - len(failed)}/{total} succeeded.")
    if failed:
        logger.warning(f"Failed combinations: {failed}")
        sys.exit(1)
    else:
        logger.info("All combinations completed successfully.")


if __name__ == "__main__":
    main()
