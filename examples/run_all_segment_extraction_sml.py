"""Run all SML pipeline combinations experiment module.

Reads a config file where any parameter value can be a list of alternatives.
This script generates the cartesian product of all such lists and invokes 
run_segment_extraction_sml.py once per combination.
"""

import argparse
import itertools
import logging
import subprocess
import sys
import yaml
import copy
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Path to the target script relative to this script
PIPELINE_SCRIPT = Path(__file__).parent / "run_segment_extraction_sml.py"


def _flatten_lists(config: dict, parent_key: str = "") -> dict:
    """Recursively collect all keys whose value is a list (sweep params)."""
    sweep_params = {}
    EXCLUDED_KEYS = {"channels"}
    
    for k, v in config.items():
        if k in EXCLUDED_KEYS:
            continue
            
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, list) and len(v) > 0:
            sweep_params[full_key] = v
        elif isinstance(v, dict):
            sweep_params.update(_flatten_lists(v, parent_key=full_key))
    return sweep_params


def _set_nested(config: dict, dotted_key: str, value):
    """Set a value in a nested dict using a dot-separated key path."""
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def build_combinations(config: dict):
    """
    Find all list-valued parameters and return an iterable of dicts.
    """
    sweep_params = _flatten_lists(config)

    if not sweep_params:
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


def get_predicted_run_id(config: dict) -> str:
    """
    Replicates the run_id generation logic from run_segment_extraction_sml.py.
    """
    model = config.get('model', 'iforest')
    detector = config.get('detector', 'threshold')
    window_size = config.get('window_size', 50)
    step_size = config.get('step_size', 50)
    base_params = config.get('base_classifier_params', {})
    
    run_id = f"SML_{model}"
    
    # Model specific type/mode
    for attr in ['type', 'mode']:
        val = config.get(f"{model}_{attr}", config.get(attr, None))
        if val: run_id += f"_{val}"
            
    run_id += f"_{detector}_w{window_size}_s{step_size}"
    
    if base_params.get('dynamic_scaling', False):
        run_id += "_ds"
        
    return run_id


def run_pipeline_for_config(config: dict, combo_index: int, total: int, extra_args: list):
    """Write a temporary config and invoke the target script as a subprocess."""
    tmp_config_dir = Path("tmp_configs")
    tmp_config_dir.mkdir(exist_ok=True)
    tmp_config_path = tmp_config_dir / f"grid_combo_{combo_index}.yaml"
    
    with open(tmp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

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
    
    # Cleanup temp config
    if tmp_config_path.exists():
        os.remove(tmp_config_path)

    if result.returncode != 0:
        logger.warning(
            f"Script exited with code {result.returncode} "
            f"for combination {combo_index}/{total}."
        )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Grid Search Runner for SML Experiments")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config")
    parser.add_argument("--skip-existent", action="store_true", help="Skip existing results")
    parser.add_argument("--dry-run", action="store_true", help="Print combinations only")
    args, extra_args = parser.parse_known_args()

    if not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f) or {}

    combinations = list(build_combinations(base_config))
    total = len(combinations)
    exp_dir = Path(base_config.get("exp_dir", "experiments_sml"))

    if args.dry_run:
        logger.info(f"Dry run: {total} combinations planned.")
        return

    failed = []
    for i, cfg in enumerate(combinations, 1):
        run_id = get_predicted_run_id(cfg)
        
        if args.skip_existent:
            results_path = exp_dir / run_id / "results.csv"
            if results_path.exists():
                logger.info(f"Skipping {run_id} (already exists)")
                continue

        rc = run_pipeline_for_config(cfg, i, total, extra_args)
        if rc != 0:
            failed.append(i)

    logger.info(f"Done. {total - len(failed)}/{total} succeeded.")
    if failed: sys.exit(1)


if __name__ == "__main__":
    main()
