"""Run all parameter combinations for model selection.

Reads a YAML config file where any parameter value can be a list of alternatives.
Generates the cartesian product of all such lists and invokes an experiment script 
(e.g., run_pipeline.py) once per combination.
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
from typing import Dict, Any, Generator, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def _flatten_lists(config: Dict[str, Any], parent_key: str = "") -> Dict[str, List[Any]]:
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

def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any):
    """Set a value in a nested dict using a dot-separated key path."""
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        if k not in d:
             d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def build_combinations(config: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Find all list-valued parameters and yield substituted config dicts."""
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
        combo_parts = []
        for k, v in zip(keys, combo):
            _set_nested(cfg_copy, k, v)
            short_k = k.split(".")[-1]
            if short_k == "type":
                combo_parts.append(str(v))
            else:
                combo_parts.append(f"{short_k}_{v}")
        
        original_run_id = cfg_copy.get("run_id", "")
        
        ch = cfg_copy.get("challenge", False)
        if "challenge" not in keys:
            prefix = "ch" + ("T" if ch else "F")
            if original_run_id:
                original_run_id += f"_{prefix}"
            else:
                original_run_id = prefix

        if original_run_id:
            cfg_copy["run_id"] = f"{original_run_id}_{'_'.join(combo_parts)}"
        else:
            cfg_copy["run_id"] = "_".join(combo_parts)
        
        yield cfg_copy

def write_temp_config(config: Dict[str, Any], tmp_path: Path) -> Path:
    """Write a config dict to a temporary YAML file."""
    with open(tmp_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return tmp_path

def run_script_for_config(script: str, config: Dict[str, Any], index: int, total: int, extra_args: List[str]):
    """Invoke the experiment script as a subprocess."""
    tmp_dir = Path("./.tmp_configs")
    tmp_dir.mkdir(exist_ok=True)
    tmp_config_path = tmp_dir / f"sweep_combo_{index}.yaml"
    write_temp_config(config, tmp_config_path)

    cmd = [
        sys.executable,
        script,
        "--config", str(tmp_config_path),
    ] + extra_args

    logger.info(f"\n{'='*60}")
    logger.info(f"Combination {index}/{total}: {config.get('run_id')}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.warning(f"Script exited with code {result.returncode} for combo {index}. Continuing...")
    else:
        logger.info(f"Combination {index}/{total} completed successfully.")

    return result.returncode

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-parameter sweep for model selection.")
    parser.add_argument("--config", type=str, required=True, help="Path to base YAML config.")
    parser.add_argument("--script", type=str, default="examples/run_pipeline.py", 
                        help="Experiment script to run (default: run_pipeline.py).")
    parser.add_argument("--dry-run", action="store_true", help="Print combinations without running.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop if a run fails.")
    
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

    combinations = list(build_combinations(base_config))
    total = len(combinations)

    if args.dry_run:
        logger.info(f"\nDry run: {total} combination(s) generated.")
        for i, cfg in enumerate(combinations, 1):
            logger.info(f"Combo {i}: {cfg.get('run_id')}")
        return

    failed = []
    for i, cfg in enumerate(combinations, 1):
        rc = run_script_for_config(args.script, cfg, i, total, extra_args)
        if rc != 0:
            failed.append(i)
            if args.stop_on_error:
                sys.exit(rc)

    logger.info(f"\nSweep finished. Success: {total - len(failed)}/{total}")
    if failed:
        logger.warning(f"Failed combos: {failed}")
        sys.exit(1)

if __name__ == "__main__":
    main()
