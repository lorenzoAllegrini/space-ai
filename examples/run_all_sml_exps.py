"""Run all SML remote experiments script."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import logging

from examples.run_sml_exp import (
    DATASET_LIST,
    DPMM_MODE,
    DPMM_MODEL_TYPE,
    MODEL_LIST,
    parse_exp_args,
    run_exp,
)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Batch runner for unified SML remote experiments")
    arg_parser.add_argument("--server_ip", required=True, help="IP of the SML server")
    arg_parser.add_argument("--port", type=int, default=5555, help="Port of the SML server")
    arg_parser.add_argument("--base_dir", required=True)
    arg_parser.add_argument("--n-workers", type=int, default=1)
    arg_parser.add_argument("--segmentator", action="store_true")
    arg_parser.add_argument("--feature-extractor", type=str, default="none")
    arg_parser.add_argument("--n-kernels", type=int)
    arg_parser.add_argument("--output-dir", type=str, default="experiments")
    arg_parser.add_argument("--datasets", type=eval, default=f"{DATASET_LIST}")
    arg_parser.add_argument("--models", type=eval, default=f"{MODEL_LIST}")
    arg_parser.add_argument("--dpmm-types", type=eval, default=f"{DPMM_MODEL_TYPE}")
    arg_parser.add_argument("--dpmm-modes", type=eval, default=f"{DPMM_MODE}")
    arg_parser.add_argument("--window-size", type=int, default=50)
    arg_parser.add_argument("--step-size", type=int, default=50)
    arg_parser.add_argument("--skip-existing", action="store_true", help="Skip already executed experiments")

    args, other_exp_args = arg_parser.parse_known_args()

    n_kernels = args.n_kernels
    dataset_list = args.datasets
    model_list = args.models
    dpmm_types_list = args.dpmm_types
    dpmm_modes_list = args.dpmm_modes
    segmentator = args.segmentator
    feature_extractor = args.feature_extractor
    window_size = args.window_size
    step_size = args.step_size

    exp_dir = f"batch_sml_{segmentator}_{feature_extractor}"
    segmentator_args = f" --feature-extractor {feature_extractor}"
    if segmentator:
        segmentator_args += " --segmentator"
        if window_size:
            segmentator_args += f" --window-size {window_size}"
        if step_size:
            segmentator_args += f" --step-size {step_size}"

    if feature_extractor == "rocket":
        if n_kernels:
            exp_dir += f"_nkernels{n_kernels}"
            segmentator_args += f" --n-kernel {n_kernels}"

    if len(other_exp_args) > 0:
        exp_dir += "_" + "_".join(
            sorted(
                [
                    other_exp_args[i][2:].replace("_", "") + (other_exp_args[i + 1] if i+1 < len(other_exp_args) else "")
                    for i in range(0, len(other_exp_args), 2)
                ]
            )
        )

    exp_path = os.path.join(args.output_dir, exp_dir)

    command_args_list = []
    for dataset in dataset_list:
        for model in model_list:
            common_args = (
                f"--server_ip {args.server_ip} --port {args.port} "
                f"--base_dir {args.base_dir} --exp-dir {exp_path} "
                f"--dataset {dataset} --model {model}"
            )

            if model == "rockad":
                if args.skip_existing:
                    run_id = f"sml_{dataset}_{model}"
                    if os.path.exists(os.path.join(exp_path, run_id, "results.csv")):
                        print(f"Skipping {run_id} as it already exists.")
                        continue
                # Special handling for rockad if needed, similar to run_all_exps_1.py
                command_args_list.append(
                    common_args + f" --n-kernel {n_kernels} --feature-extractor rocket"
                )
            elif model == "dpmm":
                for dpmm_type in dpmm_types_list:
                    for dpmm_mode in dpmm_modes_list:
                        if args.skip_existing:
                            run_id = f"sml_{dataset}_{model}_{dpmm_type}_{dpmm_mode}"
                            if os.path.exists(os.path.join(exp_path, run_id, "results.csv")):
                                print(f"Skipping {run_id} as it already exists.")
                                continue
                        dpmm_args = (
                            f" --dpmm-type {dpmm_type} --dpmm-mode {dpmm_mode}"
                        )
                        command_args_list.append(
                            common_args + segmentator_args + dpmm_args
                        )
            else:
                if args.skip_existing:
                    run_id = f"sml_{dataset}_{model}"
                    if os.path.exists(os.path.join(exp_path, run_id, "results.csv")):
                        print(f"Skipping {run_id} as it already exists.")
                        continue
                command_args_list.append(common_args + segmentator_args)

    print(f"Starting {len(command_args_list)} experiments with {args.n_workers} workers...")

    pool = ProcessPoolExecutor(max_workers=args.n_workers)
    FINISHED = 0

    def get_callback(cmd_args):
        """Get callback function."""
        def callback(future):
            global FINISHED
            FINISHED += 1
            if future.exception():
                print(
                    f"{FINISHED}/{len(command_args_list)}\t"
                    f"Error in {cmd_args}: {future.exception()}"
                )
            else:
                print(
                    f"{FINISHED}/{len(command_args_list)}\t"
                    f"Success: {cmd_args}"
                )
        return callback

    futures = []
    for C_ARGS in command_args_list:
        exp_args, _ = parse_exp_args(C_ARGS.split(" "))
        f = pool.submit(run_exp, exp_args, other_args=other_exp_args)
        f.add_done_callback(get_callback(C_ARGS))
        futures.append(f)

    pool.shutdown(wait=True)
    print("All batch SML experiments completed.")
