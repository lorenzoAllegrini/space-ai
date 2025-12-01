"""Run all experiments script."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from .config import OMP_NUM_THREADS

os.environ["OMP_NUM_THREADS"] = f"{OMP_NUM_THREADS}"

from .run_exp import (  
    DATASET_LIST,
    DPMM_MODE,
    DPMM_MODEL_TYPE,
    MODEL_LIST,
    parse_exp_args,
    run_exp,
)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--base_dir", required=True)
    arg_parser.add_argument("--n-workers", type=int, default=1)
    arg_parser.add_argument("--segmentator", action="store_true")
    arg_parser.add_argument("--feature-extractor", type=str, default="none")
    arg_parser.add_argument("--n-kernels", type=int)
    arg_parser.add_argument("--output-dir", type=str, default="")
    arg_parser.add_argument("--datasets", type=eval, default=f"{DATASET_LIST}")
    arg_parser.add_argument("--models", type=eval, default=f"{MODEL_LIST}")
    arg_parser.add_argument("--dpmm-types", type=eval, default=f"{DPMM_MODEL_TYPE}")
    arg_parser.add_argument("--dpmm-modes", type=eval, default=f"{DPMM_MODE}")
    arg_parser.add_argument("--window-size", type=int)
    arg_parser.add_argument("--step-size", type=int)

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

    EXP_DIR = f"experiments_{segmentator}_{feature_extractor}"
    SEGMENTATOR_ARGS = f" --feature-extractor {feature_extractor}"
    if segmentator:
        SEGMENTATOR_ARGS += " --segmentator"
        if window_size:
            SEGMENTATOR_ARGS += f" --window-size {window_size}"
        if step_size:
            SEGMENTATOR_ARGS += f" --step-size {step_size}"

    if feature_extractor == "rocket":
        EXP_DIR += f"_nkernels{n_kernels}"
        SEGMENTATOR_ARGS += f" --n-kernel {n_kernels}"

    if len(other_exp_args) > 0:
        EXP_DIR += "_" + "_".join(
            sorted(
                [
                    other_exp_args[i][2:].replace("_", "") + other_exp_args[i + 1]
                    for i in range(0, len(other_exp_args), 2)
                ]
            )
        )

    exp_path = os.path.join(args.output_dir, EXP_DIR)

    command_args_list = []
    for dataset in dataset_list:
        for model in model_list:
            COMMAND_ARGS = f"--base_dir {args.base_dir} --exp-dir {exp_path} --dataset {dataset} --model {model}"

            if model == "rockad":
                command_args_list.append(
                    COMMAND_ARGS + f" --n-kernel {n_kernels} --feature-extractor rocket"
                )
            else:
                if model != "dpmm":
                    command_args_list.append(COMMAND_ARGS + SEGMENTATOR_ARGS)
                else:
                    for dpmm_type in dpmm_types_list:
                        for dpmm_mode in args.dpmm_modes:
                            DPMM_ARGS = (
                                f" --dpmm-type {dpmm_type} --dpmm-mode {dpmm_mode}"
                            )
                            command_args_list.append(
                                COMMAND_ARGS + SEGMENTATOR_ARGS + DPMM_ARGS
                            )

    print("START")

    pool = ProcessPoolExecutor(max_workers=args.n_workers)
    FINISHED = 0
    C_ARGS = ""

    for C_ARGS in command_args_list:
        exp_args, _ = parse_exp_args(C_ARGS.split(" "))
        f = pool.submit(
            run_exp, exp_args, other_args=other_exp_args, suppress_output=True
        )

        def get_callback(cmd_args):
            """Get callback function."""

            def callback(future):
                global FINISHED
                FINISHED += 1
                if future.exception():
                    print(
                        f"{FINISHED}/{len(command_args_list)}\t"
                        f"Errore nell'esecuzione di {cmd_args}: {future.exception()}"
                    )
                else:
                    print(
                        f"{FINISHED}/{len(command_args_list)}\t"
                        f"Esecuzione terminata con successo per {cmd_args}"
                    )

            return callback

        f.add_done_callback(get_callback(C_ARGS))

    pool.shutdown()
