import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from .run_exp import (
    DATASET_LIST,
    DPMM_MODE,
    DPMM_MODEL_TYPE,
    MODEL_LIST,
    SEGMENTATOR_LIST,
    parse_exp_args,
    run_exp,
)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--n-workers", type=int, default=1)
    args = arg_parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"

    pool = ProcessPoolExecutor(max_workers=args.n_workers)

    n_kernels = 100

    command_args_list = []
    for dataset in DATASET_LIST:
        for segmentator in SEGMENTATOR_LIST:
            for model in MODEL_LIST:
                command_args = (
                    f"--dataset {dataset} --segmentator {segmentator} --model {model}"
                )
                if segmentator == "rocket" or model == "rockad":
                    command_args += f" --n-kernel {n_kernels}"

                if model != "dpmm":
                    command_args_list.append(command_args)
                else:
                    for dpmm_type in DPMM_MODEL_TYPE:
                        for dpmm_mode in DPMM_MODE:
                            dpmm_args = (
                                f"--dpmm-type {dpmm_type} --dpmm-mode {dpmm_mode}"
                            )
                            command_args_list.append(command_args + " " + dpmm_args)

    print("START")
    finished = 0
    c_args = ""
    for c_args in command_args_list:
        exp_args = parse_exp_args(c_args.split(" "))
        f = pool.submit(run_exp, exp_args, suppress_output=True)

        def get_callback(command_args):

            def callback(f):
                global finished
                finished += 1
                if f.exception():
                    print(
                        f"{finished}/{len(command_args_list)}\tErrore nell'esecuzione di {command_args}: {f.exception()}"
                    )
                else:
                    print(
                        f"{finished}/{len(command_args_list)}\tEsecuzione terminata con successo per {command_args}"
                    )

            return callback

        f.add_done_callback(get_callback(c_args))

    pool.shutdown()
