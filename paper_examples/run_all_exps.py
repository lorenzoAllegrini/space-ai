from config import OMP_NUM_THREADS
import os
os.environ['OMP_NUM_THREADS'] = f'{OMP_NUM_THREADS}'
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
    arg_parser.add_argument("--segmentator", type=str, required=True)
    arg_parser.add_argument("--n-kernels", type=int)
    arg_parser.add_argument("--output-dir", type=str, default='')
    arg_parser.add_argument("--datasets", type=eval, default=f'{DATASET_LIST}')
    arg_parser.add_argument("--models", type=eval, default=f'{MODEL_LIST}')
    arg_parser.add_argument("--dpmm-types", type=eval, default=f'{DPMM_MODEL_TYPE}')
    arg_parser.add_argument("--dpmm-modes", type=eval, default=f'{DPMM_MODE}')

    args, other_exp_args = arg_parser.parse_known_args()
    print(args)

    n_kernels = args.n_kernels
    dataset_list = args.datasets
    model_list = args.models
    dpmm_types_list = args.dpmm_types
    dpmm_modes_list = args.dpmm_modes
    segmentator = args.segmentator

    exp_dir = f"experiments_{segmentator}"
    segmentator_args = f' --segmentator {segmentator}'
    if segmentator == "rocket":
        exp_dir += f"_nkernels{n_kernels}"
        segmentator_args += f' --n-kernel {n_kernels}'

    if len(other_exp_args) > 0:
        exp_dir += '_' +  '_'.join(sorted([other_exp_args[i][2:].replace('_','')+other_exp_args[i+1]
                                           for i in range(0, len(other_exp_args),2)]))

    exp_path = os.path.join(args.output_dir, exp_dir)

    command_args_list = []
    for dataset in dataset_list:
        for model in model_list:
            command_args = f'--exp-dir {exp_path} --dataset {dataset} --model {model}'

            if model == 'rockad':
                command_args_list.append(command_args + f' --n-kernel {n_kernels} --segmentator rocket')
            else:
                if model != "dpmm":
                    command_args_list.append(command_args + segmentator_args)
                else:
                    for dpmm_type in dpmm_types_list:
                        for dpmm_mode in args.dpmm_modes:
                            dpmm_args = f' --dpmm-type {dpmm_type} --dpmm-mode {dpmm_mode}'
                            command_args_list.append(command_args + segmentator_args + dpmm_args)

    print('START')
    pool = ProcessPoolExecutor(max_workers=args.n_workers)
    finished = 0
    c_args = ''

    for c_args in command_args_list:
        exp_args, _  = parse_exp_args(c_args.split(' '))
        f = pool.submit(run_exp, exp_args, other_args=other_exp_args, suppress_output=True)

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
