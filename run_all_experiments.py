import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--n_cpu", type=int, default=1)
    args = arg_parser.parse_args()

    # Cartella degli script
    script_dir = "paper_examples"

    # Recupera tutti i file .py nella cartella, ignorando quelli che iniziano con "_"
    scripts = sorted([
        f for f in os.listdir(script_dir)
        if f.endswith(".py") #and not f.startswith("esa")
    ])

    # Esporta il PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    env['OMP_NUM_THREADS'] = '1'

    pool = ProcessPoolExecutor(max_workers=args.n_cpu)

    for script in scripts:
        if script == 'results_reader.py':
            continue

        script_path = os.path.join(script_dir, script)
        print(f"\n>> Eseguendo {script_path}...\n")
        f = pool.submit(subprocess.run, ["python3", script_path], check=True, env=env,
                        stdout=None, stderr=None)

        def callback(f):
            if f.exception():
                print(f"Errore nell'esecuzione di {script_path}: {f.exception()}")
            else:
                print(f"Esecuzione terminata per {script_path}")

        f.add_done_callback(callback)

    pool.shutdown()