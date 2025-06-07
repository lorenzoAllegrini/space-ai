import os
import subprocess

# Cartella degli script
script_dir = "paper_examples"

# Recupera tutti i file .py nella cartella, ignorando quelli che iniziano con "_"
scripts = sorted([
    f for f in os.listdir(script_dir)
    if f.endswith(".py") and not f.startswith("_")
])

# Esporta il PYTHONPATH
env = os.environ.copy()
env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

for script in scripts:
    script_path = os.path.join(script_dir, script)
    print(f"\n>> Eseguendo {script_path}...\n")
    try:
        result = subprocess.run(
            ["python3", script_path],
            check=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"\n!! Errore nell'esecuzione di {script}: {e}\n")
