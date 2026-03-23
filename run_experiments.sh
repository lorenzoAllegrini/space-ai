#!/bin/bash
cd ~/
poetry run python examples/run_all_sml_exps.py --server_ip 172.20.200.100 --port 5557 --base_dir ./datasets --segmentator --feature-extractor base_statistics --skip-existing --dpmm-types "['diagonal', 'single', 'unit']"
read -p "Espresso finito. Premi invio per chiudere."
