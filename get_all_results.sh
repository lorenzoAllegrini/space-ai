#!/bin/bash

export PYTHONPATH='.'

for exp_dir in "$1"/*; do
  echo "$exp_dir"
  python paper_examples/results_reader.py --base-dir "$exp_dir"  --output-dir "$2" --channel-cat-path "$3"
done