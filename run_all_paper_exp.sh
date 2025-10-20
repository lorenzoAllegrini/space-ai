#!/bin/bash

export PYTHONPATH='.'
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 30 --segmentator base_stats &
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 30 --segmentator rocket --n-kernels 10 &
#python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 30 --segmentator rocket --n-kernels 50
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 30 --segmentator rocket --n-kernels 100  --dpmm-types "['diagonal', 'single', 'unit']" &
#python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 30 --segmentator rocket --n-kernels 1000 --dpmm-types "['diagonal', 'single', 'unit']"

# experiments quantile 0.01
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 12 --models "['dpmm']" --dpmm-modes "['likelihood_threshold']" --quantile 0.01 --segmentator base_stats &
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 12 --models "['dpmm']" --dpmm-modes "['likelihood_threshold']" --quantile 0.01 --segmentator rocket --n-kernels 10 &
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 12 --models "['dpmm']" --dpmm-modes "['likelihood_threshold']" --quantile 0.01 --segmentator rocket --n-kernels 100 --dpmm-types "['diagonal', 'single', 'unit']" &

# experiments quantile 0.001
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 12 --models "['dpmm']" --dpmm-modes "['likelihood_threshold']" --quantile 0.001 --segmentator base_stats &
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 12 --models "['dpmm']" --dpmm-modes "['likelihood_threshold']" --quantile 0.001 --segmentator rocket --n-kernels 10 &
python paper_examples/run_all_exps.py --output-dir "$1" --n-workers 12 --models "['dpmm']" --dpmm-modes "['likelihood_threshold']" --quantile 0.001 --segmentator rocket --n-kernels 100 --dpmm-types "['diagonal', 'single', 'unit']" &
