```markdown
# DPMM Wrapper for space-ai

This module isolates anomaly detection logic using `torch-dpmm`, which requires specific versions of numpy, sklearn, etc.

## When to use this
Use this if you're running anomaly detection models based on DPMM in a separate environment due to dependency constraints.

---

## Required Environment
We recommend creating a conda environment named `dpmm_env`:

### Create it with conda:
```bash
conda create -n dpmm_env python=3.10
conda activate dpmm_env
pip install -r requirements_dpmm.txt
```

### Or with pip and venv:
```bash
python -m venv dpmm_env
source dpmm_env/bin/activate  # or .\dpmm_env\Scripts\activate on Windows
pip install -r requirements_dpmm.txt
```

---

##  How to run manually (optional)
```bash
python run_dpmm.py test.csv train.csv output.csv likelihood Full 100 50 0.8
```

## File Structure
```
external/
└── dpmm_wrapper/
    ├── run_dpmm.py
    ├── dpmm_core.py
    ├── benchmark_utils.py
    ├── requirements_dpmm.txt
    └── README.md
```

---

## Used from space-ai
The module is launched from within `space-ai` using the `DPMMWrapperDetector`, which runs the wrapper in subprocess. See `examples/example_dpmm_wrapper.py`.
```
