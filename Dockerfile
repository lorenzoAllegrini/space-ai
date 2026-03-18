FROM --platform=linux/arm64 python:3.10-slim

WORKDIR /workspace

# Installiamo TUTTE le dipendenze per compilare da zero
RUN apt-get update && apt-get install -y \
    gcc g++ git make cmake \
    libgomp1 libopenblas-dev liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Installiamo pip, setuptools e pyinstaller
RUN pip install --upgrade pip "setuptools<70.0.0" wheel
RUN pip install pyinstaller

# 1. Installiamo NumPy dai sorgenti (ci metterà un po')
RUN pip install --no-binary numpy numpy==1.26.4

# 3. Installiamo le dipendenze (incluse quelle per i Benchmark locali)
RUN pip install \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    torch==2.3.1 \
    pyzmq==26.0.3 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    more-itertools==10.3.0 \
    sktime==0.32.0 \
    numba==0.59.1 \
    xgboost==1.7.6 \
    psutil==5.9.8

# 4. Copiamo il codice (incluso torch-dpmm)
COPY . /workspace/

# 5. Installiamo la dipendenza locale torch-dpmm
RUN pip install --no-deps -e ./torch-dpmm

# 6. Installiamo il package spaceai
RUN pip install --no-deps .

# 7. Generiamo i binari standalone con PyInstaller
ENV PYTHONPATH=/workspace

# A. IL SERVER SML (Streaming & Continual Learning)
RUN pyinstaller --onefile --log-level INFO \
    --name sml_server \
    --hidden-import ipaddress \
    --hidden-import xgboost \
    --hidden-import torch_dpmm \
    --hidden-import spaceai.models.anomaly_classifier \
    --hidden-import spaceai.models.anomaly_classifier.ndpm_internal \
    --hidden-import spaceai.preprocessing.ts_splitter \
    --hidden-import examples.utils.model_creators \
    --hidden-import spaceai.preprocessing.feature_extractors.utils \
    --hidden-import spaceai.benchmark.callbacks \
    --add-binary '/usr/local/lib/python3.10/site-packages/xgboost/lib/libxgboost.so:xgboost/lib' \
    --add-data '/usr/local/lib/python3.10/site-packages/xgboost/VERSION:xgboost' \
    examples/sml_inference/sml_server.py

# B. IL SERVER BATCH INFERENCE (Benchmark training/testing)
RUN pyinstaller --onefile --log-level INFO \
    --name sml_server_inference \
    --hidden-import ipaddress \
    --hidden-import xgboost \
    --hidden-import torch_dpmm \
    --hidden-import spaceai.models.anomaly_classifier \
    --hidden-import spaceai.models.anomaly_classifier.ndpm_internal \
    --hidden-import spaceai.preprocessing.ts_splitter \
    --hidden-import examples.utils.model_creators \
    --hidden-import spaceai.preprocessing.feature_extractors.utils \
    --hidden-import spaceai.benchmark.callbacks \
    --add-binary '/usr/local/lib/python3.10/site-packages/xgboost/lib/libxgboost.so:xgboost/lib' \
    --add-data '/usr/local/lib/python3.10/site-packages/xgboost/VERSION:xgboost' \
    examples/sml_inference/sml_server_inference.py

# C. L'ESPERIMENTO STANDALONE (Local Benchmarking)
RUN pyinstaller --onefile --log-level INFO \
    --name run_exp \
    --hidden-import ipaddress \
    --hidden-import xgboost \
    --hidden-import torch_dpmm \
    --hidden-import spaceai.models.anomaly_classifier \
    --hidden-import spaceai.models.anomaly_classifier.ndpm_internal \
    --hidden-import spaceai.preprocessing.ts_splitter \
    --hidden-import examples.utils.model_creators \
    --hidden-import spaceai.preprocessing.feature_extractors.utils \
    --hidden-import utils.dataset_exp \
    --hidden-import spaceai.benchmark.callbacks \
    --add-binary '/usr/local/lib/python3.10/site-packages/xgboost/lib/libxgboost.so:xgboost/lib' \
    --add-data '/usr/local/lib/python3.10/site-packages/xgboost/VERSION:xgboost' \
    examples/run_segment_extraction_exp.py

