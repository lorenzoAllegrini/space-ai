import numpy as np
import torch as th
import torch.optim as optim
from torch_dpmm.models import FullGaussianDPMM, DiagonalGaussianDPMM, SingleGaussianDPMM, UnitGaussianDPMM
from tqdm import tqdm

def _init_model(model_type, K, D, alphaDP=10):
    if model_type == "Full":
        return FullGaussianDPMM(K, D, alphaDP, mu_prior=0, mu_prior_strength=0.01, var_prior=0.1, var_prior_strength=10)
    elif model_type == "Diagonal":
        return DiagonalGaussianDPMM(K, D, alphaDP, mu_prior=0, mu_prior_strength=0.01, var_prior=0.1, var_prior_strength=10)
    elif model_type == "Single":
        return SingleGaussianDPMM(K, D, alphaDP, mu_prior=0, mu_prior_strength=0.01, var_prior=0.1, var_prior_strength=10)
    elif model_type == "Unit":
        return UnitGaussianDPMM(K, D, alphaDP, mu_prior=0, mu_prior_strength=0.01)
    else:
        raise ValueError("Invalid model_type")

def get_trained_dpmm_model(X_train, model_type="Full", K=100, num_iterations=50, lr=0.8):
    D = X_train.shape[1]
    dpmm_model = _init_model(model_type, K, D)

    X_train_tensor = th.tensor(X_train, dtype=th.float32)
    dpmm_model.init_var_params(X_train_tensor)
    optimizer = optim.SGD(dpmm_model.parameters(), lr=lr)

    for epoch in tqdm(range(num_iterations), desc=f"Fitting {model_type} DPMM", unit="epoch"):
        optimizer.zero_grad()
        _, elbo_loss, _ = dpmm_model(X_train_tensor)
        elbo_loss.backward()
        optimizer.step()

    return dpmm_model

# Nuova versione: predizione da modello salvato, con soglia log-likelihood

def run_dpmm_likelihood(model, X_test):
    X_test_tensor = th.tensor(X_test, dtype=th.float32)
    with th.no_grad():
        _, _, test_log_likelihood = model(X_test_tensor)
        threshold = np.percentile(test_log_likelihood.numpy(), 1)
        return (test_log_likelihood.numpy() < threshold).astype(int)

# Nuova versione: predizione da modello salvato, con logica cluster assegnati

def run_dpmm_new_cluster(model, X_test, active_clusters, resp_threshold=0.1):
    X_test_tensor = th.tensor(X_test, dtype=th.float32)
    with th.no_grad():
        r_test, _, _ = model(X_test_tensor)
        assigned_clusters = th.argmax(r_test, dim=1).numpy()
        max_resp = th.max(r_test, dim=1).values.numpy()

        return np.array([
            1 if (cluster not in active_clusters or max_resp[i] < resp_threshold) else 0
            for i, cluster in enumerate(assigned_clusters)
        ])