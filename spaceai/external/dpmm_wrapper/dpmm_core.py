import numpy as np
import torch as th
import torch.optim as optim
from torch_dpmm.models import FullGaussianDPMM, DiagonalGaussianDPMM, SingleGaussianDPMM, UnitGaussianDPMM
from tqdm import tqdm

def _init_model(model_type, K, D, alphaDP, var_prior, var_prior_strength):
    mu_prior_strength = 0.001
    if model_type == "Full":
        return FullGaussianDPMM(
            K,
            D,
            alphaDP,
            mu_prior=0,
            mu_prior_strength=mu_prior_strength,
            var_prior=var_prior,
            var_prior_strength=var_prior_strength,
        )
    elif model_type == "Diagonal":
        return DiagonalGaussianDPMM(
            K,
            D,
            alphaDP,
            mu_prior=0,
            mu_prior_strength=mu_prior_strength,
            var_prior=var_prior,
            var_prior_strength=var_prior_strength,
        )
    elif model_type == "Single":
        return SingleGaussianDPMM(
            K,
            D,
            alphaDP,
            mu_prior=0,
            mu_prior_strength=mu_prior_strength,
            var_prior=var_prior,
            var_prior_strength=var_prior_strength,
        )
    elif model_type == "Unit":
        return UnitGaussianDPMM(
            K,
            D,
            alphaDP,
            mu_prior=0,
            mu_prior_strength=mu_prior_strength,
        )
    else:
        raise ValueError("Invalid model_type")

def get_trained_dpmm_model(X_train_tensor, model_type, K, num_iterations, lr,
                           alphaDP, var_prior, var_prior_strength):
    D = X_train_tensor.shape[1]
    dpmm_model = _init_model(model_type, K, D, alphaDP, var_prior, var_prior_strength)
   
    dpmm_model.init_var_params(X_train_tensor)
    optimizer = optim.SGD(dpmm_model.parameters(), lr=lr)

    for epoch in tqdm(range(num_iterations), desc=f"Fitting {model_type} DPMM", unit="epoch"):
        optimizer.zero_grad()
        _, elbo_loss, _ = dpmm_model(X_train_tensor)
        elbo_loss.backward()
        optimizer.step()

    return dpmm_model