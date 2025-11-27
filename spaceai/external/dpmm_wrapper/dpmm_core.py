import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch_dpmm.models import FullGaussianDPMM, DiagonalGaussianDPMM, SingleGaussianDPMM, UnitGaussianDPMM
from tqdm import tqdm


def _init_model(model_type, K, D, alphaDP, var_prior, var_prior_strength):
    mu_prior_strength = 0.001
    if model_type == "full":
        return FullGaussianDPMM(
            K,
            D,
            alphaDP,
            mu_prior=0,
            mu_prior_strength=mu_prior_strength,
            var_prior=var_prior,
            var_prior_strength=var_prior_strength,
        )
    elif model_type == "diagonal":
        return DiagonalGaussianDPMM(
            K,
            D,
            alphaDP,
            mu_prior=0,
            mu_prior_strength=mu_prior_strength,
            var_prior=var_prior,
            var_prior_strength=var_prior_strength,
        )
    elif model_type == "single":
        return SingleGaussianDPMM(
            K,
            D,
            alphaDP,
            mu_prior=0,
            mu_prior_strength=mu_prior_strength,
            var_prior=var_prior,
            var_prior_strength=var_prior_strength,
        )
    elif model_type == "unit":
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

    tr_dataset = TensorDataset(X_train_tensor)
    tr_data_loader = DataLoader(tr_dataset, batch_size=1000, shuffle=True)
    last_elbos = []
    patience = 3
    tol = 1e-3

    print(num_iterations)
    print('Training start')
    pbar = tqdm(range(num_iterations), desc=f"Fitting {model_type} DPMM", unit="epoch")
    for epoch in pbar:
        elbo_tot = 0.0
        for x_batch, in tr_data_loader:
            optimizer.zero_grad()
            _, elbo_loss, _ = dpmm_model(x_batch)
            elbo_loss.backward()
            optimizer.step()
            elbo_tot += elbo_loss.item()
        pbar.set_postfix(elbo=elbo_tot)
        # save elbos and check for early stopping
        last_elbos.append(elbo_tot)
        if len(last_elbos) > patience:
            last_elbos = last_elbos[-patience:]
            npz_elbos = np.array(last_elbos)
            if np.all(np.diff(npz_elbos)>-tol):
                break

    print('Training end')
    return dpmm_model