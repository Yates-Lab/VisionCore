import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .eval import PoissonMaskedLoss

class LNPModel(nn.Module):
    def __init__(self, input_dim, n_units):
        super(LNPModel, self).__init__()
        self.kernel = nn.Parameter(torch.randn(n_units, input_dim))
        self.bias = nn.Parameter(torch.zeros(n_units))
        
    def forward(self, x):
        stim = x['stim'].flatten(start_dim=1)
        generator = torch.einsum('td,ud->tu', stim, self.kernel)
        x['rhat'] = F.softplus(generator + self.bias)
        return x

def fit_lnp_lbfgs(data, n_steps = 10, lbfgs_kwargs = {}, device='cpu'):
    n_units = data['robs'].shape[1]
    input_dim = np.prod(list(data['stim'].shape[1:]))
    model = LNPModel(input_dim, n_units)
    model.to(device)
    loss_fn = PoissonMaskedLoss()
    optimizer = torch.optim.LBFGS(model.parameters(), **lbfgs_kwargs)

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model(data))
        loss.backward()
        return loss

    prev_loss = np.inf
    for i in range(n_steps):
        loss = optimizer.step(closure)
        if i == 0:
            print(f'Epoch {i}: {loss.item():.4e}')
        else:
            print(f'Epoch {i}: {loss.item():.4e} ({(prev_loss - loss).item():.4e})')
        prev_loss = loss

    return model    
