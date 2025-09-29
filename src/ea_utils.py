
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Tuple
import random

from .network import NeuralNet


def _model_to_vec(model: torch.nn.Module) -> torch.Tensor:
    """Flatten model params to a single 1-D tensor (shares device/dtype)."""
    return parameters_to_vector(model.parameters())

def _vec_to_model(vec: torch.Tensor, model: torch.nn.Module) -> None:
    """Write a 1-D tensor back into model params (in-place)."""
    vector_to_parameters(vec, model.parameters())

def _clamp_(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return x.clamp_(min=low, max=high)


@torch.no_grad()
def _sbx_vectors(
    v1: torch.Tensor,
    v2: torch.Tensor,
    eta: float,
    p_c: float,
    low: float,
    high: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulated Binary Crossover (SBX), applied per-weight.
    v1, v2: parent parameter vectors (same shape/device/dtype)
    eta:    distribution index (typical 10-20)
    p_c:    crossover probability applied per weight
    low, high: scalar bounds for clamping
    """
    assert v1.shape == v2.shape, "Parent vectors must have same shape."
    device = v1.device
    dtype = v1.dtype

    do = torch.rand_like(v1) < p_c

    u = torch.rand_like(v1)

    beta = torch.empty_like(v1, dtype=dtype, device=device)
    mask = (u <= 0.5)

    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))

    mean = 0.5 * (v1 + v2)
    diff = 0.5 * (v2 - v1)

    c1 = mean - beta * diff
    c2 = mean + beta * diff

    c1 = torch.where(do, c1, v1)
    c2 = torch.where(do, c2, v2)

    _clamp_(c1, low, high)
    _clamp_(c2, low, high)
    return c1, c2


@torch.no_grad()
def _poly_mutate_vector(
    v: torch.Tensor,
    eta: float,
    p_m: float,
    low: float,
    high: float,
) -> torch.Tensor:
    """
    Deb's polynomial mutation (bound-aware).
    v:    parameter vector (modified and returned)
    eta:  distribution index (typical 10-20)
    p_m:  mutation probability per weight
    low, high: scalar bounds
    """
    device = v.device
    dtype = v.dtype
    span = (high - low)

    m = (torch.rand_like(v) < p_m)

    if not torch.any(m):
        return v

    y = v.clone()
    dy1 = (y - low) / span
    dy2 = (high - y) / span

    dy1 = dy1.clamp(0.0, 1.0)
    dy2 = dy2.clamp(0.0, 1.0)

    r = torch.rand_like(v)

    idx = m.nonzero(as_tuple=False).squeeze(-1)
    r_m = r[idx]
    dy1_m = dy1[idx]
    dy2_m = dy2[idx]

    mask_left = (r_m <= 0.5)
    mask_right = ~mask_left

    deltaq = torch.empty_like(r_m, dtype=dtype, device=device)

    if torch.any(mask_left):
        rl = r_m[mask_left]
        val = 2.0 * rl + (1.0 - 2.0 * rl) * (1.0 - dy1_m[mask_left]).pow(eta + 1.0)
        deltaq[mask_left] = (val.pow(1.0 / (eta + 1.0)) - 1.0)

    if torch.any(mask_right):
        rr = r_m[mask_right]
        val = 2.0 * (1.0 - rr) + 2.0 * (rr - 0.5) * (1.0 - dy2_m[mask_right]).pow(eta + 1.0)
        deltaq[mask_right] = (1.0 - val.pow(1.0 / (eta + 1.0)))

    y[idx] = (y[idx] + deltaq * span).clamp_(low, high)

    v.copy_(y)
    return v

@torch.no_grad()
def simulated_binary_crossover(
    parent1: NeuralNet,
    parent2: NeuralNet,
    eta: float,
    crossover_prob: float,
    model_config: dict,
    bounds: Tuple[float, float],
) -> tuple:
    """
    Create two children via per-weight SBX from two parent NeuralNet objects.
    - Computes SBX on *flattened* parameter vectors
    - Clamps to bounds *immediately* after crossover
    - Returns two brand-new NeuralNet instances with the mixed weights
    """

    device = parent1.device

    low, high = float(bounds[0]), float(bounds[1])

    v1 = _model_to_vec(parent1.model)
    v2 = _model_to_vec(parent2.model)

    c1_vec, c2_vec = _sbx_vectors(v1, v2, eta=eta, p_c=crossover_prob, low=low, high=high)

    child1 = NeuralNet(model_config=model_config, device=str(device))
    child2 = NeuralNet(model_config=model_config, device=str(device))

    _vec_to_model(c1_vec, child1.model)
    _vec_to_model(c2_vec, child2.model)

    return child1, child2


@torch.no_grad()
def polynomial_mutation(
    net: NeuralNet,
    eta: float,
    mutation_prob: float,
    weight_bounds: Tuple[float, float],
) -> None:
    """
    In-place Deb polynomial mutation on a NeuralNet's parameter vector.
    - Applies per-weight mutation probability
    - Bound-aware step size
    - Clamps after mutation
    """
    low, high = float(weight_bounds[0]), float(weight_bounds[1])

    v = _model_to_vec(net.model)
    _poly_mutate_vector(v, eta=eta, p_m=mutation_prob, low=low, high=high)
    _vec_to_model(v, net.model)
