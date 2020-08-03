from typing import Union

import torch
from torch import Tensor


def valid_lb_ub(lb: Union[float, Tensor], ub: Union[float, Tensor], eps: float = 1e-5) -> bool:
    """ To be valid:
        (1) Size ==
        (2) LB <= UB
    :param eps: added for numerical instability.
    """
    if isinstance(lb, float) and isinstance(ub, float):
        return lb <= ub + eps

    if lb.size() != ub.size():
        return False

    # '<=' will return a uint8 tensor of 1 or 0 for each element, it should have all 1s.
    return (lb <= ub + eps).all()


def sample_points(lb: Tensor, ub: Tensor, K: int) -> Tensor:
    """ Uniformly sample K points for each region.
    :param lb: Lower bounds, batched
    :param ub: Upper bounds, batched
    :param K: how many pieces to sample
    """
    assert valid_lb_ub(lb, ub)
    assert K >= 1

    repeat_dims = [1] * (len(lb.size()) - 1)
    base = lb.repeat(K, *repeat_dims)  # repeat K times in the batch, preserving the rest dimensions
    width = (ub - lb).repeat(K, *repeat_dims)

    coefs = torch.rand_like(base)
    pts = base + coefs * width
    return pts
