from argparse import Namespace
from math import floor
from typing import Union
from timeit import default_timer as timer

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


def total_area(lb: Tensor, ub: Tensor, eps: float = 1e-8, by_batch: bool = False) -> float:
    """ Return the total area constrained by LB/UB. Area = \Sum_{batch}{ \Prod{Element} }.
    :param lb: <Batch x ...>
    :param ub: <Batch x ...>
    :param by_batch: if True, return the areas of individual abstractions
    """
    assert valid_lb_ub(lb, ub)
    diff = ub - lb
    diff += eps  # some dimensions may be degenerated, then * 0 becomes 0.

    while diff.dim() > 1:
        diff = diff.prod(dim=-1)

    if by_batch:
        return diff
    else:
        return diff.sum().item()


def fmt_args(args: Namespace) -> str:
    title = args.stamp
    s = [f'\n===== {title} configuration =====']
    d = vars(args)
    for k, v in d.items():
        if k == 'stamp':
            continue
        s.append(f'  {k}: {v}')
    s.append(f'===== end of {title} configuration =====\n')
    return '\n'.join(s)


def pp_time(duration: float) -> str:
    """
    :param duration: in seconds
    """
    m = floor(duration / 60)
    s = duration - m * 60
    return '%dm %ds (%.3f seconds)' % (m, s, duration)


def time_since(since, existing=None):
    t = timer() - since
    if existing is not None:
        t += existing
    return pp_time(t)
