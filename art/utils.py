from typing import Union

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
