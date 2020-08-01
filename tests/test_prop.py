""" Validating the algorithms of lbub exclusion and AndProp join. """

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import lbub_exclude, AndProp


def test_lbub_exclusion_1():
    """ Validate that LB/UB intersection/exclusion ops are correct. """
    lb1, ub1 = torch.Tensor([1, 1]), torch.Tensor([4, 4])
    lb2, ub2 = torch.Tensor([2, 2]), torch.Tensor([3, 3])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)

    # each dimension adds 2 pieces (left & right), no overlapping
    assert len(res_lb) == len(res_ub)
    assert len(res_lb) == 2 * len(lb1)
    return


def test_lbub_exclusion_2():
    lb1, ub1 = torch.Tensor([1, 1]), torch.Tensor([4, 4])
    lb2, ub2 = torch.Tensor([2, 2]), torch.Tensor([3, 4])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)

    # overlapped on one dimension
    assert len(res_lb) == len(res_ub)
    assert len(res_lb) == 2 * (len(lb1) - 1) + 1
    return


def test_lbub_exclusion_3():
    lb1, ub1 = torch.Tensor([1, 1, 1]), torch.Tensor([4, 4, 4])
    lb2, ub2 = torch.Tensor([2, 2, 2]), torch.Tensor([3, 3, 3])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)
    # each dimension adds 2 pieces (left & right), no overlapping
    assert len(res_lb) == len(res_ub)
    assert len(res_lb) == 2 * len(lb1)
    return
