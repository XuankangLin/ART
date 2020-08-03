""" Common utilities and functions used in multiple experiments. """

from abc import ABC, abstractmethod

from torch.utils import data
from typing import Union, List

from torch import Tensor

from art.utils import valid_lb_ub


class PseudoLenDataset(data.Dataset, ABC):
    """ The 'claimed_len' in AbsIns and ConcIns are used to enumerate two datasets simultaneously with true shuffling
        during joint-training.

        torch.data.ConcatDataset cannot do this because it may zip data points from two arbitrary dataset, which would
        result in [Abs, Abs], [Conc, Conc], [Abs, Conc], [Conc, Abs] in the same enumeration. So it is only for two
        homogeneous datasets.

        I was using Combined(Abs, Conc) that re-enumerate one dataset from beginning until both are enumerated. However,
        this is not true shuffling. No matter what shuffled enumeration order it is, idx-1 Abs and idx-1 Conc are always
        zipped together.

        With an extra variable that tracks the claimed length, it can have individual enumeration order for both
        datasets, thus achieving true shuffling.
    """
    def __init__(self, pivot_ls):
        self.pivot_ls = pivot_ls
        self.claimed_len = self.real_len()
        return

    def real_len(self):
        return len(self.pivot_ls)

    def reset_claimed_len(self):
        self.claimed_len = self.real_len()
        return

    def __len__(self):
        """ Allowing enumerating for more than once, so as to co-train with ConcIns. """
        return self.claimed_len

    def __getitem__(self, idx):
        """ There is only split sub-region, no label. """
        return self._getitem(idx % self.real_len())

    @abstractmethod
    def _getitem(self, idx):
        raise NotImplementedError()
    pass


class AbsIns(PseudoLenDataset):
    """ Storing the split LB/UB boxes/abstractions. """
    def __init__(self, boxes_lb: Tensor, boxes_ub: Tensor, boxes_extra: Tensor = None):
        assert valid_lb_ub(boxes_lb, boxes_ub)
        self.boxes_lb = boxes_lb
        self.boxes_ub = boxes_ub
        self.boxes_extra = boxes_extra
        super().__init__(self.boxes_lb)
        return

    def _getitem(self, idx):
        if self.boxes_extra is None:
            return self.boxes_lb[idx], self.boxes_ub[idx]
        else:
            return self.boxes_lb[idx], self.boxes_ub[idx], self.boxes_extra[idx]
    pass


class ConcIns(PseudoLenDataset):
    """ Storing the concrete data points """
    def __init__(self, inputs: Union[List[Tensor], Tensor], labels: Union[List[Tensor], Tensor]):
        assert len(inputs) == len(labels)
        self.inputs = inputs
        self.labels = labels
        super().__init__(self.inputs)
        return

    def _getitem(self, idx):
        return self.inputs[idx], self.labels[idx]
    pass
