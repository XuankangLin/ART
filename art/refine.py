""" Basic skeleton for refinement workflow, no matter using gradient heuristic
    or clustering as the underlying refinement technique.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
from torch import Tensor, nn, autograd
from torch.utils.data import DataLoader

from diffabs import AbsDom, AbsData
from diffabs.utils import valid_lb_ub

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import AbsProp
from art.utils import total_area


class Refiner(object):
    """ No matter which underlying refinement technique to use, it always depends on safety/violation distances for
        certification. To unify this workflow and utilize batch processing, the general algorithm is:

        (1) Given any new input abstractions (batched), compute its safety/violation distances.
            May also collect gradients in between.
        (2) Save the computed info into a heap, max or min, indexed by safety distances.
            Was merging everything into a Tensor and call torch.topk(), it should be much faster now.
        (3) Pick the top-k max or min items of the heap for refinement.
            verify() will pick top-k min items like DFS while split() will pick top-k max items like BFS.
        (4) Apply specific refinement technique on these selected items.
            Left for specific refinement tools.
            May raise CexError at any time if a cex is observed.
        (5) Go to (1) with the newly refined input abstractions.
    """

    def __init__(self, domain: AbsDom, prop: AbsProp):
        """
        :param domain: the abstract domain module to use
        :param prop: the safety property to consider
        """
        self.d = domain
        self.prop = prop
        return

    def _skeleton(self):

        return
    pass
