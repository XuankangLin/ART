""" Basic skeleton for refinement workflow, regardless of which refinement method to use. """

import heapq
import logging
import sys
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Tuple, Optional, Union, List

import torch
from torch import Tensor, nn, autograd
from torch.utils.data import DataLoader

from diffabs import AbsDom, AbsData
from diffabs.utils import valid_lb_ub

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import AbsProp
from art.utils import total_area


class CexError(Exception):
    """ Raised when a counter-example is observed. """
    def __init__(self, cex: Tensor = None,
                 cex_lb: Tensor = None, cex_ub: Tensor = None, cex_extra: Tensor = None):
        """ Either found a concrete cex, or a region of cexs that are certified to violate. """
        assert cex is not None or (cex_lb is not None and cex_ub is not None)
        self.cex = cex
        self.cex_lb = cex_lb
        self.cex_ub = cex_ub
        self.cex_extra = cex_extra
        return
    pass


class _WorkList(object):
    """ Maintaining the worklist of (unbatched) input abstractions during refinement as a heap. """
    def __init__(self, pick_max: bool):
        """
        :param pick_max: If True, those with largest safety distances will be picked for refinement at each step,
                         otherwise, smallest safety distance ones are picked.
                         This is needed because heapq only has support for min-heap, which is surprising.
        """
        self.pick_max = pick_max
        self.wl = []
        return

    def __len__(self) -> int:
        return len(self.wl)

    def push(self, one_lb: Tensor, one_ub: Tensor, one_extra: Optional[Tensor],
             one_safe_dist: float, one_viol_dist: float, one_grad: Tensor = None):
        """ All unbatched. """
        # heapq module is only min-heap, to use max-heap, invert the key value
        key = -1 * one_safe_dist if self.pick_max else one_safe_dist
        heapq.heappush(self.wl, (key,
                                 (one_lb, one_ub, one_extra, one_safe_dist, one_viol_dist, one_grad)))
        return

    def pop(self) -> Tuple:
        return heapq.heappop(self.wl)[1]  # discard key

    def all_items(self) -> List[Tuple]:
        return [p[1] for p in self.wl]  # discard keys

    def pop_all(self) -> List[Tuple]:
        items = self.all_items()
        self.wl = []
        return items

    def max_safe_dist(self) -> Tensor:
        if self.pick_max:
            # key == -1 * safe_dist, hence pick smallest key for max safe_dist
            key = heapq.nsmallest(1, self.wl)[0]
            return -1 * key
        else:
            # key == safe_dist, hence pick largest key for max safe_dist
            key = heapq.nlargest(1, self.wl)[0]
            return key

    def min_safe_dist(self) -> Tensor:
        if self.pick_max:
            # key == -1 * safe_dist, hence pick largest key for min safe_dist
            key = heapq.nlargest(1, self.wl)[0]
            return -1 * key
        else:
            # key == safe_dist, hence pick smallest key for max safe_dist
            key = heapq.nsmallest(1, self.wl)[0]
            return key
    pass


def stack_ls(ts: Optional[Union[Tensor, List]], device) -> Optional[Tensor]:
    """ Stack or concatenate a list of tensors for batch processing. """
    if ts is None:
        return None
    elif isinstance(ts, Tensor):
        return ts
    elif isinstance(ts, List):
        if len(ts) == 0:
            return torch.tensor([], device=device)
        elif None in ts:
            # a list of None, treated as None
            return None
        else:
            return torch.stack(ts, dim=0)
    else:
        raise NotImplementedError(f'Unknown type argument: {ts}')


def cat0(*ts: Tensor) -> Tensor:
    """ Usage: simplify `torch.cat((ts1, ts2), dim=0)` to `cat0(ts1, ts2)`. """
    return torch.cat(ts, dim=0)


class Refiner(ABC):
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

    @abstractmethod
    def batch_dists_of(self, batch_lb: Tensor, batch_ub: Tensor, batch_extra: Optional[Tensor],
                       forward_fn: nn.Module) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """ Compute a batch of input abstractions and return their safety/violation distances as well as grads. """
        raise NotImplementedError()

    def dists_of(self, new_lb: Tensor, new_ub: Tensor, new_extra: Optional[Tensor], forward_fn: nn.Module,
                 batch_size: int) -> Tuple[List[Tensor], List[Tensor], List[Optional[Tensor]]]:
        """ Compute the (batched) safety / violation distances of input abstractions, as well as grads if needed.
        :return: lists, not batched tensors
        """
        ds = AbsData(new_lb, new_ub, new_extra)
        abs_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        all_safe_dists, all_viol_dists, all_grads = [], [], []
        for batch in abs_loader:
            if new_extra is None:
                batch_lb, batch_ub = batch
                batch_extra = None
            else:
                batch_lb, batch_ub, batch_extra = batch

            batch_safe_dist, batch_viol_dist, batch_grad = self.batch_dists_of(batch_lb, batch_ub, batch_extra,
                                                                               forward_fn)
            all_safe_dists.extend(list(batch_safe_dist))
            all_viol_dists.extend(list(batch_viol_dist))
            if batch_grad is None:
                batch_grad = [None] * len(batch_safe_dist)
            all_grads.extend(batch_grad)

        return all_safe_dists, all_viol_dists, all_grads

    @abstractmethod
    def refine(self, picked_lb: List[Tensor], picked_ub: List[Tensor], picked_extra: List[Optional[Tensor]],
               picked_grad: List[Optional[Tensor]], tiny_width: Optional[float]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """ Given lists of (unbatched) input abstractions to refine (may also come with the computed gradients, compute
            their refined (batched) input abstractions.
            CexError can be raised when a cex is observed.
        """
        raise NotImplementedError()

    def skeleton(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module,
                 batch_size: int, pick_max: bool, stop_on_cex: bool,
                 stop_on_k_all: int = None, stop_on_k_new: int = None, stop_on_k_ops: int = None,
                 tiny_width: Optional[float] = 1e-4,
                 ret_all: bool = True) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """ Basic skeleton for both verify() and split().

        :param lb: batched LB, from props.lbub(), but may need additional normalization
        :param ub: batched UB, from props.lbub(), but may need additional normalization
        :param extra: batched extra info such as the bit vectors for each LB/UB cube showing which safety property
                      it should satisfy in AndProp; or just None
        :param forward_fn: differentiable forward propagation, not passing in net and call net(input) because different
                           applications may have different net(input, **kwargs)
        :param batch_size: how many abstractions to pick for refinement at each iteration;
                           larger batch_size => faster to reach more refined abstractions but can be unnecessarily
                           precise in some refined abstractions, leaving it unexplored in more needed abstractions.
        :param pick_max: if True, those with largest safety distances will be picked for refinement at each step (BFS),
                         otherwise, smallest safety distance ones are picked (DFS).
        :param stop_on_cex: if True, stop when a cex is observed, then raise CexError with that cex.
        :param stop_on_k_all: if not None, split() stops after total amount of abstractions exceeds this bar.
        :param stop_on_k_new: if not None, split() stops after the amount of abstractions introduced by this split()
                              call exceeds this bar.
        :param stop_on_k_ops: if not None, split() stops after this many refinement steps have been applied.
        :param tiny_width: if not None, stop refining one dimension if its width is already <= this bar,
                           e.g., setting tiny_width=1e-3 would ensure all refined abstraction dimension width > 5e-4.
        :param ret_all: if True, all refined abstractions are returned, otherwise, return only those not certified-safe;
                        can be used as an indicator for all certified or not.
        :return: <LB, UB> when extra is None, otherwise <LB, UB, extra>
        """
        assert valid_lb_ub(lb, ub)
        assert batch_size > 0

        def _validate_stop_criterion(v, pivot: int):
            assert v is None or (isinstance(v, int) and v > pivot)
            return
        _validate_stop_criterion(stop_on_k_all, 0)
        _validate_stop_criterion(stop_on_k_new, 0)
        _validate_stop_criterion(stop_on_k_ops, -1)  # allow 0 refinement steps, i.e., just evaluate, no refine

        # track how much have been certified/falsified for logs
        tot_area = total_area(lb, ub)
        assert tot_area > 0
        safe_area, viol_area, tiny_area = 0., 0., 0.

        # worklist maintained as a heap, other decided ones are stored in lists
        wl = _WorkList(pick_max)
        safe_lb, safe_ub, safe_extra = [], [], []
        viol_lb, viol_ub, viol_extra = [], [], []
        tiny_lb, tiny_ub, tiny_extra = [], [], []

        new_lb, new_ub, new_extra = lb, ub, extra  # init to process all (batched) given abstractions as the first step
        n_orig_abs = len(lb)
        iter = 0
        while True:
            iter += 1

            if len(new_lb) > 0:
                ''' (1) Given any new input abstractions (batched), compute its safety/violation distances.
                    May also collect gradients in between.
                '''
                new_safe_dist, new_viol_dist, new_grad = self.dists_of(new_lb, new_ub, new_extra, forward_fn, batch_size)
                if new_extra is None:
                    new_extra = [None] * len(new_lb)

                ''' (2) Save the computed info into a heap, max or min, indexed by safety distances.
                    Was merging everything into a Tensor and call torch.topk(), it should be much faster now.
                '''
                new_processed_cnt = len(new_lb)
                new_safe_cnt, new_viol_cnt, new_tiny_cnt = 0, 0, 0
                for one_lb, one_ub, one_extra, one_safe_dist, one_viol_dist, one_grad \
                        in zip(new_lb, new_ub, new_extra, new_safe_dist, new_viol_dist, new_grad):
                    if one_safe_dist <= 0.:
                        # certified safe
                        safe_lb.append(one_lb)
                        safe_ub.append(one_ub)
                        safe_extra.append(one_extra)

                        new_safe_cnt += 1
                        safe_area += total_area(one_lb.unsqueeze(dim=0), one_ub.unsqueeze(dim=0))
                    elif one_viol_dist <= 0.:
                        # certified violation
                        viol_lb.append(one_lb)
                        viol_ub.append(one_ub)
                        viol_extra.append(one_extra)

                        new_viol_cnt += 1
                        viol_area += total_area(one_lb.unsqueeze(dim=0), one_ub.unsqueeze(dim=0))
                        if stop_on_cex:
                            raise CexError(cex_lb=one_lb, cex_ub=one_ub, cex_extra=one_extra)
                    else:
                        is_tiny = False
                        if tiny_width is not None:
                            one_width = one_ub - one_lb
                            assert one_lb.dim() == 2, 'Otherwise, I need to reduce the >2 dims to compute dim width?'
                            is_tiny = (one_width > tiny_width).all()
                        if is_tiny:
                            tiny_lb.append(one_lb)
                            tiny_ub.append(one_ub)
                            tiny_extra.append(one_extra)

                            new_tiny_cnt += 1
                            tiny_area += total_area(one_lb.unsqueeze(dim=0), one_ub.unsqueeze(dim=0))
                        else:
                            # still to refine
                            wl.push(one_lb, one_ub, one_extra, one_safe_dist, one_viol_dist, one_grad)

                logging.debug(f'At iter {iter}, another {new_processed_cnt} boxes are processed, in which ' +
                              f'{new_safe_cnt} confirmed safe, {new_viol_cnt} confirmed to violate, ' +
                              f'{new_tiny_cnt} found too tiny and stopped.')

            safe_area_percent = safe_area / tot_area * 100
            viol_area_percent = viol_area / tot_area * 100
            tiny_area_percent = tiny_area / tot_area * 100
            wl_area_percent = 100. - safe_area_percent - viol_area_percent - tiny_area_percent
            # was also printing max/min violation distances, omitted now because those aren't indexed in the heap
            logging.debug(f'After iter {iter}, total #{len(wl)} ({wl_area_percent:.2f}%) in worklist, ' +
                          f'total #{len(safe_lb)} ({safe_area_percent:.2f}%) safe, ' +
                          f'total #{len(viol_lb)} ({viol_area_percent:.2f}%) to violate, ' +
                          f'total #{len(tiny_lb)} ({tiny_area_percent:.2f}%) found too tiny. ' +
                          f'Safe dist min: {wl.min_safe_dist()} ~ max: {wl.max_safe_dist()}.\n')

            ''' (2.5) early stopping checks '''
            if len(wl) == 0:
                # nothing to refine anymore
                break

            n_curr_abs = len(wl) + len(safe_lb) + len(viol_lb) + len(tiny_lb)
            if stop_on_k_all is not None and n_curr_abs >= stop_on_k_all:
                # has collected enough abstractions
                break
            if stop_on_k_new is not None and n_curr_abs - n_orig_abs >= stop_on_k_new:
                # has collected enough new abstractions
                break
            if stop_on_k_ops is not None and iter > stop_on_k_ops:
                # has run enough refinement iterations
                break

            ''' (3) Pick the top-k max or min items of the heap for refinement.
                verify() will pick top-k min items like DFS while split() will pick top-k max items like BFS.
            '''
            picked_lb, picked_ub, picked_extra, picked_grad = [], [], [], []
            if batch_size >= len(wl):
                # the entire wl items will be refined, thus no need to pop one by one
                all_items = wl.pop_all()
                picked_lb.extend([p[0] for p in all_items])
                picked_ub.extend([p[1] for p in all_items])
                picked_extra.extend([p[2] for p in all_items])
                # safe/viol dists are not needed
                picked_grad.extend(p[5] for p in all_items)
            else:
                for _ in range(batch_size):
                    p = wl.pop()
                    picked_lb.append(p[0])
                    picked_ub.append(p[1])
                    picked_extra.append(p[2])
                    # safe/viol dists are not needed
                    picked_grad.append(p[5])

            ''' (4) Apply specific refinement technique on these selected items.
                Left for specific refinement tools. May raise CexError at any time if a cex is observed.
                (5) Go to (1) with the newly refined (batched) input abstractions.
            '''
            new_lb, new_ub, new_extra = self.refine(picked_lb, picked_ub, picked_extra, picked_grad, tiny_width)
            pass  # end of big while loop

        ''' (6) collect return values '''
        all_lb, all_ub, all_extra = [], [], []

        items = wl.all_items()
        all_lb.extend([p[0] for p in items])
        all_ub.extend([p[1] for p in items])
        all_extra.extend([p[2] for p in items])

        all_lb.extend(viol_lb)
        all_ub.extend(viol_ub)
        all_extra.extend(viol_extra)

        all_lb.extend(tiny_lb)
        all_ub.extend(tiny_ub)
        all_extra.extend(tiny_extra)

        if ret_all:
            # return all refined abstractions in one batch, which includes certified safe ones
            all_lb.extend(safe_lb)
            all_ub.extend(safe_ub)
            all_extra.extend(safe_extra)

        with torch.no_grad():
            all_lb = stack_ls(all_lb, lb.device)
            all_ub = stack_ls(all_ub, ub.device)
            all_extra = stack_ls(all_extra, lb.device)
        if all_extra is None:
            return all_lb, all_ub
        else:
            return all_lb, all_ub, all_extra

    def verify(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module, batch_size: int,
               stop_on_cex: bool = True) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """ DFS, dig into small distance abstractions until they are certified safe or violation.
            Similar to safety certification, if the violation distance <= 0, it's certified to violate.
        """
        return self.skeleton(lb, ub, extra, forward_fn, batch_size, pick_max=False, stop_on_cex=stop_on_cex)

    def split(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module, batch_size: int,
              stop_on_cex: bool = False,
              stop_on_k_all: int = None,
              stop_on_k_new: int = None,
              stop_on_k_ops: int = None,
              tiny_width: Optional[float] = 1e-4,
              ret_all: bool = True) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """ BFS, the goal is to have roughly even abstractions with small safety losses for the optimization later. """
        return self.skeleton(lb, ub, extra, forward_fn, batch_size,
                             pick_max=True, stop_on_cex=stop_on_cex, stop_on_k_all=stop_on_k_all,
                             stop_on_k_new=stop_on_k_new, stop_on_k_ops=stop_on_k_ops,
                             tiny_width=tiny_width, ret_all=ret_all)

    def try_certify(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module, batch_size: int,
                    stop_on_k_all: int = None) -> bool:
        """
        :return: True if it can successfully certify the property on lb/ub within certain limits
        """
        wl_lb = self.split(lb, ub, extra, forward_fn, batch_size, stop_on_k_all=stop_on_k_all, ret_all=False)[0]
        # empty => all certified
        return len(wl_lb) == 0
    pass


class Bisecter(Refiner):
    """ Refinement by gradient-based heuristic for bisection.

        **Some Experiment History**

        I once experimented "what to pick" and "what to grad" using ACAS prop2 and property3 to see the explored area
        after 100 iterations. Results show that:
        *   Pick small violation loss ones is the worst option, perhaps because for most of the time, violation losses
            are always small.
        *   Use gradients from violation loss is highest in the rest 2 picking cases. Having safety loss may or may not
            help. Using safe+viol loss may or may not help. And min(safe, viol) is basically equal to safe loss.
        So I chose to pick small safe loss boxes and bisect based on gradients from violation loss.

        But now, after refactoring, I just follow the same practice as split() for code brevity -- pick by safety losses
        and refine by safety losses as well. This is less relevant, as Cluster.py would apply a different technique that
        doesn't need to worry about such gradients.

        For gradient-based heuristic, I also tried using a 'factor' tensor, with LB = LB * factor and UB = UB * factor,
        to compute gradient w.r.t. 'factor'. However, that is much worse than the grad w.r.t. LB and UB directly. One
        possible reason is that 'factor' can only shrink the space in one direction towards its mid point. This has
        little to do with actual bisection later on. Grads w.r.t. LB and UB is more directly related.
    """

    def __init__(self, dom: AbsDom, prop: AbsProp):
        super().__init__(dom, prop)

        self.grad_src = 'safe'  # safe? viol? both? pick source for gradients
        return

    def batch_dists_of(self, new_lb: Tensor, new_ub: Tensor, new_extra: Optional[Tensor],
                       forward_fn: nn.Module) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """ Get the gradients for each abstraction as heuristic, as well as gradients if needed. """
        new_lb = new_lb.detach().requires_grad_()
        new_ub = new_ub.detach().requires_grad_()
        if new_lb.grad is not None:
            new_lb.grad.zero_()  # in case any previous grads are unexpectedly accumulated
        if new_ub.grad is not None:
            new_ub.grad.zero_()

        ins = self.d.Ele.by_intvl(new_lb, new_ub)
        outs = forward_fn(ins)

        new_safe_dist = self.prop.safe_dist(outs) if new_extra is None else self.prop.safe_dist(outs, new_extra)
        new_viol_dist = self.prop.viol_dist(outs) if new_extra is None else self.prop.viol_dist(outs, new_extra)
        if self.grad_src == 'safe':
            grad_dist = new_safe_dist
        elif self.grad_src == 'viol':
            grad_dist = new_viol_dist
        elif self.grad_src == 'both':
            grad_dist = new_safe_dist + new_viol_dist
        else:
            raise ValueError(f'Invalid grad_src = {self.grad_src}.')

        ''' Sum safe/viol_dists to get one single value for backprop. Otherwise it needs to pass in 'grad_outputs'
            argument for autograd.grad(). e.g., use ones_like(dists).
            1st order summation will distribute the original output distance to each corresponding input.
            After all, it only needs to relatively compare input grads.
        '''
        assert grad_dist.dim() == 1, 'Do I need to squeeze the losses into <Batch> vector first?'
        losses = grad_dist.sum()

        # back-propagate safety loss to inputs
        grads = autograd.grad(losses, [new_lb, new_ub])
        # Clip grads below. LB should ++. Thus grad for LB should < 0 (LB' = LB - lr * grad). Similar for UB.
        grads[0].clamp_(max=0.)  # LB
        grads[1].clamp_(min=0.)  # UB
        new_grad = sum([g.abs() for g in grads])  # get one value for each abstraction
        return new_safe_dist, new_viol_dist, new_grad

    def refine(self, picked_lb: List[Tensor], picked_ub: List[Tensor], picked_extra: List[Optional[Tensor]],
               picked_grad: List[Optional[Tensor]],
               tiny_width: Optional[float]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        assert len(picked_lb) > 0 and len(picked_lb) == len(picked_ub) == len(picked_extra) == len(picked_grad)

        # make all batched
        device = picked_lb[0].device
        picked_lb = stack_ls(picked_lb, device)
        picked_ub = stack_ls(picked_ub, device)
        picked_extra = stack_ls(picked_extra, device)
        picked_grad = stack_ls(picked_grad, device)

        refined_outs = self.by_smear(picked_lb, picked_ub, picked_extra, picked_grad, tiny_width)
        new_lb, new_ub = refined_outs[:2]
        new_extra = None if picked_extra is None else refined_outs[2]
        return new_lb, new_ub, new_extra

    def by_smear(self, new_rem_lb: Tensor, new_rem_ub: Tensor, new_rem_extra: Optional[Tensor], new_rem_grad: Tensor,
                 tiny_width: float = None) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """ Experiment shows that smear = grad * dim_width as in ReluVal is the best heuristic tried so far. It's
            better than either one alone, and better than other indirect loss e.g., introduced over-approximated area.
        :return: if new_rem_extra is None, return <refined LB, UB> without extra, otherwise return with extra
        """
        with torch.no_grad():
            width = new_rem_ub - new_rem_lb
            assert new_rem_lb.dim() == 2, 'Otherwise, I need to reduce the >2 dims to compute dim width?'
            smears = new_rem_grad * width / 2

            if tiny_width is not None:
                # consider only those dimensions that are not tiny
                not_tiny_bits = width > tiny_width
                smears = smears * not_tiny_bits.float()

            _, split_idxs = smears.max(dim=-1)
            return self.bisect_by(new_rem_lb, new_rem_ub, split_idxs, new_rem_extra)

    @staticmethod
    def bisect_by(lb: Tensor, ub: Tensor, idxs: Tensor,
                  extra: Tensor = None) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """ Bisect specific columns.
        :param idxs: <Batch>, as the indices from torch.max()
        :param extra: if not None, it contains the bit vector for each LB/UB piece showing which prop they should obey
        :return: <New LB, New UB> if extra is None, otherwise <New LB, New UB, New Extra>
        """
        # scatter_() to convert indices into one-hot encoding
        split_idxs = idxs.unsqueeze(dim=-1)  # Batch x 1
        onehot_idxs = torch.zeros_like(lb).byte().scatter_(-1, split_idxs, 1)

        # then bisect the specified cols only
        mid = (lb + ub) / 2.0
        lefts_lb = lb
        lefts_ub = torch.where(onehot_idxs, mid, ub)
        rights_lb = torch.where(onehot_idxs, mid, lb)
        rights_ub = ub

        newlb = cat0(lefts_lb, rights_lb)
        newub = cat0(lefts_ub, rights_ub)
        if extra is None:
            return newlb, newub

        newextra = cat0(extra, extra)
        return newlb, newub, newextra
    pass
