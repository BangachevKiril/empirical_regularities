from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import FunctionType
from typing import Iterator

import torch
from torch import Tensor

import cumulant_propagation._arc_mlp_kprop.factor_k4 as _factor_k4


_ORIGINAL_FACTORED_NONLIN_KPROP_K4: FunctionType | None = None
_OPTIMIZED_FACTORED_NONLIN_KPROP_K4: FunctionType | None = None


def _accumulate_sum(acc: Tensor | None, term: Tensor) -> Tensor:
    if acc is None:
        return term.clone()
    acc.add_(term)
    return acc


def _leaf_left_factor(A_term: Tensor) -> Tensor:
    n = A_term.shape[0]
    A = torch.zeros((n, n, n), device=A_term.device, dtype=A_term.dtype)
    idx = torch.arange(n, device=A_term.device)
    A[:, idx, idx] = A_term
    return A


def expected_grouped_hard2_simple_flops(
    n: int,
    *,
    include_repeated_211: bool = False,
) -> int:
    """Manual FLOP formula from the hard-2 edge brief for simple K=4."""
    n_i = int(n)
    flops = 62 * n_i**3 + 80 * n_i**2
    if include_repeated_211:
        rank = 2 * n_i
        flops += (8 * rank + 3) * n_i**3
    return flops


def expected_ungrouped_hard2_simple_flops(n: int) -> int:
    """Manual FLOP formula for the old ungrouped simple K=4 hard-edge block."""
    n_i = int(n)
    return 434 * n_i**3 + 6 * n_i**2


def _optimized_hard2_block_source() -> str:
    return '''    with flop_name("nonlin_sum 1111 factored"):
        # Optimized exact hard-2 leaf-splittable edge construction.
        #
        # This is algebraically identical to the old path/star/3+2 incidence
        # loops, but groups all incidence choices at the n^2 level before doing
        # dense n^3 products. It also builds the leaf factor by assignment
        # instead of _einsum_delta and adds the two rank-n blocks in one call.
        w_cache = {q: w(q) for q in range(1, 7)}
        w12_cache = {q: 12.0 * w_cache[q] for q in range(1, 7)}
        w4_cache = {q: 4.0 * w_cache[q] for q in range(1, 7)}

        pair = {}
        for a in (1, 2):
            for b in (1, 2):
                pair[(a, b)] = dsWK(a, b)

        triple = {}
        for a in (1, 2):
            for b in (1, 2):
                for c in (1, 2):
                    triple[(a, b, c)] = dsWK(a, b, c)

        path_Y = {}
        for yk in (1, 2):
            acc = None
            for yl in (1, 2):
                Y = pair[(yk, yl)]
                if Y is None:
                    continue
                acc = _hard2_accumulate_sum(acc, Y * w_cache[yl][None, :])
            path_Y[yk] = acc

        star_X = {}
        for xj in (1, 2):
            acc = None
            for xk in (1, 2):
                X = pair[(xj, xk)]
                if X is None:
                    continue
                acc = _hard2_accumulate_sum(acc, X * w_cache[xk][None, :])
            star_X[xj] = acc

        star_Y = {}
        for yj in (1, 2):
            acc = None
            for yl in (1, 2):
                Y = pair[(yj, yl)]
                if Y is None:
                    continue
                acc = _hard2_accumulate_sum(acc, Y * w_cache[yl][None, :])
            star_Y[yj] = acc

        new_As = []
        new_Bs = []
        tmp = torch.empty((n, n, n), device=mean.device, dtype=mean.dtype)

        for A_j_inc in (1, 2):
            A_term = None
            dsA1 = pair[(1, A_j_inc)]
            if dsA1 is not None:
                A_term = _hard2_accumulate_sum(A_term, dsA1 * w_cache[1][:, None])
            dsA2 = pair[(2, A_j_inc)]
            if dsA2 is not None:
                A_term = _hard2_accumulate_sum(A_term, dsA2 * w_cache[2][:, None])
            if A_term is None:
                continue

            A = _hard2_leaf_left_factor(A_term)
            B_jkl = None

            path_sum = None
            for yk in (1, 2):
                Ysum = path_Y[yk]
                if Ysum is None:
                    continue

                Xsum = None
                for xj in (1, 2):
                    for xk in (1, 2):
                        X = pair[(xj, xk)]
                        if X is None:
                            continue
                        term = X * w_cache[A_j_inc + xj][:, None]
                        term = term * w_cache[xk + yk][None, :]
                        Xsum = _hard2_accumulate_sum(Xsum, term)
                if Xsum is None:
                    continue

                Xsum.mul_(12.0)
                torch.mul(Xsum[:, :, None], Ysum[None, :, :], out=tmp)
                path_sum = _hard2_accumulate_sum(path_sum, tmp)

            if path_sum is not None:
                B_jkl = _hard2_accumulate_sum(B_jkl, path_sum)

            star_sum = None
            for xj in (1, 2):
                Xbase = star_X[xj]
                if Xbase is None:
                    continue
                for yj in (1, 2):
                    Ybase = star_Y[yj]
                    if Ybase is None:
                        continue
                    Xweighted = Xbase * w4_cache[A_j_inc + xj + yj][:, None]
                    torch.mul(Xweighted[:, :, None], Ybase[:, None, :], out=tmp)
                    star_sum = _hard2_accumulate_sum(star_sum, tmp)

            if star_sum is not None:
                B_jkl = _hard2_accumulate_sum(B_jkl, star_sum)

            triple_sum = None
            for bj in (1, 2):
                for bk in (1, 2):
                    for bl in (1, 2):
                        T = triple[(bj, bk, bl)]
                        if T is None:
                            continue
                        tmp.copy_(T)
                        tmp.mul_(w12_cache[A_j_inc + bj][:, None, None])
                        tmp.mul_(w_cache[bk][None, :, None])
                        tmp.mul_(w_cache[bl][None, None, :])
                        triple_sum = _hard2_accumulate_sum(triple_sum, tmp)

            if triple_sum is not None:
                B_jkl = _hard2_accumulate_sum(B_jkl, triple_sum)

            if B_jkl is None:
                continue

            new_As.append(A)
            new_Bs.append(B_jkl.permute(1, 2, 0).contiguous())

        if new_As:
            pK_1111.add_factors_((torch.cat(new_As, dim=2), torch.cat(new_Bs, dim=2)))

'''


def build_optimized_factored_nonlin_kprop_k4() -> FunctionType:
    """Build the optimized K=4 nonlinear function without editing source files."""
    global _OPTIMIZED_FACTORED_NONLIN_KPROP_K4
    if _OPTIMIZED_FACTORED_NONLIN_KPROP_K4 is not None:
        return _OPTIMIZED_FACTORED_NONLIN_KPROP_K4

    source = inspect.getsource(_factor_k4.factored_nonlin_kprop_k4)
    start_marker = '    with flop_name("nonlin_sum 1111 factored"):\n'
    end_marker = "        # WK2111 -> pK1111\n"
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    source = source[:start] + _optimized_hard2_block_source() + source[end:]

    namespace = dict(vars(_factor_k4))
    namespace["_hard2_accumulate_sum"] = _accumulate_sum
    namespace["_hard2_leaf_left_factor"] = _leaf_left_factor
    compiled = compile(
        source,
        f"{__file__}:optimized_factored_nonlin_kprop_k4",
        "exec",
    )
    exec(compiled, namespace)
    optimized = namespace["factored_nonlin_kprop_k4"]
    optimized.__name__ = "optimized_factored_nonlin_kprop_k4"
    optimized.__qualname__ = "optimized_factored_nonlin_kprop_k4"
    _OPTIMIZED_FACTORED_NONLIN_KPROP_K4 = optimized
    return optimized


def install() -> FunctionType:
    """Install the grouped hard-2 implementation for the current Python process."""
    global _ORIGINAL_FACTORED_NONLIN_KPROP_K4
    if _ORIGINAL_FACTORED_NONLIN_KPROP_K4 is None:
        _ORIGINAL_FACTORED_NONLIN_KPROP_K4 = _factor_k4.factored_nonlin_kprop_k4
    optimized = build_optimized_factored_nonlin_kprop_k4()
    _factor_k4.factored_nonlin_kprop_k4 = optimized
    return optimized


def uninstall() -> None:
    """Restore the repository implementation if this module installed a patch."""
    global _ORIGINAL_FACTORED_NONLIN_KPROP_K4
    if _ORIGINAL_FACTORED_NONLIN_KPROP_K4 is not None:
        _factor_k4.factored_nonlin_kprop_k4 = _ORIGINAL_FACTORED_NONLIN_KPROP_K4


@contextmanager
def patched() -> Iterator[FunctionType]:
    optimized = install()
    try:
        yield optimized
    finally:
        uninstall()
