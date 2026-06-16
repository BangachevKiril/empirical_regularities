from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import FunctionType
from typing import Iterator

import cumulant_propagation._arc_mlp_kprop.factor_k4 as _factor_k4


_ORIGINAL_FACTORED_NONLIN_KPROP_K4: FunctionType | None = None
_REDUCED_FACTORED_NONLIN_KPROP_K4: FunctionType | None = None


def vec_under_degree(vec_part) -> int:
    return int(sum(sum(block) for block in vec_part))


def _reduced_1111_block_source() -> str:
    return '''    with flop_name("nonlin_sum 1111 factored"):
        # In the literal |k_under| <= 4 algorithm, the carried single K4
        # all-distinct factor above is retained, but every leaf-splittable
        # path/star/3+2 addition has underlying degree at least five:
        #
        #   path/star: three pair edges, minimum 2 + 2 + 2 = 6
        #   3+2:      one pair edge plus one triple block, minimum 2 + 3 = 5
        #
        # So no extra all-distinct hard-edge factor bank is added here.
        pass

'''


def build_reduced_factored_nonlin_kprop_k4() -> FunctionType:
    """Build a K=4 nonlinear step with explicit |k_under| <= 4 filtering."""
    global _REDUCED_FACTORED_NONLIN_KPROP_K4
    if _REDUCED_FACTORED_NONLIN_KPROP_K4 is not None:
        return _REDUCED_FACTORED_NONLIN_KPROP_K4

    source = inspect.getsource(_factor_k4.factored_nonlin_kprop_k4)
    old_filter = "        and int_part != (1, 1, 1, 1)   # Factor this manually\n"
    new_filter = (
        old_filter
        + "        and _rank0_optimal_vec_under_degree(vec_part) <= 4\n"
    )
    if old_filter not in source:
        raise RuntimeError("Could not locate K4 nonlinearity term filter.")
    source = source.replace(old_filter, new_filter, 1)

    start_marker = '    with flop_name("nonlin_sum 1111 factored"):\n'
    end_marker = "        # WK2111 -> pK1111\n"
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    source = source[:start] + _reduced_1111_block_source() + source[end:]

    namespace = dict(vars(_factor_k4))
    namespace["_rank0_optimal_vec_under_degree"] = vec_under_degree
    compiled = compile(
        source,
        f"{__file__}:reduced_factored_nonlin_kprop_k4",
        "exec",
    )
    exec(compiled, namespace)
    reduced = namespace["factored_nonlin_kprop_k4"]
    reduced.__name__ = "reduced_factored_nonlin_kprop_k4"
    reduced.__qualname__ = "reduced_factored_nonlin_kprop_k4"
    _REDUCED_FACTORED_NONLIN_KPROP_K4 = reduced
    return reduced


def install() -> FunctionType:
    global _ORIGINAL_FACTORED_NONLIN_KPROP_K4
    if _ORIGINAL_FACTORED_NONLIN_KPROP_K4 is None:
        _ORIGINAL_FACTORED_NONLIN_KPROP_K4 = _factor_k4.factored_nonlin_kprop_k4
    reduced = build_reduced_factored_nonlin_kprop_k4()
    _factor_k4.factored_nonlin_kprop_k4 = reduced
    return reduced


def uninstall() -> None:
    if _ORIGINAL_FACTORED_NONLIN_KPROP_K4 is not None:
        _factor_k4.factored_nonlin_kprop_k4 = _ORIGINAL_FACTORED_NONLIN_KPROP_K4


@contextmanager
def patched() -> Iterator[FunctionType]:
    reduced = install()
    try:
        yield reduced
    finally:
        uninstall()
