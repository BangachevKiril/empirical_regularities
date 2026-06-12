from __future__ import annotations

import argparse

import torch

from experimental_cumulant_propagation.cp4_input import direct_ica_cp4_cumulants_from_samples
from experimental_cumulant_propagation.structured_cp import (
    EdgeCompressionConfig,
    HardHadamardEdge,
    batch_diagonal_quadratic_compressed,
    batch_diagonal_quadratic_exact,
    compress_hard_edge,
    cp_contract_W,
    cp_slice_D,
    cp_to_dense,
    dense_hard_edge,
)
from experimental_cumulant_propagation.structured_k4_approx import compress_hard_edges


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tol: float) -> None:
    err = torch.max(torch.abs(actual - expected)).item()
    if err > tol:
        raise AssertionError(f"{name} max error {err:.3e} exceeds tolerance {tol:.3e}")


def test_edge_compression_exactness(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    n = 5
    J = 4
    edge = torch.randn(n, n, device=device, dtype=dtype)
    U = torch.randn(n, J, device=device, dtype=dtype)
    V = torch.randn(n, J, device=device, dtype=dtype)
    T = torch.randn(n, J, device=device, dtype=dtype)
    term = HardHadamardEdge(edge=edge, left_factor=U, right_factor=V, trailing_factors=(T,), output_degree=3)
    compressed, report = compress_hard_edge(
        term,
        EdgeCompressionConfig(explicit_edge_rank=n, allow_exact_when_affordable=True),
    )
    _assert_close("hard-edge reconstruction", cp_to_dense(compressed), dense_hard_edge(term), 2e-5)
    if report.edge_rank != n or report.residual_fro != 0.0:
        raise AssertionError("full-rank compression should report exact residuals")


def test_batch_diagonal_quadratic(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(12)
    n = 6
    J = 7
    x = 4
    edge = torch.randn(n, n, device=device, dtype=dtype)
    U = torch.randn(n, J, device=device, dtype=dtype)
    V = torch.randn(n, J, device=device, dtype=dtype)
    L = torch.randn(x, n, device=device, dtype=dtype)
    R = torch.randn(x, n, device=device, dtype=dtype)
    term = HardHadamardEdge(edge=edge, left_factor=U, right_factor=V, output_degree=2)

    exact = batch_diagonal_quadratic_exact(L, R, edge, U, V)
    compressed_full, _ = compress_hard_edge(
        term,
        EdgeCompressionConfig(explicit_edge_rank=n, allow_exact_when_affordable=True),
    )
    approx_full = batch_diagonal_quadratic_compressed(L, R, compressed_full, edge_rank=n)
    _assert_close("batch diagonal full rank", approx_full, exact, 3e-5)

    errors = []
    residuals = []
    for rank in range(1, n + 1):
        compressed, report = compress_hard_edge(
            term,
            EdgeCompressionConfig(explicit_edge_rank=rank, allow_exact_when_affordable=False),
        )
        approx = batch_diagonal_quadratic_compressed(L, R, compressed, edge_rank=rank)
        errors.append(torch.linalg.vector_norm(approx - exact).item())
        residuals.append(report.residual_fro)
    if any(residuals[index + 1] > residuals[index] + 1e-5 for index in range(len(residuals) - 1)):
        raise AssertionError(f"SVD residuals are not monotone nonincreasing: {residuals}")
    if errors[-1] > 3e-5:
        raise AssertionError(f"full-rank primitive error should be tiny, got {errors[-1]:.3e}")


def test_cp4_input_and_slices(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(13)
    samples = torch.randn(8, 4, device=device, dtype=dtype)
    cumulants = direct_ica_cp4_cumulants_from_samples(samples=samples, p=16)
    if cumulants.sample_fourth.rank != samples.shape[0]:
        raise AssertionError("CP4 sample rank should equal the sample count")

    W = torch.randn(4, 4, device=device, dtype=dtype) / 2.0
    propagated = cp_contract_W(cumulants.sample_fourth, W)
    dense_propagated = cp_to_dense(propagated)
    dense_direct = torch.einsum("ia,jb,kc,ld,abcd->ijkl", W, W, W, W, cp_to_dense(cumulants.sample_fourth))
    _assert_close("CP4 linear contraction", dense_propagated, dense_direct, 3e-5)

    slices = cp_slice_D(cumulants.sample_fourth, (2, 1, 1))
    if not slices:
        raise AssertionError("symmetric CP4 slice should produce ordered slices")


def test_hard_edge_batch_report(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(14)
    n = 4
    J = 3
    edge = torch.randn(n, n, device=device, dtype=dtype)
    U = torch.randn(n, J, device=device, dtype=dtype)
    V = torch.randn(n, J, device=device, dtype=dtype)
    hard_edge = HardHadamardEdge(edge=edge, left_factor=U, right_factor=V, output_degree=2)
    terms, report = compress_hard_edges(
        (hard_edge,),
        EdgeCompressionConfig(explicit_edge_rank=n),
        sample_budget_m=J,
    )
    if report.hard_edges_compressed != 1 or len(terms) != 1:
        raise AssertionError("hard-edge batch compression report has wrong counts")
    _assert_close("batched hard-edge reconstruction", cp_to_dense(terms[0]), dense_hard_edge(hard_edge), 2e-5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    test_edge_compression_exactness(device, dtype)
    test_batch_diagonal_quadratic(device, dtype)
    test_cp4_input_and_slices(device, dtype)
    test_hard_edge_batch_report(device, dtype)
    print("experimental_cumulant_propagation smoke tests passed")


if __name__ == "__main__":
    main()
