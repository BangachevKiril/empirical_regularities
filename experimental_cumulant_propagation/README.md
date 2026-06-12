# Experimental CP4 cumulant propagation

This package is a prototype for approximate structured CP4 propagation of the
unknown-parameter ICA fourth cumulant.

The mode is approximate. It follows the K=4 cumulant recurrence idea, but when
a dense two-index edge connects two rank-dependent endpoints it projects that
edge to low rank before the term is allowed to cross a layer boundary. Setting
the edge rank to `n`, when affordable, recovers the exact hard-edge
representation for that primitive.

The implementation here is intentionally separate from the existing exact
`FactoredTensor4`/Pair4 path. It provides:

- ordered CP tensor terms with shape `(n, J)` factors,
- direct empirical ICA fourth-cumulant input construction in CP4 form,
- hard Hadamard edge compression by SVD/eigh/randomized SVD-compatible API,
- reconstruction and batch diagonal quadratic smoke tests,
- simple cost bookkeeping for the target `O(L m n^2 + poly(L) n^4)` regime.

The full K=4 nonlinear recurrence should be wired through the structured
product builder in this package once the primitive tests and rank-budget
behavior are satisfactory.
