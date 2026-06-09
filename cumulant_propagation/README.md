# Cumulant Propagation

This folder adapts the ARC `mlp_cumulant_propagation` implementation to this
repository's `DeepReLUMLP`.

```python
import torch

from cumulant_propagation import propagate_cumulants
from inference_models import DeepReLUMLP

n = 16
model = DeepReLUMLP(n=n, L=3, dtype=torch.float64)

# Caller-controlled initialization. These can later come from ICA.
K_in = {
    1: torch.zeros(n, dtype=torch.float64),
    2: torch.eye(n, dtype=torch.float64),
    3: torch.zeros(n, n, n, dtype=torch.float64),
}

K_out = propagate_cumulants(
    model,
    K_in,
    k_max=3,
    return_tensors=True,
)
estimated_mean = K_out[1]
```

`K_in` is not assumed to be Gaussian. Orders 1 and 2 must be present because
the ReLU Wick coefficients are expanded around the initialized mean and
variance. Higher missing orders are treated as absent/zero by the propagation
routine.

The local `DeepReLUMLP` applies ReLU after every weight matrix, so layer labels
are `pre0`, `act0`, ..., `pre{L-1}`, `act{L-1}`. The final output is
`act{L-1}`.

The vendored ARC core is MIT licensed; see
`ARC_MLP_CUMULANT_PROPAGATION_LICENSE`.
