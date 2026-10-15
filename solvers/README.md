# Solvers

[Paper experiment commands](../paper/README.md)

Every solver uses an unweighted data-fit term. Set `mu0` and `mu1` directly;
there is no `kappa` or combined `mu` argument.

All time-varying solvers (`FixedTimeAdjoint`, `FixedTimeGradient`,
`JumpTimeGradient`, `BirthDeathEmbedder`, and `BirthDeathJumpTimeGradient`)
use the full Sobolev penalty

```text
R(A) = (mu0 * dt / 2) sum_{k=0}^{N-1} ||A_k||_F^2
     + (mu1 / (2 * dt)) sum_{k=0}^{N-1} ||A_{k+1} - A_k||_F^2.
```

This uses a simple left-endpoint Riemann sum for the magnitude integral and
forward differences for the derivative integral, on the existing N+1 point grid.
Both norms include all generator entries, including diagonals. The final grid
point contributes to the last derivative difference but not the left Riemann sum.
The birth-death solver enforces its absorbing-state constraint in both terms.

Pass keyword arguments `mu0` (magnitude) and `mu1` (derivative) to set separate
nonnegative, finite weights. Both default to `.01` in time-varying solvers.
`mu0=0` disables the magnitude term. The constant-generator
`StrongBirthDeathEmbedder` uses only the magnitude weight `mu0`.

`BirthDeathEmbedder` fits periodic snapshots, while `BirthDeathJumpTimeGradient`
fits compressed jump histories using the discrete Euler jump/no-jump likelihood
of `JumpTimeGradient`. Both fit only `2*(rank-1)` neighbouring-state rates per
grid point and construct their row-sum-zero diagonals from those rates. They
default to an absorbing final state; use `absorbing_final=False` for Ehrenfest.

```python
from solvers import BirthDeathJumpTimeGradient

solver = BirthDeathJumpTimeGradient(6, num_intervals=190,
                                    absorbing_final=False, mu0=.01, mu1=.01)
# jump_times[m] contains strictly increasing interval indices, not real times.
# states[m] contains the initial state followed by one new state per jump.
estimate, status = solver.optimise_A(jump_times, states, return_result=True)
```

Stays, including the interval after the final recorded jump through time 1,
contribute to the likelihood. Transition counts are prepared once per fit.

```sh
python -m snippets.fixed_time_adjoint
python -m snippets.fixed_time_gradient
python -m snippets.jump_time_gradient
python -m snippets.weak_birth_death
python -m snippets.strong_birth_death
```
