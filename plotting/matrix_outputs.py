"""LaTeX formatting for numerical matrix outputs."""

from pathlib import Path

import numpy as np


def save_latex_matrices(path, matrices, decimal_places=6):
    """Write (LaTeX label, matrix) pairs as display equations in a text file."""
    if decimal_places < 0:
        raise ValueError("decimal_places must be nonnegative")
    equations = []
    for label, matrix in matrices:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("each matrix must be two-dimensional")
        rows = [
            " & ".join(f"{value:.{decimal_places}f}" for value in row)
            for row in matrix
        ]
        body = " \\\\\n".join(rows)
        equations.append(
            "\\[\n" + label + " = \\begin{bmatrix}\n"
            + body + "\n\\end{bmatrix}\n\\]"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(equations) + "\n", encoding="utf-8")


def save_birth_death_results(
    output_prefix: str | Path,
    A_hat: np.ndarray,
    P_hat: np.ndarray,
    rates_hat: np.ndarray,
    decimal_places: int = 6,
    *, labels=None,
) -> None:
    """Save CSV results and a text file containing the matrices in LaTeX."""
    if decimal_places < 0:
        raise ValueError("decimal_places must be nonnegative.")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(f"{prefix}_generator.csv", A_hat, delimiter=",", fmt=f"%.{decimal_places}f")
    np.savetxt(f"{prefix}_fitted_transition.csv", P_hat, delimiter=",", fmt=f"%.{decimal_places}f")
    save_latex_matrices(
        f"{prefix}_matrices_latex.txt",
        [(r"\widehat{A}", A_hat), (r"\exp(\widehat{A})", P_hat)],
        decimal_places,
    )

    rank = A_hat.shape[0]
    labels = [str(i) for i in range(rank)] if labels is None else labels
    rate_rows = []
    for i in range(rank - 1):
        rate_rows.append((f"lambda_{i}", labels[i], labels[i + 1], rates_hat[i]))
    for i in range(1, rank):
        idx = (rank - 1) + (i - 1)
        rate_rows.append((f"mu_{i}", labels[i], labels[i - 1], rates_hat[idx]))

    rate_path = Path(f"{prefix}_rates.csv")
    with rate_path.open("w", encoding="utf-8") as f:
        f.write("rate,from_state,to_state,value\n")
        for name, from_state, to_state, value in rate_rows:
            f.write(f"{name},{from_state},{to_state},{value:.{decimal_places}f}\n")


def print_birth_death_summary(
    Pi: np.ndarray,
    A_hat: np.ndarray,
    P_hat: np.ndarray,
    rates_hat: np.ndarray,
    result,
    decimal_places: int = 6,
    *, labels=None,
) -> None:
    rank = Pi.shape[0]
    labels = [str(i) for i in range(rank)] if labels is None else labels

    if decimal_places < 0:
        raise ValueError("decimal_places must be nonnegative.")

    print("\nOptimisation status")
    print("-------------------")
    print("success :", result.success)
    print("message :", result.message)
    print("nit     :", result.nit)
    print(f"fun     : {result.fun:.{decimal_places}f}")

    residual = P_hat - Pi
    print("\nFit")
    print("---")
    print(f"||exp(A)-Pi||_L1         = {np.linalg.norm(residual, 1):.{decimal_places}f}")
    print(f"||exp(A)-Pi||_L2         = {np.linalg.norm(residual, 2):.{decimal_places}f}")
    print(f"||exp(A)-Pi||_Frob       = {np.linalg.norm(residual, 'fro'):.{decimal_places}f}")
    print(f"||exp(A)-Pi||_infty      = {np.max(np.abs(residual)):.{decimal_places}f}")
    print(f"generator row-sum max = {np.max(np.abs(A_hat.sum(axis=1))):.{decimal_places}f}")

    print("\nEstimated rates")
    print("---------------")
    for i in range(rank - 1):
        print(f"lambda_{i:>2}  {labels[i]:>4} -> {labels[i+1]:<4}  {rates_hat[i]:.{decimal_places}f}")
    for i in range(1, rank):
        idx = (rank - 1) + (i - 1)
        print(f"mu_{i:>6}  {labels[i]:>4} -> {labels[i-1]:<4}  {rates_hat[idx]:.{decimal_places}f}")

    with np.printoptions(precision=decimal_places, floatmode="fixed",
                         suppress=True, linewidth=140):
        print("\nEstimated generator A")
        print(A_hat)
        print("\nFitted transition matrix exp(A)")
        print(P_hat)
