"""Fit an absorbing birth-death model to the bundled S&P 1993 rating transition matrix."""

from __future__ import annotations
import argparse
from pathlib import Path
from experiments import load_transition_matrix
from plotting.matrix_outputs import save_birth_death_results, print_birth_death_summary
from solvers import StrongBirthDeathEmbedder

DEFAULT_RATING_LABELS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "sp1993_embeddability_8x8_numeric.csv"
DEFAULT_OUTPUT_PREFIX = Path(__file__).resolve().parents[1] / "results" / "strong_birth_death_fit"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a constant birth-death generator A to Pi via exp(A)."
    )
    parser.add_argument(
        "pi_csv",
        nargs="?",
        default=DEFAULT_INPUT,
        help="numeric CSV containing the target transition matrix Pi",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0e4,
        help="weight on ||exp(A)-Pi||_F^2 (default: 1e4)",
    )
    parser.add_argument(
        "--mu0",
        type=float,
        default=1.0e-4,
        help="weight on ||A||_F^2 (default: 1e-4)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="maximum L-BFGS-B iterations",
    )
    parser.add_argument(
        "--rate-upper-bound",
        type=float,
        default=None,
        help="optional common upper bound for all rates",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="prefix for output CSV files",
    )
    parser.add_argument(
        "--decimal-places", type=int, default=6,
        help="decimal places in console, CSV and LaTeX output (default: 6)",
    )
    args = parser.parse_args()
    if args.decimal_places < 0:
        parser.error("--decimal-places must be nonnegative")

    Pi = load_transition_matrix(args.pi_csv)
    print(f"Loaded Pi from {args.pi_csv} with shape {Pi.shape}.")

    solver = StrongBirthDeathEmbedder(
        kappa=args.kappa,
        mu0=args.mu0,
        max_iter=args.max_iter,
        rate_upper_bound=args.rate_upper_bound,
    )
    A_hat, P_hat, rates_hat, result = solver.optimise(Pi)

    labels = DEFAULT_RATING_LABELS if Pi.shape[0] == 8 else None
    print_birth_death_summary(Pi, A_hat, P_hat, rates_hat, result, args.decimal_places, labels=labels)
    save_birth_death_results(args.output_prefix, A_hat, P_hat, rates_hat, args.decimal_places, labels=labels)
    print(f"\nSaved results with prefix: {args.output_prefix}")

if __name__ == "__main__":
    main()
