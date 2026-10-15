"""Compare adjoint and direct-gradient fixed-time fitting.

Run: python run_example.py adjoint_vs_gradient
Quick run: python run_example.py adjoint_vs_gradient --individuals 100 1000 --max-iter 20
Plot saved data: python run_example.py adjoint_vs_gradient --plot-only
Plot one method: python run_example.py adjoint_vs_gradient --plot-only --method adjoint

Both methods fit the same panel in each trial. The simulator caches Euler
transition matrices and draws observations in batches.
"""

import argparse
from pathlib import Path

from experiments.comparisons import generate_adjoint_comparison, ADJOINT_RESULTS_FILE
from plotting.comparison_results import plot_adjoint_vs_gradient_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-file", type=Path, default=ADJOINT_RESULTS_FILE)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--method", choices=("both", "adjoint", "gradient"), default="both",
                        help="Method(s) to plot; data generation still fits both (default: both)")
    parser.add_argument("--figure-file", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--individuals", type=int, nargs="+",
                        help="Panel sizes; default: 12 geometrically spaced sizes from 1 to 1,000,000")
    parser.add_argument("--num-intervals", type=int, default=100)
    parser.add_argument("--num-observations", type=int, default=20)
    parser.add_argument("--target-jumps", type=float, default=1)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--num-trials", type=int, default=5,
                        help="Independent trials per sample size (default: 5)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mu0", type=float, default=.01, help="Generator magnitude penalty weight")
    parser.add_argument("--mu1", type=float, default=.01, help="Generator derivative penalty weight")
    args = parser.parse_args()
    if not args.plot_only:
        generate_adjoint_comparison(
            args.results_file, target_jumps=args.target_jumps,
            num_observations=args.num_observations, num_intervals=args.num_intervals,
            num_individuals_list=args.individuals, max_iter=args.max_iter, seed=args.seed,
            num_trials=args.num_trials, mu0=args.mu0, mu1=args.mu1,
        )
    plot_adjoint_vs_gradient_results(args.results_file, args.figure_file, show=not args.no_show,
                           method=args.method)


if __name__ == "__main__":
    main()
