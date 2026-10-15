"""Fixed versus jump-time experiments with batched simulation.

Run: python run_example.py fixed_vs_jump
Quick run: python run_example.py fixed_vs_jump --individuals 100 1000 --max-iter 20
Plot saved data: python run_example.py fixed_vs_jump --plot-only

Each method receives a separate sample from the same Euler model. Jump
observations are stored as compressed paths rather than a full time-grid panel.
"""

import argparse
from pathlib import Path

from experiments.comparisons import generate_jump_comparison, JUMP_RESULTS_FILE
from plotting.comparison_results import plot_fixed_vs_jump_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-file", type=Path, default=JUMP_RESULTS_FILE)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--figure-file", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--individuals", type=int, nargs="+",
                        help="Sample sizes; default: six log-spaced sizes from 1 to 1,000,000")
    parser.add_argument("--num-intervals", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--num-trials", type=int, default=5,
                        help="Independent trials per sample size (default: 5)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mu0", type=float, default=1., help="Generator magnitude penalty weight")
    parser.add_argument("--mu1", type=float, default=1., help="Generator derivative penalty weight")
    args = parser.parse_args()
    if not args.plot_only:
        generate_jump_comparison(args.results_file, num_individuals_list=args.individuals,
                      num_intervals=args.num_intervals, max_iter=args.max_iter,
                      seed=args.seed, batch_size=args.batch_size, num_trials=args.num_trials,
                      mu0=args.mu0, mu1=args.mu1)
    plot_fixed_vs_jump_results(args.results_file, args.figure_file, show=not args.no_show)


if __name__ == "__main__":
    main()
