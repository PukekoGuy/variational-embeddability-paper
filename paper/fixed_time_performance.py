"""Run the fixed-time performance experiment and plot MISE and fitting time.

Run: python -m paper.fixed_time_performance --mu0 .01 --mu1 .01 --no-show
Replot: python -m paper.fixed_time_performance --plot-only --no-show
"""

import argparse
from pathlib import Path

from experiments.comparisons import generate_fixed_time_performance
from plotting.fixed_time_performance import plot_fixed_time_performance

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
DEFAULT_RESULTS_FILE = RESULTS_DIR / 'fixed_time_performance_results.csv'
DEFAULT_FIGURE_FILE = RESULTS_DIR / 'fixed_time_performance.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-file', type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument('--figure-file', type=Path, default=DEFAULT_FIGURE_FILE)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--mu0', type=float, default=.01)
    parser.add_argument('--mu1', type=float, default=.01)
    parser.add_argument('--individuals', type=int, nargs='+')
    parser.add_argument('--num-observations', type=int, default=20)
    parser.add_argument('--num-intervals', type=int, default=100)
    parser.add_argument('--num-trials', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1008)
    args = parser.parse_args()
    if not args.plot_only:
        generate_fixed_time_performance(
            args.results_file, mu0=args.mu0, mu1=args.mu1,
            num_observations=args.num_observations, num_intervals=args.num_intervals,
            num_individuals_list=args.individuals, num_trials=args.num_trials,
            max_iter=args.max_iter, seed=args.seed,
        )
    plot_fixed_time_performance(args.results_file, args.figure_file, show=not args.no_show)


if __name__ == '__main__':
    main()
