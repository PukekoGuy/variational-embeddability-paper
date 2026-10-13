"""Four-panel comparison of fixed-time and jump-time generator estimates.

Run: python run_example.py comparison --no-show
Replot: python run_example.py comparison --plot-only --no-show
"""
import argparse
from pathlib import Path
from plotting.comparison_panels import plot_comparison_results
from experiments.comparisons import generate_comparison, COMPARISON_RESULTS_FILE

DEFAULT_FIGURE_FILE = COMPARISON_RESULTS_FILE.with_suffix('.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-file', type=Path, default=COMPARISON_RESULTS_FILE)
    parser.add_argument('--figure-file', type=Path, default=DEFAULT_FIGURE_FILE)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--individuals', type=int, nargs='+', default=[1000] * 10)
    parser.add_argument('--num-observations', type=int, nargs=2, default=[20, 10])
    parser.add_argument('--constant', type=float, default=10.)
    parser.add_argument('--num-intervals', type=int, default=1000)
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if not args.plot_only:
        generate_comparison(args.results_file, args.individuals, args.num_observations, args.constant,
                      num_intervals=args.num_intervals, max_iter=args.max_iter, seed=args.seed)
    plot_comparison_results(args.results_file, args.figure_file, show=not args.no_show)


if __name__ == '__main__':
    main()
