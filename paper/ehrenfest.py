"""Ehrenfest generator estimation from periodic observations.

Run: python run_example.py ehrenfest --no-show
Replot: python run_example.py ehrenfest --plot-only --no-show

Custom volumes:
    generate_data(volume_a_fn=lambda t: np.exp(-t),
                  volume_b_fn=lambda t: 1.)
    plot_ehrenfest_results("results/ehrenfest_results.csv",
                          "results/ehrenfest_mise.png")

Both volumes default to 1. Volumes must be finite and positive.
num_particles counts particles within a chain; individuals counts independent
chains. A Poisson clock of rate jump_frequency converts move probabilities
into continuous-time rates. The reported error is the trial average of
mean((estimate[:-1] - true_A[:-1])**2) / num_intervals, including diagonals.
This normalisation is not the integrated squared Frobenius error.
"""
import argparse
from pathlib import Path

from plotting.ehrenfest_results import plot_ehrenfest_results
from experiments.ehrenfest import DEFAULT_OUTPUT_PREFIX, generate_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-prefix', type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--num-particles', type=int, default=6)
    parser.add_argument('--jump-frequency', type=float, default=6.)
    parser.add_argument('--individuals', type=int, nargs='+', default=[100, 300, 1000, 3000, 10000])
    parser.add_argument('--num-observations', type=int, default=20)
    parser.add_argument('--num-intervals', type=int, default=190)
    parser.add_argument('--num-trials', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1008)
    args = parser.parse_args()
    if not args.plot_only:
        generate_data(output_prefix=args.output_prefix, num_particles=args.num_particles,
                      jump_frequency=args.jump_frequency, individuals=args.individuals,
                      num_observations=args.num_observations, num_intervals=args.num_intervals,
                      num_trials=args.num_trials, max_iter=args.max_iter, seed=args.seed)
    plot_ehrenfest_results(f'{args.output_prefix}_results.csv',
                           f'{args.output_prefix}_mise.png', show=not args.no_show)


if __name__ == '__main__':
    main()
