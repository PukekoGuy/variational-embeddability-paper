"""Six-state Ehrenfest rates over time from one fixed-size observed sample.

Run: python run_example.py ehrenfest_rates --no-show
Replot: python run_example.py ehrenfest_rates --plot-only --no-show

Five particles give six states and ten directed neighbouring-state rates.
The chamber volumes are V_A(t)=exp(-t) and V_B(t)=1 on [0, 1].
"""

import argparse
from pathlib import Path
import numpy as np

from experiments.ehrenfest import DEFAULT_OUTPUT_PREFIX, generate_data
from plotting.ehrenfest_rates import plot_ehrenfest_rates

DEFAULT_RATES_OUTPUT_PREFIX = DEFAULT_OUTPUT_PREFIX.with_name('ehrenfest_rates')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-prefix', type=Path, default=DEFAULT_RATES_OUTPUT_PREFIX)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--individuals', type=int, default=1000)
    parser.add_argument('--jump-frequency', type=float, default=6.)
    parser.add_argument('--num-observations', type=int, default=20)
    parser.add_argument('--num-intervals', type=int, default=190)
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1008)
    args = parser.parse_args()
    if not args.plot_only:
        if args.individuals < 1:
            parser.error('--individuals must be positive')
        if args.num_observations < 2:
            parser.error('--num-observations must be at least 2')
        if args.num_intervals < 1 or args.num_intervals % (args.num_observations - 1):
            parser.error('--num-intervals must be positive and divisible by --num-observations minus 1')
        generate_data(num_particles=5, jump_frequency=args.jump_frequency,
                      volume_a_fn=lambda t: np.exp(-t), volume_b_fn=lambda t: 1.,
                      individuals=(args.individuals,), num_observations=args.num_observations,
                      num_intervals=args.num_intervals, num_trials=1,
                      max_iter=args.max_iter, seed=args.seed, output_prefix=args.output_prefix)
    plot_ehrenfest_rates(f'{args.output_prefix}_generators.npz',
                         f'{args.output_prefix}_rates.png', show=not args.no_show)


if __name__ == '__main__':
    main()
