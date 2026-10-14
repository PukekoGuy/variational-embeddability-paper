"""Run an application stored in the paper package.

Usage: python run_example.py EXAMPLE [EXAMPLE OPTIONS]
"""
import argparse
import importlib
from pathlib import Path
import sys

EXAMPLES = ('comparison', 'fixed_vs_jump', 'adjoint_vs_gradient',
            'ehrenfest', 'ehrenfest_rates', 'financial_birth_death')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', choices=EXAMPLES)
    parser.add_argument('options', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = importlib.import_module(f'paper.{args.example}')
    sys.argv = [f'python {Path(__file__).name} {args.example}', *args.options]
    module.main()


if __name__ == '__main__':
    main()
