"""Plot an error curve from numbers you have already computed."""
import numpy as np
from plotting import plot_mise


def main():
    errors = [0.1, 0.03, 0.01]  # illustrative values, not paper results
    fig, ax = plot_mise([10, 30, 100], errors, show=False)
    # fig.savefig('mise.png')
    print(f'plot axes=1, finite={np.isfinite(errors).all()}')
    return fig


if __name__ == '__main__':
    main()
