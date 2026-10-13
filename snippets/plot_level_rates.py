"""Draw birth and death rates from already computed generator arrays."""
import numpy as np
from experiments import birth_death_generator
from plotting import plot_level_rates


def main():
    generator = birth_death_generator(3, [0.8, 0.4], [0.3, 0.5])
    true_path = np.stack([generator(t) for t in np.linspace(0., 1., 21)])
    example_estimate = 0.9 * true_path
    fig, axes = plot_level_rates([example_estimate], true_path, show=False)
    # fig.savefig('level_rates.png')
    print(f'plot axes={axes.shape}, finite={np.isfinite(example_estimate).all()}')
    return fig


if __name__ == '__main__':
    main()
