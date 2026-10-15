import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from experiments import ehrenfest as e
from plotting.ehrenfest_results import plot_ehrenfest_results
from solvers.weak_birth_death_adjoint import BirthDeathEmbedder
from solvers import BirthDeathJumpTimeGradient

class EhrenfestTests(unittest.TestCase):
    def test_paired_observations_are_snapshots_of_the_same_jump_histories(self):
        from experiments.observations import Observation
        times = np.array([0., .5, 1.])
        sampler = Observation(e.ehrenfest_generator(2, 2.), 3, times, 10, seed=8)
        fixed, jumps, histories = e.simulate_paired_observations(
            sampler, times, 30, np.ones(3)/3)
        for individual, (events, states) in enumerate(zip(jumps, histories)):
            expected = [states[np.count_nonzero(np.asarray(events) < index)]
                        for index in (0, 5, 10)]
            np.testing.assert_array_equal(fixed[:, individual], expected)

    def test_mise_plot_includes_both_methods_without_refitting(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/'results.csv'
            filename.write_text('num_individuals,fixed_mise,jump_mise\n10,.2,.1\n20,.1,.03\n')
            fig = plot_ehrenfest_results(filename, show=False)
            self.addCleanup(plt.close, fig)
            fixed, jump = fig.axes[0].lines
            np.testing.assert_allclose(fixed.get_ydata(), [.2, .1])
            np.testing.assert_allclose(jump.get_ydata(), [.1, .03])
            self.assertEqual([fixed.get_label(), jump.get_label()], ['Fixed-time', 'Jump-time'])

    def test_mise_averages_trial_errors_over_all_grid_points_and_entries(self):
        # Three grid points, four entries per matrix. The only nonzero errors
        # are at the final point: squared sums are 8 and 72, hence trial errors
        # 2/3 and 6. Their mean is 10/3, rather than the mean path's error 8/3.
        first = np.zeros((3, 2, 2))
        first[-1] = [[-2., 2.], [0., 0.]]
        second = 3 * first
        status = SimpleNamespace(success=True, nit=1, message='test fit')
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'experiment'
            with patch.object(BirthDeathEmbedder, 'optimise_A',
                              side_effect=[(first, status), (second, status)]), \
                 patch.object(BirthDeathJumpTimeGradient, 'optimise_A',
                              side_effect=[(2*first, status), (2*second, status)]):
                _, paths, rows = e.run_experiments(
                    lambda t: np.zeros((2, 2)), 2, [0., 1.], 2, [3],
                    num_trials=2, output_prefix=prefix,
                )
            self.assertAlmostEqual(rows[0]['fixed_mise'], 10/3)
            self.assertAlmostEqual(rows[0]['jump_mise'], 40/3)
            trials = np.genfromtxt(f'{prefix}_trials.csv', delimiter=',', names=True,
                                  dtype=None, encoding='utf-8')
            np.testing.assert_allclose(trials['fixed_ise'], [2/3, 6.])
            np.testing.assert_allclose(trials['jump_ise'], [8/3, 24.])
            np.testing.assert_allclose(paths['fixed'][0], (first + second) / 2)
            with np.load(f'{prefix}_generators.npz') as saved:
                np.testing.assert_allclose(saved['mean_jump_estimated_generators'][0], first + second)

    def test_model_endpoints_and_volume_effect(self):
        q = e.ehrenfest_generator(6, 6., volume_a_fn=lambda t: np.exp(-t))
        self.assertGreater(q(0)[6,5], 0)
        self.assertGreater(q(0)[0,1], 0)
        np.testing.assert_allclose(q(0).sum(axis=1), 0)
        self.assertGreater(q(1)[3,2], q(0)[3,2])
        self.assertLess(q(1)[3,4], q(0)[3,4])

    def test_default_volumes_are_constant_one(self):
        q = e.ehrenfest_generator(num_particles=4, jump_frequency=3.)
        expected_births = 3.*np.array([4.,3.,2.,1.])/4.
        np.testing.assert_allclose(np.diag(q(0),1),expected_births)
        np.testing.assert_allclose(q(1),q(0))
        import inspect
        for function in (e.ehrenfest_generator,e.generate_data):
            parameters = inspect.signature(function).parameters
            for removed in ('volume_a_initial','volume_decay','volume_b'):
                self.assertNotIn(removed,parameters)
            self.assertEqual(parameters['volume_a_fn'].default(.5),1.)
            self.assertEqual(parameters['volume_b_fn'].default(.5),1.)

    def test_custom_volume_functions(self):
        q = e.ehrenfest_generator(num_particles=4, jump_frequency=3.,
                                   volume_a_fn=lambda t: 2.+t,
                                   volume_b_fn=lambda t: 1.+2.*t)
        t = .5
        down = 1./(2.+t)
        up = 3./(1.+2.*t)
        self.assertAlmostEqual(q(t)[1,0],3.*down/(down+up))
        self.assertAlmostEqual(q(t)[1,2],3.*up/(down+up))
        for bad in (0., -1., float('nan'), float('inf')):
            with self.subTest(volume=bad), self.assertRaises(ValueError):
                e.ehrenfest_generator(volume_a_fn=lambda t: bad)(t)

    def test_simulation_matches_constant_generator(self):
        q = e.ehrenfest_generator(2, 2.)
        times = np.array([0.,1.])
        ps = e.transition_matrices(q, 3, times, 1e-9, 1e-11)
        np.testing.assert_allclose(ps[0], expm(q(0)), atol=1e-8)
        panel = e.simulate_panel(ps, 50000, np.random.default_rng(4), np.array([1.,0.,0.]))
        np.testing.assert_allclose(np.bincount(panel[-1], minlength=3)/50000, ps[0][0], atol=.01)

    def test_nonabsorbing_adjoint_gradient(self):
        emb = BirthDeathEmbedder(3, num_intervals=4, absorbing_final=False)
        x = np.full(20, .8)
        data = np.array([[2,1,0],[1,2,1]])
        _, g = emb.objective_grad(x, [0,4], data)
        h = 1e-6
        numerical = np.array([(emb.objective_grad(x+np.eye(20)[j]*h,[0,4],data)[0]-emb.objective_grad(x-np.eye(20)[j]*h,[0,4],data)[0])/(2*h) for j in range(20)])
        np.testing.assert_allclose(g, numerical, atol=1e-6)
        self.assertNotEqual(g[3], 0)
        old = BirthDeathEmbedder(3, num_intervals=4)
        self.assertEqual(old.objective_grad(x,[0,4],data)[1][3], 0)

    def test_end_to_end_saved_results_and_nonabsorbing_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)/'experiment'
            rows = e.generate_data(num_particles=1, jump_frequency=1.,
                                    individuals=(200,),num_observations=3,
                                    num_intervals=10,num_trials=2,max_iter=500,
                                    output_prefix=prefix,
                                    volume_a_fn=lambda t: np.exp(-t),
                                    volume_b_fn=lambda t: 1.)
            self.assertTrue(np.isfinite(rows[0]['fixed_mise']))
            self.assertEqual(rows[0]['fixed_converged_trials'],2)
            data = np.load(f'{prefix}_generators.npz')
            self.assertTrue(np.all(data['mean_fixed_estimated_generators'][0,:,1,0]>0))
            jump = data['mean_jump_estimated_generators']
            self.assertEqual(jump.shape, data['mean_fixed_estimated_generators'].shape)
            self.assertTrue(np.all(jump[0,:,1,0]>0))
            np.testing.assert_allclose(jump.sum(axis=-1), 0., atol=1e-12)
            np.testing.assert_allclose(data['mean_fixed_estimated_generators'].sum(axis=-1),0,atol=1e-12)
            trials = np.genfromtxt(f'{prefix}_trials.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
            self.assertAlmostEqual(rows[0]['fixed_mise'],np.mean(trials['fixed_ise']))
            self.assertAlmostEqual(rows[0]['jump_mise'], np.mean(trials['jump_ise']))
            fig = plot_ehrenfest_results(f'{prefix}_results.csv', f'{prefix}_mise.png', show=False)
            self.assertTrue(Path(f'{prefix}_mise.png').exists())
            import matplotlib.pyplot as plt
            plt.close(fig)

if __name__ == '__main__':
    unittest.main()
