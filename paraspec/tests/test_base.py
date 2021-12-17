import copy
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from paraspec import IR12SpeciationModel


@pytest.fixture
def params():
    return {
        'slope_trait_env': 0.95,
        'lifespan': 1,
        'random_seed': 1234,
        'always_direct_parent': True,
        'nb_radius': 5,
        'car_cap': 5,
        'sigma_w': 0.5,
        'sigma_mov': 4,
        'sigma_mut': 0.5,
        'mut_prob': 0.04,
        'on_extinction': 'ignore'
    }


@pytest.fixture(scope='session')
def grid():
    X, Y = np.meshgrid(np.linspace(0, 20, 10), np.linspace(0, 10, 20))
    return X, Y


@pytest.fixture(scope='session')
def env_field(grid):
    return np.random.uniform(0, 1, grid[0].shape)


@pytest.fixture
def model(params, grid):
    X, Y = grid
    return IR12SpeciationModel(X, Y, 10, **params)


@pytest.fixture
def initialized_model(model, env_field):
    m = copy.deepcopy(model)
    m.initialize([0, 1])
    return m


@pytest.fixture(scope='session')
def model_repr():
    return dedent("""\
    <IR12SpeciationModel (population: not initialized)>
    Parameters:
        slope_trait_env: 0.95
        lifespan: 1
        random_seed: 1234
        always_direct_parent: True
        nb_radius: 5
        car_cap: 5
        sigma_w: 0.5
        sigma_mov: 4
        sigma_mut: 0.5
        mut_prob: 0.04
        on_extinction: ignore
    """)


@pytest.fixture(scope='session')
def initialized_model_repr():
    return dedent("""\
    <IR12SpeciationModel (population: 10)>
    Parameters:
        slope_trait_env: 0.95
        lifespan: 1
        random_seed: 1234
        always_direct_parent: True
        nb_radius: 5
        car_cap: 5
        sigma_w: 0.5
        sigma_mov: 4
        sigma_mut: 0.5
        mut_prob: 0.04
        on_extinction: ignore
    """)


def _in_bounds(grid_coord, pop_coord):
    return (pop_coord.min() >= grid_coord.min()
            and pop_coord.max() <= grid_coord.max())


class TestParapatricSpeciationModel(object):

    def test_constructor(self):

        with pytest.raises(ValueError, match="invalid value"):
            IR12SpeciationModel([0, 1, 2], [0, 1, 2], 10,
                                on_extinction='invalid')

        rs = np.random.default_rng(0)

        m = IR12SpeciationModel([0, 1, 2], [0, 1, 2], 10, random_seed=rs)
        assert m._rng is rs

        m2 = IR12SpeciationModel([0, 1, 2], [0, 1, 2], 10, random_seed=0)
        np.testing.assert_equal(m2._rng.__getstate__()['state'], rs.__getstate__()['state'])

    def test_params(self, params, model):
        assert model.params == params

    def test_initialize_population(self, grid, initialized_model):
        assert initialized_model.population_size == 10

        assert initialized_model.population['step'] == 0
        np.testing.assert_equal(initialized_model.population['id'],
                                np.arange(0, 10))
        np.testing.assert_equal(initialized_model.population['parent'],
                                np.arange(0, 10))

        trait = initialized_model.population['trait']
        assert np.all((trait >= 0) & (trait <= 1))

        assert _in_bounds(grid[0], initialized_model.population['x'])
        assert _in_bounds(grid[1], initialized_model.population['y'])

    @pytest.mark.parametrize("x_range,y_range,error", [
        (None, None, False),
        ([0, 15], None, False),
        (None, [2, 7], False),
        ([0, 15], [2, 7], False),
        ([-1, 100], None, True),
        (None, [-1, 100], True),
        ([-1, 100], [-1, 100], True)
    ])
    def test_xy_range(self, model, grid, x_range, y_range, error):
        if error:
            expected = "x_range and y_range must be within model bounds"
            with pytest.raises(ValueError, match=expected):
                model.initialize(
                    [0, 1], x_range=x_range, y_range=y_range
                )

        else:
            model.initialize(
                [0, 1], x_range=x_range, y_range=y_range
            )
            x_r = x_range or grid[0]
            y_r = y_range or grid[1]
            assert _in_bounds(np.array(x_r), model.population['x'])
            assert _in_bounds(np.array(y_r), model.population['y'])

    def test_to_dataframe(self, initialized_model):
        expected = pd.DataFrame(initialized_model.population)
        actual = initialized_model.to_dataframe()
        pd.testing.assert_frame_equal(actual, expected)

        expected = pd.DataFrame({'x': initialized_model.population['x']})
        actual = initialized_model.to_dataframe(varnames='x')
        pd.testing.assert_frame_equal(actual, expected)

        data = {k: initialized_model.population[k] for k in ['x', 'y']}
        expected = pd.DataFrame(data)
        actual = initialized_model.to_dataframe(varnames=['x', 'y'])
        pd.testing.assert_frame_equal(actual, expected)

    #def test_scaled_params(self, model):
    #    params = model._get_scaled_params(4)
    #    expected = (0.5, 8., 1)

    #   assert params == expected

    #def test_scaled_params_not_effective(self, params, grid):
    #    X, Y = grid
    #    params.pop('lifespan')

    #    model = IR12SpeciationModel(X, Y, 10, **params)

    #    expected = (params['sigma_w'], params['sigma_mov'], params['sigma_mut'])
    #    assert model._get_scaled_params(4) == expected

    def test_count_neighbors(self, model, grid):
        points = np.column_stack([[0, 4, 8, 12], [0, 2, 4, 6]])
        expected = [2, 3, 3, 2]

        np.testing.assert_equal(model._count_neighbors(points), expected)

    def test_get_optimal_trait(self, model, grid, env_field):
        # using points = grid points + offset less than grid spacing
        # expected: env_field and optimal trait should be equal
        X, Y = grid
        points = np.column_stack([X.ravel() + 0.1, Y.ravel() + 0.1])

        opt_trait = model._get_local_env_value(env_field, points)

        np.testing.assert_array_equal(opt_trait, env_field.ravel())

    def test_evaluate_fitness(self, model, env_field):
        # TODO: more comprehensive testing

        model.initialize([0, 1])
        model.evaluate_fitness(env_field, 0, 1, 1)
        pop = model.population.copy()

        for k in ['r_d', 'opt_trait', 'fitness', 'n_offspring']:
            assert k in pop

    def test_update_population(self, model, grid, env_field):
        # do many runs to avoid favorable random conditions
        trait_diff = []

        for i in range(1000):
            model._rng = np.random.default_rng(i)

            model.initialize([0, 1])
            init_pop = model.population.copy()
            model.evaluate_fitness(env_field, 0, 1, 1)
            model.update_population(1)
            current_pop = model.population.copy()

            # test step
            assert current_pop['step'] == 1
            assert current_pop['id'][0] == init_pop['id'].size

            # test dispersal (only check within domain)
            assert _in_bounds(grid[0], current_pop['x'])
            assert _in_bounds(grid[1], current_pop['y'])

            # test mutation
            model.evaluate_fitness(env_field, 0, 1, 1)
            model.update_population(1)
            last_pop = model.population.copy()
            idx = np.searchsorted(current_pop['id'], last_pop['parent'])
            trait_diff.append(current_pop['trait'][idx] - last_pop['trait'])

        trait_diff = np.concatenate(trait_diff)
        trait_rms = np.sqrt(np.mean(trait_diff**2))
        scaled_sigma_mut = 1   # sigma_mut * sqrt(m_freq) * 1
        assert pytest.approx(trait_rms, scaled_sigma_mut)

        # test reset fitness data
        for k in ['r_d', 'opt_trait', 'fitness', 'n_offspring']:
            np.testing.assert_array_equal(last_pop[k], np.array([]))

    @pytest.mark.parametrize('direct_parent', [True, False])
    def test_updade_population_parents(self, grid, params, env_field,
                                       direct_parent):
        X, Y = grid
        params['always_direct_parent'] = direct_parent

        model = IR12SpeciationModel(X, Y, 10, **params)
        model.initialize([0, 1])

        model.evaluate_fitness(env_field, 0, 1, 1)
        parents0 = model.to_dataframe(varnames='parent')
        model.update_population(1)

        model.evaluate_fitness(env_field, 0, 1, 1)
        model.update_population(1)

        model.evaluate_fitness(env_field, 0, 1, 1)
        parents2 = model.to_dataframe(varnames='parent')
        model.update_population(1)

        model.evaluate_fitness(env_field, 0, 1, 1)
        parents3 = model.to_dataframe(varnames='parent')
        model.update_population(1)

        if direct_parent:
            assert parents2.values.max() > parents0.values.max()
        else:
            #assert parents2.values.max() <= parents0.values.max()
            assert parents3.values.max() > parents2.values.max()

    @pytest.mark.parametrize('car_cap_mul,env_field_mul,on_extinction', [
        (0., 1, 'raise'),
        (0., 1, 'warn'),
        (0., 1, 'ignore'),
        #(1., 1e3, 'ignore')
    ])
    def test_update_population_extinction(self,
                                          initialized_model,
                                          env_field,
                                          car_cap_mul,
                                          env_field_mul,
                                          on_extinction):

        subset_keys = ('id', 'parent', 'x', 'y', 'trait')

        def get_pop_subset():
            pop = initialized_model.population.copy()
            return {k: pop[k] for k in subset_keys}

        initialized_model._params['on_extinction'] = on_extinction

        # no offspring via either r_d values = 0 or very low fitness values
        initialized_model._params['car_cap'] *= car_cap_mul
        field = env_field * env_field_mul

        if on_extinction == 'raise':
            with pytest.raises(RuntimeError, match="no offspring"):
                initialized_model.evaluate_fitness(field, 0, 1, 1)
                initialized_model.update_population(1)
            return

        elif on_extinction == 'warn':
            with pytest.warns(RuntimeWarning, match="no offspring"):
                initialized_model.evaluate_fitness(field, 0, 1, 1)
                initialized_model.update_population(1)
                current = get_pop_subset()
                initialized_model.evaluate_fitness(field, 0, 1, 1)
                initialized_model.update_population(1)
                next = get_pop_subset()

        else:
            initialized_model.evaluate_fitness(field, 0, 1, 1)
            initialized_model.update_population(1)
            current = get_pop_subset()
            initialized_model.evaluate_fitness(field, 0, 1, 1)
            initialized_model.update_population(1)
            next = get_pop_subset()

        for k in subset_keys:
            assert current[k].size == 0
            assert next[k].size == 0

    def test_repr(self, model, model_repr,
                  initialized_model, initialized_model_repr):
        assert repr(model) == model_repr
        assert repr(initialized_model) == initialized_model_repr
