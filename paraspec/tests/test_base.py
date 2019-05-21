import copy
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from paraspec import ParapatricSpeciationModel


@pytest.fixture(scope='session')
def params():
    return {
        'nb_radius': 5,
        'lifespan': 1,
        'capacity': 5,
        'sigma_w': 0.5,
        'sigma_d': 4,
        'sigma_mut': 0.5,
        'm_freq': 0.04,
        'random_seed': 1234
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
    return ParapatricSpeciationModel(X, Y, 10, **params)


@pytest.fixture
def initialized_model(model, env_field):
    m = copy.deepcopy(model)
    m.initialize_population([env_field.min(), env_field.max()])
    return m


@pytest.fixture(scope='session')
def model_repr():
    return dedent("""\
    <ParapatricSpeciationModel (population: not initialized)>
    Parameters:
        nb_radius: 5
        lifespan: 1
        capacity: 5
        sigma_w: 0.5
        sigma_d: 4
        sigma_mut: 0.5
        m_freq: 0.04
        random_seed: 1234
    """)


@pytest.fixture(scope='session')
def initialized_model_repr():
    return dedent("""\
    <ParapatricSpeciationModel (population: 10)>
    Parameters:
        nb_radius: 5
        lifespan: 1
        capacity: 5
        sigma_w: 0.5
        sigma_d: 4
        sigma_mut: 0.5
        m_freq: 0.04
        random_seed: 1234
    """)


def _in_bounds(grid_coord, pop_coord):
    return (pop_coord.min() >= grid_coord.min()
            and pop_coord.max() <= grid_coord.max())


class TestParapatricSpeciationModel(object):

    def test_constructor(self):
        with pytest.raises(KeyError, match="not valid model parameters"):
            ParapatricSpeciationModel([0, 1, 2], [0, 1, 2], 10,
                                      invalid_param=0, invlaid_param2='1')

        rs = np.random.RandomState(0)

        m = ParapatricSpeciationModel([0, 1, 2], [0, 1, 2], 10, random_seed=rs)
        assert m._random is rs

        m2 = ParapatricSpeciationModel([0, 1, 2], [0, 1, 2], 10, random_seed=0)
        np.testing.assert_equal(m2._random.get_state()[1], rs.get_state()[1])

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

    def test_to_dataframe(self, initialized_model):
        expected = pd.DataFrame(initialized_model.population)
        actual = initialized_model.to_dataframe()
        pd.testing.assert_frame_equal(actual, expected)

    def test_scaled_params(self, model):
        params = model._get_scaled_params(4)
        expected = (1., 8., 0.2)

        assert params == expected

    def test_count_neighbors(self, model, grid):
        points = np.column_stack([[0, 4, 8, 12], [0, 2, 4, 6]])
        expected = [2, 3, 3, 2]

        np.testing.assert_equal(model._count_neighbors(points), expected)

    def test_get_optimal_trait(self, model, grid, env_field):
        # using points = grid points + offset less than grid spacing
        # expected: env_field and optimal trait should be equal
        X, Y = grid
        points = np.column_stack([X.ravel() + 0.1, Y.ravel() + 0.1])

        opt_trait = model._get_optimal_trait(env_field, points)

        np.testing.assert_array_equal(opt_trait, env_field.ravel())

    def test_update_population(self, model, grid, env_field):
        # do many runs to avoid favorable random conditions
        trait_diff = []

        for i in range(1000):
            model._random = np.random.RandomState(i)

            model.initialize_population([env_field.min(), env_field.max()])
            init_pop = model.population.copy()
            model.update_population(env_field, 1)
            current_pop = model.population.copy()

            # test step
            assert current_pop['step'] == 1
            assert current_pop['id'][0] == init_pop['id'].size

            # test dispersal (only check within domain)
            assert _in_bounds(grid[0], current_pop['x'])
            assert _in_bounds(grid[1], current_pop['y'])

            # test mutation
            model.update_population(env_field, 1)
            last_pop = model.population.copy()
            idx = np.searchsorted(current_pop['id'], last_pop['parent'])
            trait_diff.append(current_pop['trait'][idx] - last_pop['trait'])

        trait_diff = np.concatenate(trait_diff)
        trait_rms = np.sqrt(np.mean(trait_diff**2))
        scaled_sigma_mut = 0.2   # sigma_mut * sqrt(m_freq) * 1
        assert pytest.approx(trait_rms, scaled_sigma_mut)

    @pytest.mark.parametrize('capacity_factor,env_field_factor',
                             [(0., 1), (1., 1e3)])
    def test_update_population_no_offspring(self, initialized_model,
                                            env_field, capacity_factor,
                                            env_field_factor):
        # no offspring via either r_d values = 0 or very low fitness values
        initialized_model._params['capacity'] *= capacity_factor

        init_pop = initialized_model.population.copy()
        initialized_model.update_population(env_field * env_field_factor, 1)
        current_pop = initialized_model.population.copy()

        np.testing.assert_array_equal(init_pop['id'], current_pop['id'])

    def test_repr(self, model, model_repr,
                  initialized_model, initialized_model_repr):
        assert repr(model) == model_repr
        assert repr(initialized_model) == initialized_model_repr
