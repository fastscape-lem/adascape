import copy
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from paraspec.base import IR12SpeciationModel, DD03SpeciationModel


@pytest.fixture
def params_IR12():
    return {
        'slope_trait_env': [0.95],
        'lifespan': 1,
        'random_seed': 1234,
        'always_direct_parent': True,
        'distance_method': 'ward',
        'distance_value': 0.5,
        'nb_radius': 5,
        'car_cap': 5,
        'sigma_env_trait': 0.5,
        'sigma_mov': 4,
        'sigma_mut': 0.5,
        'mut_prob': 0.04,
        'on_extinction': 'ignore'
    }

@pytest.fixture
def params_DD03():
    return {
        'slope_trait_env': [0.95],
        'lifespan': 1,
        'random_seed': 1234,
        'always_direct_parent': True,
        'distance_method': 'ward',
        'distance_value': 0.5,
        'sigma_env_trait': 0.5,
        'sigma_mov': 4,
        'sigma_mut': 0.5,
        'mut_prob': 0.04,
        'birth_rate': 1,
        'movement_rate': 5,
        'car_cap_max': 100,
        'sigma_comp_trait': 0.9,
        'sigma_comp_dist': 0.2,
    }

@pytest.fixture(scope='session')
def grid():
    X, Y = np.meshgrid(np.linspace(0, 20, 10), np.linspace(0, 10, 20))
    return X, Y


@pytest.fixture(scope='session')
def env_field(grid):
    return np.random.uniform(0, 1, np.expand_dims(grid[0], 0).shape)


@pytest.fixture
def model_IR12(params_IR12, grid):
    X, Y = grid
    return IR12SpeciationModel(X, Y, 10, **params_IR12)


@pytest.fixture
def model_DD03(params_DD03, grid):
    X, Y = grid
    return DD03SpeciationModel(X, Y, 10, **params_DD03)

@pytest.fixture
def initialized_model_IR12(model_IR12, env_field):
    m = copy.deepcopy(model_IR12)
    m.initialize([[0.5, 0.5]])
    return m

@pytest.fixture
def initialized_model_DD03(model_DD03, env_field):
    m = copy.deepcopy(model_DD03)
    m.initialize([[0.5, 0.5]])
    return m

@pytest.fixture(scope='session')
def model_IR12_repr():
    return dedent("""\
    <IR12SpeciationModel (individuals: not initialized)>
    Parameters:
        slope_trait_env: [0.95]
        lifespan: 1
        random_seed: 1234
        always_direct_parent: True
        distance_method: ward
        distance_value: 0.5
        nb_radius: 5
        car_cap: 5
        sigma_env_trait: 0.5
        sigma_mov: 4
        sigma_mut: 0.5
        mut_prob: 0.04
        on_extinction: ignore
    """)


@pytest.fixture(scope='session')
def initialized_model_IR12_repr():
    return dedent("""\
    <IR12SpeciationModel (individuals: 10)>
    Parameters:
        slope_trait_env: [0.95]
        lifespan: 1
        random_seed: 1234
        always_direct_parent: True
        distance_method: ward
        distance_value: 0.5
        nb_radius: 5
        car_cap: 5
        sigma_env_trait: 0.5
        sigma_mov: 4
        sigma_mut: 0.5
        mut_prob: 0.04
        on_extinction: ignore
    """)

@pytest.fixture(scope='session')
def model_DD03_repr():
    return dedent("""\
    <DD03SpeciationModel (individuals: not initialized)>
    Parameters:
        slope_trait_env: [0.95]
        lifespan: 1
        random_seed: 1234
        always_direct_parent: True
        distance_method: ward
        distance_value: 0.5
        birth_rate: 1
        movement_rate: 5
        car_cap_max: 100
        sigma_env_trait: 0.5
        mut_prob: 0.04
        sigma_mut: 0.5
        sigma_mov: 4
        sigma_comp_trait: 0.9
        sigma_comp_dist: 0.2
    """)


@pytest.fixture(scope='session')
def initialized_model_DD03_repr():
    return dedent("""\
    <DD03SpeciationModel (individuals: 10)>
    Parameters:
        slope_trait_env: [0.95]
        lifespan: 1
        random_seed: 1234
        always_direct_parent: True
        distance_method: ward
        distance_value: 0.5
        birth_rate: 1
        movement_rate: 5
        car_cap_max: 100
        sigma_env_trait: 0.5
        mut_prob: 0.04
        sigma_mut: 0.5
        sigma_mov: 4
        sigma_comp_trait: 0.9
        sigma_comp_dist: 0.2
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

    def test_params(self, params_IR12, model_IR12, params_DD03, model_DD03):
        assert model_IR12.params == params_IR12
        assert model_DD03.params == params_DD03

    def test_initialize_population(self, grid, initialized_model_IR12):
        assert initialized_model_IR12.abundance == 10

        assert initialized_model_IR12.individuals['step'] == 0
        np.testing.assert_equal(initialized_model_IR12.individuals['taxon_id'],
                                np.repeat(1, 10))
        np.testing.assert_equal(initialized_model_IR12.individuals['ancestor_id'],
                                np.repeat(0, 10))

        trait = initialized_model_IR12.individuals['trait']
        assert np.all((trait >= 0) & (trait <= 1))

        assert _in_bounds(grid[0], initialized_model_IR12.individuals['x'])
        assert _in_bounds(grid[1], initialized_model_IR12.individuals['y'])

    @pytest.mark.parametrize("x_range,y_range,error", [
        (None, None, False),
        ([0, 15], None, False),
        (None, [2, 7], False),
        ([0, 15], [2, 7], False),
        ([-1, 100], None, True),
        (None, [-1, 100], True),
        ([-1, 100], [-1, 100], True)
    ])
    def test_xy_range(self, model_IR12, grid, x_range, y_range, error):
        if error:
            expected = "x_range and y_range must be within model bounds"
            with pytest.raises(ValueError, match=expected):
                model_IR12.initialize(
                    [[0, 1]], x_range=x_range, y_range=y_range
                )

        else:
            model_IR12.initialize(
                [[0, 1]], x_range=x_range, y_range=y_range
            )
            x_r = x_range or grid[0]
            y_r = y_range or grid[1]
            assert _in_bounds(np.array(x_r), model_IR12.individuals['x'])
            assert _in_bounds(np.array(y_r), model_IR12.individuals['y'])

    def test_to_dataframe(self, initialized_model_IR12):
        individuals_data = initialized_model_IR12.individuals.copy()
        for i in range(initialized_model_IR12.individuals['trait'].shape[1]):
            individuals_data['trait_' + str(i)] = individuals_data['trait'][:, i]
        individuals_data.pop('trait')
        expected = pd.DataFrame(individuals_data)
        actual = initialized_model_IR12.to_dataframe()
        pd.testing.assert_frame_equal(actual, expected)

        expected = pd.DataFrame({'x': initialized_model_IR12.individuals['x']})
        actual = initialized_model_IR12.to_dataframe(varnames='x')
        pd.testing.assert_frame_equal(actual, expected)

        data = {k: initialized_model_IR12.individuals[k] for k in ['x', 'y']}
        expected = pd.DataFrame(data)
        actual = initialized_model_IR12.to_dataframe(varnames=['x', 'y'])
        pd.testing.assert_frame_equal(actual, expected)

    def test_scaled_params(self, model_IR12, params_IR12):
        test_params = ['sigma_env_trait', 'mut_prob', 'sigma_mov', 'sigma_mut']
        rescaled_params = [model_IR12._scaled_param(params_IR12[p], 1) for p in test_params]
        expected = [0.5, 0.04, 4.0, 0.5]
        assert rescaled_params == expected

    def test_scaled_params_not_effective(self, params_IR12, grid):
        X, Y = grid
        params_IR12.pop('lifespan')
        test_params = ['sigma_env_trait', 'mut_prob', 'sigma_mov', 'sigma_mut']
        model = IR12SpeciationModel(X, Y, 10, **params_IR12)
        actual = [model._scaled_param(params_IR12[p], 1) for p in test_params]
        expected = [params_IR12['sigma_env_trait'], params_IR12['mut_prob'], params_IR12['sigma_mov'], params_IR12['sigma_mut']]
        assert actual == expected

    def test_count_neighbors(self, model_IR12, grid):
        points = np.column_stack([[0, 4, 8, 12], [0, 2, 4, 6]])
        expected = [2, 3, 3, 2]

        np.testing.assert_equal(model_IR12._count_neighbors(points), expected)

    def test_get_optimal_trait(self, model_IR12, grid, env_field):
        # using points = grid points + offset less than grid spacing
        # expected: env_field and optimal trait should be equal
        X, Y = grid
        points = np.column_stack([X.ravel() + 0.1, Y.ravel() + 0.1])

        opt_trait = model_IR12._get_local_env_value(env_field, points)

        np.testing.assert_array_equal(opt_trait, env_field.ravel())

    @pytest.mark.parametrize('model', ['IR12', 'DD03'])
    def test_evaluate_fitness(self, model, model_IR12, model_DD03, env_field):

        if model == 'IR12':
            model_IR12.initialize([[0.5, 0.5]])
            init_pop = model_IR12.individuals.copy()
            model_IR12.evaluate_fitness(env_field, [0], [1], 1)
            eval_pop = model_IR12.individuals.copy()
            for k in ['fitness', 'n_offspring']:
                assert k in eval_pop
            with pytest.raises(ValueError):
                model_IR12.evaluate_fitness(env_field.squeeze(), [0], [1], 1)

        elif model == 'DD03':
            model_DD03.initialize([[0.5, 0.5]])
            init_pop = model_DD03.individuals.copy()
            model_DD03.evaluate_fitness(env_field, [0], [1], 1)
            eval_pop = model_DD03.individuals
            for k in ['events_i', 'death_i']:
                assert k in eval_pop
            with pytest.raises(ValueError):
                model_DD03.evaluate_fitness(env_field.squeeze(), [0], [1], 1)

        assert init_pop['n_offspring'].max() == 0
        assert init_pop['n_offspring'].max() < eval_pop['n_offspring'].max()

    def test_update_individuals(self, model_IR12, grid, env_field):
        # do many runs to avoid favorable random conditions
        trait_diff = []

        for i in range(1000):
            model_IR12._rng = np.random.default_rng(i)

            model_IR12.initialize([[0.5, 0.5]])
            init_pop = model_IR12.individuals.copy()
            model_IR12.evaluate_fitness(env_field, [0], [1], 1)
            model_IR12.update_individuals(1)
            current_pop = model_IR12.individuals.copy()

            # test step
            assert current_pop['step'] == 1
            assert current_pop['time'] == init_pop['time'] + 1

            # test dispersal (only check within domain)
            assert _in_bounds(grid[0], current_pop['x'])
            assert _in_bounds(grid[1], current_pop['y'])

            # test mutation
            model_IR12.evaluate_fitness(env_field, [0], [1], 1)
            model_IR12.update_individuals(1)
            last_pop = model_IR12.individuals.copy()
            idx = np.searchsorted(current_pop['taxon_id'], last_pop['ancestor_id'])-1
            trait_diff.append(current_pop['trait'][idx, 0] - last_pop['trait'][:, 0])

        trait_diff = np.concatenate(trait_diff)
        trait_rms = np.sqrt(np.mean(trait_diff ** 2))
        scaled_sigma_mut = 1  # sigma_mut * sqrt(m_freq) * 1
        assert pytest.approx(trait_rms, scaled_sigma_mut)

        # test reset fitness data
        for k in ['fitness', 'n_offspring']:
            np.testing.assert_array_equal(last_pop[k], np.zeros(last_pop['trait'][:, 0].size))

    @pytest.mark.parametrize('direct_parent', [True, False])
    def test_updade_population_parents(self, grid, params_IR12, env_field,
                                       direct_parent):
        X, Y = grid
        params_IR12['always_direct_parent'] = direct_parent

        model = IR12SpeciationModel(X, Y, 10, **params_IR12)
        model.initialize([[0.5, 0.5]])

        model.evaluate_fitness(env_field, [0], [1], 1)
        parents0 = model.to_dataframe(varnames='ancestor_id')
        model.update_individuals(1)

        model.evaluate_fitness(env_field, [0], [1], 1)
        parents1 = model.to_dataframe(varnames='ancestor_id')
        model.update_individuals(1)

        model.evaluate_fitness(env_field, [0], [1], 1)
        parents2 = model.to_dataframe(varnames='ancestor_id')
        model.update_individuals(1)

        model.evaluate_fitness(env_field, [0], [1], 1)
        parents3 = model.to_dataframe(varnames='ancestor_id')
        model.update_individuals(1)

        if direct_parent:
            assert parents1.values.max() > parents0.values.max()
            assert parents2.values.max() > parents1.values.max()
            assert parents3.values.max() > parents2.values.max()
        else:
            assert parents1.values.max() > parents0.values.max()
            assert parents2.values.max() == parents1.values.max()
            assert parents3.values.max() == parents2.values.max()

    @pytest.mark.parametrize('car_cap_mul,env_field_mul,on_extinction', [
        (0., 1, 'raise'),
        (0., 1, 'warn'),
        (0., 1, 'ignore'),
        (1., 1e3, 'ignore')
    ])
    def test_update_individuals_extinction(self,
                                           initialized_model_IR12,
                                           env_field,
                                           car_cap_mul,
                                           env_field_mul,
                                           on_extinction):

        subset_keys = ('taxon_id', 'ancestor_id', 'x', 'y', 'trait')

        def get_pop_subset():
            pop = initialized_model_IR12.individuals.copy()
            return {k: pop[k] for k in subset_keys}

        initialized_model_IR12._params['on_extinction'] = on_extinction

        # no offspring via either r_d values = 0 or very low fitness values
        initialized_model_IR12._params['car_cap'] *= car_cap_mul
        field = env_field * env_field_mul

        if on_extinction == 'raise':
            with pytest.raises(RuntimeError, match="no offspring"):
                initialized_model_IR12.evaluate_fitness(field, [0], [1], 1)
                initialized_model_IR12.update_individuals(1)
            return

        elif on_extinction == 'warn':
            with pytest.warns(RuntimeWarning, match="no offspring"):
                initialized_model_IR12.evaluate_fitness(field, [0], [1], 1)
                initialized_model_IR12.update_individuals(1)
                current = get_pop_subset()
                initialized_model_IR12.evaluate_fitness(field, [0], [1], 1)
                initialized_model_IR12.update_individuals(1)
                next = get_pop_subset()

        else:
            initialized_model_IR12.evaluate_fitness(field, [0], [1], 1)
            initialized_model_IR12.update_individuals(1)
            current = get_pop_subset()
            initialized_model_IR12.evaluate_fitness(field, [0], [1], 1)
            initialized_model_IR12.update_individuals(1)
            next = get_pop_subset()

        for k in subset_keys:
            assert current[k].size == 0
            assert next[k].size == 0

    def test_repr(self, model_IR12, model_IR12_repr,
                  initialized_model_IR12, initialized_model_IR12_repr,
                  model_DD03, model_DD03_repr, initialized_model_DD03,
                  initialized_model_DD03_repr):
        assert repr(model_IR12) == model_IR12_repr
        assert repr(initialized_model_IR12) == initialized_model_IR12_repr
        assert repr(model_DD03) == model_DD03_repr
        assert repr(initialized_model_DD03) == initialized_model_DD03_repr

