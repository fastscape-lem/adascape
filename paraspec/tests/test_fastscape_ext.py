import numpy as np
import pytest

pytest.importorskip("fastscape")  # isort:skip

from paraspec.fastscape_ext import (IR12Speciation,
                                    EnvironmentElevation,
                                    ir12spec_model)

@pytest.fixture
def ps_process():
    params = {
        'init_size': 10,
        'nb_radius': 5,
        'car_cap': 10,
        'mut_prob': 1.0,
        'sigma_w': 0.5,
        'sigma_mov': 4,
        'sigma_mut': 0.5,
        'random_seed': 1234,
        'rescale_rates': True
    }

    x = np.linspace(0, 20, 10)
    y = np.linspace(0, 10, 20)
    elev = np.random.uniform(0, 1, (20, 10))
    return IR12Speciation(env_field=elev, grid_x=x, grid_y=y,
                          init_min_trait=0, init_max_trait=1, **params)


def test_parapatric_speciation(ps_process):
    ps_process.initialize()
    ps_process.run_step(1)

    assert ps_process.size == 10
    np.testing.assert_equal(ps_process._get_id(), np.arange(0, 10))
    np.testing.assert_equal(ps_process._get_parent(), np.arange(0, 10))

    for vname in ["x", "y", "trait", "opt_trait", "r_d", "fitness", "n_offspring"]:
        getter = getattr(ps_process, "_get_" + vname)
        assert getter() is ps_process.population[vname]

    ps_process.finalize_step(1)

    assert ps_process.size != len(ps_process.population["id"])


def test_parapatric_environment_elevation():
    elev = np.random.rand(10, 20)
    p = EnvironmentElevation(elevation=elev)

    p.initialize()

    assert p.env_field is p.elevation
    np.testing.assert_array_equal(p.env_field, p.elevation)


def test_paraspec_model():
    assert isinstance(ir12spec_model["life"], IR12Speciation)
    assert isinstance(ir12spec_model["life_env"], EnvironmentElevation)
