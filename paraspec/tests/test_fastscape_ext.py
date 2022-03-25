import copy

import numpy as np
import pytest

pytest.importorskip("fastscape")  # isort:skip

from paraspec.fastscape_ext import (IR12Speciation,
                                    DD03Speciation,
                                    CompoundEnvironment,
                                    ElevationEnvField1,
                                    ElevationEnvField2,
                                    ir12spec_model,
                                    dd03spec_model
                                    )


@pytest.fixture
def specDD03_process():
    params = {
        'slope_trait_env': [0.95],
        'init_abundance': 10,
        'birth_rate': 1,
        'movement_rate': 5,
        'car_cap_max': 100,
        'mut_prob': 1.0,
        'sigma_env_trait': 0.5,
        'sigma_mov': 4,
        'sigma_mut': 0.5,
        'random_seed': 1234,
        'rescale_rates': True,
        'sigma_comp_trait': 0.9,
        'sigma_comp_dist': 0.2
    }

    x = np.linspace(0, 20, 10)
    y = np.linspace(0, 10, 20)
    elev = np.random.uniform(0, 1, (1, 20, 10))
    return DD03Speciation(env_field=elev, grid_x=x, grid_y=y,
                          init_min_trait=[0.5], init_max_trait=[0.5],
                          min_env=[0], max_env=[1],
                          **params)


@pytest.fixture
def specIR12_process():
    params = {
        'slope_trait_env': [0.95],
        'init_abundance': 10,
        'nb_radius': 5,
        'car_cap': 10,
        'mut_prob': 1.0,
        'sigma_env_trait': 0.5,
        'sigma_mov': 4,
        'sigma_mut': 0.5,
        'random_seed': 1234,
        'rescale_rates': True
    }

    x = np.linspace(0, 20, 10)
    y = np.linspace(0, 10, 20)
    elev = np.random.uniform(0, 1, (1, 20, 10))
    return IR12Speciation(env_field=elev, grid_x=x, grid_y=y,
                          init_min_trait=[0.5], init_max_trait=[0.5],
                          min_env=[0], max_env=[1],
                          **params)


@pytest.mark.parametrize('speciation', ['IR12', 'DD03'])
def test_parapatric_speciation(speciation, specIR12_process, specDD03_process):
    if speciation == 'IR12':
        spec = copy.deepcopy(specIR12_process)
    elif speciation == 'DD03':
        spec = copy.deepcopy(specDD03_process)
    spec.initialize()
    spec.run_step(1)

    assert spec.abundance == 10
    np.testing.assert_equal(spec._get_taxon_id(), np.ones(10))
    np.testing.assert_equal(spec._get_ancestor_id(), np.zeros(10))

    for vname in ["x", "y", "trait", "n_offspring"]:
        getter = getattr(spec, "_get_" + vname)
        assert getter() is spec.individuals[vname]
    if speciation == 'IR12':
        getter = getattr(spec, "_get_" + 'fitness')
        assert getter() is spec.individuals['fitness']
    spec.finalize_step(1)

    assert spec.abundance != len(spec.individuals["x"])


@pytest.mark.parametrize('field', ['elev_field01', 'elev_field02'])
def test_parapatric_environment_elevation(field):
    elev = np.random.uniform(0, 1, (1, 20, 10))

    if field == 'elev_field01':
        p = ElevationEnvField1(elevation=elev)
    elif field == 'elev_field02':
        p = ElevationEnvField2(elevation=elev)
    p.initialize()

    assert p.field is p.elevation
    np.testing.assert_array_equal(p.field, p.elevation)


def test_compound_environment():
    elev = np.random.uniform(0, 1, (1, 20, 10))
    field01 = ElevationEnvField1(elevation=elev)
    field02 = ElevationEnvField2(elevation=elev)
    dic_fields = {'elevation01': field01, 'elevation02': field02}
    comp_field = CompoundEnvironment(field_arrays=dic_fields)
    comp_field.initialize()

    for ef in comp_field.env_field:
        np.testing.assert_array_equal(ef.elevation, elev)


def test_paraspec_model():
    assert isinstance(ir12spec_model["life"], IR12Speciation)
    assert isinstance(ir12spec_model["life_env"], CompoundEnvironment)
    assert isinstance(dd03spec_model["life"], DD03Speciation)
    assert isinstance(dd03spec_model["life_env"], CompoundEnvironment)
