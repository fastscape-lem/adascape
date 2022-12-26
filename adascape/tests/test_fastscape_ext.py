import copy

import numpy as np
import pytest

pytest.importorskip("fastscape")  # isort:skip

from adascape.fastscape_ext import (IR12Speciation, DD03Speciation,
                                    CompoundEnvironment, ElevationEnvField, PrecipitationField,
                                    FastscapeElevationTrait, FastscapePrecipitationTrait,
                                    adaspec_IR12_model, adaspec_DD03_model)


@pytest.fixture
def specDD03_process(trait_funcs):
    params = {
        'init_abundance': 10,
        'birth_rate': 1,
        'movement_rate': 5,
        'car_cap_max': 100,
        'mut_prob': 1.0,
        'sigma_env_fitness': 0.5,
        'sigma_mov': 4,
        'sigma_mut': 0.5,
        'random_seed': 1234,
        'rescale_rates': True,
        'sigma_comp_trait': 0.9,
        'sigma_comp_dist': 0.2,
        'taxon_threshold': 0.05,
        'rho': 0
    }

    x = np.linspace(0, 20, 10)
    y = np.linspace(0, 10, 20)
    init_trait_funcs, opt_trait_funcs = trait_funcs
    return DD03Speciation(env_field=env_field, grid_x=x, grid_y=y,
                          init_trait_funcs=init_trait_funcs,
                          opt_trait_funcs=opt_trait_funcs,
                          **params)


@pytest.fixture
def specIR12_process(trait_funcs):
    params = {
        'init_abundance': 10,
        'nb_radius': 5,
        'car_cap': 10,
        'mut_prob': 1.0,
        'sigma_env_fitness': 0.5,
        'sigma_mov': 4,
        'sigma_mut': 0.5,
        'random_seed': 1234,
        'rescale_rates': True,
        'taxon_threshold': 0.05,
        'rho': 0
    }

    x = np.linspace(0, 20, 10)
    y = np.linspace(0, 10, 20)
    init_trait_funcs, opt_trait_funcs = trait_funcs
    return IR12Speciation(env_field=env_field, grid_x=x, grid_y=y,
                          init_trait_funcs=init_trait_funcs,
                          opt_trait_funcs=opt_trait_funcs,
                          **params)


@pytest.fixture()
def env_field():
    return np.random.uniform(0, 1, (1, 20, 10))


@pytest.fixture()
def trait_funcs(env_field):
    trait_01 = FastscapeElevationTrait(topo_elevation=env_field,
                                       init_trait_min=0.5,
                                       init_trait_max=0.5,
                                       lin_slope=0.95,
                                       norm_min=env_field.min(),
                                       norm_max=env_field.max(),
                                       random_seed=1234)
    trait_01.initialize()

    trait_02 = FastscapePrecipitationTrait(oro_precipitation=env_field,
                                           init_trait_min=0.5,
                                           init_trait_max=0.5,
                                           lin_slope=0.95,
                                           norm_min=env_field.min(),
                                           norm_max=env_field.max(),
                                           random_seed=1234)
    trait_02.initialize()

    init_trait_funcs = {'trait_01': trait_01.init_trait_func, 'trait_02': trait_02.init_trait_func}
    opt_trait_funcs = {'trait_01': trait_01.opt_trait_func, 'trait_02': trait_02.opt_trait_func}

    return init_trait_funcs, opt_trait_funcs


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
        p = ElevationEnvField(elevation=elev)
    elif field == 'elev_field02':
        p = ElevationEnvField(elevation=elev)
    p.initialize()

    assert p.field is p.elevation
    np.testing.assert_array_equal(p.field, p.elevation)


@pytest.mark.parametrize('field', ['precip_field01', 'precip_field02'])
def test_parapatric_environment_precipitation(field):
    precip = np.random.uniform(0, 1, (1, 20, 10))

    if field == 'precip_field01':
        p = PrecipitationField(precip=precip)
    elif field == 'precip_field02':
        p = PrecipitationField(precip=precip)
    p.initialize()

    assert p.field is p.precip
    np.testing.assert_array_equal(p.field, p.precip)


def test_compound_environment():
    elev = np.random.uniform(0, 1, (1, 20, 10))
    prec = np.random.uniform(0, 1, (1, 20, 10))
    field01 = ElevationEnvField(elevation=elev)
    field02 = ElevationEnvField(elevation=elev)
    field03 = PrecipitationField(precip=prec)
    field04 = PrecipitationField(precip=prec)
    dic_fields = {'elevation01': field01, 'elevation02': field02, 'precip_rate01': field03, 'precip_rate02': field04}
    comp_field = CompoundEnvironment(field_arrays=dic_fields)
    comp_field.initialize()

    for ef in comp_field.env_field:
        if 'elevation' in ef.__dict__:
            np.testing.assert_array_equal(ef.elevation, elev)
        elif 'precip' in ef.__dict__:
            np.testing.assert_array_equal(ef.precip, prec)


def test_paraspec_model():
    assert isinstance(adaspec_IR12_model["life"], IR12Speciation)
    assert isinstance(adaspec_IR12_model["life_env"], CompoundEnvironment)
    assert isinstance(adaspec_DD03_model["life"], DD03Speciation)
    assert isinstance(adaspec_DD03_model["life_env"], CompoundEnvironment)
