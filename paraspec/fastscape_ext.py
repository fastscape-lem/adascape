from fastscape.models import basic_model
from fastscape.processes import SurfaceTopography, UniformRectilinearGrid2D
# from orographic_precipitation.fastscape_ext import (
#     OrographicPrecipitation,
#     OrographicDrainageDischarge
# )
import numpy as np
import xsimlab as xs
from paraspec.base import IR12SpeciationModel
from paraspec.base import DD03SpeciationModel


@xs.process
class Speciation:
    """
    Speciation model as a fastscape extension
    """
    init_abundance = xs.variable(description="initial number of individuals", static=True)
    init_min_trait = xs.variable(dims='trait', description="initial min trait value", static=True)
    init_max_trait = xs.variable(dims='trait', description="initial max trait value", static=True)
    min_env = xs.variable(dims='field',
                          description="Minimum value for the environmental field throughout simulation", static=True)
    max_env = xs.variable(dims='field',
                          description="Maximum value for the environmental field throughout simulation", static=True)
    slope_trait_env = xs.variable(dims='field',
                                  description="slope of the linear relationship between "
                                              "optimum trait and environmental field",
                                  static=True)
    random_seed = xs.variable(
        default=None,
        description="random number generator seed",
        static=True
    )
    rescale_rates = xs.variable(default=True, description="whether to rescale rates", static=True)

    env_field = xs.variable(dims=(('field', "y", "x"), ("y", "x")))

    grid_x = xs.foreign(UniformRectilinearGrid2D, "x")
    grid_y = xs.foreign(UniformRectilinearGrid2D, "y")

    _model = xs.any_object(description="speciation model instance")
    _individuals = xs.any_object(description="speciation model state dictionary")

    id = xs.on_demand(
        dims='ind',
        description="individual's id"
    )
    parent = xs.on_demand(
        dims='ind',
        description="individual's ancestor"
    )
    x = xs.on_demand(
        dims='ind',
        description="individual's x-position"
    )
    y = xs.on_demand(
        dims='ind',
        description="individual's y-position"
    )
    trait = xs.on_demand(
        dims=('ind', 'trait'),
        description="individual's actual trait value"
    )
    n_offspring = xs.on_demand(
        dims='ind',
        description="number of offspring"
    )

    taxon_id = xs.on_demand(
        dims='ind',
        description="taxon id number"
    )

    ancestor_id = xs.on_demand(
        dims='ind',
        description="ancestor taxa id number",
        encoding={'fill_value': -1}
    )

    @property
    def individuals(self):
        if self._individuals is None:
            self._individuals = self._model.individuals
        return self._individuals

    @id.compute
    def _get_id(self):
        return self.individuals["id"]

    @parent.compute
    def _get_parent(self):
        return self.individuals["parent"]

    @x.compute
    def _get_x(self):
        return self.individuals["x"]

    @y.compute
    def _get_y(self):
        return self.individuals["y"]

    @trait.compute
    def _get_trait(self):
        return self.individuals["trait"]

    @n_offspring.compute
    def _get_n_offspring(self):
        return self.individuals["n_offspring"]

    @taxon_id.compute
    def _get_taxon_id(self):
        return self.individuals["taxon_id"]

    @ancestor_id.compute
    def _get_ancestor_id(self):
        return self.individuals["ancestor_id"]


@xs.process
class IR12Speciation(Speciation):
    """Irwin (2012) Speciation model as a fastscape extension.
    For more info, see :class:`paraspec.base.IR12SpeciationModel`.
    """
    nb_radius = xs.variable(description="fixed neighborhood radius")
    car_cap = xs.variable(description="carrying capacity within a neighborhood")
    sigma_mov = xs.variable(description="controls dispersal magnitude")
    sigma_mut = xs.variable(description="controls mutation magnitude")
    sigma_env_trait = xs.variable(description="controls strength abiotic filtering")
    mut_prob = xs.variable(description="mutation probability")

    abundance = xs.variable(intent="out", description="abundance")

    fitness = xs.on_demand(
        dims='ind',
        description="individual's fitness value"
    )

    def _get_model_params(self):
        return {
            "nb_radius": self.nb_radius,
            "car_cap": self.car_cap,
            "sigma_mov": self.sigma_mov,
            "sigma_mut": self.sigma_mut,
            "sigma_env_trait": self.sigma_env_trait,
            "random_seed": self.random_seed,
            "slope_trait_env": self.slope_trait_env,
        }

    def initialize(self):
        X, Y = np.meshgrid(self.grid_x, self.grid_y)

        self._model = IR12SpeciationModel(
            X, Y,
            self.init_abundance,
            # TODO: maybe expose kwargs below as process inputs
            lifespan=None,
            always_direct_parent=False,
            **self._get_model_params()
        )
        traits_range = [[min_t, max_t] for min_t, max_t in zip(self.init_min_trait, self.init_max_trait)]
        self._model.initialize(traits_range)

    @xs.runtime(args='step_delta')
    def run_step(self, dt):
        # reset individuals "cache"
        self._individuals = None

        # maybe update model parameters
        self._model.params.update(self._get_model_params())

        self.abundance = self._model.abundance
        self._model.evaluate_fitness(self.env_field, self.min_env, self.max_env, dt)

    @xs.runtime(args='step_delta')
    def finalize_step(self, dt):
        self._model.update_individuals(dt)

    @fitness.compute
    def _get_fitness(self):
        return self.individuals["fitness"]


@xs.process
class DD03Speciation(Speciation):
    """Doebeli & Dieckmann (2003) Speciation model as a fastscape extension.
    For more info, see :class:`paraspec.base.DD03SpeciationModel`.
    """
    birth_rate = xs.variable(description="birth rate of individuals")
    movement_rate = xs.variable(description="movement/dispersion rate of individuals")
    car_cap_max = xs.variable(description="maximum carrying capacity")
    sigma_env_trait = xs.variable(description="controls strength abiotic filtering")
    mut_prob = xs.variable(description="mutation probability")
    sigma_mut = xs.variable(description="controls mutation magnitude")
    sigma_mov = xs.variable(description="controls movement/dispersal magnitude")
    sigma_comp_trait = xs.variable(description="controls competition strength among individuals and its based on trait")
    sigma_comp_dist = xs.variable(description="controls competition strength among individuals and its based on "
                                              "spatial distance")
    abundance = xs.variable(intent="out", description="abundance of individuals")

    def _get_model_params(self):
        return {
            'birth_rate': self.birth_rate,
            'movement_rate': self.movement_rate,
            'car_cap_max': self.car_cap_max,
            'sigma_env_trait': self.sigma_env_trait,
            'mut_prob': self.mut_prob,
            'sigma_mut': self.sigma_mut,
            'sigma_mov': self.sigma_mov,
            'sigma_comp_trait': self.sigma_comp_trait,
            'sigma_comp_dist': self.sigma_comp_dist,
            "random_seed": self.random_seed,
            "slope_trait_env": self.slope_trait_env,
        }

    def initialize(self):
        X, Y = np.meshgrid(self.grid_x, self.grid_y)

        self._model = DD03SpeciationModel(
            X, Y,
            self.init_abundance,
            lifespan=None,
            always_direct_parent=False,
            **self._get_model_params()
        )

        traits_range = [[min_t, max_t] for min_t, max_t in zip(self.init_min_trait, self.init_max_trait)]
        self._model.initialize(traits_range)

    @xs.runtime(args='step_delta')
    def run_step(self, dt):
        # reset individuals "cache"
        self._individuals = None

        # maybe update model parameters
        self._model.params.update(self._get_model_params())

        self.abundance = self._model.abundance
        self._model.evaluate_fitness(self.env_field, self.min_env, self.max_env, dt)

    @xs.runtime(args='step_delta')
    def finalize_step(self, dt):
        self._model.update_individuals(dt)


@xs.process
class CompoundEnvironment:
    """Multiple environment fields defined on the same grid.
    """
    field_arrays = xs.group_dict("env_field")
    env_field = xs.foreign(Speciation, "env_field", intent="out")

    def initialize(self):
        self.env_field = np.stack(list(self.field_arrays.values()))

    def run_step(self):
        self.env_field = np.stack(list(self.field_arrays.values()))


@xs.process
class ElevationEnvField1:
    """Topographic elevation used as one environment field for the
    speciation model.

    """
    elevation = xs.foreign(SurfaceTopography, "elevation")
    field = xs.variable(dims=("y", "x"), intent="out", groups="env_field")

    def initialize(self):
        self.field = self.elevation

    def run_step(self):
        self.field = self.elevation


@xs.process
class ElevationEnvField2:
    """Topographic elevation used as one environment field for the
    speciation model.

    """
    elevation = xs.foreign(SurfaceTopography, "elevation")
    field = xs.variable(dims=("y", "x"), intent="out", groups="env_field")

    def initialize(self):
        self.field = self.elevation

    def run_step(self):
        self.field = self.elevation


ir12spec_model = basic_model.update_processes(
    {"life": IR12Speciation, "life_env": CompoundEnvironment}
)

dd03spec_model = basic_model.update_processes(
    {"life": DD03Speciation, "life_env": CompoundEnvironment}
)
