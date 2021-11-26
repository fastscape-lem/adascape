from fastscape.models import basic_model
from fastscape.processes import SurfaceTopography, UniformRectilinearGrid2D
import numpy as np
import xsimlab as xs
from .base import IR12SpeciationModel
from .base import DD03SpeciationModel


@xs.process
class Speciation:
    """
    Speciation model as a fastscape extension
    """
    init_size = xs.variable(description="initial population size", static=True)
    init_min_trait = xs.variable(description="initial min trait value", static=True)
    init_max_trait = xs.variable(description="initial max trait value", static=True)
    random_seed = xs.variable(
        default=None,
        description="random number generator seed",
        static=True
    )
    env_field = xs.variable(dims=("y", "x"))

    grid_x = xs.foreign(UniformRectilinearGrid2D, "x")
    grid_y = xs.foreign(UniformRectilinearGrid2D, "y")

    _model = xs.any_object(description="speciation model instance")
    _population = xs.any_object(description="speciation model state dictionary")

    id = xs.on_demand(
        dims='pop',
        description="individual's id"
    )
    parent = xs.on_demand(
        dims='pop',
        description="individual's ancestor"
    )
    x = xs.on_demand(
        dims='pop',
        description="individual's x-position"
    )
    y = xs.on_demand(
        dims='pop',
        description="individual's y-position"
    )
    trait = xs.on_demand(
        dims='pop',
        description="individual's actual trait value"
    )

    @property
    def population(self):
        if self._population is None:
            self._population = self._model.population
        return self._population

    @id.compute
    def _get_id(self):
        return self.population["id"]

    @parent.compute
    def _get_parent(self):
        return self.population["parent"]

    @x.compute
    def _get_x(self):
        return self.population["x"]

    @y.compute
    def _get_y(self):
        return self.population["y"]

    @trait.compute
    def _get_trait(self):
        return self.population["trait"]


@xs.process
class IR12Speciation(Speciation):
    """Irwin (2012) Speciation model as a fastscape extension.
    For more info, see :class:`paraspec.base.IR12SpeciationModel`.
    """
    nb_radius = xs.variable(description="fixed neighborhood radius")
    capacity = xs.variable(description="carrying capacity within a neighborhood")
    sigma_d = xs.variable(description="controls dispersal magnitude")
    sigma_mut = xs.variable(description="controls mutation magnitude")
    sigma_w = xs.variable(description="scales fitness")

    size = xs.variable(intent="out", description="population size")

    opt_trait = xs.on_demand(
        dims='pop',
        description="individual's optimal trait value"
    )
    r_d = xs.on_demand(
        dims='pop',
        description="individual's r_d value"
    )
    fitness = xs.on_demand(
        dims='pop',
        description="individual's fitness value"
    )
    n_offspring = xs.on_demand(
        dims='pop',
        description="number of offsrping"
    )

    def _get_model_params(self):
        return {
            "nb_radius": self.nb_radius,
            "capacity": self.capacity,
            "sigma_d": self.sigma_d,
            "sigma_mut": self.sigma_mut,
            "sigma_w": self.sigma_w,
            "random_seed": self.random_seed,
        }

    def initialize(self):
        X, Y = np.meshgrid(self.grid_x, self.grid_y)

        self._model = IR12SpeciationModel(
            X, Y,
            self.init_size,
            # TODO: maybe expose kwargs below as process inputs
            m_freq=1.,
            lifespan=None,
            always_direct_parent=False,
            **self._get_model_params()
        )

        self._model.initialize([self.init_min_trait, self.init_max_trait])

    @xs.runtime(args='step_delta')
    def run_step(self, dt):
        # reset population "cache"
        self._population = None

        # maybe update model parameters
        self._model.params.update(self._get_model_params())

        self.size = self._model.population_size
        self._model.evaluate_fitness(self.env_field, dt)

    @xs.runtime(args='step_delta')
    def finalize_step(self, dt):
        self._model.update_population(dt)

    @opt_trait.compute
    def _get_opt_trait(self):
        return self.population["opt_trait"]

    @r_d.compute
    def _get_r_d(self):
        return self.population["r_d"]

    @fitness.compute
    def _get_fitness(self):
        return self.population["fitness"]

    @n_offspring.compute
    def _get_n_offspring(self):
        return self.population["n_offspring"]


@xs.process
class DD03Speciation(Speciation):
    """Doebeli & Dieckmann (2003) Speciation model as a fastscape extension.
    For more info, see :class:`paraspec.base.DD03SpeciationModel`.
    """
    birth_rate = xs.variable(description="birth rate of individuals")
    movement_rate = xs.variable(description="movement/dispersion rate of individuals")
    slope_topt_env = xs.variable(description="slope of the relationship between optimum trait and environmental field")
    car_cap_max = xs.variable(description="maximum carrying capacity")
    sigma_opt_trait = xs.variable(description="controls strength abiotic filtering")
    mut_prob = xs.variable(description="mutation probability")
    sigma_mut = xs.variable(description="controls mutation magnitude")
    sigma_mov = xs.variable(description="controls movement/dispersal magnitude")
    sigma_comp_trait = xs.variable(description="controls competition strength among individual based trait")
    sigma_comp_dist = xs.variable(description="controls competition strength among individual based distance")
    size = xs.variable(intent="out", description="abundance of individuals")

    def _get_model_params(self):
        return {
            'birth_rate': self.birth_rate,
            'movement_rate': self.movement_rate,
            'slope_topt_env': self.slope_topt_env,
            'car_cap_max': self.car_cap_max,
            'sigma_opt_trait': self.sigma_opt_trait,
            'mut_prob': self.mut_prob,
            'sigma_mut': self.sigma_mut,
            'sigma_mov': self.sigma_mov,
            'sigma_comp_trait': self.sigma_comp_trait,
            'sigma_comp_dist': self.sigma_comp_dist,
            "random_seed": self.random_seed
        }

    def initialize(self):
        X, Y = np.meshgrid(self.grid_x, self.grid_y)

        self._model = DD03SpeciationModel(
            X, Y,
            self.init_size,
            **self._get_model_params()
        )

        self._model.initialize([self.init_min_trait, self.init_max_trait])

    @xs.runtime(args='step_delta')
    def run_step(self, dt):
        # reset population "cache"
        self._population = None

        # maybe update model parameters
        self._model.params.update(self._get_model_params())

        self.size = self._model.population_size
        self._model.update(self.env_field)


@xs.process
class IR12EnvironmentElevation:
    """Topographic elevation used as the environment field for the
    speciation model.

    """
    elevation = xs.foreign(SurfaceTopography, "elevation")
    env_field = xs.foreign(IR12Speciation, "env_field", intent="out")

    def initialize(self):
        self.env_field = self.elevation


ir12spec_model = basic_model.update_processes(
    {"life": IR12Speciation, "life_env": IR12EnvironmentElevation}
)


@xs.process
class DD03EnvironmentElevation:
    """Topographic elevation used as the environment field for the
    speciation model.

    """
    elevation = xs.foreign(SurfaceTopography, "elevation")
    env_field = xs.foreign(DD03Speciation, "env_field", intent="out")

    def initialize(self):
        self.env_field = self.elevation


dd03spec_model = basic_model.update_processes(
    {"life": DD03Speciation, "life_env": DD03EnvironmentElevation}
)
