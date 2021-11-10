from fastscape.models import basic_model
from fastscape.processes import SurfaceTopography, UniformRectilinearGrid2D
import numpy as np
import xsimlab as xs

from .base import IR12SpeciationModel


@xs.process
class ParapatricSpeciation:
    """Parapatric speciation model as a fastscape extension.

    For more info, see :class:`paraspec.ParapatricSpeciationModel`.

    """
    init_size = xs.variable(description="initial population size", static=True)
    random_seed = xs.variable(
        default=None,
        description="random number generator seed",
        static=True
    )

    nb_radius = xs.variable(description="fixed neighborhood radius")
    capacity = xs.variable(description="population capacity within neighborhood")
    sigma_d = xs.variable(description="controls dispersal magnitude")
    sigma_mut = xs.variable(description="controls mutation magnitude")
    sigma_w = xs.variable(description="scales fitness")

    env_field = xs.variable(dims=("y", "x"))

    grid_x = xs.foreign(UniformRectilinearGrid2D, "x")
    grid_y = xs.foreign(UniformRectilinearGrid2D, "y")

    size = xs.variable(intent="out", description="population size")

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

        self._model.initialize([self.env_field.min(), self.env_field.max()])

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

    @opt_trait.compute
    def _get_opt_trait(self):
        return self.population["opt_trait"]

    @trait.compute
    def _get_trait(self):
        return self.population["trait"]

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
class ParapatricEnvironmentElevation:
    """Topographic elevation used as the environment field for the parapatric
    speciation model.

    """
    elevation = xs.foreign(SurfaceTopography, "elevation")
    env_field = xs.foreign(ParapatricSpeciation, "env_field", intent="out")

    def initialize(self):
        self.env_field = self.elevation


paraspec_model = basic_model.update_processes(
    {"life": ParapatricSpeciation, "life_env": ParapatricEnvironmentElevation}
)
