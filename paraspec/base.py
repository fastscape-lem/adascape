import textwrap
import warnings

import numpy as np
import pandas as pd

import scipy.stats as stats
import scipy.spatial as spatial


class ParapatricSpeciationModel(object):
    """Model of speciation along an environmental gradient defined on a
    2-d grid.

    This model is adapted from:

    Irwin D.E., 2012. Local Adaptation along Smooth Ecological
    Gradients Causes Phylogeographic Breaks and Phenotypic Clustering.
    The American Naturalist Vol. 180, No. 1, pp. 35-49.
    DOI: 10.1086/666002

    A model run starts with a given number of individuals with random
    positions (x, y) generated uniformly within the grid bounds and
    initial "trait" values generated uniformly within a given range.

    Then, at each step, the number of offspring for each individual is
    determined using a fitness value computed from the comparison of
    environmental ("trait" vs. "optimal trait") and population density
    ("number of individuals in the neighborhood" vs "capacity")
    variables measured locally. Environmental variables are given on a
    grid, while the neighborhood is defined by a circle of a given
    radius centered on each individual.

    New individuals are generated from the offspring, which undergo
    some random dispersion (position) - and mutation (trait
    value). Dispersion is constrained so that all individuals stay
    within the domain delineated by the grid.
    """

    def __init__(self, grid_x, grid_y, init_pop_size, nb_radius=500.,
                 lifespan=1., capacity=1000., sigma_w=500., sigma_d=5.,
                 sigma_mut=500., m_freq=0.05, random_seed=None,
                 on_extinction='warn', always_direct_parent=True):
        """Setup a new speciation model.

        Parameters
        ----------
        grid_x : array-like
            Grid x-coordinates.
        grid_y : array_like
            Grid y-coordinates.
        init_pop_size : int
            Total number of indiviuals generated as the initial population.
        nb_radius: float
            Fixed radius of the circles that define the neighborhood
            around each individual.
        capacity: int
            Capacity of population within the neighborhood area.
        lifespan: float
            Reproductive lifespan of organism. Used to scale the
            parameters below with time step length.
        sigma_w: float
            Width of fitness curve.
        sigma_d: float
            Width of dispersal curve.
        sigma_mut: float
            Width of mutation curve.
        m_freq: float
            Probability of mutation occurrring in offspring.
        random_seed : int or :class:`numpy.random.RandomState` object
            Fixed random state for reproducible experiments.
            If None (default), results will differ from one run
            to another.
        on_extinction : {'warn', 'raise', 'ignore'}
            Behavior when no offspring is generated (total extinction of
            population) during model runtime. 'warn' (default) displays
            a RuntimeWarning, 'raise' raises a RuntimeError (the simulation
            stops) or 'ignore' silently continues the simulation
            doing nothing (no population).
        always_direct_parent : bool, optional
            If True (default), the id of the parent set for each individual
            of the current population will always correspond to its direct
            parent. If False, those id values may correspond to older
            ancestors. Set this parameter to False if you want to preserve the
            connectivity of the generation tree built by calling
            ``.population`` or ``.to_dataframe()`` at arbitrary steps of a
            model run.

        """
        grid_x = np.asarray(grid_x)
        grid_y = np.asarray(grid_y)
        self._grid_bounds = {'x': np.array([grid_x.min(), grid_x.max()]),
                             'y': np.array([grid_y.min(), grid_y.max()])}
        self._grid_index = self._build_grid_index([grid_x, grid_y])

        self._population = {}
        self._init_pop_size = init_pop_size

        valid_on_extinction = ('warn', 'raise', 'ignore')

        if on_extinction not in valid_on_extinction:
            raise ValueError(
                "invalid value found for 'on_extinction' parameter. "
                "Found {!r}, must be one of {!r}"
                .format(on_extinction, valid_on_extinction)
            )

        # default parameter values
        self._params = {
            'nb_radius': nb_radius,
            'lifespan': lifespan,
            'capacity': capacity,
            'sigma_w': sigma_w,
            'sigma_d': sigma_d,
            'sigma_mut': sigma_mut,
            'm_freq': m_freq,
            'random_seed': random_seed,
            'on_extinction': on_extinction,
            'always_direct_parent': always_direct_parent
        }

        self._set_direct_parent = True

        if isinstance(random_seed, np.random.RandomState):
            self._random = random_seed
        else:
            self._random = np.random.RandomState(random_seed)

        # https://stackoverflow.com/questions/16016959/scipy-stats-seed
        self._truncnorm = stats.truncnorm
        self._truncnorm.random_state = self._random

    def _build_grid_index(self, grid_coords):
        grid_points = np.column_stack([c.ravel() for c in grid_coords])

        return spatial.cKDTree(grid_points)

    @property
    def params(self):
        """Model parameters (dict)."""
        return self._params

    @property
    def population(self):
        """Population data (dict) at the current time step."""
        self._set_direct_parent = True

        return self._population

    @property
    def population_size(self):
        """Number of individuals in the population at the current time
        step (return None if the population is not yet initialized).

        """
        if not self._population:
            return None
        else:
            return self._population['trait'].size

    def to_dataframe(self, varnames=None):
        """Return the population data at the current time step as a
        :class:`pandas.Dataframe`.

        Parameters
        ----------
        varnames : list or string, optional
            Only export those variable name(s) as dataframe column(s).
            Default: export all variables.

        """
        if varnames is None:
            data = self.population
        elif isinstance(varnames, str):
            data = {varnames: self.population[varnames]}
        else:
            data = {k: self.population[k] for k in varnames}

        return pd.DataFrame(data)

    def _sample_in_range(self, range):
        return self._random.uniform(range[0], range[1], self._init_pop_size)

    def initialize_population(self, trait_range, x_range=None, y_range=None):
        """Initialize population data.

        The positions (x, y) of population individuals are generated
        uniformly within the grid bounds.

        Parameters
        ----------
        trait_range : tuple
            Range (min, max) within which initial trait values
            are uniformly sampled for the population individuals.
        x_range : tuple, optional
            Range (min, max) to define initial spatial bounds
            of population in the x direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the x direction.
        y_range : tuples, optional
            Range (min, max) to define initial spatial bounds
            of population in the y direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the y direction.

        """
        population = {}
        population['step'] = 0
        population['time'] = 0.
        population['id'] = np.arange(0, self._init_pop_size)
        population['parent'] = np.arange(0, self._init_pop_size)

        x_bounds = self._grid_bounds['x']
        y_bounds = self._grid_bounds['y']
        x_range = x_range or x_bounds
        y_range = y_range or y_bounds

        if ((x_range[0] < x_bounds[0]) or (x_range[1] > x_bounds[1]) or
                (y_range[0] < y_bounds[0]) or (y_range[1] > y_bounds[1])):
            raise ValueError("x_range and y_range must be within model bounds")

        population['x'] = self._sample_in_range(x_range)
        population['y'] = self._sample_in_range(y_range)
        population['trait'] = self._sample_in_range(trait_range)

        self._population.update(population)

    def _get_scaled_params(self, dt):
        # Scale sigma parameters according to the number of generations that
        # succeed to each other during a time step.

        n_gen = dt / self._params['lifespan']

        sigma_w = self._params['sigma_w']
        sigma_d = self._params['sigma_d'] * np.sqrt(n_gen)
        sigma_mut = self._params['sigma_mut'] * np.sqrt(n_gen)

        return sigma_w, sigma_d, sigma_mut

    def _count_neighbors(self, pop_points):
        index = spatial.cKDTree(pop_points)
        neighbors = index.query_ball_tree(index, self._params['nb_radius'])

        return np.array([len(nb) for nb in neighbors])

    def _get_optimal_trait(self, env_field, pop_points):
        # the optimal trait value is given by the environmental field
        # defined on the grid (nearest grid node).

        _, idx = self._grid_index.query(pop_points)

        return env_field.ravel()[idx]

    def evaluate_fitness(self, env_field, dt):
        """Evaluate fitness and generate offspring number for population and
        with environmental conditions both taken at the current time step.

        Parameters
        ----------
        env_field : array-like
            Environmental field defined on the grid.
        dt : float
            Time step duration.

        """
        sigma_w, _, _ = self._get_scaled_params(dt)

        if self.population_size:
            pop_points = np.column_stack([self._population['x'],
                                          self._population['y']])

            # compute offspring sizes
            r_d = self._params['capacity'] / self._count_neighbors(pop_points)

            opt_trait = self._get_optimal_trait(env_field, pop_points)

            fitness = np.exp(-(self._population['trait'] - opt_trait)**2
                             / (2 * sigma_w**2))

            n_gen = dt / self._params['lifespan']
            n_offspring = np.round(
                r_d * fitness * np.sqrt(n_gen)
            ).astype('int')

        else:
            r_d = np.array([])
            opt_trait = np.array([])
            fitness = np.array([])
            n_offspring = np.array([], dtype='int')

        self._population.update({
            'r_d': r_d,
            'opt_trait': opt_trait,
            'fitness': fitness,
            'n_offspring': n_offspring
        })

    def update_population(self, dt):
        """Update population data (generate, mutate and disperse offspring).

        """
        _, sigma_d, sigma_mut = self._get_scaled_params(dt)

        n_offspring = self._population['n_offspring']

        if not n_offspring.sum():
            # population total extinction
            if self._params['on_extinction'] == 'raise':
                raise RuntimeError("no offspring generated. "
                                   "Model execution has stopped.")

            if self._params['on_extinction'] == 'warn':
                warnings.warn("no offspring generated. "
                              "Model execution continues with no population.",
                              RuntimeWarning)

            new_population = {k: np.array([])
                              for k in ('id', 'parent', 'x', 'y', 'trait')}

        else:
            # generate offspring
            new_population = {k: np.repeat(self._population[k], n_offspring)
                              for k in ('x', 'y', 'trait')}

            # set parents either to direct parents or older ancestors
            if self._set_direct_parent:
                parents = self._population['id']
            else:
                parents = self._population['parent']

            new_population['parent'] = np.repeat(parents, n_offspring)

            last_id = self._population['id'][-1] + 1
            new_population['id'] = np.arange(
                last_id, last_id + n_offspring.sum())

            # mutate offspring
            new_population['trait'] = self._random.normal(
                new_population['trait'], sigma_mut)

            # disperse offspring within grid bounds
            for k in ('x', 'y'):
                bounds = self._grid_bounds[k][:, None] - new_population[k]

                new_k = self._truncnorm.rvs(*(bounds / sigma_d),
                                            loc=new_population[k],
                                            scale=sigma_d)

                new_population[k] = new_k

        self._population['step'] += 1
        self._population['time'] += dt
        self._population.update(new_population)

        # reset fitness / offspring data
        self._population.update({
            'r_d': np.array([]),
            'opt_trait': np.array([]),
            'fitness': np.array([]),
            'n_offspring': np.array([])
        })

        if not self._params['always_direct_parent']:
            self._set_direct_parent = False

    def __repr__(self):
        class_str = type(self).__name__
        population_str = "population: {}".format(
            self.population_size or 'not initialized')
        params_str = "\n".join(["{}: {}".format(k, v)
                                for k, v in self._params.items()])

        return "<{} ({})>\nParameters:\n{}\n".format(
            class_str, population_str, textwrap.indent(params_str, '    '))
