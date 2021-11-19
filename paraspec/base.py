import textwrap
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial as spatial
from numba import jit


class SpeciationModelBase(object):
    """
    Speciation Model base class with common methods for the different types of speciation models.
    """
    def __init__(self, grid_x, grid_y, init_pop_size,
                 random_seed=None):
        """
        Initialization of based model.
        :param grid_x:
        :param grid_y:
        :param init_pop_size:
        :param random_seed:
        """
        grid_x = np.asarray(grid_x)
        grid_y = np.asarray(grid_y)
        self._grid_bounds = {'x': np.array([grid_x.min(), grid_x.max()]),
                             'y': np.array([grid_y.min(), grid_y.max()])}
        self._grid_index = self._build_grid_index([grid_x, grid_y])

        self._population = {}
        self._init_pop_size = init_pop_size
        self._rng = np.random.default_rng(random_seed)

        # https://stackoverflow.com/questions/16016959/scipy-stats-seed
        self._truncnorm = stats.truncnorm
        self._truncnorm.random_state = self._rng

    def initialize(self, trait_range=(0.5, 0.5), x_range=None, y_range=None):
        """
        Initialization of a group of individuals with randomly distributed traits, which are
        randomly located in two dimensional space.

        :param trait_range: tuple
            trait range of initial population
        :param x_range: tuple, optional
            Range (min, max) to define initial spatial bounds
            of population in the x direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the x direction.
        :param y_range:tuples, optional
            Range (min, max) to define initial spatial bounds
            of population in the y direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the y direction.
        :return:
        """

        x_bounds = self._grid_bounds['x']
        y_bounds = self._grid_bounds['y']
        x_range = x_range or x_bounds
        y_range = y_range or y_bounds

        if ((x_range[0] < x_bounds[0]) or (x_range[1] > x_bounds[1]) or
                (y_range[0] < y_bounds[0]) or (y_range[1] > y_bounds[1])):
            raise ValueError("x_range and y_range must be within model bounds")

        population = {'step': 0,
                      'time': 0.,
                      'dt': 0.,
                      'id': np.arange(0, self._init_pop_size),
                      'parent': np.arange(0, self._init_pop_size),
                      'x': self._sample_in_range(x_range),
                      'y': self._sample_in_range(y_range),
                      'trait': self._sample_in_range(trait_range)}
        self._population.update(population)

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

        :param varnames: list or string, optional
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

    @staticmethod
    def _build_grid_index(grid_coords):
        grid_points = np.column_stack([c.ravel() for c in grid_coords])
        return spatial.cKDTree(grid_points)

    def _get_optimal_env_value(self, env_field, pop_points):
        """
        Optimal environmental value defined on the grid
        of the respective environmental field and taken
        as the nearest grid node to the location of
        each individual.
        :param env_field: array-like
        :param pop_points: array-like
        :return:
        """
        _, idx = self._grid_index.query(pop_points)
        return env_field.ravel()[idx]

    def _sample_in_range(self, values_range):
        return self._rng.uniform(values_range[0], values_range[1], self._init_pop_size)

    def _within_bounds(self, x, y, sigma):
        # TODO: check effects of movement and boundary conditions
        # TODO: Make boundary conditions of speciation model to match those of LEM
        delta_bounds_x = self._grid_bounds['x'][:, None] - x
        delta_bounds_y = self._grid_bounds['y'][:, None] - y
        new_x = self._truncnorm.rvs(*(delta_bounds_x/sigma), loc=x, scale=sigma)
        new_y = self._truncnorm.rvs(*(delta_bounds_y/sigma), loc=y, scale=sigma)
        return new_x, new_y


class IR12SpeciationModel(SpeciationModelBase):
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

    def __init__(self, grid_x, grid_y, init_pop_size, nb_radius=500., lifespan=None, capacity=1000., sigma_w=500.,
                 sigma_d=5., sigma_mut=500., m_freq=0.05, random_seed=None, on_extinction='warn',
                 always_direct_parent=True):
        """Setup a new speciation model.

        Parameters
        ----------
        grid_x : array-like
            Grid x-coordinates.
        grid_y : array_like
            Grid y-coordinates.
        init_pop_size : int
            Total number of individuals generated as the initial population.
        nb_radius: float
            Fixed radius of the circles that define the neighborhood
            around each individual.
        capacity: int
            Capacity of population within the neighborhood area.
        lifespan: float, optional
            Reproductive lifespan of organism. Used to scale the
            parameters below with time step length. If None (default), the
            lifespan will always match time step length so the parameters
            won't be scaled.
        sigma_w: float
            Width of fitness curve.
        sigma_d: float
            Width of dispersal curve.
        sigma_mut: float
            Width of mutation curve.
        m_freq: float
            Probability of mutation occurring in offspring.
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
        super().__init__(grid_x, grid_y, init_pop_size, random_seed)

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

    def _get_n_gen(self, dt):
        # number of generations during one time step.

        if self._params['lifespan'] is None:
            return 1.
        else:
            return dt / self._params['lifespan']

    def _get_scaled_params(self, dt):
        # Scale sigma parameters according to the number of generations that
        # succeed to each other during a time step.

        n_gen = self._get_n_gen(dt)

        sigma_w = self._params['sigma_w']
        sigma_d = self._params['sigma_d'] * np.sqrt(n_gen)
        sigma_mut = self._params['sigma_mut'] * np.sqrt(n_gen)

        return sigma_w, sigma_d, sigma_mut

    def _count_neighbors(self, pop_points):
        index = spatial.cKDTree(pop_points)
        neighbors = index.query_ball_tree(index, self._params['nb_radius'])

        return np.array([len(nb) for nb in neighbors])

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

            opt_trait = self._get_optimal_env_value(env_field, pop_points)

            fitness = np.exp(-(self._population['trait'] - opt_trait) ** 2
                             / (2 * sigma_w ** 2))

            n_gen = self._get_n_gen(dt)
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
            new_population['trait'] = self._rng.normal(
                new_population['trait'], sigma_mut)

            # disperse offspring within grid bounds
            new_x, new_y = self._within_bounds(new_population['x'],
                                               new_population['y'],
                                               sigma_d)
            new_population['x'] = new_x
            new_population['y'] = new_y

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


class DD03SpeciationModel(SpeciationModelBase):
    """
    Speciation model for asexual populations based on the model by:
        Doebeli, M., & Dieckmann, U. (2003).
        Speciation along environmental gradients.
        Nature, 421, 259–264.
        https://doi.org/10.1038/nature01312.Published.
    """

    def __init__(self, grid_x, grid_y, init_pop_size, random_seed=None, lifespan=None, birth_rate=1, movement_rate=5,
                 slope_topt_env=0.95, car_cap_max=500, sigma_opt_trait=0.3, mut_prob=0.005, sigma_mut=0.05,
                 sigma_mov=0.12, sigma_comp_trait=0.9, sigma_comp_dist=0.19):
        """
        Initialization of
        :param grid_x: array-like
            grid x-coordinate
        :param grid_y: array-like
            grid y-coordinate
        :param init_pop_size: integer
            initial number of individuals
        :param random_seed: integer
            seed used in random number generator
        :param lifespan: float, optional
            Reproductive lifespan of organism. If None (default), the
            lifespan will always match time step length
        :param birth_rate: integer or float
            birth rate of individuals
        :param movement_rate: integer of float
            movement/dispersion rate of individuals
        :param slope_topt_env: float
            slope of the relationship between optimum trait and environmental field
        :param car_cap_max: integer
            maximum carrying capacity
        :param sigma_opt_trait: float
            variability of carrying capacity
        :param mut_prob: float
            mutation probability
        :param sigma_mut: float
            variability of mutated trait
        :param sigma_mov: float
            variability of movement distance
        :param sigma_comp_trait: float
            variability of competition trait distance between individuals
        :param sigma_comp_dist: float
            variability of competition spatial distance between individuals
        """
        super().__init__(grid_x, grid_y, init_pop_size, random_seed)
        self._params = {
                'lifespan': lifespan,
                'birth_rate': birth_rate,
                'movement_rate': movement_rate,
                'slope_topt_env': slope_topt_env,
                'car_cap_max': car_cap_max,
                'sigma_opt_trait': sigma_opt_trait,
                'mut_prob': mut_prob,
                'sigma_mut': sigma_mut,
                'sigma_mov': sigma_mov,
                'sigma_comp_trait': sigma_comp_trait,
                'sigma_comp_dist': sigma_comp_dist
            }

        self.dtf = pd.DataFrame({
            'time': np.array([]),
            'step': np.array([]),
            'dt': np.array([]),
            'id': np.array([]),
            'parent': np.array([]),
            'x': np.array([]),
            'y': np.array([]),
            'trait': np.array([])
        })

    def _x_within_bounds(self, x):
        x_bounds = self._grid_bounds['x']
        return np.where(np.logical_and(x_bounds[0] < x, x < x_bounds[1]), x,
                        np.where(x < x_bounds[0], x_bounds[0], x_bounds[1]))

    def _y_within_bounds(self, y):
        y_bounds = self._grid_bounds['y']
        return np.where(np.logical_and(y_bounds[0] < y, y < y_bounds[1]), y,
                        np.where(y < y_bounds[0], y + y_bounds[1], y - y_bounds[1]))

    def update(self, Z):
        """
        Update method
        :param Z:
        :return:
        """
        # Compute local individual environmental field
        z_i = self._get_optimal_env_value(Z, np.column_stack([self._population['x'], self._population['y']]))

        # Compute event probabilities
        birth_i = self._population['trait'].size * [self._params['birth_rate']]
        death_i = death_rate(trait=self._population['trait'],
                             x=self._population['x'], y=self._population['y'], z=z_i,
                             xmin=self._grid_bounds['x'][0], xmax=self._grid_bounds['x'][1],
                             ymin=self._grid_bounds['y'][0], ymax=self._grid_bounds['y'][1],
                             zmin=np.min(Z), zmax=np.max(Z),
                             slope_topt_env=self._params['slope_topt_env'],
                             K0=self._params['car_cap_max'],
                             sigma_opt_trait=self._params['sigma_opt_trait'],
                             sigma_comp_trait=self._params['sigma_comp_trait'],
                             sigma_comp_dist=self._params['sigma_comp_dist'], )
        movement_i = self._population['trait'].size * [self._params['movement_rate']]
        events_tot = np.sum(birth_i) + np.sum(death_i) + np.sum(movement_i)
        events_i = self._rng.choice(a=['B', 'D', 'M'], size=self._population['trait'].size,
                                    p=[np.sum(birth_i) / events_tot, np.sum(death_i) / events_tot,
                                       np.sum(movement_i) / events_tot])
        delta_t = self._rng.exponential(1 / events_tot, self._population['id'].size)

        # initialize temporary dictionaries
        offspring = {k: np.array([]) for k in ('id', 'parent', 'x', 'y', 'trait')}
        extant = self._population.copy()

        # Birth
        offspring['id'] = np.arange(self._population['id'].max() + 1,
                                    self._population['id'][events_i == 'B'].size + self._population['id'].max() + 1)
        offspring['parent'] = self._population['id'][np.where(events_i == 'B')]
        offspring['x'] = self._x_within_bounds(self._population['x'][events_i == 'B'])
        offspring['y'] = self._y_within_bounds(self._population['y'][events_i == 'B'])
        mu_prob = self._rng.uniform(0, 1, self._population['trait'][events_i == 'B'].size) < self._params['mut_prob']

        offspring['trait'] = np.where(mu_prob,
                                      self._rng.normal(self._population['trait'][events_i == 'B'],
                                                       self._params['sigma_mut']),
                                      self._population['trait'][events_i == 'B'])

        # Movement
        new_x, new_y = self._within_bounds(self._population['x'][events_i == 'M'],
                                           self._population['y'][events_i == 'M'],
                                           self._params['sigma_mov'])
        extant['x'][events_i == 'M'] = new_x
        extant['y'][events_i == 'M'] = new_y

        # Death
        todie = self._rng.choice(extant['id'], size=self._population['id'][events_i == 'D'].size,
                                 p=death_i / death_i.sum(), replace=False)
        todie_ma = np.logical_not(np.any(extant['id'] == todie.repeat(extant['id'].size).reshape(todie.size,
                                                                                                 extant['id'].size),
                                         axis=0))
        extant['id'] = extant['id'][todie_ma]
        extant['parent'] = extant['parent'][todie_ma]
        extant['x'] = extant['x'][todie_ma]
        extant['y'] = extant['y'][todie_ma]
        extant['trait'] = extant['trait'][todie_ma]

        if self._params['lifespan'] is None:
            lifespan = 1
        else:
            lifespan = self._params['lifespan']

        dt = np.sum(lifespan * delta_t / np.sum(delta_t))
        # Update dictionary
        self._population.update({k: np.append(extant[k], offspring[k]) for k in offspring.keys()})
        self._population.update({'time': self._population['time'] + dt})
        self._population.update({'step': self._population['step'] + 1})
        self._population.update({'dt': dt})


@jit(nopython=True)
def death_rate(trait, x, y, z, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, slope_topt_env=0.95,
               car_cap_max=500., sigma_opt_trait=0.3, sigma_comp_trait=0.9, sigma_comp_dist=0.19):
    """
    Logistic death rate for DD03 Speciation model
    implemented using numba
    :param trait: 1d array, floats
        trait value per individual
    :param x: 1d array, floats
        location along the x coord
    :param y: 1d array, floats
        location along the y coord
    :param z: 1d array, floats
        optimal environmental value
    :param zmax: value, float
        maximum value of environmental field
    :param zmin: value, float
        maximum value of environmental field
    :param ymax: value, float
        maximum value of y coordinate
    :param ymin: value, float
        minimum value of y coordinate
    :param xmax: value, float
        maximum value of y coordinate
    :param xmin: value, float
        minimum value of y coordinate
    :param slope_topt_env: value, float
        slope of the relationship between environmental field and optimal trait value
    :param car_cap_max: value, float
        maximum carrying capacity
    :param sigma_opt_trait: value, float
        variability of trait to abiotic condition
    :param sigma_comp_trait: value, float
        competition variability as a measure of the trait difference between individuals
    :param sigma_comp_dist: value, float
        competition variability as a measure of the spatial distance between individuals
    :return: 1d array, floats
        death rate per individual
    """
    x = (x-xmin)/(xmax-xmin)
    y = (y-ymin)/(ymax-ymin)
    z = (z-zmin)/(zmax-zmin)
    delta_trait = np.expand_dims(trait, 1) - trait
    delta_xy = np.sqrt((np.expand_dims(x, 1) - x) ** 2 + (np.expand_dims(y, 1) - y) ** 2)
    delta_trait_norm = np.exp(-0.5 * delta_trait ** 2 / sigma_comp_trait ** 2)
    delta_xy_norm = np.exp(-0.5 * delta_xy ** 2 / sigma_comp_dist ** 2)
    n_eff = 1 / (2 * np.pi * sigma_comp_dist ** 2) * np.sum(delta_trait_norm * delta_xy_norm, axis=1)
    topt = ((slope_topt_env * (z - 0.5)) + 0.5)
    k = car_cap_max * np.exp(-0.5 * (trait - topt) ** 2 / sigma_opt_trait ** 2)
    return n_eff / k
