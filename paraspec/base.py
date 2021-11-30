import textwrap
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial as spatial
from numba import jit


class SpeciationModelBase:
    """
    Speciation Model base class with common methods for the different
    types of speciation models.
    """

    def __init__(self, grid_x, grid_y, init_pop_size,
                 random_seed=None):
        """
        Initialization of based model.

        Parameters
        ----------
        grid_x : array-like
            Grid x-coordinates.
        grid_y : array_like
            Grid y-coordinates.
        init_pop_size : int
            Total number of individuals generated as the initial population.
        random_seed : int or :class:`numpy.random.default_rng` object
            Fixed random state for reproducible experiments.
            If None (default), results will differ from one run
            to another.
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

        self._params = {}
        self._env_field_bounds = None

    def initialize(self, trait_range=(0.5, 0.5), x_range=None, y_range=None):
        """
        Initialization of a group of individuals with randomly distributed traits,
        and which are randomly located in two dimensional grid.

        Parameters
        ----------
        trait_range : tuple
            trait range of initial population
        x_range : tuple, optional
            Range (min, max) to define initial spatial bounds
            of population in the x direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the x direction.
        y_range : tuple, optional
            Range (min, max) to define initial spatial bounds
            of population in the y direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the y direction.
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
        """Model parameters.

        Returns
        -------
        dict
            Model parameters
        """
        return self._params

    @property
    def population(self):
        """Population data at the current time step.

        Returns
        -------
        dict
            Population data
        """
        self._set_direct_parent = True
        return self._population

    @property
    def population_size(self):
        """Number of individuals in the population at the current time
        step.

        Returns
        -------
        int or None
            size of the population
            (return None if the population is not yet initialized)
        """
        if not self._population:
            return None
        else:
            return self._population['trait'].size

    def to_dataframe(self, varnames=None):
        """Population data at the current time step as a
        pandas Dataframe.

        Parameters
        ----------
        varnames : list or string, optional
            Only export those variable name(s) as dataframe column(s).
            Default: export all variables.

        Returns
        -------
        pandas.Dataframe
            Population data
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
        """
        Builds scipy kd-tre for a quick indexing of all
        points in a grid.

        Parameters
        ----------
        grid_coords : list of arrays
            x and y points in the grid

        Returns
        -------
        scipy.spatial.cKDTree
            kd-tree index object for the set of grid points
        """
        grid_points = np.column_stack([c.ravel() for c in grid_coords])
        return spatial.cKDTree(grid_points)

    def _get_local_env_value(self, env_field, pop_points):
        """
        Local environmental value defined on the grid
        of the respective environmental field and taken
        as the nearest grid node to the location of
        each individual.

        Parameters
        ----------
        env_field : array-like
            Environmental field defined on the grid.
        pop_points : array-like
            x and y location of individuals on the grid.

        Returns
        -------
        array_like
            value of environmental field near the individual location.
        """
        _, idx = self._grid_index.query(pop_points)
        return env_field.ravel()[idx]

    def _sample_in_range(self, values_range):
        """
        Draw a random sample of values for a given range following
        a uniform distribution.

        Parameters
        ----------
        values_range : list or tuple
            max and min value from with to draw random values

        Returns
        -------
        array_like
            random sample of values between given range
        """
        return self._rng.uniform(values_range[0], values_range[1], self._init_pop_size)

    def mov_within_bounds(self, x, y, sigma):
        """
        Move and check if the location of individuals are within grid range.

        Parameters
        ----------
        x : array_like
            locations along the x coordinate
        y : array_like
            locations along the y coordinate
        sigma : float
            movement variability
        Returns
        -------
        array-like
            new coordinate for the moved individuals.
        """
        # TODO: check effects of movement and boundary conditions
        # TODO: Make boundary conditions of speciation model to match those of LEM
        delta_bounds_x = self._grid_bounds['x'][:, None] - x
        delta_bounds_y = self._grid_bounds['y'][:, None] - y
        new_x = self._truncnorm.rvs(*(delta_bounds_x / sigma), loc=x, scale=sigma)
        new_y = self._truncnorm.rvs(*(delta_bounds_y / sigma), loc=y, scale=sigma)
        return new_x, new_y

    @staticmethod
    def _optimal_trait_lin(env_field, local_env_val, slope=0.95):
        """
        Normalized optimal trait value as a linear relationship
        with environmental field. Noticed that the local
        environmental field has been computed as a normalized
        local environmental field value based on the maximum and
        minimum values of the complete environmental field.

        Parameters
        ----------
        env_field : array-like
            Environmental field defined on the grid.
         local_env_val : array-like
            local environmental field for each individual
        slope : float
            slope of the linear relationship between the
            environmental field and the optimal trait value

        Returns
        -------
        array-like
            optimal trait values for each individual.
        """
        norm_loc_env_field = (local_env_val - env_field.min()) / (env_field.max() - env_field.min())
        opt_trait = ((slope * (norm_loc_env_field - 0.5)) + 0.5)
        return opt_trait


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

    def __init__(self, grid_x, grid_y, init_pop_size, nb_radius=500., lifespan=None, car_cap=1000., sigma_w=500.,
                 sigma_mov=5., sigma_mut=500., mut_prob=0.05, random_seed=None, on_extinction='warn',
                 always_direct_parent=True):
        """Initialization of speciation model without competition.

        Parameters
        ----------
        nb_radius: float
            Fixed radius of the circles that define the neighborhood
            around each individual.
        car_cap: int
            Carrying capacity of group of individuals within the neighborhood area.
        lifespan: float, optional
            Reproductive lifespan of organism. Used to scale the
            parameters below with time step length. If None (default), the
            lifespan will always match time step length so the parameters
            won't be scaled.
        sigma_w: float
            Width of fitness curve.
        sigma_mov: float
            Width of dispersal curve.
        sigma_mut: float
            Width of mutation curve.
        mut_prob: float
            Probability of mutation occurring in offspring.
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
            'car_cap': car_cap,
            'sigma_w': sigma_w,
            'sigma_mov': sigma_mov,
            'sigma_mut': sigma_mut,
            'mut_prob': mut_prob,
            'random_seed': random_seed,
            'on_extinction': on_extinction,
            'always_direct_parent': always_direct_parent
        }

        self._set_direct_parent = True

    def _get_n_gen(self, dt):
        """
        Number of generations during one time step.

        Parameters
        ----------
        dt : float
            Time step duration.

        Returns
        -------
        float
            number of generations per time step.
        """

        if self._params['lifespan'] is None:
            return 1.
        else:
            return dt / self._params['lifespan']

    def _get_scaled_params(self, dt):
        """Scale sigma parameters according to the number of generations that
        succeed to each other during a time step.

        Parameters
        ----------
        dt : float
            Time step duration.

        Returns
        -------
        float
            scaled rates.

        """

        n_gen = self._get_n_gen(dt)

        sigma_w = self._params['sigma_w']
        sigma_d = self._params['sigma_mov'] * np.sqrt(n_gen)
        sigma_mut = self._params['sigma_mut'] * np.sqrt(n_gen)

        return sigma_w, sigma_d, sigma_mut

    def _count_neighbors(self, pop_points):
        """
        count number of neighbouring individual in a given radius.

        Parameters
        ----------
        pop_points : list of array
            location of individuals in a grid.

        Returns
        -------
        array-like
            number of neighbouring individual in a give radius.

        """
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
            r_d = self._params['car_cap'] / self._count_neighbors(pop_points)

            local_env = self._get_local_env_value(env_field, pop_points)
            opt_trait = self._optimal_trait_lin(env_field, local_env)
            #opt_trait = self._get_local_env_value(env_field, pop_points)

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

        Parameters
        ----------
        dt : float
            Time step duration.
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
            new_x, new_y = self.mov_within_bounds(new_population['x'],
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
        Initialization of speciation model with competition.

        Parameters
        ----------
        grid_x : array-like
            grid x-coordinate
        grid_y : array-like
            grid y-coordinate
        init_pop_size : integer
            initial number of individuals
        random_seed : integer
            seed used in random number generator
        lifespan : float, optional
            Reproductive lifespan of organism. If None (default), the
            lifespan will always match time step length
        birth_rate : integer or float
            birth rate of individuals
        movement_rate : integer of float
            movement/dispersion rate of individuals
        slope_topt_env : float
            slope of the relationship between optimum trait and environmental field
        car_cap_max : integer
            maximum carrying capacity
        sigma_opt_trait : float
            variability of carrying capacity
        mut_prob : float
            mutation probability
        sigma_mut : float
            variability of mutated trait
        sigma_mov : float
            variability of movement distance
        sigma_comp_trait : float
            variability of competition trait distance between individuals
        sigma_comp_dist : float
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

    def update(self, env_field):
        """
        Update of individuals' properties for a given environmental field.
        The computation is based on the Gillespie algorithm for a population
        of individuals that grows, move, and dies.

        Parameters
        ----------
        env_field : array-like
            Environmental field defined on the grid.

        """
        # Compute local individual environmental field
        local_env = self._get_local_env_value(env_field, np.column_stack([self._population['x'], self._population['y']]))
        # Compute optimal trait value
        opt_trait = self._optimal_trait_lin(env_field, local_env, slope=self._params['slope_topt_env'])

        # Compute event probabilities
        birth_i = self._population['trait'].size * [self._params['birth_rate']]
        death_i = death_rate(trait=self._population['trait'], x=self._population['x'], y=self._population['y'],
                             opt_trait=opt_trait,
                             xmin=self._grid_bounds['x'][0], xmax=self._grid_bounds['x'][1],
                             ymin=self._grid_bounds['y'][0], ymax=self._grid_bounds['y'][1],
                             car_cap_max=self._params['car_cap_max'],
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
        offspring['parent'] = self._population['id'][events_i == 'B']
        offspring['x'] = self._population['x'][events_i == 'B']
        offspring['y'] = self._population['y'][events_i == 'B']
        mu_prob = self._rng.uniform(0, 1, self._population['trait'][events_i == 'B'].size) < self._params['mut_prob']

        offspring['trait'] = np.where(mu_prob,
                                      self._rng.normal(self._population['trait'][events_i == 'B'],
                                                       self._params['sigma_mut']),
                                      self._population['trait'][events_i == 'B'])

        # Movement
        new_x, new_y = self.mov_within_bounds(self._population['x'][events_i == 'M'],
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
def death_rate(trait, x, y, opt_trait, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
               car_cap_max=500., sigma_opt_trait=0.3, sigma_comp_trait=0.9, sigma_comp_dist=0.19):
    """
    Logistic death rate for DD03 Speciation model
    implemented using numba

    Parameters
    ----------

    trait : 1d array, floats
        trait value per individual
    x : 1d array, floats
        location along the x coord
    y : 1d array, floats
        location along the y coord
    opt_trait : 1d array, floats
        optimal trait value
    ymax : value, float
        maximum value of y coordinate
    ymin : value, float
        minimum value of y coordinate
    xmax : value, float
        maximum value of y coordinate
    xmin : value, float
        minimum value of y coordinate
    car_cap_max : value, float
        maximum carrying capacity
    sigma_opt_trait : value, float
        variability of trait to abiotic condition
    sigma_comp_trait : value, float
        competition variability as a measure of the trait difference between individuals
    sigma_comp_dist : value, float
        competition variability as a measure of the spatial distance between individuals

    Returns
    -------
    1d array, floats
        death rate per individual
    """
    x = (x - xmin) / (xmax - xmin)
    y = (y - ymin) / (ymax - ymin)
    delta_trait = np.expand_dims(trait, 1) - trait
    delta_xy = np.sqrt((np.expand_dims(x, 1) - x) ** 2 + (np.expand_dims(y, 1) - y) ** 2)
    delta_trait_norm = np.exp(-0.5 * delta_trait ** 2 / sigma_comp_trait ** 2)
    delta_xy_norm = np.exp(-0.5 * delta_xy ** 2 / sigma_comp_dist ** 2)
    n_eff = 1 / (2 * np.pi * sigma_comp_dist ** 2) * np.sum(delta_trait_norm * delta_xy_norm, axis=1)
    k = car_cap_max * np.exp(-0.5 * (trait - opt_trait) ** 2 / sigma_opt_trait ** 2)
    return n_eff / k
