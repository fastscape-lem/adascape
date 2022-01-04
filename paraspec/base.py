import textwrap
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial as spatial


class SpeciationModelBase:
    """
    Speciation Model base class with common methods for the different
    types of speciation models.
    """

    def __init__(self, grid_x, grid_y, init_pop_size, slope_trait_env=0.95, lifespan=None,
                 random_seed=None, rescale_rates=True, always_direct_parent=True):
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
        slope_trait_env : float
            slope of the relationship between optimum trait and environmental field
        lifespan : float, optional
            Reproductive lifespan of organism. If None (default), the
            lifespan will always match time step length
        random_seed : int or :class:`numpy.random.default_rng` object
            Fixed random state for reproducible experiments.
            If None (default), results will differ from one run
            to another.
        rescale_rates : bool
            If True (default) rates and parameters will be rescaled as
            a fraction of the square root number of generations per
            time step.
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
        self._rng = np.random.default_rng(random_seed)

        # https://stackoverflow.com/questions/16016959/scipy-stats-seed
        self._truncnorm = stats.truncnorm
        self._truncnorm.random_state = self._rng

        self._params = {
            'slope_trait_env':  slope_trait_env,
            'lifespan': lifespan,
            'random_seed': random_seed,
            'always_direct_parent': always_direct_parent
        }
        self._env_field_bounds = None
        self._rescale_rates = rescale_rates
        self._set_direct_parent = True

    def initialize(self, trait_range, x_range=None, y_range=None):
        """
        Initialization of a group of individuals with randomly distributed traits,
        and which are randomly located in a two-dimensional grid.

        Parameters
        ----------
        trait_range : list of lists,
            with trait ranges of initial population
        x_range : tuple, optional
            Spatial range (min, max) to define initial spatial bounds
            of population in the x direction. Values must be contained
            within grid bounds. Default ('None') will initialize population
            within grid bounds in the x direction.
        y_range : tuple, optional
            Spatial range (min, max) to define initial spatial bounds
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

        if not all([isinstance(i, (list, tuple)) for i in trait_range]):
            raise ValueError("Range of trait values for each trait "
                             "must be provided as a list of lists, "
                             "where each sublist contains the minimum and "
                             "maximum value for each trait, e.g.:"
                             "[[trait1.min, trait1.max], [trait2.min, trait2.max] ... ]. "
                             "Instead got {!r}".format(trait_range))

        #if num_traits == 1:
        #    init_traits = self._sample_in_range(trait_range)
        #else:
        init_traits = np.zeros((self._init_pop_size, len(trait_range)))
        for i, tg in enumerate(trait_range):
            init_traits[:, i] = self._sample_in_range(tg)

        population = {'step': 0,
                      'time': 0.,
                      'dt': 0.,
                      'id': np.arange(0, self._init_pop_size),
                      'parent': np.arange(0, self._init_pop_size),
                      'x': self._sample_in_range(x_range),
                      'y': self._sample_in_range(y_range),
                      'trait': init_traits}
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
        """Individuals data at the current time step as a
        pandas Dataframe.

        Parameters
        ----------
        varnames : list or string, optional
            Only export those variable name(s) as dataframe column(s).
            Default: export all variables.

        Returns
        -------
        pandas.Dataframe
            Individuals data
        """

        individuals_data = self.population.copy()
        for i in range(self.population['trait'].shape[1]):
            individuals_data['trait_'+str(i)] = individuals_data['trait'][:, i]

        individuals_data.pop('trait')

        if varnames is None:
            data = individuals_data
        elif isinstance(varnames, str):
            data = {varnames: individuals_data[varnames]}
        else:
            data = {k: individuals_data[k] for k in varnames}
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

    def _optimal_trait_lin(self, env_field_min, env_field_max, local_env_val):
        """
        Normalized optimal trait value as a linear relationship
        with environmental field. Noticed that the local
        environmental field has been computed as a normalized
        local environmental field value based on the maximum and
        minimum values of the complete environmental field.

        Parameters
        ----------
        env_field_min : float
            Minimum value for the environmental field throughout simulation
        env_field_max : float
            Maximum value for the environmental field throughout simulation
         local_env_val : array-like
            local environmental field for each individual

        Returns
        -------
        array-like
            optimal trait values for each individual.
        """
        norm_loc_env_field = (local_env_val - env_field_min) / (env_field_max - env_field_min)
        opt_trait = ((self._params['slope_trait_env'] * (norm_loc_env_field - 0.5)) + 0.5)
        return opt_trait

    def _scaled_param(self, param, dt):
        """ Rescale a parameter as a fraction of the square root
        of the number of generations per time step.
        param : float
            parameter value.
        dt : float
            time step.
        """
        if self._params['lifespan'] is None:
            n_gen = 1.
        else:
            n_gen = dt / self._params['lifespan']

        return param/np.sqrt(n_gen)

    def __repr__(self):
        class_str = type(self).__name__
        population_str = "population: {}".format(
            self.population_size or 'not initialized')
        params_str = "\n".join(["{}: {}".format(k, v)
                                for k, v in self._params.items()])

        return "<{} ({})>\nParameters:\n{}\n".format(
            class_str, population_str, textwrap.indent(params_str, '    '))


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

    def __init__(self, grid_x, grid_y, init_pop_size, lifespan=None, random_seed=None, always_direct_parent=True,
                 slope_trait_env=0.95, nb_radius=500.,  car_cap=1000., sigma_w=500., sigma_mov=5., sigma_mut=500.,
                 mut_prob=0.05, on_extinction='warn'):
        """Initialization of speciation model without competition.

        Parameters
        ----------
        nb_radius: float
            Fixed radius of the circles that define the neighborhood
            around each individual.
        car_cap: int
            Carrying capacity of group of individuals within the neighborhood area.
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

        """
        super().__init__(grid_x, grid_y, init_pop_size, slope_trait_env, lifespan, random_seed, always_direct_parent)

        valid_on_extinction = ('warn', 'raise', 'ignore')

        if on_extinction not in valid_on_extinction:
            raise ValueError(
                "invalid value found for 'on_extinction' parameter. "
                "Found {!r}, must be one of {!r}"
                    .format(on_extinction, valid_on_extinction)
            )

        # default parameter values
        self._params.update({
            'nb_radius': nb_radius,
            'car_cap': car_cap,
            'sigma_w': sigma_w,
            'sigma_mov': sigma_mov,
            'sigma_mut': sigma_mut,
            'mut_prob': mut_prob,
            'on_extinction': on_extinction,
        })

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

    def evaluate_fitness(self, env_field, env_field_min, env_field_max, dt):
        """Evaluate fitness and generate offspring number for population and
        with environmental conditions both taken at the current time step.

        Parameters
        ----------
        env_field : array-like
            Environmental field defined on the grid.
        env_field_min : float
            Minimum value for the environmental field throughout simulation
        env_field_max : float
            Maximum value for the environmental field throughout simulation
        dt : float
            Time step duration.

        """
        # sigma_w, _, _ = self._get_scaled_params(dt)

        if self.population_size:
            pop_points = np.column_stack([self._population['x'],
                                          self._population['y']])

            # compute offspring sizes
            r_d = self._params['car_cap'] / self._count_neighbors(pop_points)

            local_env = self._get_local_env_value(env_field, pop_points)
            opt_trait = self._optimal_trait_lin(env_field_min, env_field_max, local_env)
            # opt_trait = self._get_local_env_value(env_field, pop_points)

            trait_fitness = []
            for i in range(self.population['trait'].shape[1]):
                delta_trait = self._population['trait'][:, i].flatten() - opt_trait
                trait_fitness.append(np.exp(-delta_trait ** 2 / (2 * self.params['sigma_w'] ** 2)))

            fitness = np.prod(trait_fitness, axis=0)
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
        """Update population data (generate, mutate, and disperse offspring).

        Parameters
        ----------
        dt : float
            Time step duration.
        """
        # _, sigma_mov, sigma_mut = self._get_scaled_params(dt)
        if self._rescale_rates:
            mut_prob = self._scaled_param(self._params['mut_prob'], dt)
            sigma_mov = self._scaled_param(self._params['sigma_mov'], dt)
            sigma_mut = self._scaled_param(self._params['sigma_mut'], dt)
        else:
            mut_prob = self._params['mut_prob']
            sigma_mov = self._params['sigma_mov']
            sigma_mut = self._params['sigma_mut']

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
                              for k in ('x', 'y')}
            new_population['trait'] = np.repeat(self._population['trait'], n_offspring, axis=0)

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
            to_mutate = self._rng.uniform(0, 1, new_population['trait'].shape[0]) < mut_prob
            for i in range(new_population['trait'].shape[1]):
                new_population['trait'][:, i] = np.where(to_mutate,
                                                   self._rng.normal(new_population['trait'][:, i], sigma_mut),
                                                   new_population['trait'][:, i])

            # disperse offspring within grid bounds
            new_x, new_y = self.mov_within_bounds(new_population['x'],
                                                  new_population['y'],
                                                  sigma_mov)
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


class DD03SpeciationModel(SpeciationModelBase):
    """
    Speciation model for asexual populations based on the model by:
        Doebeli, M., & Dieckmann, U. (2003).
        Speciation along environmental gradients.
        Nature, 421, 259–264.
        https://doi.org/10.1038/nature01312.Published.
    """

    def __init__(self, grid_x, grid_y, init_pop_size, lifespan=None, random_seed=None, always_direct_parent=True,
                 slope_trait_env=0.95, birth_rate=1, movement_rate=5, car_cap_max=500, sigma_opt_trait=0.3,
                 mut_prob=0.005, sigma_mut=0.05, sigma_mov=0.12, sigma_comp_trait=0.9, sigma_comp_dist=0.19):
        """
        Initialization of speciation model with competition.

        Parameters
        ----------
        birth_rate : integer or float
            birth rate of individuals
        movement_rate : integer of float
            movement/dispersion rate of individuals
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
        super().__init__(grid_x, grid_y, init_pop_size, slope_trait_env, lifespan, random_seed, always_direct_parent)
        self._params.update({
            'birth_rate': birth_rate,
            'movement_rate': movement_rate,
            'car_cap_max': car_cap_max,
            'sigma_opt_trait': sigma_opt_trait,
            'mut_prob': mut_prob,
            'sigma_mut': sigma_mut,
            'sigma_mov': sigma_mov,
            'sigma_comp_trait': sigma_comp_trait,
            'sigma_comp_dist': sigma_comp_dist,
        })

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

    def update(self, env_field, env_field_min, env_field_max, dt):
        """
        Update of individuals' properties for a given environmental field.
        The computation is based on the Gillespie algorithm for a group
        of individuals that grows, moves, and dies.

        Parameters
        ----------
        env_field : array-like
            Environmental field defined on the grid.
        env_field_min : float
            Minimum value for the environmental field throughout simulation
        env_field_max : float
            Maximum value for the environmental field throughout simulation
        dt : float
            Time step duration.

        """
        # rescale parameters
        if self._rescale_rates:
            mut_prob = self._scaled_param(self._params['mut_prob'], dt)
            sigma_mov = self._scaled_param(self._params['sigma_mov'], dt)
            sigma_mut = self._scaled_param(self._params['sigma_mut'], dt)

        else:
            mut_prob = self._params['mut_prob']
            sigma_mov = self._params['sigma_mov']
            sigma_mut = self._params['sigma_mut']

        # Compute local individual environmental field
        local_env = self._get_local_env_value(env_field,
                                              np.column_stack([self._population['x'], self._population['y']]))
        # Compute optimal trait value
        opt_trait = self._optimal_trait_lin(env_field_min, env_field_max, local_env)

        # Compute events probabilities
        birth_i = self._population['trait'].size * [self._params['birth_rate']]
        death_i = self.death_rate(opt_trait=opt_trait, dt=dt)
        movement_i = self._population['trait'].size * [self._params['movement_rate']]
        events_tot = np.sum(birth_i) + np.sum(death_i) + np.sum(movement_i)
        events_i = self._rng.choice(a=['B', 'D', 'M'], size=self._population['trait'].shape[0],
                                    p=[np.sum(birth_i) / events_tot, np.sum(death_i) / events_tot,
                                       np.sum(movement_i) / events_tot])
        # delta_t = self._rng.exponential(1 / events_tot, self._population['id'].size)

        # initialize temporary dictionaries
        offspring = {k: np.array([]) for k in ('id', 'parent', 'x', 'y')}
        extant = self._population.copy()
        # Birth
        offspring['id'] = np.arange(self._population['id'].max() + 1,
                                    self._population['id'][events_i == 'B'].size + self._population['id'].max() + 1)

        # set parents either to direct parents or older ancestors
        if self._set_direct_parent:
            parents = self._population['id'][events_i == 'B']
        else:
            parents = self._population['parent'][events_i == 'B']
        offspring['parent'] = parents
        offspring['x'] = self._population['x'][events_i == 'B']
        offspring['y'] = self._population['y'][events_i == 'B']

        to_mutate = self._rng.uniform(0, 1, self._population['trait'][events_i == 'B', :].size) < mut_prob
        offspring.update({'trait': np.empty([offspring['id'].size, extant['trait'].shape[1]])})
        for i in range(extant['trait'].shape[1]):
            offspring['trait'][:, i] = np.where(to_mutate,
                                      self._rng.normal(self._population['trait'][events_i == 'B', i], sigma_mut),
                                      self._population['trait'][events_i == 'B', i])

        # Movement
        new_x, new_y = self.mov_within_bounds(self._population['x'][events_i == 'M'],
                                              self._population['y'][events_i == 'M'],
                                              sigma_mov)
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
        extant['trait'] = extant['trait'][todie_ma, :]

        # Update dictionary
        self._population.update({k: np.append(extant[k], offspring[k]) for k in offspring.keys()})
        self._population['trait'] = np.expand_dims(self._population['trait'], 1)
        # reset the id number for the tree creation
        #self._population['id'] = np.arange(self._population['id'][-1]+1,
        #                                   self._population['id'][-1]+1+self._population['id'].size)
        self._population.update({'time': self._population['time'] + dt})
        self._population.update({'step': self._population['step'] + 1})
        self._population.update({'dt': dt})

    def death_rate(self, opt_trait, dt):
        """
        Logistic death rate

        Parameters
        ----------
        opt_trait : 1d array, floats
            optimal trait value
        Returns
        -------
        1d array, floats
            death rate per individual
        """

        # rescale parameters
        if self._rescale_rates:
            sigma_opt_trait = self._scaled_param(self._params['sigma_opt_trait'], dt)
            sigma_comp_trait = self._scaled_param(self._params['sigma_comp_trait'], dt)
            sigma_comp_dist = self._scaled_param(self._params['sigma_comp_dist'], dt)
        else:
            sigma_opt_trait = self._params['sigma_opt_trait']
            sigma_comp_trait = self._params['sigma_comp_trait']
            sigma_comp_dist = self._params['sigma_comp_dist']

        x = (self.population['x'] - self._grid_bounds['x'][0]) / (self._grid_bounds['x'][1] - self._grid_bounds['x'][0])
        y = (self.population['y'] - self._grid_bounds['y'][0]) / (self._grid_bounds['y'][1] - self._grid_bounds['y'][0])

        delta_trait = []
        for i in range(self.population['trait'].shape[1]):
            trait = self.population['trait'][:, i]
            delta_trait.append((np.expand_dims(trait, 1) - trait)**2)
        delta_trait = np.sqrt(np.sum(delta_trait, axis=0))
        delta_trait_norm = np.exp(-0.5 * delta_trait / sigma_comp_trait ** 2)

        delta_xy = np.sqrt((np.expand_dims(x, 1) - x) ** 2 + (np.expand_dims(y, 1) - y) ** 2)
        delta_xy_norm = np.exp(-0.5 * delta_xy ** 2 / sigma_comp_dist ** 2)
        n_eff = 1 / (2 * np.pi * sigma_comp_dist ** 2) * np.sum(delta_trait_norm * delta_xy_norm, axis=1)

        trait_fitness = []
        for i in range(self.population['trait'].shape[1]):
            delta_trait = self.population['trait'][:, i].flatten() - opt_trait
            trait_fitness.append(np.exp(-delta_trait ** 2 / (2 * sigma_opt_trait ** 2)))

        fitness = np.prod(trait_fitness, axis=0)

        k = self._params['car_cap_max'] * fitness

        return n_eff / k
