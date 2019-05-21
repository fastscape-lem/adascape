import textwrap

import numpy as np
import pandas as pd

import scipy.stats as stats
import scipy.spatial as spatial


class ParapatricSpeciationModel(object):
    """
    Model of speciation along an environmental gradient.

    This is adapted from Irwin (2012).

    Environmental factors are given on a grid.
    Population individuals are all generated within the bounds
    of this grid.

    """

    def __init__(self, grid_x, grid_y, init_pop_size, **kwargs):
        """Setup a new Parapatric Speciation Model.

        Parameters
        ----------
        grid_x : array-like
            Grid x-coordinates.
        grid_y : array_like
            Grid y-coordinates.
        init_pop_size : int
            Total number of indiviuals generated as the initial population.
        **kwargs
            nb_radius: float
                radius of window to obtain population size around an individual
            lifespan: int
                reproductive lifespan of organism, to scale with dt
            capacity: int
                capacity of population in window with radius (nb_radius)
            sigma_w: float
                width of fitness curve
            sigma_d: float
                width of dispersal curve
            sigma_mut: float
                width of mutation curve
            m_freq: float
                probability of mutation occurrring in offspring

        """
        grid_x = np.asarray(grid_x)
        grid_y = np.asarray(grid_y)
        self._grid_bounds = {'x': np.array([grid_x.min(), grid_x.max()]),
                             'y': np.array([grid_y.min(), grid_y.max()])}
        self._grid_index = self._build_grid_index([grid_x, grid_y])

        self._population = {}
        self._init_pop_size = init_pop_size

        # default parameter values
        self._params = {
            'nb_radius': 500,
            'lifespan': 1,
            'capacity': 1000,
            'sigma_w': 500,
            'sigma_d': 5,
            'sigma_mut': 500,
            'm_freq': 0.05,
            'random_seed': None
        }

        invalid_params = list(set(kwargs) - set(self._params))
        if invalid_params:
            raise KeyError("{} are not valid model parameters"
                           .format(", ".join(invalid_params)))

        self._params.update(kwargs)

        if isinstance(self._params['random_seed'], np.random.RandomState):
            self._random = self._params['random_seed']
        else:
            self._random = np.random.RandomState(self._params['random_seed'])

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

    def to_dataframe(self):
        """Return the population data at the current time step as a
        :class:`pandas.Dataframe`.

        """
        return pd.DataFrame(self._population)

    def _sample_in_range(self, range):
        return self._random.uniform(range[0], range[1], self._init_pop_size)

    def initialize_population(self, trait_range):
        """Initialize population data.

        The positions (x, y) of population individuals are generated
        uniformly within the grid bounds.

        Parameters
        ----------
        trait_range : tuple
            Range (min, max) within which initial trait values
            are uniformly sampled for the population individuals.

        """
        population = {}
        population['step'] = 0
        population['time'] = 0.
        population['id'] = np.arange(0, self._init_pop_size)
        population['parent'] = np.arange(0, self._init_pop_size)
        population['x'] = self._sample_in_range(self._grid_bounds['x'])
        population['y'] = self._sample_in_range(self._grid_bounds['y'])
        population['trait'] = self._sample_in_range(trait_range)

        self._population.update(population)

    def _get_scaled_params(self, dt):
        # Scale sigma parameters according to the number of generations that
        # succeed to each other during a time step.

        n_gen = dt / self._params['lifespan']

        sigma_w = self._params['sigma_w'] * np.sqrt(n_gen)
        sigma_d = self._params['sigma_d'] * np.sqrt(n_gen)
        sigma_mut = (self._params['sigma_mut'] * np.sqrt(n_gen)
                     * np.sqrt(self._params['m_freq']))

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

    def update_population(self, env_field, dt):
        """Update population data (generate offspring) during a time step,
        depending on the current population state and environmental factors.

        Parameters
        ----------
        env_field : array-like
            Environmental field defined on the grid.
        dt : float
            Time step duration.

        """
        sigma_w, sigma_d, sigma_mut = self._get_scaled_params(dt)

        pop_points = np.column_stack([self._population['x'],
                                      self._population['y']])

        # compute offspring sizes
        r_d = self._params['capacity'] / self._count_neighbors(pop_points)

        opt_trait = self._get_optimal_trait(env_field, pop_points)

        fitness = np.exp(-(self._population['trait'] - opt_trait)**2 /
                         (2 * sigma_w**2))

        n_offspring = np.round(r_d * fitness).astype('int')

        # no offspring? keep population unchanged
        if n_offspring.sum() == 0:
            return

        # generate offspring
        new_population = {k: np.repeat(self._population[k], n_offspring)
                          for k in ('x', 'y', 'trait')}

        new_population['parent'] = np.repeat(self._population['id'],
                                             n_offspring)

        last_id = self._population['id'][-1] + 1
        new_population['id'] = np.arange(last_id, last_id + n_offspring.sum())

        # mutate offspring
        new_population['trait'] = self._random.normal(new_population['trait'],
                                                      sigma_mut)

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

    def __repr__(self):
        class_str = type(self).__name__
        population_str = "population: {}".format(
            self.population_size or 'not initialized')
        params_str = "\n".join(["{}: {}".format(k, v)
                                for k, v in self._params.items()])

        return "<{} ({})>\nParameters:\n{}\n".format(
            class_str, population_str, textwrap.indent(params_str, '    '))
