import numpy as np
import pandas as pd

import scipy.stats as stats
import scipy.spatial as spatial


class ParapatricSpeciationModel(object):
    """
    Model of ...

    This is adapted from Irwin (2012).

    Environmental factors are given on a grid, which is
    assumed static and rectangular in this implementation.
    Population individuals are all generated within the bounds
    of this grid.

    TODO: complete docstrings.

    """

    def __init__(self, grid_x, grid_y, init_pop_size, **kwargs):
        """Setup a new parapatric specification model.

        Parameters
        ----------
        grid_x : array-like
            Grid x-coordinates.
        grid_y : array_like
            Grid y-coordinates.
        init_pop_size : int
            Total number of indiviuals generated as the initial population.
        **kwargs
            Model parameters.

        TODO: document the list of model parameters
        """

        self.grid_x = grid_x
        self.grid_y = grid_y
        self._grid_extent = {'x': np.array([grid_x.min(), grid_x.max()]),
                             'y': np.array([grid_y.min(), grid_y.max()])}

        self.init_pop_size = init_pop_size

        # default parameter values
        self.params = {
            'nb_radius': 500,
            'lifespan': 1,
            'capacity': 1000,
            'sigma_w': 500,
            'sigma_d': 5,
            'sigma_mut': 500,
            'm_freq': 0.05
        }

        self.params.update(kwargs)
        self._grid_index = self._build_grid_index()

    def _build_grid_index(self):
        grid_points = np.column_stack([self.grid_x.ravel(),
                                       self.grid_y.ravel()])
        return spatial.cKDTree(grid_points)

    def initialize_population(self, trait_range):
        """Initialize population data.

        Inital trait value is generated randomly (uniform distribution
        bounded by ``trait_range``).

        """
        sample = lambda minmax: np.random.uniform(minmax[0], minmax[1],
                                                  self.init_pop_size)

        population = {}
        population['generation'] = 0
        population['x'] = sample(self._grid_extent['x'])
        population['y'] = sample(self._grid_extent['y'])
        population['trait'] = sample(trait_range)

        self.population = population

    def _get_scaled_params(self, dt):
        """Scale sigma parameters according to the number of generations that
        succeed to each other during a time step.

        """
        n_gen = dt / self.params['lifespan']

        sigma_w = self.params['sigma_w'] * np.sqrt(n_gen)
        sigma_d = self.params['sigma_d'] * np.sqrt(n_gen)
        sigma_mut = (self.params['sigma_mut'] * np.sqrt(n_gen)
                     * np.sqrt(self.params['m_freq']))

        return sigma_w, sigma_d, sigma_mut

    def _count_neighbors(self, pop_points):
        index = spatial.cKDTree(pop_points)
        neighbors = index.query_ball_tree(index, self.params['nb_radius'])

        return np.array([len(nb) for nb in neighbors])

    def _get_optimal_trait(self, env_field, pop_points):
        """Get the optimal trait value, i.e., an environmental factor
        given by a field defined on the static grid at the position
        of each individual (the chosen value is the one at the nearest
        grid node).

        """
        _, idx = self._grid_index.query(pop_points)

        return env_field.ravel()[idx]

    def update_population(self, env_field, dt):
        """Update population data (offsprings) during a time step,
        given environmental factors.

        """
        sigma_w, sigma_d, sigma_mut = self._get_scaled_params(dt)

        pop_points = np.column_stack([self.population['x'],
                                      self.population['y']])

        # compute offspring sizes
        r_d = self.params['capacity'] / self._count_neighbors(pop_points)

        opt_trait = self._get_optimal_trait(env_field, pop_points)

        fitness = np.exp(-(self.population['trait'] - opt_trait)**2 /
                         (2 * sigma_w**2))

        n_offspring = np.round(r_d * fitness).astype('int')

        # no offspring? keep population unchanged
        if n_offspring.sum() == 0:
            return

        # generate offspring
        new_population = {k : np.repeat(self.population[k], n_offspring)
                          for k in ('x', 'y', 'trait')}

        # mutate offspring
        new_population['trait'] = np.random.normal(new_population['trait'],
                                                   sigma_mut)

        # disperse offspring within grid bounds
        for k in ('x', 'y'):
            bounds = self._grid_extent[k][:, None] - new_population[k]

            new_k = stats.truncnorm.rvs(*(bounds / sigma_d),
                                        loc=new_population[k],
                                        scale=sigma_d)

            new_population[k] = new_k

        self.population['generation'] += 1
        self.population.update(new_population)

    def to_dataframe(self):
        """Return the population data at the current time step as a
        pandas.Dataframe.

        """
        return pd.DataFrame(self.population)
