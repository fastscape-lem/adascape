import textwrap
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial as spatial
import scipy.spatial.distance as dist
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.vq import kmeans2


class SpeciationModelBase:
    """
    Speciation Model base class with common methods for the different
    types of speciation models.
    """

    def __init__(self, grid_x, grid_y, init_trait_funcs, opt_trait_funcs, init_abundance,
                 lifespan=None, random_seed=None, rescale_rates=False, always_direct_parent=True,
                 on_extinction='warn', distance_metric='ward', distance_value=0.5, taxon_def='spec_clus'):
        """
        Initialization of based model.

        Parameters
        ----------
        grid_x : array-like
            Grid x-coordinates.
        grid_y : array_like
            Grid y-coordinates.
        init_trait_funcs : dict
            with callables to generate initial values of each trait for each individual.
        opt_trait_funcs : dict
            with callables to compute optimal trait values of each trait for each individual.
        init_abundance : int
            Total number of individuals generated as the initial population.
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
        on_extinction : {'warn', 'raise', 'ignore'}
            Behavior when no offspring is generated (total extinction of
            population) during model runtime. 'warn' (default) displays
            a RuntimeWarning, 'raise' raises a RuntimeError (the simulation
            stops) or 'ignore' silently continues the simulation
            doing nothing (no population).
        distance_metric : str, optional
            distance metric to calculate pairwise distance between individuals
            in n-dimensional trait space. For further details see documentation
            of method scipy.cluster.hierarchy.fclusterdata.
            default = 'ward'
        distance_value : float, optional
            distance threshold to construct clusters. For further details see documentation
            of method scipy.cluster.hierarchy.fclusterdata.
            default  = 0.5
        taxon_def: {'hier_clus', 'spec_clus'}
                   Method use to define a taxon based on individual's traits
                   and their shared common ancestry, using a hierarchical clustering 'hier_clus'
                   of a spectral clustering 'spec_clus', default 'spec_clus'
        """
        valid_on_extinction = ('warn', 'raise', 'ignore')

        if on_extinction not in valid_on_extinction:
            raise ValueError(
                "invalid value found for 'on_extinction' parameter. "
                "Found {!r}, must be one of {!r}"
                    .format(on_extinction, valid_on_extinction)
            )

        grid_x = np.asarray(grid_x)
        grid_y = np.asarray(grid_y)
        self._grid_bounds = {'x': np.array([grid_x.min(), grid_x.max()]),
                             'y': np.array([grid_y.min(), grid_y.max()])}
        self._grid_index = self._build_grid_index([grid_x, grid_y])
        self._individuals = {}
        self._init_abundance = init_abundance
        self._rng = np.random.default_rng(random_seed)

        # https://stackoverflow.com/questions/16016959/scipy-stats-seed
        self._truncnorm = stats.truncnorm
        self._truncnorm.random_state = self._rng

        self._params = {
            'lifespan': lifespan,
            'random_seed': random_seed,
            'always_direct_parent': always_direct_parent,
            'on_extinction': on_extinction,
            'distance_metric': distance_metric,
            'distance_value': distance_value,
            'taxon_def': taxon_def
        }
        self._env_field_bounds = None
        self._rescale_rates = rescale_rates
        self._set_direct_parent = True

        # dict of callables to generate initial values for each trait
        self._init_trait_funcs = init_trait_funcs
        # dict of callables to compute optimal values for each trait
        self._opt_trait_funcs = opt_trait_funcs

        # number of traits
        self.n_traits = len(self._init_trait_funcs)
        assert len(self._init_trait_funcs) == len(self._opt_trait_funcs)

        # for test of taxon definition
        valid_taxon_def = ('spec_clus', 'hier_clus')
        if taxon_def not in valid_taxon_def:
            raise ValueError(
                "invalid value found for 'valid_taxon_def' parameter. "
                "Found {!r}, must be one of {!r}"
                    .format(taxon_def, valid_taxon_def)
            )

    def initialize(self, x_range=None, y_range=None):
        """
        Initialization of a group of individuals with randomly distributed traits,
        and which are randomly located in a two-dimensional grid.

        Parameters
        ----------
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

        # array of shape (n_individuals, n_traits)
        init_traits = np.column_stack(
            [func(self._init_abundance) for func in self._init_trait_funcs.values()]
        )

        if self.params['taxon_def'] == 'hier_clus':
            clus = fclusterdata(init_traits,
                                method=self._params['distance_metric'],
                                t=self._params['distance_value'],
                                criterion='distance')
            taxon_id = clus.copy()
            ancestor_id = clus - 1

        elif self.params['taxon_def'] == 'spec_clus':
            clus = self._spect_clus(init_traits,
                                    taxon_treshold=self._params['distance_value'])
            taxon_id = clus + 1
            ancestor_id = clus.copy()

        population = {'step': 0,
                      'time': 0.,
                      'dt': 0.,
                      'x': self._sample_in_range(x_range),
                      'y': self._sample_in_range(y_range),
                      'trait': init_traits,
                      'taxon_id': taxon_id,
                      'ancestor_id': ancestor_id,
                      'n_offspring': np.zeros(init_traits.shape[0])
                      }
        self._individuals.update(population)

    def _compute_taxon_ids(self):
        """
        Method to define taxa based on individual's traits and their shared common ancestry
        using 1) a hierarchical clustering algorithm from scipy.spatial.hierarchy or
        2) spectral clustering algorithm. For hierarchical clustering a distance value
        and distance metric needs to be specified,the latter by default is set to 'ward' distance.
        For spectral clustering only distance value needs to be specified.

        Returns
        -------
        taxon_ids based on the clustering of individuals with similar trait values
        and common ancestry.
        """
        if self._set_direct_parent:
            new_id_key = 'taxon_id'
        else:
            new_id_key = 'ancestor_id'
        current_ancestor_id = np.repeat(self._individuals[new_id_key], self._individuals['n_offspring'].astype('int'))
        max_clus = self._individuals['taxon_id'].max()

        if self.params['taxon_def'] == 'hier_clus':
            clus_dat = np.column_stack([self._individuals['trait'], current_ancestor_id/current_ancestor_id.max()])
            clus = fclusterdata(clus_dat,
                                method=self._params['distance_metric'],
                                t=self._params['distance_value'],
                                criterion='distance')
            new_taxon_id = clus + max_clus
        elif self.params['taxon_def'] == 'spec_clus':
            new_taxon_id = np.zeros_like(current_ancestor_id)
            for ans in np.unique(current_ancestor_id):
                ans_indx = np.where(current_ancestor_id==ans)[0]
                clus = self._spect_clus(self._individuals['trait'][ans_indx],
                                        taxon_treshold=self._params['distance_value'])
                if max_clus < np.max(clus):
                    new_clus = clus + 1
                    new_taxon_id[ans_indx] = new_clus.astype(int)
                elif max_clus >= np.max(clus):
                    new_clus = clus + 1 + max_clus
                    new_taxon_id[ans_indx] = new_clus.astype(int)
                max_clus = np.max(new_clus).astype(int)

        else:
            new_taxon_id = np.empty_like(current_ancestor_id)
        return new_taxon_id, current_ancestor_id

    def _spect_clus(self, clus_data, taxon_treshold=0.1, split_size=10):
        """
        Spectral clustering algorithm based on von Luxburg (2007), which we modified to:
            1) only divide groups of individuals larger than "split_size",
            2) if division occurs only divide into a maximum of two clusters or  taxon_ids,
            3) the minimum size of the divided clusters or taxon_ids must be half of split_size.

        Ulrike von Luxburg (2007) A tutorial on spectral clustering.Statistics and Computing,
        17(4):395-416. doi:10.1007/s11222-007-9033-z

        Parameters
        ----------
        clus_data : array-like
                    individual's trait data to be used in the clustering
        split_size : int
                    minimum number of individuals (observations) in cluster data to perform the clustering algorithm

        Returns
        -------
        array-like of int
            with taxon id

        """
        if clus_data.shape[0] > split_size:
            try:
                D2Mat = dist.squareform(dist.pdist(clus_data))
            except:
                D2Mat = dist.squareform(dist.pdist(clus_data[:, np.newaxis]))

            sigma = np.std(D2Mat.flatten())
            mean = np.mean(D2Mat.flatten())
            if sigma <= 0:
                W = np.exp(-D2Mat)  # Exponential similarity function
            else:
                W = np.exp(-(D2Mat - mean) ** 2 / (2 * sigma ** 2))  # Gaussian similarity function
            W[D2Mat > taxon_treshold] = 0  # rule to connect individuals as well as ancestor

            D = np.diag(np.sum(W, axis=1))
            L = D - W
            E, U = np.linalg.eig(L)
            E = np.real(E)  # remove tiny imaginary numbers
            U = np.real(U)  # remove tiny imaginary numbers
            E[np.isclose(E, np.zeros(len(E)))] = 0  # remove tiny numbers that are too close to zeros
            n_comp = np.sum(E == 0)
            if n_comp > 1:
                k = min(2, n_comp)
                centroid, labels = kmeans2(U[:, :k], k=k, minit='points', seed=self._rng)
                count1 = np.sum(labels == 0)
                count2 = np.sum(labels == 1)
                if count1 < split_size//2:
                    labels[labels == 0] = 1
                if count2 < split_size//2:
                    labels[labels == 1] = 0
                if np.all(labels.astype(int) == 1):
                    labels = np.zeros(clus_data.shape[0])
            else:
                labels = np.zeros(clus_data.shape[0])
        else:
            labels = np.zeros(clus_data.shape[0])
        return labels

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
    def individuals(self):
        """Individuals' data at the current time step.

        Returns
        -------
        dict
            Individuals data
        """
        self._set_direct_parent = True
        return self._individuals

    @property
    def abundance(self):
        """Number of individuals at the current time
        step.

        Returns
        -------
        int or None
            Number of individuals
            (return None if a group of individuals has not yet been initialized)
        """
        if not self._individuals:
            return None
        else:
            return self._individuals['trait'][:, 0].size

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

        individuals_data = self._individuals.copy()
        for i in range(self._individuals['trait'].shape[1]):
            individuals_data['trait_' + str(i)] = individuals_data['trait'][:, i]

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
        return self._rng.uniform(values_range[0], values_range[1], self._init_abundance)

    def _mov_within_bounds(self, x, y, sigma):
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

    def _mutate_trait(self, trait, sigma):
        """
        Mutate individual trait values within a range between 0 and 1

        Parameters
        ----------
        trait: array_like
               trait values
        sigma: float
               trait variability
        Returns
        -------
        array-like
            Mutate trait values
        """
        a, b = (0 - trait) / sigma, (1 - trait) / sigma
        mut_trait = self._truncnorm.rvs(a, b, loc=trait, scale=sigma)
        return mut_trait

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

        return param / np.sqrt(n_gen)

    def _update_individuals(self, dt):
        """Require implementation in subclasses."""
        raise NotImplementedError()

    def update_individuals(self, dt):
        """Update individuals' data (generate, mutate, and disperse).

        Parameters
        ----------
        dt : float
            Time step duration.

        """
        self._update_individuals(dt)

        if not self._params['always_direct_parent']:
            self._set_direct_parent = False

    def __repr__(self):
        class_str = type(self).__name__
        population_str = "individuals: {}".format(
            self.abundance or 'not initialized')
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

    def __init__(self, grid_x, grid_y, init_trait_funcs, opt_trait_funcs, init_abundance,
                 lifespan=None, random_seed=None, always_direct_parent=True,
                 on_extinction='warn', distance_metric='ward', distance_value=0.5,
                 nb_radius=500., car_cap=1000., sigma_env_trait=0.3, sigma_mov=5.,
                 sigma_mut=0.05, mut_prob=0.05, taxon_def='spec_clus'):
        """Initialization of speciation model without competition.

        Parameters
        ----------
        nb_radius: float
            Fixed radius of the circles that define the neighborhood
            around each individual.
        car_cap: int
            Carrying capacity of group of individuals within the neighborhood area.
        sigma_env_trait: float
            Width of fitness curve.
        sigma_mov: float
            Width of dispersal curve.
        sigma_mut: float
            Width of mutation curve.
        mut_prob: float
            Probability of mutation occurring in offspring.
        """
        super().__init__(grid_x=grid_x, grid_y=grid_y,
                         init_trait_funcs=init_trait_funcs,
                         opt_trait_funcs=opt_trait_funcs,
                         init_abundance=init_abundance,
                         lifespan=lifespan,
                         random_seed=random_seed,
                         always_direct_parent=always_direct_parent,
                         on_extinction=on_extinction,
                         distance_metric=distance_metric,
                         distance_value=distance_value,
                         taxon_def=taxon_def)

        # default parameter values
        self._params.update({
            'nb_radius': nb_radius,
            'car_cap': car_cap,
            'sigma_env_trait': sigma_env_trait,
            'sigma_mov': sigma_mov,
            'sigma_mut': sigma_mut,
            'mut_prob': mut_prob,
            'always_direct_parent': always_direct_parent,
            'on_extinction': on_extinction,
            'distance_metric': distance_metric,
            'distance_value': distance_value
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

    def evaluate_fitness(self, dt):
        """Evaluate fitness and generate offspring number for group of individuals and
        with environmental conditions both taken at the current time step.

        Parameters
        ----------
        dt : float
            Time step duration.

        """

        if self.abundance:
            individual_positions = np.column_stack([self._individuals['x'], self._individuals['y']])
            _, grid_positions = self._grid_index.query(individual_positions)

            # array of shape (n_individuals, n_traits)
            opt_trait = np.column_stack(
                [func(grid_positions) for func in self._opt_trait_funcs.values()]
            )

            # compute offspring sizes
            r_d = self._params['car_cap'] / self._count_neighbors(individual_positions)

            trait_fitness = []
            for i in range(self._individuals['trait'].shape[1]):
                delta_trait = self._individuals['trait'][:, i].flatten() - opt_trait[:, i]
                trait_fitness.append(np.exp(-delta_trait ** 2 / (2 * self.params['sigma_env_trait'] ** 2)))

            fitness = np.prod(trait_fitness, axis=0)
            n_gen = self._get_n_gen(dt)
            n_offspring = np.round(r_d * fitness * np.sqrt(n_gen)).astype('int')

        else:
            fitness = np.zeros(self._individuals['trait'].shape[1])
            n_offspring = np.zeros(self._individuals['trait'].shape[1])

        self._individuals.update({
            'fitness': fitness,
            'n_offspring': n_offspring
        })

    def _update_individuals(self, dt):
        """Update individuals' data (generate, mutate, and disperse).

        Parameters
        ----------
        dt : float
            Time step duration.
        """
        if self._rescale_rates:
            mut_prob = self._scaled_param(self._params['mut_prob'], dt)
            sigma_mov = self._scaled_param(self._params['sigma_mov'], dt)
            sigma_mut = self._scaled_param(self._params['sigma_mut'], dt)
        else:
            mut_prob = self._params['mut_prob']
            sigma_mov = self._params['sigma_mov']
            sigma_mut = self._params['sigma_mut']

        n_offspring = self._individuals['n_offspring']

        if not n_offspring.sum():
            # population total extinction
            if self._params['on_extinction'] == 'raise':
                raise RuntimeError("no offspring generated. "
                                   "Model execution has stopped.")

            if self._params['on_extinction'] == 'warn':
                warnings.warn("no offspring generated. "
                              "Model execution continues with no population.",
                              RuntimeWarning)

            new_individuals = {k: np.array([])
                               for k in ('x', 'y', 'trait')}
            new_individuals['trait'] = np.expand_dims(new_individuals['trait'], 1)
        else:
            # generate offspring
            new_individuals = {k: np.repeat(self._individuals[k], n_offspring)
                               for k in ('x', 'y')}
            new_individuals['trait'] = np.repeat(self._individuals['trait'], n_offspring, axis=0)

            # mutate offspring
            to_mutate = self._rng.uniform(0, 1, new_individuals['trait'].shape[0]) < mut_prob
            for i in range(new_individuals['trait'].shape[1]):
                new_individuals['trait'][:, i] = np.where(to_mutate,
                                                          self._mutate_trait(new_individuals['trait'][:, i], sigma_mut),
                                                          new_individuals['trait'][:, i])

            # disperse offspring within grid bounds
            new_x, new_y = self._mov_within_bounds(new_individuals['x'],
                                                   new_individuals['y'],
                                                   sigma_mov)
            new_individuals['x'] = new_x
            new_individuals['y'] = new_y

        self._individuals['step'] += 1
        self._individuals['time'] += dt
        self._individuals.update(new_individuals)
        if not n_offspring.sum():
            taxon_id, ancestor_id = np.array([]), np.array([])
        else:
            taxon_id, ancestor_id = self._compute_taxon_ids()
        self._individuals.update({'taxon_id': taxon_id})
        self._individuals.update({'ancestor_id': ancestor_id})

        # reset fitness / offspring data
        self._individuals.update({
            'fitness': np.zeros(self._individuals['trait'].shape[0]),
            'n_offspring': np.zeros(self._individuals['trait'].shape[0])
        })


class DD03SpeciationModel(SpeciationModelBase):
    """
    Speciation model for asexual populations adapted from:
        Doebeli, M., & Dieckmann, U. (2003).
        Speciation along environmental gradients.
        Nature, 421, 259–264.
        https://doi.org/10.1038/nature01312.Published.
    """

    def __init__(self, grid_x, grid_y, init_trait_funcs, opt_trait_funcs, init_abundance,
                 lifespan=None, random_seed=None, always_direct_parent=True,
                 on_extinction='warn', distance_metric='ward', distance_value=0.5,
                 birth_rate=1, movement_rate=5, car_cap_max=500, sigma_env_trait=0.3,
                 mut_prob=0.005, sigma_mut=0.05, sigma_mov=0.12, sigma_comp_trait=0.9,
                 sigma_comp_dist=0.19, taxon_def='spec_clus'):
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
        sigma_env_trait : float
            variability of trait-environment relationship
        mut_prob : float
            mutation probability
        sigma_mut : float
            variability of mutated trait
        sigma_mov : float
            variability of movement distance
        sigma_comp_trait : float
            competition variability for trait distance between individuals
        sigma_comp_dist : float
            competition variability for spatial distance between individuals
        """

        super().__init__(grid_x=grid_x, grid_y=grid_y,
                         init_trait_funcs=init_trait_funcs,
                         opt_trait_funcs=opt_trait_funcs,
                         init_abundance=init_abundance,
                         lifespan=lifespan,
                         random_seed=random_seed,
                         always_direct_parent=always_direct_parent,
                         on_extinction=on_extinction,
                         distance_metric=distance_metric,
                         distance_value=distance_value,
                         taxon_def=taxon_def)

        self._params.update({
            'birth_rate': birth_rate,
            'movement_rate': movement_rate,
            'car_cap_max': car_cap_max,
            'sigma_env_trait': sigma_env_trait,
            'mut_prob': mut_prob,
            'sigma_mut': sigma_mut,
            'sigma_mov': sigma_mov,
            'sigma_comp_trait': sigma_comp_trait,
            'sigma_comp_dist': sigma_comp_dist,
            'always_direct_parent': always_direct_parent,
            'on_extinction': on_extinction,
            'distance_metric': distance_metric,
            'distance_value': distance_value
        })

    def evaluate_fitness(self, dt):
        """
        Evaluate fitness of individuals' in a given environmental field.
        The computation is based on the Gillespie algorithm for a group
        of individuals that randomly grows, moves, and dies.

        Parameters
        ----------
        dt : float
            Time step duration.

        """
        if self.abundance:
            individual_positions = np.column_stack([self._individuals['x'], self._individuals['y']])
            _, grid_positions = self._grid_index.query(individual_positions)

            # array of shape (n_individuals, n_traits)
            opt_trait = np.column_stack(
                [func(grid_positions) for func in self._opt_trait_funcs.values()]
            )

            # Compute events probabilities
            net_gain, death_i = self.logistic_growth_comp(opt_trait=opt_trait, dt=dt)
            birth_i = np.array(self.abundance * [self._params['birth_rate']]) + net_gain
            movement_i = self.abundance * [self._params['movement_rate']]
            events_tot = np.sum(birth_i) + np.sum(death_i) + np.sum(movement_i)
            events_i = self._rng.choice(a=['B', 'D', 'M'], size=self._individuals['trait'].shape[0],
                                        p=[np.sum(birth_i) / events_tot, np.sum(death_i) / events_tot,
                                           np.sum(movement_i) / events_tot])

            n_offspring = np.where(events_i == 'B', 2, np.where(events_i == 'M', 1, 0))

        else:
            events_i = np.zeros(self._individuals['trait'].shape[1])
            death_i = np.zeros(self._individuals['trait'].shape[1])
            n_offspring = np.zeros(self._individuals['trait'].shape[1])

        self._individuals.update({'events_i': events_i,
                                  'death_i': death_i,
                                  'n_offspring': n_offspring
                                  })

    def _update_individuals(self, dt):
        """
        Update individuals' data (birth, death, and move).
        Parameters
        ----------
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

        n_offspring = self._individuals['n_offspring']
        if not n_offspring.sum():
            # population total extinction
            if self._params['on_extinction'] == 'raise':
                raise RuntimeError("no offspring generated. "
                                   "Model execution has stopped.")

            if self._params['on_extinction'] == 'warn':
                warnings.warn("no offspring generated. "
                              "Model execution continues with no population.",
                              RuntimeWarning)

            new_individuals = {k: np.array([])
                               for k in ('x', 'y', 'trait')}
            new_individuals['trait'] = np.expand_dims(new_individuals['trait'], 1)
        else:
            events_i = self._individuals['events_i']
            death_i = self._individuals['death_i']

            # initialize temporary dictionaries
            offspring = {k: np.array([]) for k in ('x', 'y')}
            extant = self._individuals.copy()

            # Birth
            offspring['x'] = self._individuals['x'][events_i == 'B']
            offspring['y'] = self._individuals['y'][events_i == 'B']

            to_mutate = self._rng.uniform(0, 1, self._individuals['trait'][events_i == 'B', :].shape[0]) < mut_prob
            offspring.update({'trait': np.empty([offspring['x'].size, extant['trait'].shape[1]])})
            for i in range(extant['trait'].shape[1]):
                offspring['trait'][:, i] = np.where(to_mutate,
                                                    self._mutate_trait(self._individuals['trait'][events_i == 'B', i],
                                                                       sigma_mut),
                                                    self._individuals['trait'][events_i == 'B', i])

            # Movement
            new_x, new_y = self._mov_within_bounds(self._individuals['x'][events_i == 'M'],
                                                   self._individuals['y'][events_i == 'M'],
                                                   sigma_mov)
            extant['x'][events_i == 'M'] = new_x
            extant['y'][events_i == 'M'] = new_y

            # Death
            ids = np.arange(extant['x'].size)
            todie = self._rng.choice(ids, size=self._individuals['x'][events_i == 'D'].size,
                                     p=death_i / death_i.sum(), replace=False)
            todie_ma = np.logical_not(np.any(ids == todie.repeat(ids.size).reshape(todie.size, ids.size), axis=0))
            extant['x'] = extant['x'][todie_ma]
            extant['y'] = extant['y'][todie_ma]
            extant['trait'] = extant['trait'][todie_ma, :]
            new_individuals = {k: np.append(extant[k], offspring[k]) for k in offspring.keys()}
            new_individuals['trait'] = new_individuals['trait'].reshape(
                [extant['trait'].shape[0] + offspring['trait'].shape[0], extant['trait'].shape[1]])

        # Update dictionary
        self._individuals.update(new_individuals)
        self._individuals.update({'time': self._individuals['time'] + dt})
        self._individuals.update({'step': self._individuals['step'] + 1})
        self._individuals.update({'dt': dt})
        if not n_offspring.sum():
            taxon_id, ancestor_id = np.array([]), np.array([])
        else:
            taxon_id, ancestor_id = self._compute_taxon_ids()
        self._individuals.update({'taxon_id': taxon_id})
        self._individuals.update({'ancestor_id': ancestor_id})

        # reset offspring data
        self._individuals.update({
            'n_offspring': np.zeros(self._individuals['trait'].shape[0])
        })

    def logistic_growth_comp(self, opt_trait, dt):
        """
        Logistic growth components

        N = (1 - N/K_max) (1 - N_eff/K_eff)

        Parameters
        ----------
        opt_trait : 2d array, floats
            optimal trait value
        Returns
        -------
        1d array, floats
            death rate per individual
        """

        # rescale parameters
        if self._rescale_rates:
            sigma_env_trait = self._scaled_param(self._params['sigma_env_trait'], dt)
            sigma_comp_trait = self._scaled_param(self._params['sigma_comp_trait'], dt)
            sigma_comp_dist = self._scaled_param(self._params['sigma_comp_dist'], dt)
        else:
            sigma_env_trait = self._params['sigma_env_trait']
            sigma_comp_trait = self._params['sigma_comp_trait']
            sigma_comp_dist = self._params['sigma_comp_dist']

        # normalize spatial dimensions
        x = (self._individuals['x'] - self._grid_bounds['x'][0]) / (
                self._grid_bounds['x'][1] - self._grid_bounds['x'][0])
        y = (self._individuals['y'] - self._grid_bounds['y'][0]) / (
                self._grid_bounds['y'][1] - self._grid_bounds['y'][0])

        # trait distance among individuals
        delta_comp_trait = dist.squareform(dist.pdist(self._individuals['trait']))
        delta_trait_norm = np.exp(-0.5 * delta_comp_trait ** 2 / sigma_comp_trait ** 2)
        # spatial distance among individuals
        delta_xy = dist.squareform(dist.pdist(np.column_stack([x, y])))
        delta_xy_norm = np.exp(-0.5 * delta_xy ** 2 / sigma_comp_dist ** 2)
        # number of individual with similar traits and in proximity to each other
        n_eff = np.sum(delta_trait_norm * delta_xy_norm, axis=1)
        # environmental fitness to local environmental fields
        delta_env_trait = np.exp(-0.5 * (self._individuals['trait'] - opt_trait) ** 2 / sigma_env_trait ** 2)
        env_fitness = np.prod(delta_env_trait, axis=1)
        # global carrying capacity
        a = self.abundance / self._params['car_cap_max']
        # local carrying capacity
        k_eff = self._params['car_cap_max'] * env_fitness
        b = n_eff / k_eff
        net_gain = b * a
        net_loss = b + a
        return net_gain, net_loss
