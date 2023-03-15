#############################################################################
# Custom-made functions to compute and plot the solution of the eco-evo model
#############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adascape.base import IR12SpeciationModel
from adascape.fastscape_ext import FastscapeElevationTrait


def single_model_run(environment, x, y, num_gen=500, init_abundance=10, dt=1e0,
                     mut_prob=0.005, sigma_mut=0.05, sigma_env_fitness=0.2,
                     sigma_disp=30, sigma_comp_trait=1.0,
                     nb_radius=50, car_cap=50, random_seed=1234):
    """
    Function to execute a single run of the speciation model.

    Parameters
    ----------
    environment: array-like
                 environmental field
    x: array-like
       x coordinates
    y: array-like
       y coordiante
    num_gen: int
             maximum number of generations to compute.
    init_abundance: int
                    initial number of individuals.
    dt: float
        time step of the simulation
    sigma_env_fitness: float
                       environmental fitness variability controlling
                       the selection width around optimal trait value.
    mut_prob: float
              probability that a given ofspring will mutate or keep its ancestor trait value.
    sigma_mut: float
               trait variability of mutated offspring.
    sigma_disp: float
               dispersal variability of offspring in meters
    sigma_comp_trait: float
                      competition variability based on trait among individuals
    nb_radius: int or float
               Radius of the local neighbourhood centred at each individual.
    car_cap: int
             Carrying capacity of individuals in the local neighbourhood
    random_seed: int
                 seed of the random number generator

    Returns
    -------
    pandas.DataFrame with results of the run and where the columns are step, time, dt, x, y,
    taxon_id, ancestor_id, n_offspring, fitness, n_eff, n_all, trait_#
    """

    # we initialize one trait that we will assume is associated with elevation
    # where the initial min/max provides the range of trait values that the individual will have.
    # The parameter lin_slope determines the linear relationship between the environmental field
    # and the optimal trait value for each individual on that field.
    # The parameter *norm_min* and *norm_max* is the minimum and maximum values of the environmental field.
    trait = FastscapeElevationTrait(topo_elevation=environment,
                                    init_trait_min=0.5,
                                    init_trait_max=0.5,
                                    lin_slope=0.95,
                                    norm_min=environment.min(),
                                    norm_max=environment.max(),
                                    random_seed=1234)
    # initialization of traits
    trait.initialize()

    # dict of callables to generate initial values for each trait
    init_trait_funcs = {
        'trait': trait.init_trait_func
    }

    # dict of callables to compute optimal values for each trait
    opt_trait_funcs = {
        'trait': trait.opt_trait_func
    }

    # define a speciation model using the specified parameter values
    model = IR12SpeciationModel(grid_x=x, grid_y=y, init_trait_funcs=init_trait_funcs,
                                opt_trait_funcs=opt_trait_funcs, init_abundance=init_abundance,
                                nb_radius=nb_radius, car_cap=car_cap,
                                sigma_env_fitness=sigma_env_fitness, sigma_comp_trait=sigma_comp_trait,
                                sigma_disp=sigma_disp, sigma_mut=sigma_mut,
                                mut_prob=mut_prob, random_seed=random_seed)
    print(model)
    # initialize the speciation model
    model.initialize()
    # run the speciation model for the number of generations
    dfs = []
    for step in range(num_gen):
        # compute environmental fitness for each individual
        model.evaluate_fitness(dt)
        # append results to pandas.DataFrame
        dfs.append(model.to_dataframe())
        # mutate and disperse offspring
        model.update_individuals(dt)
    return pd.concat(dfs)


def plot_spatial_dist(df, environment):
        """
        Plot spatial distribution of individuals over the environmental field in selected time steps.

        Parameters
        ----------
        df: pd.DataFrame
             with results of the speciation model
        environment: array-like
                     environmental field
        """
        fig, axes = plt.subplots(nrows=2, ncols=5, sharex='col', sharey='row', figsize=(12, 6))
        for ax, print_step in zip(axes.ravel(), np.linspace(0, df.step.unique().size - 1, 10).astype(int)):
            pop = df.groupby('step').get_group(print_step)
            ax.pcolormesh(environment)
            ax.scatter(pop.x, pop.y, c=pop.trait_0, edgecolor='w', vmin=0, vmax=1)
            ax.set_title(f't = {print_step}, size = {len(pop)}')
        fig.tight_layout()


def plot_sol(dtf, environment, X):
    """
        Plot solution of the model result.

        Parameters
        ----------
        dtf: pd.DataFrame
             with results of the speciation model
        environment: array-like
                     environmental field
        X: array-like
           x coordinate
        """
    num_gen = dtf.step.unique().size
    fig1, axs1 = plt.subplots(2, 2, sharex="col", figsize=(12, 6))
    axs1[0, 0].plot(dtf.groupby('step').size())
    h, xedge, yedge, _ = axs1[1, 0].hist2d(x=dtf['step'], y=dtf['trait_0'],
                                           range=((0, num_gen), (0, 1)),
                                           bins=(num_gen, 100), cmap='bone_r')
    axs1[0, 1].pcolormesh(environment)
    axs1[0, 1].scatter(dtf['x'].loc[dtf['step'] == max(dtf['step'])],
                       dtf['y'].loc[dtf['step'] == max(dtf['step'])],
                       c=dtf['trait_0'].loc[dtf['step'] == max(dtf['step'])],
                       edgecolor='w', vmin=0, vmax=1)
    h, xedge, yedge, _ = axs1[1, 1].hist2d(x=dtf['x'].loc[dtf['step'] == max(dtf['step'])],
                                           y=dtf['trait_0'].loc[dtf['step'] == max(dtf['step'])],
                                           range=((0, X.max()), (0, 1)),
                                           bins=(25, 250), cmap='bone_r')
    axs1[0, 1].yaxis.set_label_position("right")
    axs1[0, 1].yaxis.tick_right()
    axs1[1, 1].yaxis.set_label_position("right")
    axs1[1, 1].yaxis.tick_right()
    axs1[0, 1].set_ylabel('Y', weight='bold')
    axs1[0, 0].set_ylabel('Abundance (No. ind)', weight='bold')
    axs1[1, 1].set_ylabel('Trait', weight='bold')
    axs1[1, 0].set_ylabel('Trait', weight='bold')
    axs1[1, 1].set_xlabel('X', weight='bold')
    axs1[1, 0].set_xlabel('Time (generations)', weight='bold')
