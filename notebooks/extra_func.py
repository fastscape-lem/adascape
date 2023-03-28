#############################################################################
# Custom-made functions to compute and plot the solution of the eco-evo model
#############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toytree
import toyplot
import scipy.stats as sc

from adascape.base import IR12SpeciationModel
from adascape.fastscape_ext import FastscapeElevationTrait


def single_model_run(environment, x, y, num_gen=500, init_abundance=10, dt=1e0,
                     p_m=0.005, sigma_m=0.05, sigma_f=0.2,
                     sigma_d=30, sigma_u=1.0, r=50, K=50,
                     random_seed=1234, taxon_threshold=0.075):
    """
    Function to execute a single run of the speciation model.

    Parameters
    ----------
    environment: array-like
                 environmental field
    x: array-like
       x coordinates
    y: array-like
       y coordinate
    num_gen: int
             maximum number of generations to compute.
    init_abundance: int
                    initial number of individuals.
    dt: float
        time step of the simulation
    sigma_f: float
                       environmental fitness variability controlling
                       the selection width around optimal trait value.
    p_m: float
              probability that a given ofspring will mutate or keep its ancestor trait value.
    sigma_m: float
               trait variability of mutated offspring.
    sigma_d: float
               dispersal variability of offspring in meters
    sigma_u: float
                      competition variability based on trait among individuals
    r: int or float
               Radius of the local neighbourhood centred at each individual.
    K: int
             Carrying capacity of individuals in the local neighbourhood
    random_seed: int
                 seed of the random number generator
    taxon_threshold: float
            threshold value to split a group of organisms into two taxon clusters

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
                                r=r, K=K, taxon_threshold=taxon_threshold,
                                sigma_f=sigma_f, sigma_u=sigma_u,
                                sigma_d=sigma_d, sigma_m=sigma_m, p_m=p_m,
                                random_seed=random_seed)
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


def plot_spatial_dist_ind(df, environment):
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


def plot_temp_dyn_1trait(dtf, environment, X):
    """
    Function to plot the abundance and trait distribution through time and along the X coordinate.

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


def toytree_plot(tree, ind_dtf):
    """
    Plot of phylogenetic tree using library toytree and following the cookbook example
    https://toytree.readthedocs.io/en/latest/

    Parameters
    ----------
    tree : class 'dendropy.tree'
        Phylogenetic tree as class dendropy.tree
    ind_dtf : class 'pandas.DataFrame'
        Eco-evo model output with data for all individuals

    """
    # convert dendropy-tree to toytree format
    ttree = toytree.tree(tree.as_string(schema='newick'))
    # generate a distribution between 0 and 1 for each tip in the tree
    points = np.linspace(0, 1, 50)
    dists = {}
    for tip in ttree.get_tip_labels():
        trait_values = ind_dtf[ind_dtf.taxon_id == int(tip)].trait_0.values
        if trait_values.size > 1 and trait_values.std() / trait_values.mean() > 1e-10:
            kernel = sc.gaussian_kde(trait_values)
            dists[tip] = kernel(points)
        else:
            dists[tip] = None
    # set up canvas for two panel plot
    canvas = toyplot.Canvas(width=300, height=400)

    # add tree to canvas
    ax0 = canvas.cartesian(bounds=(50, 180, 50, 350), ymin=0, ymax=ttree.ntips, padding=15)
    ttree.draw(axes=ax0, tip_labels=False)
    ax0.show = False

    # add histograms to canvas
    ax1 = canvas.cartesian(bounds=(200, 275, 50, 350), ymin=0, ymax=ttree.ntips, padding=15)

    # iterate from top to bottom (ntips to 0)
    for tip in range(ttree.ntips)[::-1]:

        # select a color for hist
        color = toytree.colors[int((tip) / 100)]

        # get tip name and get hist from dict
        tipname = ttree.get_tip_labels()[tip]
        probs = dists[tipname]

        if probs is not None:
            # fill histogram with slightly overlapping histograms
            ax1.fill(
                points, probs / probs.max() * 1.25,
                baseline=[tip] * len(points),
                style={"fill": color, "stroke": "white", "stroke-width": 0.5},
                title=tipname,
            )
            # add horizontal line at base
            ax1.hlines(tip, opacity=0.5, color="grey", style={"stroke-width": 0.5})

    # hide y axis, show x
    ax1.y.show = False
    ax1.x.label.text = "Trait value"
    ax1.x.ticks.show = True


def get_dataframe(ds, out_vars=['life__taxon_id', 'life__ancestor_id',
                                'life__trait_elev', 'life__trait_prep', 'life__y', 'life__x']):
    """
    Function to convert xarray.Dataset with results of a couple eco-evolutionary model with FastScape LEM

    Parameters
    ----------
    ds: xr.Dataset
             with results of the eco-evo model
    out_vars: list-like
              with variable to be extracted. The variable names must be equal as they appear in ds
    Returns
    -------
    pandas.DataFrame with columns out (time step), taxon_id, ancestor_id, trait associated with elevation (trait_elev)
    trait associated with precipitation (trait_prep), x and y location of individuals
    """
    individuals_data = {}
    for i in range(ds.life__traits.shape[2]):
        individuals_data['life__' + str(ds.trait[i].values.astype(str))] = ds.life__traits[:, :, i]
    ds = ds.assign(individuals_data)
    out_ds = ds[out_vars]

    dtf = (
        out_ds
            .to_dataframe()
            .rename(columns=lambda name: name.replace('life__', ''))
            .reset_index()
            .dropna()
            .drop('ind', axis=1)
    )
    return dtf


def plot_topo_taxa(ds, dtf, time_sel):
    """
    Function to plot the spatial distribution of taxa, where the taxa are depicted with different markers and
    colors.

    Parameters
    ----------
     ds: xr.Dataset
             with results of the eco-evo model
     dtf: pd.DataFrame
         with results of the eco-evo model
     time_sel: array-like
               with time steps to plot the results
    """
    mkrs = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8',
            's', 'p', '*', '+', 'h', 'H', 'D', 'd', 'P', 'X',
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    fig1 = (ds
            .sel(out=time_sel)
            .topography__elevation.plot(col='out', col_wrap=2, figsize=(8, 8), cmap='bone')
            )

    for ax1, t in zip(fig1.axes.ravel(), time_sel):
        pop = dtf[dtf.out == t].groupby('taxon_id')
        max_no_grp = max(list(pop.groups.keys()))
        for k, v in pop:
            ax1.scatter(v.x, v.y, marker=mkrs[int(max_no_grp - k)], s=20)


def plot_temp_dyn_2traits(dtf):
    """
    Function to plot the abundance, richness and trait distribution through time and along the Y coordinate.

    Parameters
    ----------
    dtf: pd.DataFrame
         with results of the speciation model
    """
    fig2, axs2 = plt.subplots(3, 3, sharex="col", figsize=(12, 6))
    gs2 = axs2[1, 1].get_gridspec()
    for ax in axs2[0:, 1:].flatten():
        ax.remove()
    axbig0 = fig2.add_subplot(gs2[0:, -2])
    axbig1 = fig2.add_subplot(gs2[0:, -1])
    axs2[0, 0].plot(dtf.groupby('out').size())
    axs2_2 = axs2[0, 0].twinx()
    axs2_2.plot(dtf.groupby(['out']).apply(lambda x: x.taxon_id.unique().size), c='red', alpha=0.75)
    h, xedge, yedge, _ = axs2[1, 0].hist2d(x=dtf['out'], y=dtf['trait_elev'],
                                           range=((0, 1e6), (0, 1)),
                                           bins=(100, 100), cmap='bone_r')
    h, xedge, yedge, _ = axs2[2, 0].hist2d(x=dtf['out'], y=dtf['trait_prep'],
                                           range=((0, 1e6), (0, 1)),
                                           bins=(100, 100), cmap='bone_r')
    h, xedge, yedge, _ = axbig0.hist2d(x=dtf['trait_elev'].loc[dtf['out'] == max(dtf['out'])],
                                       y=dtf['y'].loc[dtf['out'] == max(dtf['out'])],
                                       range=((0, 1), (0, 1e5)),
                                       bins=(100, 100), cmap='bone_r')
    h, xedge, yedge, _ = axbig1.hist2d(x=dtf['trait_prep'].loc[dtf['out'] == max(dtf['out'])],
                                       y=dtf['y'].loc[dtf['out'] == max(dtf['out'])],
                                       range=((0, 1), (0, 1e5)),
                                       bins=(100, 100), cmap='bone_r')
    axbig0.yaxis.set_tick_params(labelleft=False)
    axbig1.yaxis.set_label_position("right")
    axbig0.yaxis.tick_right()
    axbig1.yaxis.tick_right()
    axbig0.set_xlabel('Trait\nElevation', weight='bold')
    axbig1.set_xlabel('Trait\nPrecipitation', weight='bold')
    axbig1.set_ylabel('Y [m]', weight='bold')
    axs2[0, 0].set_ylabel('Abundance\n(No. ind)', weight='bold', color='blue')
    axs2_2.set_ylabel('Taxon richness', weight='bold', color='red')
    axs2[1, 0].set_ylabel('Trait\nElevation', weight='bold')
    axs2[2, 0].set_ylabel('Trait\nPrecipitation', weight='bold')
    axs2[2, 0].set_xlabel('Time [years]', weight='bold')