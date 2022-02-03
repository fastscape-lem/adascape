import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata
import dendropy


def taxon_definition(dtf, distance_method='ward', distance_value=0.5):
    """
    Taxon definition for lineage reconstruction based on the similarity of trait values
    and common ancestry among modelled individuals.

    Criteria adapted from:
        Pontarp & Wiens (2017) The origin of species richness patterns along environmental
        gradients: uniting explanations based on time, diversification rate and carrying capacity.
        Journal of Biogeography 44(4), 722-753.

    Parameters
    ----------
    dtf: pandas.DataFrame
        Output dataFrame of the speciation model with data of individuals
    distance_method: str
        Linkage algorithm to use to compute the distance matrix.
        See detail documentation from scipy.cluster.hierarchy.fclusterdata
    distance_value:

    Returns
    ----------
        Dataframe with taxon and ancestor definition.
    """
    col_traits = dtf.columns[dtf.columns.str.contains('trait_')].to_list()
    initial_data = dtf.groupby('time').get_group(0)
    dtf_cols = initial_data.columns.to_list()
    dtf_cols.append("taxon_id")
    dtf_cols.append("ancestor_id")
    out_dtf = pd.DataFrame(columns=dtf_cols)
    col_traits.append('ancestor_id')
    _clus = fclusterdata(initial_data[["trait_0"]].to_numpy(),
                         method=distance_method,
                         t=distance_value,
                         criterion='distance')
    out_dtf = out_dtf.append(initial_data.assign(taxon_id=_clus, ancestor_id=0))

    for i in dtf['time'].unique()[1:].astype(int):
        ancestor_data = out_dtf.groupby('time').get_group(i - 1)
        current_ancestor_id = np.repeat(ancestor_data['taxon_id'], ancestor_data['n_offspring'])
        current_data = dtf.groupby('time').get_group(i)
        current_data = current_data.assign(ancestor_id=current_ancestor_id.values)
        _clus = fclusterdata(current_data[["trait_0", "ancestor_id"]].to_numpy(),
                             method=distance_method,
                             t=distance_value,
                             criterion='distance')
        current_data = current_data.assign(taxon_id=_clus + current_ancestor_id.max())
        out_dtf = out_dtf.append(current_data)

    return out_dtf


def node_type(dtf):
    """Return the type of each node (i.e., root, leaf or node) as a
        :class:`pandas.Series` object (categorical variable).
        """
    is_root = dtf.parent == 0
    is_node = dtf.id.isin(dtf.parent)

    type_values = np.where(is_root,
                           'root',
                           np.where(is_node, 'node', 'leaf'))

    node_type = pd.Series(pd.Categorical(type_values), index=dtf.index)
    return node_type


def dtf_taxon(dtf, min_branch_lenght=0.001):
    """
    Construct a dataframe with information on nodes, leaf, and branch length
    for each taxon.

     Returns
    ----------
        Dataframe with tree data for each taxon over the simulation time.
    """
    col_traits = dtf.columns[dtf.columns.str.contains('trait_')].to_list()
    traits_taxon = (dtf
                    .groupby(['time', 'taxon_id', 'ancestor_id'])[col_traits]
                    .mean()
                    .reset_index()
                    )
    abundace_taxon = (dtf.groupby(['time', 'taxon_id', 'ancestor_id'])
                      .size().rename('abundance')
                      .reset_index()
                      )

    dtf_out = pd.merge(traits_taxon, abundace_taxon).rename(columns={'taxon_id': 'id', 'ancestor_id': 'parent'})
    dtf_out = (dtf_out.assign(node_type=node_type(dtf_out),
                              branch_length=dtf_out.time.diff().fillna(0) + min_branch_lenght
                              ))
    dtf_out['id'] = dtf_out.id.astype(str)
    dtf_out['parent'] = dtf_out.parent.astype(str)

    return dtf_out


def pandas_dtf_to_dendropy_tree(df, taxon_col='id', taxon_annotations=[], node_col='node_type',
                                node_annotations=[], branch_lengths=True):
    """
    Turn a pandas dataframe with tree information (node and leaf) into a dendropy tree.
    Adapted from phylopandas project
    https://github.com/Zsailer/phylopandas/blob/master/phylopandas/treeio/write.py

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing tree data.
    taxon_col : str (optional)
        Column in dataframe to label the taxon. If None, the index will be used.
    taxon_annotations : str
        List of columns to annotation in the tree taxon.
    node_col : str (optional)
        Column in dataframe to label the nodes. If None, the index will be used.
    node_annotations : str
        List of columns to annotation in the node taxon.
    branch_lengths : bool
        If True, includes branch lengths.
    """
    if isinstance(taxon_col, str) is False:
        raise Exception("taxon_col must be a string.")

    if isinstance(node_col, str) is False:
        raise Exception("taxon_col must be a string.")

    # Construct a list of nodes from dataframe.
    taxon_namespace = dendropy.TaxonNamespace()
    nodes = {}
    for idx in df.index:
        # Get node data.
        data = df.loc[idx]

        # Get taxon for node (if leaf node).
        taxon = None
        if data['node_type'] == 'leaf':
            taxon = dendropy.Taxon(label=data[taxon_col])
            # Add annotations data.
            for ann in taxon_annotations:
                taxon.annotations.add_new(ann, data[ann])
            taxon_namespace.add_taxon(taxon)

        # Get label for node.
        label = data[node_col]

        # Get edge length.
        edge_length = None
        if branch_lengths is True:
            edge_length = data['branch_length']

        # Build a node
        n = dendropy.Node(
            taxon=taxon,
            label=label,
            edge_length=edge_length
        )

        # Add node annotations
        for ann in node_annotations:
            n.annotations.add_new(ann, data[ann])

        nodes[idx] = n
    # return nodes
    # Build branching pattern for nodes.
    root = None
    for idx, node in nodes.items():
        # Get node data.
        data = df.loc[idx]

        # Get children nodes
        children_idx = df[df['parent'] == data['id']].index
        children_nodes = [nodes[i] for i in children_idx]

        # Set child nodes
        nodes[idx].set_child_nodes(children_nodes)

        # Check if this is root.
        if data['node_type'] == 'root':
            root = nodes[idx]

    # Build tree.
    tree = dendropy.Tree(
        seed_node=root,
        taxon_namespace=taxon_namespace
    )
    return tree
