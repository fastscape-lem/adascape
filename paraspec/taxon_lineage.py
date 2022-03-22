import numpy as np
import pandas as pd
import dendropy


def node_type(dtf, parent_col='parent', id_col='id'):
    """
    Construct a pandas.Series with categories for each type of nodes,
    i.e., root, leaf or node to construct a phylogenetic Tree

    Parameters
    ----------
    dtf : pandas.DataFrame
        DataFrame with two integer columns for the parent and individual id number
    parent_col : string
        Name of the column with parent or ancestor id number
    id_col : string
        Name of the column with individual id number

    Returns
    ----------
    :class:`pandas.Series` object (categorical variable) with the type of each node

    """
    is_root = dtf[parent_col] == 0
    is_node = dtf[id_col].isin(dtf[parent_col])

    type_values = np.where(is_root,
                           'root',
                           np.where(is_node, 'node', 'leaf'))

    node_type = pd.Series(pd.Categorical(type_values), index=dtf.index)
    return node_type


def dtf_taxon(dtf):
    """
    Construct a dataframe with information on nodes, leaf, and branch length
    for each taxon based on the results of Speciation model.

    Parameters
    ----------
    dtf : pandas.DataFrame
        DataFrame with output of Speciation model.

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

    dtf_out = pd.merge(traits_taxon, abundace_taxon)
    dtf_out = (dtf_out.assign(node_type=node_type(dtf_out, parent_col='ancestor_id', id_col='taxon_id')
                              )
               )
    dtf_out['taxon_id'] = 'S' + dtf_out['taxon_id'].astype(int).astype(str)
    dtf_out['ancestor_id'] = 'S' + dtf_out['ancestor_id'].astype(int).astype(str)

    return dtf_out


def pandas_dtf_to_dendropy_tree(dtf, taxon_col='taxon_id', taxon_annotations=[],
                                node_col='ancestor_id', node_annotations=[],
                                branch_lengths=False, branch_length_col='time',
                                parent_col='ancestor_id'):
    """
    Turn a pandas dataframe with tree information (node type, i.e. root, node and leaf) into a dendropy tree.
    Adapted from phylopandas project
    https://github.com/Zsailer/phylopandas/blob/master/phylopandas/treeio/write.py

    Parameters
    ----------
    dtf : pandas.DataFrame
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
    branch_length_col : str
        Name of the column with branch length information in the data frame
    parent_col : str
        Name of the column with parent or ancestor id values.
    """
    name_par_to_check = ['taxon_col', 'node_col', 'parent_col', 'branch_length_col']
    val_par_to_check = [taxon_col, node_col, parent_col, branch_length_col]
    str_check_bool = [isinstance(i, str) for i in val_par_to_check]
    if not all(str_check_bool):
        name_par = [name_par_to_check[i] for i, l in enumerate(str_check_bool) if not l]
        value_par = [val_par_to_check[i] for i, l in enumerate(str_check_bool) if not l]
        raise TypeError("{} must be a string instead got {}.".format(repr(name_par).strip('[]'),
                                                                     repr(value_par).strip('[]'))
                        )

    # Construct a list of nodes from dataframe.
    taxon_namespace = dendropy.TaxonNamespace()
    nodes = {}
    for idx in dtf.index:
        # Get node data.
        data = dtf.loc[idx]

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
            if data['node_type'] == 'leaf':
                edge_length = 0
            elif data['node_type'] == 'node':
                edge_length = dtf[branch_length_col].max() - data[branch_length_col]
            else:
                edge_length = dtf[branch_length_col].max()

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
    # Build branching pattern for nodes.
    root = None
    for idx, node in nodes.items():
        # Get node data.
        data = dtf.loc[idx]

        # Get children nodes
        children_idx = dtf[dtf[parent_col] == data[taxon_col]].index
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
