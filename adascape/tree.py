import numpy as np
import pandas as pd
import dendropy


@pd.api.extensions.register_dataframe_accessor("ptree")
class PTreeAccessor:
    """Pandas.DataFrame extension for handling tree (and forest) structures.

    The tree or forest structure is fully defined by the 'taxon_id' and
    'ancestor_id' columns, which are both required in the data frame. The
    'ancestor_id' column contains the ids of the ancestor_id nodes. Root nodes
    must have the same value set in both 'taxon_id' and 'ancestor_id' columns.

    The index of the data frame should match with the 'taxon_id' column.

    Other columns of the data frame are optional and represent
    either node or incoming branch (ancestor_id->taxon_id) attributes.

    """

    def __init__(self, df):
        self._df = df

    @staticmethod
    def node_type(dtf):
        """
        Construct a pandas.Series with categories for each type of nodes,
        i.e., root, leaf or node to construct a phylogenetic Tree

        Returns
        ----------
        :class:`pandas.Series` object (categorical variable) with the type of each node

        """

        is_root = dtf.ancestor_id == 0
        is_node = dtf.taxon_id.isin(dtf.ancestor_id)

        type_values = np.where(is_root,
                               'root',
                               np.where(is_node, 'node', 'leaf'))

        node_type = pd.Series(pd.Categorical(type_values), index=dtf.index)

        return node_type

    def extract_taxon_summary(self, min_abund=None):
        """
        Construct a dataframe with information on node type and summary
        statistics for each taxon based on the results of Speciation model.

         Returns
        ----------
            Dataframe with the following columns:
            A) time of the simulation,
            B) taxon ids,
            C) ancestor ids,
            D) the average trait value for all traits computed
            during the simulation each as separate column,
            E) the abundance of individuals for each taxa, and
            F) the node type.
        """

        dtf = self._df
        traits_taxon = (dtf
                        .groupby(['time', 'taxon_id', 'ancestor_id'])
                        .mean()
                        .filter(regex='trait_*')
                        .reset_index()
                        )
        abundance_taxon = (dtf.groupby(['time', 'taxon_id', 'ancestor_id'])
                           .size().rename('abundance')
                           .reset_index()
                           )

        dtf_out = pd.merge(traits_taxon, abundance_taxon)
        dtf_out = dtf_out.assign(node_type=self.node_type(dtf_out))

        dtf_out['taxon_id'] = dtf_out['taxon_id'].astype(int).astype(str)
        dtf_out['ancestor_id'] = dtf_out['ancestor_id'].astype(int).astype(str)

        if min_abund is not None:
            dtf_out = dtf_out.loc[dtf_out.abundance >= min_abund]
        return dtf_out

    def to_dendropy_tree(self, taxon_annotations=[], node_annotations=[],
                         branch_lengths=True, branch_length_col='time',
                         min_abund=None):
        """
        Turn a pandas dataframe with tree information (node type, i.e. root, node and leaf) into a dendropy tree.
        Adapted from phylopandas project
        https://github.com/Zsailer/phylopandas/blob/master/phylopandas/treeio/write.py

        Parameters
        ----------
        taxon_annotations : str
            List of columns to annotation in the tree taxon.
        node_annotations : str
            List of columns to annotation in the node taxon.
        branch_lengths : bool
            If True, includes branch lengths.
        branch_length_col : str
            Name of the column with branch length information in the data frame.
        """
        name_par_to_check = ['branch_length_col']
        val_par_to_check = [branch_length_col]
        str_check_bool = [isinstance(i, str) for i in val_par_to_check]
        if not all(str_check_bool):
            name_par = [name_par_to_check[i] for i, l in enumerate(str_check_bool) if not l]
            value_par = [val_par_to_check[i] for i, l in enumerate(str_check_bool) if not l]
            raise TypeError("{} must be a string instead got {}.".format(repr(name_par).strip('[]'),
                                                                         repr(value_par).strip('[]'))
                            )

        dtf = self.extract_taxon_summary(min_abund=min_abund)

        # Construct a list of nodes from dataframe.
        taxon_namespace = dendropy.TaxonNamespace()
        nodes = {}
        for idx in dtf.index:
            # Get node data.
            data = dtf.loc[idx]

            # Get taxon for node (if leaf node).
            taxon = None
            if data['node_type'] == 'leaf':
                taxon = dendropy.Taxon(label=data.taxon_id)
                # Add annotations data.
                for ann in taxon_annotations:
                    taxon.annotations.add_new(ann, data[ann])
                taxon_namespace.add_taxon(taxon)

            # Get label for node.
            label = data.taxon_id

            # Get edge length.
            edge_length = None
            if branch_lengths is True:
                if data['node_type'] == 'root':
                    edge_length = 0
                else:
                    edge_length = data[branch_length_col] - \
                                  dtf[dtf.taxon_id == str(data.ancestor_id)][branch_length_col].values[0]

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
            children_idx = dtf[dtf.ancestor_id == data.taxon_id].index
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
