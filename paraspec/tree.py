import numpy as np
import pandas as pd
import dendropy


@pd.api.extensions.register_dataframe_accessor("ptree")
class PTreeAccessor(object):
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

        # cache for node type (root, leaf, node) and tree id
        self._node_type = None
        self._tree_id = None

    @property
    def node_type(self):
        """
        Construct a pandas.Series with categories for each type of nodes,
        i.e., root, leaf or node to construct a phylogenetic Tree

        Returns
        ----------
        :class:`pandas.Series` object (categorical variable) with the type of each node

        """
        if self._node_type is None:
            is_root = self._df.ancestor_id == 0
            is_node = self._df.taxon_id.isin(self._df.ancestor_id)

            type_values = np.where(is_root,
                                   'root',
                                   np.where(is_node, 'node', 'leaf'))

            self._node_type = pd.Series(pd.Categorical(type_values),
                                        index=self._df.index)

        return self._node_type

    def dtf_taxon(self):
        """
        Construct a dataframe with information on nodes, leaf, and branch length
        for each taxon based on the results of Speciation model.


         Returns
        ----------
            Dataframe with tree data for each taxon over the simulation time.
        """

        dtf = self._df
        traits_taxon = (dtf
                        .groupby(['time', 'taxon_id', 'ancestor_id'])
                        .filter(regex='trait_*')
                        .mean()
                        .reset_index()
                        )
        abundance_taxon = (dtf.groupby(['time', 'taxon_id', 'ancestor_id'])
                          .size().rename('abundance')
                          .reset_index()
                          )

        dtf_out = pd.merge(traits_taxon, abundance_taxon)
        dtf_out = dtf_out.assign(node_type=self.node_type)

        dtf_out['taxon_id'] = dtf_out['taxon_id'].astype(int).astype(str)
        dtf_out['ancestor_id'] = dtf_out['ancestor_id'].astype(int).astype(str)

        return dtf_out

    def to_dendropy_tree(self, taxon_annotations=[], node_annotations=[],
                         branch_lengths=True, branch_length_col='time',
                         ):
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

        dtf = self.dtf_taxon()

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
