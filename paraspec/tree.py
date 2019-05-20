import numpy as np
import pandas as pd
import networkx as nx


def _simplify_one_step(df):
    # select all "only child" nodes for which parent has siblings
    has_siblings = df.duplicated(subset='parent', keep=False)
    child1_nodes = df[~has_siblings]

    parent_is_child1 = child1_nodes.parent.isin(child1_nodes.id)
    selection = child1_nodes[~parent_is_child1]

    if not len(selection):
        return df, True

    # remove parent nodes from selection and update their parents
    parents2remove = selection.parent

    new_parents = df.parent.copy()
    new_parents[selection.id] = df.loc[parents2remove, 'parent']

    return df.assign(parent=new_parents).drop(parents2remove), False


@pd.api.extensions.register_dataframe_accessor("ptree")
class PTreeAccessor(object):
    """Pandas.DataFrame extension for handling tree (and forest) structures.

    The tree or forest structure is fully defined by the 'id' and
    'parent' columns, which are both required in the data frame. The
    'parent' column contains the ids of the parent nodes. Root nodes
    must have the same value set in both 'id' and 'parent' columns.

    The index of the data frame should match with the 'id' column.

    Other columns of the data frame are optional and represent
    either node or incoming branch (parent->id) attributes.

    """

    def __init__(self, df):
        self._df = df

        # cache for node type (root, leaf, node) and tree id
        self._node_type = None
        self._tree_id = None

    @property
    def node_type(self):
        """Return the type of each node (i.e., root, leaf or node) as a
        :class:`pandas.Series` object (categorical variable).

        """
        if self._node_type is None:
            is_root = self._df.parent == self._df.id
            is_node = self._df.id.isin(self._df.parent)

            type_values = np.where(is_root,
                                   'root',
                                   np.where(is_node, 'node', 'leaf'))

            self._node_type = pd.Series(pd.Categorical(type_values),
                                        index=self._df.index)

        return self._node_type

    @property
    def tree_id(self):
        """Return a unique id for each tree found in the data frame as a
        :class:`pandas.Series` object.

        The id corresponds to the id of the root node.

        """
        if self._tree_id is None:
            tree_id = pd.Series(index=self._df.index, dtype=self._df.id)
            roots = self._df.loc[self.node_type == 'root']

            if len(roots) == 1:
                # fastpath (only 1 tree)
                tree_id[:] = roots.loc[0, 'id']

            else:
                tree_id.loc[roots.index] = roots.id
                nodes = (self._df.loc[self._df.parent.isin(roots.id)]
                                 .drop(roots.index))

                while len(nodes):
                    tree_id.loc[nodes.index] = tree_id.loc[nodes.parent].values
                    nodes = self._df.loc[self._df.parent.isin(nodes.id)]

            self._tree_id = tree_id

        return self._tree_id

    def branch_length(self, from_col):
        """Return the lengths of the branches of the tree(s).

        Parameters
        ----------
        from_col : str
            Name of the variable (column) used to compute the lengths
            of the branches.

        Returns
        -------
        lengths : :class:`pd.Series` object
            Length values assigned to the branches connecting
            the nodes and their respective parents.

        """
        vals = self._df.loc[:, from_col]
        lengths = vals[self._df.id].values - vals[self._df.parent].values

        return pd.Series(lengths, vals.index)

    def drop_disconnected_roots(self):
        """Drop all root nodes that are not connected to any child node.

        Returns
        -------
        dropped : :class:`pandas.DataFrame` object
            A new dataframe without the disconnected root nodes.

        """
        root_mask = self.node_type == 'root'
        not_root_sel = self._df.loc[~root_mask, 'parent']

        roots2drop = self._df.loc[root_mask & ~self._df.id.isin(not_root_sel)]

        return self._df.drop(roots2drop.index)

    def merge_forest(self, new_root_id, **kwargs):
        """Add a common parent node to every root node and return a
        single tree.

        Parameters
        ----------
        new_root_id : int
            Id of the common parent node (i.e., the root of the tree).
        **kwargs : {col_name: value}, optional
            Use keyword arguments to set variable values for the new
            root node.

        Returns
        -------
        tree : :class:`pandas.DataFrame` object
            A new dataframe representing a single tree with a new
            root node.

        """
        if new_root_id in self._df.id:
            raise ValueError("id {!r} already exists".format(new_root_id))

        kwargs.update({'id': new_root_id, 'parent': new_root_id})
        new_parent = np.where(self.node_type == 'root',
                              new_root_id,
                              self._df.parent)

        root_df = pd.DataFrame(kwargs, index=[new_root_id])
        tree_df = self._df.assign(parent=new_parent)

        return pd.concat([root_df, tree_df], sort=False)

    def simplify(self):
        """Simplify the tree(s).

        Merge non-diverging branches, i.e., drop the nodes that have
        exactly one parent and one child and repair the connections
        between the other nodes.

        Returns
        -------
        simplified : :class:`pandas.DataFrame` object
            A new dataframe with the simplified tree(s).

        """
        new_df = self._df.copy()
        is_simplified = False

        while not is_simplified:
            new_df, is_simplified = _simplify_one_step(new_df)

        return new_df

    def _check_node(self, ids):
        valid_ids = pd.unique(
            self._df.loc[:, ['id', 'parent']].values.ravel('K')
        )
        invalid_ids = pd.Index(ids).difference(valid_ids)

        if len(invalid_ids):
            raise ValueError("the following ids are not valid node ids: {}"
                             .format(list(invalid_ids)))

    def _drop_root_nodes(self, df):
        node_type = self.node_type.loc[df.id]
        root_idx = df.loc[(node_type == 'root').values].index

        return df.drop(root_idx)

    def _extract(self, nodes, dir):
        dfs = []

        if dir == 'down':
            from_col, to_col = 'parent', 'id'

            # append nodes as new roots unless they are already roots
            # or their parent is also in the node list
            df_nodes = self._df.loc[self._df.id.isin(nodes)]
            df_maybe_root = df_nodes.loc[~df_nodes.parent.isin(nodes)]
            df_no_root = self._drop_root_nodes(df_maybe_root)

            new_roots = df_no_root.assign(parent=df_no_root['id'])

            dfs.append(new_roots)

        elif dir == 'up':
            from_col, to_col = 'id', 'parent'

        while len(nodes):
            df_level = self._df.loc[self._df[from_col].isin(nodes)]
            dfs.append(df_level)

            # prevent infinite loop: drop original root nodes
            nodes = self._drop_root_nodes(df_level)[to_col]

        return dfs

    def extract_subtree(self, root=None, leaf=None):
        """Extract sub-tree(s) from the input tree(s).

        Parameters
        ----------
        root : int or list, optional
            Id(s) of the root node(s) of the extracted sub-tree(s).
        leaf : int or list, optional
            Id(s) of the leaf node(s) of the extracted sub-tree(s).

        Returns
        -------
        subtree : :class:`pandas.DataFrame` object
            A new dataframe with selected rows representing the nodes
            of the extracted sub-tree(s).
            The returned selection is the union of the extracted sub-trees
            from the all given root and leaf nodes.

        """
        def to_list(nodes):
            if nodes is None:
                return []
            elif isinstance(nodes, list):
                return nodes
            else:
                return [nodes]

        parents = to_list(root)
        children = to_list(leaf)

        self._check_node(parents + children)

        dfs = self._extract(parents, dir='down')
        dfs += self._extract(children, dir='up')

        return pd.concat(dfs).drop_duplicates().sort_index()

    def to_phylopandas(self, length_col=None, length_from_col=None,
                       uid_col=None):
        """Re-arrange the data frame so that it can be used with the
        Phylopandas package.

        More info: https://github.com/Zsailer/phylopandas

        Parameters
        ----------
        length_col : str, optional
            Name of the column used for branch lengths. If None is
            specified, the branch length will be calculated using
            ``length_from_col``.
        length_from_col : str, optional
            Name of the column used to calculate branch lengths. Ignored
            if a name is specified for ``length_col``.
        uid_col : str, optional
            Name if the column used for uid node values. If None is
            specified, the 'id' column is used.

        Returns
        -------
        phylo : :class:`pandas.DataFrame` object
            A new dataframe with 'id', 'parent', 'type', 'length' and
            'uid' cols, and where root nodes have parent values reset
            to None.

        """
        if length_col is None and length_from_col is None:
            raise ValueError("either 'length_col' or 'length_from_col' "
                             "must be set")

        if length_col is None:
            length = self.branch_length(length_from_col)
        else:
            length = self._df.loc[:, length_col]

        if uid_col is None:
            uid = self._df.id
        else:
            uid = self._df.loc[:, uid_col]

        update_cols = {
            'parent': np.where(self.node_type == 'root',
                               None, self._df.parent.astype(np.object)),
            'id': self._df.id.astype(np.object),
            'type': self._df.ptree.node_type,
            'length': length,
            'uid': uid
        }

        return self._df.assign(**update_cols)

    def to_networkx(self, node_col=None, edge_col=None):
        """Export tree data to a networkx graph.

        Parameters
        ----------
        node_col : str or tuple, optional
            Name(s) of the columns to add as node attribute(s).
        edge_col : str or tuple, optional
            Name(s) of the columns to add as edge attribute(s).

        Returns
        -------
        G : :class:`networkx.classes.graph.Graph` object
            A new networkx (undirected) graph object.

        """
        def to_list(cols):
            if cols is None:
                return []
            elif isinstance(cols, str):
                return [cols]
            else:
                return list(cols)

        G = nx.convert_matrix.from_pandas_edgelist(
            self._df, source='parent', target='id'
        )

        node_attrs = (self._df.loc[:, to_list(node_col)]
                              .to_dict(orient='index'))
        nx.set_node_attributes(G, node_attrs)

        edge_attrs = (self._df.set_index(['parent', 'id'])
                              .loc[:, to_list(edge_col)]
                              .to_dict(orient='index'))
        nx.set_edge_attributes(G, edge_attrs)

        return G
