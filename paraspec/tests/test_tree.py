import numpy as np
import networkx as nx
import pandas as pd
import pytest

import paraspec


class TestPTAccessor(object):

    def test_node_type(self):
        df = pd.DataFrame({'id': [0, 1, 2],
                           'parent': [0, 0, 1]},
                          index=[-1, 1, 2])

        actual = df.ptree.node_type
        expected = pd.Series(pd.Categorical(['root', 'node', 'leaf']),
                             index=df.index)

        pd.testing.assert_series_equal(actual, expected)

    def test_tree_id(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3],
                          'parent': [0, 0, 1, 3]})

        actual = df.ptree.tree_id
        expected = pd.Series([0, 0, 0, 3], index=df.index, dtype=df.id.dtype)

        pd.testing.assert_series_equal(actual, expected)

        # test single tree fastpath
        df = pd.DataFrame({'id': [0, 1, 2, 3],
                          'parent': [0, 0, 1, 2]})

        actual = df.ptree.tree_id
        expected = pd.Series([0, 0, 0, 0], index=df.index, dtype=df.id.dtype)

    def test_branch_length(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5],
                           'parent': [0, 0, 0, 1, 1, 0],
                           'val': [0., 1., 1., 2., 3., 2.]})

        actual = df.ptree.branch_length('val')
        expected = pd.Series([0., 1., 1., 1., 2., 2.],
                             index=df.index)

        pd.testing.assert_series_equal(actual, expected)

    def test_simplify(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6],
                           'parent': [0, 0, 0, 1, 2, 2, 3]})

        actual = df.ptree.simplify()
        expected = pd.DataFrame({'id': [0, 2, 4, 5, 6],
                                 'parent': [0, 0, 2, 2, 0]},
                                index=[0, 2, 4, 5, 6])

        pd.testing.assert_frame_equal(actual, expected)

    def test_drop_disconnected_roots(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5],
                           'parent': [0, 1, 2, 3, 3, 3]})

        actual = df.ptree.drop_disconnected_roots()
        expected = pd.DataFrame({'id': [3, 4, 5],
                                 'parent': [3, 3, 3]},
                                index=[3, 4, 5])

        pd.testing.assert_frame_equal(actual, expected)

    def test_merge_forest(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3],
                           'parent': [0, 1, 2, 2],
                           'val': [1., 1., 1., 2.]})

        actual = df.ptree.merge_forest(-1, val=0.)
        expected = pd.DataFrame({'id': [-1, 0, 1, 2, 3],
                                 'parent': [-1, -1, -1, -1, 2],
                                 'val': [0., 1., 1., 1., 2.]},
                                index=[-1, 0, 1, 2, 3])

        def sort_df(_df):
            return _df.reindex(sorted(_df.columns), axis=1)

        pd.testing.assert_frame_equal(sort_df(actual), sort_df(expected))

        with pytest.raises(ValueError, match="id 0 already exists"):
            df.ptree.merge_forest(0)

    @pytest.mark.parametrize('root,leaf,expected_idx', [
        (0, None, [0, 1, 2, 3, 4, 5]),
        (2, None, [2, 3, 4, 5]),
        (4, None, [4, 5]),
        (5, None, [5]),
        ([3, 4], None, [3, 4, 5]),
        (99, None, 'error'),
        (None, 5, [0, 1, 2, 3, 4, 5]),
        (None, 0, [0]),
        (None, 99, 'error'),
        (None, [0, 5], [0, 1, 2, 3, 4, 5]),
        (4, 2, [0, 1, 2, 4, 5])
    ])
    def test_extract_subtree(self, root, leaf, expected_idx):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5],
                           'parent': [0, 0, 1, 2, 3, 4]})

        if expected_idx == 'error':
            with pytest.raises(ValueError, match="not valid node ids"):
                df.ptree.extract_subtree(root=root, leaf=leaf)

        else:
            actual = df.ptree.extract_subtree(root=root, leaf=leaf).index
            expected = pd.Index(expected_idx, dtype=actual.dtype)

            pd.testing.assert_index_equal(actual, expected)

    @pytest.mark.parametrize('length_col,length_from_col,uid_col,length,uid', [
        (None, None, None, [], []),
        ('length', None, None, [-1, -2, -3, -4], [0, 1, 2, 3]),
        (None, 'x', 'uid', [0., 1., 2., 2.], ['a', 'b', 'c', 'd']),

    ])
    def test_to_phylopandas(self, length_col, length_from_col, uid_col,
                            length, uid):
        df = pd.DataFrame({'id': [0, 1, 2, 3],
                           'parent': [0, 0, 0, 1],
                           'x': [0., 1., 2., 3.],
                           'length': [-1, -2, -3, -4],
                           'uid': ['a', 'b', 'c', 'd']})

        if length_col is None and length_from_col is None:
            with pytest.raises(ValueError, match='must be set'):
                df.ptree.to_phylopandas(length_col=length_col,
                                        length_from_col=length_from_col,
                                        uid_col=uid_col)
            return

        actual = df.ptree.to_phylopandas(length_col=length_col,
                                         length_from_col=length_from_col,
                                         uid_col=uid_col)

        type_col = pd.Categorical(["root", "node", "leaf", "leaf"])

        expected = pd.DataFrame({
            'id': np.array([0, 1, 2, 3], dtype='O'),
            'parent': np.array([None, 0, 0, 1], dtype='O'),
            'type': type_col,
            'x': [0., 1., 2., 3.],
            'length': length,
            'uid': uid
        })

        def sort_df(_df):
            return _df.reindex(sorted(_df.columns), axis=1)

        pd.testing.assert_frame_equal(sort_df(actual), sort_df(expected))

    @pytest.mark.parametrize('node_col', [None, 'x', ['x']])
    def test_to_networkx(self, node_col):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4],
                           'parent': [0, 0, 0, 2, 2],
                           'x': [10, 11, 12, 13, 14],
                           'y': [5., 6., 7., 8., 9.]})

        G_actual = df.ptree.to_networkx(node_col=node_col, edge_col='y')
        G_expected = nx.Graph([(0, 0), (0, 1), (0, 2), (2, 3), (2, 4)])

        nx.isomorphism.is_isomorphic(G_actual, G_expected)

        # node attributes
        actual = dict(G_actual.nodes.data())

        if node_col is None:
            expected = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
        else:
            expected = {0: {'x': 10}, 1: {'x': 11}, 2: {'x': 12},
                        3: {'x': 13}, 4: {'x': 14}}

        assert actual == expected

        # edge attributes
        actual = list(G_actual.edges.data())
        expected = [(0, 0, {'y': 5.}), (0, 1, {'y': 6.}), (0, 2, {'y': 7.}),
                    (2, 3, {'y': 8.}), (2, 4, {'y': 8.})]
