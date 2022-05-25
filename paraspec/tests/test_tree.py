import numpy as np
import dendropy
import pandas as pd


class TestPTAccessor:

    def test_node_type(self):
        df = pd.DataFrame({'taxon_id': [1, 2, 3],
                           'ancestor_id': [0, 1, 2]},
                          index=[-1, 1, 2])

        actual = df.ptree.node_type(df)
        expected = pd.Series(pd.Categorical(['root', 'node', 'leaf']),
                             index=df.index)

        pd.testing.assert_series_equal(actual, expected)

    def test_extract_taxon_summary(self):
        dtf = pd.DataFrame({
            'time': np.repeat(np.arange(0, 5, 1), np.arange(10, 35, 5)),
            'taxon_id': np.repeat(np.arange(0, 5, 1), np.arange(10, 35, 5)) + 1,
            'ancestor_id': np.repeat(np.arange(0, 5, 1), np.arange(10, 35, 5)),
            'trait_0': np.repeat(np.arange(0, 5, 1), np.arange(10, 35, 5))/4
        })

        actual = dtf.ptree.extract_taxon_summary()

        expected = pd.DataFrame({
            'time': np.arange(0, 5, 1),
            'taxon_id': np.arange(0, 5, 1) + 1,
            'ancestor_id': np.arange(0, 5, 1),
            'trait_0': np.arange(0, 1.1, 0.25),
            'abundance': np.arange(10, 35, 5),
            'node_type': ['root', 'node', 'node', 'node', 'leaf']
        })
        expected['taxon_id'] = expected['taxon_id'].astype(str)
        expected['ancestor_id'] = expected['ancestor_id'].astype(str)
        expected['node_type'] = pd.Categorical(expected['node_type'])

        pd.testing.assert_frame_equal(actual, expected)

    def test_to_dendropy_tree(self):
        dtf = pd.DataFrame({
            'time': np.repeat(np.arange(0, 5, 1), np.arange(10, 35, 5)),
            'taxon_id': np.repeat(np.arange(0, 5, 1), np.arange(10, 35, 5)) + 1,
            'ancestor_id': np.repeat(np.arange(0, 5, 1), np.arange(10, 35, 5)),
            'trait_0': np.repeat(np.arange(0, 5, 1), np.arange(10, 35, 5)) / 4
        })

        tree = dtf.ptree.to_dendropy_tree()

        assert isinstance(tree, dendropy.Tree)
        assert len(tree.nodes()) == dtf.taxon_id.max()
        assert tree.max_distance_from_root() == dtf.time.max()
