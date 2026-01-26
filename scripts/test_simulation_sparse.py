import unittest
import pandas as pd
import networkx as nx
import numpy as np

class TestNodeMatching(unittest.TestCase):

    def test_successful_node_matching(self):
        """
        Tests that common nodes are found when data and graph nodes have matching GEOIDs.
        This shows the expected format for the input files.
        """
        # 1. Create a sample DataFrame with standard AFFGEOIDs in the index
        df_data = {
            'AFFGEOID': ['1500000US010010201001', '1500000US010010201002', '1500000US010010202001'],
            'some_data': [10, 20, 30]
        }
        df = pd.DataFrame(df_data).set_index('AFFGEOID')

        # 2. Create a sample graph with corresponding numeric GEOIDs (no prefix)
        G = nx.Graph()
        G.add_nodes_from(['010010201001', '010010201002', '010010203001']) # One node is different

        # 3. Mimic the node matching logic from simulation_sparse.py
        graph_nodes = {f"1500000US{node}" for node in G.nodes()}
        df_nodes = set(df.index)
        
        node_list = sorted(list(graph_nodes.intersection(df_nodes)))

        # 4. Assert that two common nodes were found
        self.assertEqual(len(node_list), 2)
        self.assertIn('1500000US010010201001', node_list)
        self.assertIn('1500000US010010201002', node_list)

    def test_node_mismatch_if_graph_nodes_have_prefix(self):
        """
        Tests for a common failure case where graph nodes already contain the prefix.
        The script adds the prefix a second time, causing the match to fail.
        """
        # 1. Create a sample DataFrame
        df_data = {
            'AFFGEOID': ['1500000US010010201001', '1500000US010010201002'],
            'some_data': [10, 20]
        }
        df = pd.DataFrame(df_data).set_index('AFFGEOID')

        # 2. Create a graph where nodes INCORRECTLY have the prefix already
        G = nx.Graph()
        G.add_nodes_from(['1500000US010010201001', '1500000US010010201002'])

        # 3. Mimic the node matching logic from the script
        # The script adds the prefix again, which will cause a mismatch.
        # It will create nodes like '1500000US1500000US010010201001'
        graph_nodes = {f"1500000US{node}" for node in G.nodes()}
        df_nodes = set(df.index)
        node_list = sorted(list(graph_nodes.intersection(df_nodes)))

        # 4. Assert that no common nodes were found
        self.assertEqual(len(node_list), 0)

    def test_node_mismatch_if_dataframe_index_is_wrong(self):
        """
        Tests for a failure case where the DataFrame's index is not the AFFGEOID.
        """
        # 1. Create a sample DataFrame with the wrong index
        df_data = {
            'AFFGEOID': ['1500000US010010201001', '1500000US010010201002'],
            'geoname': ['Block Group 1', 'Block Group 2'],
            'some_data': [10, 20]
        }
        # Here, the index is just the default RangeIndex (0, 1, 2...)
        df = pd.DataFrame(df_data) 

        # 2. Create a sample graph
        G = nx.Graph()
        G.add_nodes_from(['010010201001', '010010201002'])

        # 3. Mimic the node matching logic
        graph_nodes = {f"1500000US{node}" for node in G.nodes()}
        df_nodes = set(df.index) # df.index will be {0, 1}, not the GEOIDs
        node_list = sorted(list(graph_nodes.intersection(df_nodes)))

        # 4. Assert that no common nodes were found
        self.assertEqual(len(node_list), 0)

if __name__ == '__main__':
    unittest.main()
