import unittest
from cause2e import _graph
import networkx as nx


class TestGraphToplevel(unittest.TestCase):
    def setUp(self):
        # set up a partially undirected sprinkler graph
        G = nx.DiGraph()
        G.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
        edges = {('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E')}
        G.add_edges_from(edges)
        undirected_edges = {frozenset({'A', 'B'})}
        technical = _graph._GraphNetworkx(G, undirected_edges)
        self.graph = _graph.Graph(technical)

    def test_add_directed_edge(self):
        self.graph.add_edge('A', 'E', show=False)
        self.assertTrue(self.graph.has_edge('A', 'E'))

    def test_add_undirected_edge(self):
        self.graph.add_edge('A', 'D', directed=False, show=False)
        self.assertTrue(self.graph.has_edge('A', 'D', directed=False))

    def test_add_edge_to_new_node(self):
        self.graph.add_edge('B', 'new', show=False)
        self.assertTrue(self.graph.has_edge('B', 'new'))

    def test_has_node(self):
        # nonexistent
        self.assertFalse(self.graph.has_node('U'))
        # only part of undirected edge
        self.assertTrue(self.graph.has_node('A'))
        # part of directed and undirected edge
        self.assertTrue(self.graph.has_node('B'))
        # only part of directed edge
        self.assertTrue(self.graph.has_node('C'))
        # unconnected (nodes are implicitly generated from edges, unconnected nodes vanish)
        self.graph._technical._graph.add_node('F')
        self.assertFalse(self.graph.has_node('F'))

    def test_remove_directed_edge(self):
        self.assertTrue(self.graph.has_edge('D', 'E'))
        self.graph.remove_edge('D', 'E', show=False)
        self.assertFalse(self.graph.has_edge('D', 'E'))

    def test_remove_undirected_edge(self):
        self.assertTrue(self.graph.has_edge('A', 'B', directed=False))
        self.graph.remove_edge('B', 'A', directed=False, show=False)
        self.assertFalse(self.graph.has_edge('A', 'B', directed=False))

    def test_remove_nonexistent_edge(self):
        with self.assertRaises(nx.NetworkXError):
            self.graph.remove_edge('E', 'A', show=False)

    def test_reverse_directed_edge(self):
        self.assertTrue(self.graph.has_edge('D', 'E'))
        self.graph.reverse_edge('D', 'E', show=False)
        self.assertFalse(self.graph.has_edge('D', 'E'))
        self.assertTrue(self.graph.has_edge('E', 'D'))
        # reverse edge with non-directed call to opposite edge
        self.graph.reverse_edge('E', 'D', show=False)
        self.assertFalse(self.graph.has_edge('E', 'D'))
        self.assertTrue(self.graph.has_edge('D', 'E'))

    def test_reverse_undirected_edge(self):
        with self.assertRaises(AssertionError):
            self.graph.reverse_edge('A', 'B', direction_strict=True, show=False)
        with self.assertRaises(AssertionError):
            self.graph.reverse_edge('A', 'B', direction_strict=False, show=False)

    def test_has_edge(self):
        self.assertFalse(self.graph.has_edge('A', 'B'))
        self.assertTrue(self.graph.has_edge('A', 'B', directed=False))
        self.assertFalse(self.graph.has_edge('A', 'E'))

    def test_is_acyclic(self):
        # not fully oriented
        with self.assertRaises(AssertionError):
            self.graph.is_acyclic()
        # orient graph and retry
        self.graph.add_edge('A', 'B', show=False)
        self.assertTrue(self.graph.is_acyclic())
        # create cycle
        self.graph.add_edge('E', 'D', show=False)
        self.assertFalse(self.graph.is_acyclic())
        # break cycle
        self.graph.remove_edge('E', 'D', show=False)
        self.assertTrue(self.graph.is_acyclic())
        # add loop (should count as cycle)
        self.graph.add_edge('A', 'A', show=False)
        self.assertFalse(self.graph.is_acyclic())

    def test_orient_edge_by_adding(self):
        self.assertTrue(self.graph.has_undirected_edges())
        self.graph.add_edge('A', 'B', show=False)
        self.assertFalse(self.graph.has_undirected_edges())
        self.assertTrue(self.graph.has_edge('A', 'B'))

    def test_orient_edge_by_removing(self):
        self.assertTrue(self.graph.has_undirected_edges())
        with self.assertRaises(nx.NetworkXError):
            self.graph.remove_edge('A', 'B', show=False)
        self.graph.remove_edge('A', 'B', directed=False, show=False)
        self.assertFalse(self.graph.has_undirected_edges())
        self.assertFalse(self.graph.has_edge('A', 'B'))

    def test_fail_to_orient_edge_by_reversing(self):
        self.assertTrue(self.graph.has_undirected_edges())
        with self.assertRaises(AssertionError):
            self.graph.reverse_edge('A', 'B', show=False)
        self.assertTrue(self.graph.has_undirected_edges())


class TestGraphNetworkx(unittest.TestCase):
    def setUp(self):
        # set up the sprinkler graph
        G = nx.DiGraph()
        G.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
        edges = {('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E')}
        G.add_edges_from(edges)
        undirected_edges = {frozenset({'A', 'B'})}
        self.graph = _graph._GraphNetworkx(G, undirected_edges)

    def check_all_edges(self, undirected_edges, directed_edges):
        self.assertEqual(self.graph.undirected_edges, undirected_edges)
        self.assertEqual(self.graph.directed_edges, directed_edges)
        self.assertEqual(self.graph.edges, undirected_edges | directed_edges)
        self.graph.verify_identical_edges(undirected_edges | directed_edges)

    def test_edges(self):
        # initial config
        undirected_edges = {frozenset({'A', 'B'})}
        directed_edges = {('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E')}
        self.check_all_edges(undirected_edges, directed_edges)
        # after undirected edge is added
        self.graph.add_edge('B', 'D', directed=False)
        undirected_edges = {frozenset({'A', 'B'}), frozenset({'B', 'D'})}
        directed_edges = {('A', 'C'), ('C', 'D'), ('D', 'E')}
        self.check_all_edges(undirected_edges, directed_edges)
        # after the undirected edges are removed
        self.graph.remove_edge('A', 'B', directed=False)
        undirected_edges = {frozenset({'B', 'D'})}
        directed_edges = {('A', 'B'), ('A', 'C'), ('C', 'D'), ('D', 'E')}
        self.check_all_edges(undirected_edges, directed_edges)


if __name__ == '__main__':
    unittest.main()
