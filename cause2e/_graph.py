"""
_graph.py
================================================================
This module implements graph classes.

The goal is to reach independence of the Java Tetrad program as soon as possible after the search.
The Tetrad graphs are very powerful, but it is intransparent and cumbersome to work with wrappers
around Java functions that could as well be represented by the Python networkx package.
A drawback is that the networkx package does not allow mixed graphs, so we provide a custom
implementation as a quick workaround.
"""

import networkx as nx
import pydot
from IPython.display import Image, display
import itertools
import pandas as pd
from cause2e import knowledge
from cause2e._result_mgr import save_df_as_png


# TODO split this into several files
class Graph:
    """Top level graph class.

    Attributes:
        edges: The set of edges in the graph, stored as pairs (node_1, node_2) for directed edges
            or as sets {node_1, node_2} for undirected edges.
        undirected_edges: The set of undirected edges.
        directed_edges: The set of directed_edges.
        nodes: The set of nodes in the graph, stored as strings.
        dot: A pydot dot graph representation.
        png: A png representation.
    """

    def __init__(self, intelligent_graph, knowledge=None):
        """Inits Graph from a cause2e._GraphNetworkx."""
        self._technical = intelligent_graph
        self._knowledge = _format_knowledge(knowledge)

    @classmethod
    def from_tetrad(cls, tetrad_graph, knowledge=None):
        """Inits Graph directly from a TETRAD graph."""
        return cls(_GraphTetrad(tetrad_graph).to_GraphNetworkx(), knowledge)

    @classmethod
    def from_edges(cls, directed_edges, undirected_edges, knowledge=None):
        """Inits Graph directly from given edges."""
        intelligent_graph = _GraphNetworkx.from_edges(directed_edges, undirected_edges)
        return cls(intelligent_graph, knowledge)

    @property
    def directed_edges(self):
        return self._technical.directed_edges

    @property
    def undirected_edges(self):
        return self._technical.undirected_edges

    @property
    def edges(self):
        return self._technical.edges

    @property
    def nodes(self):
        return self._technical.nodes

    def add_edge(self, source, destination, directed=True, show=True):
        """Adds an edge to the graph.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            directed: Optional; A boolean indicating if the edge should be a directed one. If not,
                the roles of source and destination can be exchanged. Defaults to True.
            show: Optional; A boolean indicating if the resulting graph should be displayed.
                Defaults to True.
        """
        self._technical.add_edge(source, destination, directed)
        if show:
            self.show()

    def remove_edge(self, source, destination, directed=True, show=True):
        """Removes an edge from the graph.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            directed: Optional; A boolean indicating if the edge should be a directed one. If not,
                the roles of source and destination can be exchanged. Defaults to True.
            show: Optional; A boolean indicating if the resulting graph should be displayed.
                Defaults to True.
        """
        self._technical.remove_edge(source, destination, directed)
        if show:
            self.show()

    def reverse_edge(self, source, destination, direction_strict=False, show=True):
        """Reverses an edge in the graph.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            direction_strict: Optional; A boolean indicating if the edge must exist in the
                direction 'source -> destination'. If not, the edge 'destination -> source' is also
                detected and reversed if it exists. Defaults to False.
            show: Optional; A boolean indicating if the resulting graph should be displayed.
                Defaults to True.
        """
        self._technical.reverse_edge(source, destination, direction_strict)
        if show:
            self.show()

    def has_edge(self, source, destination, directed=True):
        """Checks if the graph contains a specific edge.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            directed: Optional; A boolean indicating if the edge should be a directed one. If not,
                the roles of source and destination can be exchanged. Defaults to True.
        """
        return self._technical.has_edge(source, destination, directed)

    def has_node(self, name):
        """Checks if the graph contains a specific node.

        Args:
            name: A string indicating the name of the node in question.
        """
        return self._technical.has_node(name)

    def _final_checks(self, knowledge):
        assert self.is_acyclic(), 'The graph contains at least one directed cycle!'
        return self.respects_knowledge(knowledge)

    def is_acyclic(self):
        """Checks if the graph is acyclic.

        The graph is considered acyclic if it has no undirected edges and does not
        contain any directed cycles.

        Returns:
            A boolean that is True if and only if the graph is acyclic.

        Raises:
            AssertionError: At least one edge is undirected.
        """
        assert not self.has_undirected_edges(), 'Orient all edges before checking for acyclicity!'
        return self._technical.is_acyclic()

    def has_undirected_edges(self):
        """Checks if the graph has undirected edges.

        Returns:
            A boolean that is True if and only if the graph has at least one undirected edge.
        """
        return self._technical.has_undirected_edges()

    def respects_knowledge(self, knowledge_dict):
        """Checks if the graph respects the domain knowledge.

        This means that it contains all the edges that were required in the domain knowledge
        and none of the edges that were forbidden in the domain knowledge.

        Args:
            knowledge_dict: A dictionary containing the domain knowledge.

        Returns:
            A boolean that is True if and only if the graph respects the domain knowledge.
        """
        checker = knowledge.KnowledgeChecker(self._technical.edges, knowledge_dict)
        return checker.respects_knowledge()

    @property
    def dot(self):
        return self._technical.to_dot(self._knowledge['required'])

    @property
    def png(self):
        return self.dot.create_png(prog='dot')

    @property
    def _png_display(self):
        dot = self.dot
        dot.set_size('"5,5!"')
        return dot.create_png(prog='dot')

    def show(self):
        "Shows the graph."
        print('Proposed causal graph (edges required by domain knowledge in red, undirected edges in blue):\n')
        display(Image(self._png_display))

    def save(self, name, file_extension, verbose=True, strict=True, knowledge=None):
        """Saves the graph to a file.

        Args:
            name: A string indicating the name of the file to save.
            file_extension: A string indicating the desired file extension.
            verbose: Optional; A boolean indicating if confirmation messages should be printed.
                Defaults to True.
            strict: Optional; A boolean indicating if the graph must be acyclic and in accordance
                to the domain knowledge to allow saving. Defaults to True.
            knowledge: Optional; A dictionary containing the domain knowledge. Defaults to None.
        """
        if verbose:
            print(f'Saving graph to {file_extension} file.')
        if strict:
            msg = 'Your graph is not a DAG or not compliant with the specified knowledge.'
            assert self._final_checks(knowledge), msg
        self.dot.write(name, format=file_extension)

    def print_edge_analysis(self, save_to_name=None):
        """Analyzes which part of the edges were forced by domain knowledge.

        Args:
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the edge information should be saved. Defaults to None.
        """
        self._edge_analyzer.print_edge_analysis(save_to_name)

    @property
    def _edge_analyzer(self):
        return _EdgeAnalyzer(self.nodes, self.directed_edges, self.undirected_edges, self._knowledge)

    # TODO: Add edges modified in postprocessing to reporting and give them a separate color
    def save_edge_analysis(self, save_to_name):
        """Saves the edge analysis to a png file.

        Args:
            save_to_name:A string indicating the beginning of the name of the png file
                where the edge information should be saved.
        """
        self._edge_analyzer.save_edge_analysis(save_to_name)


class GraphKnowledge(Graph):
    """A subclass of Graph that enables visualizing domain knowledge."""

    def __init__(self, knowledge, variables):
        """Inits GraphKnowledge from domain knowledge."""
        knowledge_ = _format_knowledge(knowledge)
        intelligent_graph = _GraphNetworkx.from_edges(knowledge_['required'])
        super().__init__(intelligent_graph, knowledge_)
        self._variables = variables
        self._allowed_edges = self._edge_analyzer.get_remaining_allowed_edges()

    @property
    def dot(self):
        return self._technical.to_dot(self._knowledge['required'], self._allowed_edges)

    @property
    def _edge_analyzer(self):
        return _EdgeAnalyzer(self.nodes, self.directed_edges, self.undirected_edges,
                             self._knowledge, self._variables)

    def show(self):
        "Shows the graph."
        print('Causal graph constructed from domain knowledge:')
        print('(Edges required by domain knowledge in red, remaining allowed edges dotted)\n')
        display(Image(self._png_display))

    def save(self, name, file_extension, verbose=True):
        """Saves the knowledge graph to a file.

        Args:
            name: A string indicating the name of the file to save.
            file_extension: A string indicating the desired file extension.
            verbose: Optional; A boolean indicating if confirmation messages should be printed.
                Defaults to True.
        """
        if verbose:
            print(f'Saving knowledge graph to {file_extension} file.')
        self.dot.write(name, format=file_extension)


class _GraphNetworkx:
    """An enhancement of a networkx graph which allows mixed edges.

    Attributes:
        edges: The set of edges in the graph, stored as pairs (node_1, node_2) for directed edges
            or as sets {node_1, node_2} for undirected edges.
        undirected_edges: The set of undirected edges.
        directed_edges: The set of directed_edges.
        nodes: The set of nodes in the graph, stored as strings.

    """
    def __init__(self, graph_nx, undirected_edges):
        """Inits _GraphNetworkx from a networkx digraph and a set of undirected edges."""
        self._graph = graph_nx
        self.undirected_edges = undirected_edges

    @classmethod
    def from_edges(self, directed_edges, undirected_edges=set()):
        """Inits _GraphNetworkx from a sets of directed and undirected edges."""
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(directed_edges)
        return _GraphNetworkx(nx_graph, undirected_edges)

    @property
    def edges(self):
        nx_edges = set(self._graph.edges)
        nx_edges |= self.undirected_edges
        for source, destination in self.undirected_edges:
            if (source, destination) in nx_edges:
                nx_edges.discard((source, destination))
            elif (destination, source) in nx_edges:
                nx_edges.discard((destination, source))
        return nx_edges

    @property
    def directed_edges(self):
        return self.edges - self.undirected_edges

    @property
    def nodes(self):
        return set(itertools.chain.from_iterable(self.edges))

    def to_dot(self, required_edges=set(), allowed_edges=set()):
        """Returns a pydot dot graph representation of the graph.

        Args:
            required_edges: Optional; A set of edges required by domain knowledge.
                Defaults to set().
            allowed_edges: Optional; A set of absent edges allowed by domain knowledge.
                Defaults to set().
        """
        dot = pydot.Dot(type='directed')
        self._add_directed_edges_to_dot(dot, required_edges)
        self._add_undirected_edges_to_dot(dot)
        if allowed_edges:
            self._add_allowed_edges_to_dot(dot, allowed_edges)
        dot.set_size('"25,20!"')
        return dot

    def _add_directed_edges_to_dot(self, dot, required_edges=set()):
        """Adds directed edges to a directed dot graph.

        Args:
            dot: A pydot dot graph.
            required_edges: Optional; A set of edges required by domain knowledge.
                Defaults to set().

        Returns:
            A pydot dot graph with additional directed edges.
        """
        for edge in self.directed_edges:
            color = self._get_edge_color(edge, required_edges)
            dot.add_edge(pydot.Edge(*edge, color=color))
        return dot

    def _get_edge_color(self, edge, required_edges):
        """Returns the color of an edge.

        Args:
            edge: An edge stored as pair (node_1, node_2).
            required_edges: A set of edges required by domain knowledge.

        Returns:
            A string. Red if the edge is forced by domain knowledge, black else.
        """
        if edge in required_edges:
            return "Red"
        else:
            return "Black"

    def _add_undirected_edges_to_dot(self, dot):
        """Adds undirected edges to a directed dot graph.

        Args:
            dot: A pydot dot graph, containing only directed edges.

        Returns:
            A pydot dot graph with additional undirected edges.
        """
        for edge in map(tuple, self.undirected_edges):
            dot.add_edge(pydot.Edge(*edge,
                                    dir="None",
                                    arrowhead="None",
                                    arrowtail="None",
                                    color="Blue",
                                    )
                         )
        return dot

    def _add_allowed_edges_to_dot(self, dot, allowed_edges):
        """Adds remaining allowed edges to a directed dot graph.

        Args:
            dot: A pydot dot graph.

        Returns:
            A pydot dot graph with additional allowed edges in dotted lines.
        """
        for edge in allowed_edges:
            dot.add_edge(pydot.Edge(*edge, style='dotted'))
        return dot

    def verify_identical_edges(self, edges):
        """Checks if the graph has exactly the given edges.

        Args:
            edges: A set of edges stored as pairs (node_1, node_2).

        Raises:
            ValueError: At least one edge differs.
        """
        if self.edges != edges:
            print('Networkx Edges:')
            print(self.edges)
            print('Edges of original graph:')
            print(edges)
            print('Symmetric difference:')
            print(self.edges.symmetric_difference(edges))
            raise ValueError("Edges do not match after conversion to NetworkX!")

    def add_edge(self, source, destination, directed=True):
        """Adds an edge to the graph.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            directed: Optional; A boolean indicating if the edge should be a directed one. If not,
                the roles of source and destination can be exchanged. Defaults to True.
        """
        if not directed:
            self.undirected_edges.add(frozenset({source, destination}))
        else:
            self._remove_from_undirected_edges(source, destination)
            self._graph.add_edge(source, destination)

    def remove_edge(self, source, destination, directed=True):
        """Removes an edge from the graph.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            directed: Optional; A boolean indicating if the edge should be a directed one. If not,
                the roles of source and destination can be exchanged. Defaults to True.
        """
        if not directed:
            self.undirected_edges.discard(frozenset({source, destination}))
        else:
            self._remove_from_undirected_edges(source, destination)
            self._graph.remove_edge(source, destination)

    def _remove_from_undirected_edges(self, source, destination):
        self.undirected_edges.discard(frozenset({source, destination}))

    def reverse_edge(self, source, destination, direction_strict=False):
        """Reverses an edge in the graph.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            direction_strict: Optional; A boolean indicating if the edge must exist in the
                direction 'source -> destination'. If not, the edge 'destination -> source' is also
                detected and reversed if it exists. Defaults to False.

        Raises:
            AssertionError: Given edge does not exist.
        """
        if self.has_edge(source, destination):
            self.reverse_existing_edge(source, destination)
        elif not direction_strict and self.has_edge(destination, source):
            self.reverse_existing_edge(destination, source)
        else:
            msg = f'The edge between {source} and {destination} is undirected or nonexistent!'
            raise AssertionError(msg)

    def reverse_existing_edge(self, source, destination):
        """Reverses an existing edge in the graph.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
        """
        self.remove_edge(source, destination)
        self.add_edge(destination, source)

    def has_edge(self, source, destination, directed=True):
        """Checks if the graph contains a specific edge.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            directed: Optional; A boolean indicating if the edge should be a directed one. If not,
                the roles of source and destination can be exchanged. Defaults to True.
        """
        if directed:
            return (source, destination) in self.directed_edges
        else:
            return frozenset({source, destination}) in self.undirected_edges

    def has_node(self, name):
        """Checks if the graph contains a specific node.

        Args:
            name: A string indicating the name of the node in question.
        """
        return name in self.nodes

    def is_acyclic(self):
        """Checks if the graph is acyclic.

        The graph is considered acyclic if it has no undirected edges and does not
        contain any directed cycles.

        Returns:
            A boolean that is True if and only if the graph is acyclic.
        """
        return nx.algorithms.dag.is_directed_acyclic_graph(self._graph)

    def has_undirected_edges(self):
        """Checks if the graph has undirected edges.

        Returns:
            A boolean that is True if and only if the graph has at least one undirected edge.
        """
        return bool(self.undirected_edges)


class _GraphTetrad:
    """A wrapper around a TETRAD graph that allows conversion to a Python format.

    Attributes:
        graph: the result of tetrad.getTetradGraph() after a TETRAD search
        edges: The set of edges in the graph, stored as pairs (node_1, node_2) for directed edges
            or as sets {node_1, node_2} for undirected edges.
        undirected_edges: The set of undirected edges.
        directed_edges: The set of directed_edges.
    """

    def __init__(self, graph_tetrad):
        """Inits GraphTetrad."""
        self._graph = graph_tetrad

    @property
    def undirected_edges(self):
        return {edge for edge in self.edges if type(edge) is frozenset}

    @property
    def directed_edges(self):
        return self.edges - self.undirected_edges

    @property
    def edges(self):
        return {self._format_edge(edge) for edge in self._split_edge_str()}

    def _split_edge_str(self):
        """Returns a list of strings representing all edges."""
        edge_str = str(self._graph.getEdges())
        return edge_str.strip("[").strip("]").split(", ")

    def _format_edge(self, edge):
        """Returns a representation of an edge as a pair or as a set.

        Args:
            edge: A string as returned by _split_edge_str.

        Raises:
            ValueError: The string does not conform to a known edge type.
        """
        source, arrow, destination = self._decompose_edge(edge)
        direction = self._check_edge_direction(arrow)
        if direction == 'forward':
            return (source, destination)
        elif direction == 'backwards':
            return (destination, source)
        elif direction == 'undirected':
            return frozenset({source, destination})
        else:  # is this even possible? Yes, PAGs
            raise ValueError('Something weird happened while converting dot edges!')

    def _decompose_edge(self, edge):
        return edge.split(' ')

    def _check_edge_direction(self, arrow):
        if arrow == '-->':
            return 'forward'
        elif arrow == '<--':
            return 'backwards'
        elif arrow == '---':
            return 'undirected'
        else:
            return False

    def to_GraphNetworkx(self):
        """Returns the graph represented as a cause2e._GraphNetworkx."""
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(self.directed_edges)
        graph_networkx = _GraphNetworkx(nx_graph, self.undirected_edges)
        graph_networkx.verify_identical_edges(self.edges)
        return graph_networkx


class _EdgeAnalyzer:
    """Helper class for analyzing which part of the edges were forced by domain knowledge."""
    def __init__(self, nodes, directed_edges, undirected_edges, knowledge, additional_nodes=set()):
        self._nodes = nodes
        self._directed_edges = directed_edges
        self._undirected_edges = undirected_edges
        self._knowledge = knowledge
        self._additional_nodes = additional_nodes

    def print_edge_analysis(self, save_to_name=None):
        """Analyzes which part of the edges were forced by domain knowledge.

        Args:
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the edge information should be saved. Defaults to None.
        """
        print("================================")
        self._print_undirected_edges()
        print("--------------------------------")
        self._print_forced_edges()
        print("--------------------------------")
        self._print_unforced_edges()
        print("--------------------------------")
        self._print_remaining_allowed_edges()
        print("================================")
        if save_to_name:
            self.save_edge_analysis(save_to_name)

    def _print_undirected_edges(self):
        """Prints a list of all undirected edges in the graph."""
        if self._undirected_edges:
            print('The following edges are undirected (colored in blue):')
            for edge in self._undirected_edges:
                self._print_undirected_edge(edge)
        else:
            print('The graph is fully directed.\n')

    def _print_undirected_edge(self, edge):
        node_1, node_2 = edge
        print(f'{node_1} --- {node_2}')

    def _print_forced_edges(self):
        print("The following present edges were forced by domain knowledge:")
        for edge in self._get_forced_edges():
            print(edge)

    def _print_unforced_edges(self):
        print("The following present edges were not forced by domain knowledge:")
        for edge in self._get_unforced_edges():
            print(edge)

    def _get_forced_edges(self):
        return {edge for edge in self._directed_edges.intersection(self._knowledge['required'])}

    def _get_unforced_edges(self):
        forced_edges = self._get_forced_edges()
        return self._directed_edges - forced_edges

    def _print_remaining_allowed_edges(self):
        print("The following missing edges were not forbidden by domain knowledge:")
        for edge in self.get_remaining_allowed_edges():
            print(edge)

    def get_remaining_allowed_edges(self):
        """Returns the set of absent edges that are not forbidden by domain knowledge."""
        allowed_edges = self._get_allowed_edges()
        return allowed_edges - self._directed_edges

    def _get_allowed_edges(self):
        possible_edges = self._get_all_possible_edges()
        return possible_edges - self._knowledge['forbidden']

    def _get_all_possible_edges(self):
        possible_edges = set()
        nodes = self._nodes | self._additional_nodes
        for source_node in nodes:
            for destination_node in nodes - {source_node}:
                possible_edges.add((source_node, destination_node))
        return possible_edges

    def save_edge_analysis(self, save_to_name):
        """Analyzes which part of the edges were forced by domain knowledge.

        Args:
            save_to_name: A string indicating the beginning of the name of the png file
                where the edge information should be saved.
        """
        df = pd.DataFrame(self.get_remaining_allowed_edges())
        title = "Ignored allowed edges:"
        columns = ["Source", "Destination"]
        if df.empty:
            df = pd.DataFrame(["None"])
            columns = ["No ignored allowed edges"]
        save_df_as_png(df, title, save_to_name, col_labels=columns, loc='upper left')
        print("Saving edge analysis.")


def _format_knowledge(knowledge):
    if knowledge:
        return knowledge
    else:
        return {'required': set(), 'forbidden': set()}
