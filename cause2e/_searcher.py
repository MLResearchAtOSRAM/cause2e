"""
_searcher.py
================================================================
This module implements classes for causal discovery.

It is used by the discovery module to learn the causal graph from data and domain knowledge.
Currently, algorithms from py-causal and the pc-algorithm from causal-learn are supported.
"""

from cause2e import _data_type_mgr as dtm, _graph

from abc import ABC, abstractmethod

from pycausal.pycausal import pycausal as pc
from pycausal import search as s
from pycausal import prior as p

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc as cl_pc
from causallearn.utils.cit import fisherz


class _Searcher(ABC):
    """Abstract base class for causal discovery classes.

    Attributes:
        graph_output: A cause2e.Graph representing the causal graph.
    """

    def __init__(self, data, knowledge):
        self._data = data
        self._knowledge = knowledge
        self.graph_output = None

    @abstractmethod
    def run_search(self, **kwargs):
        pass


class TetradSearcher(_Searcher):
    """Main class for causal discovery with TETRAD algorithms called by py-causal.

    Attributes:
        graph_output: A cause2e.Graph representing the causal graph.
    """
    def __init__(self, data, continuous, discrete, knowledge):
        """Inits TetradSearcher."""
        super().__init__(data, knowledge)
        self._data_types = self._get_data_types(continuous, discrete)
        self._type_mgr = dtm.DataTypeManager(data,
                                             continuous,
                                             discrete
                                             )
        pc().start_vm()
        self._tetrad = s.tetradrunner()
        self._separator = "---------------------\n"

    def show_search_algos(self):
        """Shows all search algorithms that the TETRAD program offers."""
        print("TETRAD search algos:\n")
        self._tetrad.listAlgorithms()
        print(self._separator)

    def show_search_scores(self):
        """Shows all search scores that the TETRAD program offers."""
        print("TETRAD search scores:\n")
        self._tetrad.listScores()
        print(self._separator)

    def show_independence_tests(self):
        """Shows all independence tests that the TETRAD program offers."""
        print("TETRAD search independence tests:\n")
        self._tetrad.listIndTests()
        print(self._separator)

    def show_algo_info(self, algo_name):
        """Shows information about a selected algorithm from the TETRAD program.

        Args:
            algo_name: A string indicating the name of the algorithm of interest.
        """
        self._tetrad.getAlgorithmDescription(algo_name)
        print(self._separator)

    def show_algo_params(self, algo_name, test_name=None, score_name=None):
        """Shows the parameters that are required for a causal search with the TETRAD program.

        Args:
            algo_name: A string indicating the name of the algorithm of interest.
            test_name: Optional; A string indicating the independence test that the algorithm uses.
                Use show_algo_info to find out if this is a necessary input. Defaults to None.
            score_name: Optional; A string indicating the search score that the algorithm uses.
                Use show_algo_info to find out if this is a necessary input. Defaults to None.
        """
        self._tetrad.getAlgorithmParameters(algo_name, test_name, score_name)
        print(self._separator)

    def _get_data_types(self, continuous, discrete):
        """Returns a string indicating if the data is continuous, discrete or mixed.

        Args:
            continuous: A set containing the names of all continuous variables in the data.
            discrete: A set containing the names of all discrete variables in the data.
        """
        if not continuous:
            return 'discrete'
        elif not discrete:
            return 'continuous'
        else:
            return 'mixed'

    @property
    def _tetrad_knowledge(self):
        forbidden, required = self._format_knowledge_tetrad()
        return p.knowledge(forbiddirect=forbidden,
                           requiredirect=required
                           )

    def _format_knowledge_tetrad(self):
        """Returns the knowledge from the knowledge dictionary in a format suitable for TETRAD."""
        if self._knowledge:
            forbidden = self._set_to_list(self._knowledge['forbidden'])
            required = self._set_to_list(self._knowledge['required'])
        else:
            forbidden = []
            required = []
        return forbidden, required

    def _set_to_list(self, possible_set):
        if possible_set:
            possible_set = list(possible_set)
        return possible_set

    @property
    def _type_threshold(self):
        return self._get_type_threshold(complete=True)

    @property
    def _type_threshold_incomplete(self):
        return self._get_type_threshold(complete=False)

    def _get_type_threshold(self, complete):
        self._type_mgr.enforce_desired_types(complete)
        return self._type_mgr.threshold

    def run_search(self,
                   algo='fges',
                   use_knowledge=True,
                   score='cg-bic-score',
                   verbose=True,
                   keep_vm=True,
                   **kwargs):
        """Infers the causal graph from the data and domain knowledge.

        This is where the causal discovery algorithms are invoked. Currently only algorithms from
        the TETRAD program are available. The algorithms are called via pycausal, which is a Python
        wrapper around the TETRAD program provided by the creators of the original software. It
        seems that superfluous arguments are ignored, meaning e.g. that the default score does not
        cause problems when invoking constraint based algorithms like PC. Note that you do not need
        to specify a threshold for distinguish between discrete and continuous variables, since
        this is taken care of internally by the cause2e.searcher.

        Args:
            algo: Optional; A string indicating the search algorithm. Defaults to 'fges'.
            use_knowledge: Optional; A boolean indicating if we want to use our domain knowledge
                (some TETRAD algorithms cannot use it). Defaults to True.
            verbose: Optional; A boolean indicating if we want verbose output. Defaults to True.
            keep_vm: A boolean indicating if we want to keep the Java VM (used by TETRAD) alive
                after the search.   This is required to use TETRAD objects afterwards. Defaults to
                True.
            **kwargs: Arguments that are used to further specify parameters for the search. Use
                show_algo_params to find out which ones need to be passed.
            """
        if use_knowledge:
            knowledge = self._tetrad_knowledge
        else:
            knowledge = None
        self._tetrad.run(algoId=algo,
                         dfs=self._data,
                         dataType=self._data_types,
                         numCategoriesToDiscretize=self._type_threshold,
                         priorKnowledge=knowledge,
                         verbose=verbose,
                         **kwargs
                         )
        self._get_graphs()
        if not keep_vm:  # can be opened only once (no restart after closing)
            pc().stop_vm()
        else:
            print('Remember to stop the JVM after you are completely done.')

    def _get_graphs(self):
        """Extracts the causal graph from an internal TETRAD format into a cause2e.Graph."""
        tetrad_graph = self._tetrad.getTetradGraph()
        self.graph_output = _graph.Graph.from_tetrad(tetrad_graph, knowledge=self._knowledge)

    def _stop_vm(self):
        """Stops the Java VM. No restart possible."""
        pc().stop_vm()


class CausalLearnSearcher:
    """Main class for causal discovery with causal-learn algorithms. Causal-learn is a Python translation
       and extension of the TETRAD program.

    Attributes:
        graph_output: A cause2e.Graph representing the causal graph.
    """
    def __init__(self, data, knowledge):
        """Inits CausalLearnSearcher."""
        self._data = data
        self._knowledge = knowledge
        self._translate_knowledge()
        self._colors_dict = {'directed': 'b', 'undirected': 'g', 'bidirected': 'r'}
        self._names_dict = {ind: name for ind, name in enumerate(self._data.columns)}

    def _translate_knowledge(self):
        """Translates the knowledge from the knowledge dictionary in a format suitable for causal-learn."""
        self._cl_knowledge = BackgroundKnowledge()
        self._create_nodes()
        if self._knowledge:
            for source, destination in self._knowledge['forbidden']:
                self._forbid_edge(source, destination)
            for source, destination in self._knowledge['required']:
                self._require_edge(source, destination)

    def _create_nodes(self): #COMMENT ALL THE NEW FUNCTIONS AND CHECK WHICH ARGUMENTS WE SHOULD PASS TO THE SEARCH
        nodes_numeric = [GraphNode(f'X{ind + 1}') for ind, _ in enumerate(self._data.columns)]
        self._nodes = {x: nodes_numeric[self._data.columns.get_loc(x)] for x in self._data.columns}

    def _forbid_edge(self, source, destination):
        self._cl_knowledge.add_forbidden_by_node(self._nodes[source], self._nodes[destination])

    def _require_edge(self, source, destination):
        self._cl_knowledge.add_required_by_node(self._nodes[source], self._nodes[destination])

    def run_search(self,
                   alpha=0.05,
                   use_knowledge=True,
                   verbose=True,
                   **kwargs
                   ):
        """Infers the causal graph from the data and domain knowledge.

        This is where the causal discovery algorithms are invoked. Currently only the pc algorithm
        is available, as other options do not support domain knowledge yet.

        Args:
            alpha: Optional; A float indicating the significance threshold used in conditional independence tests.
            use_knowledge: Optional; A boolean indicating if we want to use our domain knowledge
                (some algorithms cannot use it). Defaults to True.
            verbose: Optional; A boolean indicating if we want verbose output. Defaults to True.
            **kwargs: Arguments that are used to further specify parameters for the search. Use
                show_algo_params to find out which ones need to be passed.
            """
        if use_knowledge:
            knowledge = self._cl_knowledge
        else:
            knowledge = None
        self._cl_graph = cl_pc(data=self._data.to_numpy(),
                               alpha=alpha,
                               indep_test=fisherz,
                               stable=True,
                               uc_rule=0,
                               uc_priority=-1,
                               background_knowledge=knowledge
                               )
        self._get_graph()

    def _get_graph(self):
        """Extracts the causal graph from an internal causal-learn format into a cause2e.Graph."""
        edges = self._get_edges()
        if edges['bidirected']:
            raise ValueError("Watch out! There are bidirected edges. Have you included all relevant variables?")
        else:
            self.graph_output = _graph.Graph.from_edges(directed_edges=edges['directed'],
                                                        undirected_edges=edges['undirected'],
                                                        knowledge=self._knowledge
                                                        )

    def _get_edges(self):
        edges_cl = {orientation: self._get_numeric_edges_by_orientation(orientation) for orientation in self._colors_dict}
        return {orientation: self._translate_edges(edges_cl[orientation], orientation) for orientation in edges_cl}

    def _get_numeric_edges_by_orientation(self, orientation):
        color = self._colors_dict[orientation]
        edges = self._nx_graph.edges
        return {(source, destination) for source, destination in edges if self._check_color(source, destination, color)}

    @property
    def _nx_graph(self):
        self._cl_graph.to_nx_graph()
        return self._cl_graph.nx_graph

    def _check_color(self, source, destination, color):
        return self._nx_graph[source][destination]['color'] == color

    def _translate_edges(self, edges, orientation):
        return {self._translate_edge(source, destination, orientation) for source, destination in edges}

    def _translate_edge(self, source, destination, orientation):
        source_translated = self._names_dict[source]
        destination_translated = self._names_dict[destination]
        if orientation == 'directed':
            return (source_translated, destination_translated)
        else:
            return frozenset({source_translated, destination_translated})
