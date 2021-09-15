"""
discovery.py
================================================================
This module implements the StructureLearner class.

It is used to learn a causal graph from domain knowledge and data.
The proposed procedure is as follows:\n
1) Read the data from a csv or parquet file.\n
2) Preprocess the data (e.g. delete or recombine variables).\n
3) Pass the domain knowledge.\n
4) Run a causal discovery algorithm.\n
5) Check if the graph looks sensible and orient remaining undirected or misdirected edges.\n
6) Check if the graph is a DAG conforming to the domain knowledge.\n
7) Save the graph in various file formats.\n
"""

from cause2e import _reader, _preproc, _graph, _searcher, estimator


class StructureLearner():
    """Main class for performing causal discovery.

    Attributes:
        paths: A cause2e.PathManager managing paths and file names.
        data: A pandas.Dataframe containing the data.
        transformations: A list storing all the performed preprocessing transformations.
        variables: A set containing the names of all variables in the data.
        continuous: A set containing the names of all continuous variables in the data.
        discrete: A set containing the names of all discrete variables in the data.
        knowledge: A dictionary containing domain knowledge about required or forbidden edges in
            the causal graph. Known quantitative effects can be included for later validation.
        graph: A cause2e.Graph representing the causal graph.
    """

    def __init__(self, paths, spark=None):
        """Inits StructureLearner."""
        self.paths = paths
        self.data = None
        self.continuous = set()
        self.discrete = set()
        self.knowledge = None
        self.graph = None
        self._spark = spark

    @property
    def _reader(self):
        return _reader.Reader(self.paths.full_data_name,
                              self._spark
                              )

    def read_csv(self, **kwargs):
        """Reads data from a csv file."""
        self.data = self._reader.read_csv(**kwargs)

    def read_parquet(self, **kwargs):
        """Reads data rom a parquet file."""
        self.data = self._reader.read_parquet(**kwargs)

    @property
    def variables(self):
        return set(self.data.columns)

    @property
    def transformations(self):
        return self._preprocessor.transformations

    def combine_variables(self, name, input_cols, func, keep_old=True):
        """Combines data from existing variables into a new variable.

        Args:
            name: A string indicating the name of the new variable.
            input_cols: A list containing the names of the variables that are used for generating
                the new variable.
            func: A function describing how the new variable is calculated from the input variables.
            keep_old: Optional; A boolean indicating if we want to keep the input variables in our
                data. Defaults to True.
        """
        self._ensure_preprocessor()
        self._preprocessor.combine_variables(name, input_cols, func, keep_old)

    def add_variable(self, name, vals):
        """Adds a new variable to the data.

        Args:
            name: A string indicating the name of the new variable.
            vals: A column of values for the new variable.
        """
        self._ensure_preprocessor()
        self._preprocessor.add_variable(name, vals)

    def delete_variable(self, name):
        """Deletes a variable from the data.

        Args:
            name: A string indicating the name of the target variable.
        """
        self._ensure_preprocessor()
        self._preprocessor.delete_variable(name)

    def rename_variable(self, current_name, new_name):
        """Renames a variable in the data.

        Args:
            current_name: A string indicating the current name of the variable.
            new_name: A string indicating the desired new name of the variable.
        """
        self._ensure_preprocessor()
        self._preprocessor.rename_variable(current_name, new_name)

    def binarize_variable(self, name, one_val, zero_val=None):
        """Transforms a variable to a binary variable.

        Args:
            name: A string indicating the name of the target variable.
            one_val: The value that should be translated to 1.
            zero_val: Optional; the value that should be translated to 0.
                Use None if everything except for one_val should be translated to 0. Defaults to None.
        """
        self._ensure_preprocessor()
        self._preprocessor.binarize_variable(name, one_val, zero_val)

    def normalize_variables(self):
        """Replaces data for all variables by their z-scores."""
        self._ensure_preprocessor()
        self._preprocessor.normalize_variables()

    def normalize_variable(self, name):
        """Replaces a variable by its z-scores.

        Args:
            name: A string indicating the name of the target variable.
        """
        self._ensure_preprocessor()
        self._preprocessor.normalize_variable(name)

    def _ensure_preprocessor(self):
        """Ensures that a preprocessor is initialized."""
        if not hasattr(self, 'preprocessor'):
            self._preprocessor = _preproc.Preprocessor(self.data)

    def set_knowledge(self,
                      edge_creator,
                      validation_creator=None,
                      show=True,
                      save=True):
        """Sets the domain knowledge that we have a about the causal graph.

        Args:
            edge_creator: A cause2e.knowledge.EdgeCreator that has been used to create
                required and forbidden edges.
            validation_creator: Optional; A cause2e.knowledge.ValidationCreator that has been used
                to create a dictionary containing expected quantitative causal effects. These are
                evaluated after estimation of the effects. Defaults to None.
            show: Optional; A boolean indicating if information about the passed knowledge should
                be displayed. Defaults to True.
            show: Optional; A boolean indicating if information about the passed knowledge should
                be saved to a png. Defaults to True.
        """
        forbidden = edge_creator.forbidden_edges
        required = edge_creator.required_edges
        if validation_creator:
            expected_effects = validation_creator.expected_effects
        else:
            expected_effects = {}
        self.knowledge = {'forbidden': forbidden,
                          'required': required,
                          'expected_effects': expected_effects
                          }
        if show:
            self.show_knowledge()
        if save:
            self.save_knowledge()

    def show_knowledge(self):
        """Shows all domain knowledge that is used for causal discovery."""
        print("====================")
        if self.knowledge:
            print("Showing knowledge for graph search.\n")
            self.knowledge_graph.show()
            print("Required edges:")
            for edge in self.knowledge['required']:
                print(edge)
            print("--------------------")
            print("Forbidden edges:")
            for edge in self.knowledge['forbidden']:
                print(edge)
        else:
            print("No knowledge has been provided.")
        print("====================")

    @property
    def knowledge_graph(self):
        return _graph.GraphKnowledge(self.knowledge, self.variables)

    def save_knowledge(self, file_extension='png', verbose=True):
        """Saves the knowledge graph to a file.

        Args:
            file_extension: A string indicating the desired file extension.
            verbose: Optional; A boolean indicating if confirmation messages should be printed.
                Defaults to True.
        """
        name = self.paths.create_knowledge_graph_name(file_extension)
        self.knowledge_graph.save(name, file_extension, verbose)

    def erase_knowledge(self):
        """Erases all domain knowledge."""
        self.knowledge = None

    @property
    def _plain_searcher(self):
        return _searcher.TetradSearcher(self.data,
                                        self.continuous,
                                        self.discrete,
                                        self.knowledge
                                        )

    def show_search_algos(self):  # these methods are not efficient
        """Shows all search algorithms that the TETRAD program offers."""
        self._plain_searcher.show_search_algos()

    def show_search_scores(self):
        """Shows all search scores that the TETRAD program offers."""
        self._plain_searcher.show_search_scores()

    def show_independence_tests(self):
        """Shows all independence tests that the TETRAD program offers."""
        self._plain_searcher.show_independence_tests()

    def show_algo_info(self, algo_name):
        """Shows information about a selected algorithm from the TETRAD program.

        Args:
            algo_name: A string indicating the name of the algorithm of interest.
        """
        self._plain_searcher.show_algo_info(algo_name)

    def show_algo_params(self, algo_name, test_name=None, score_name=None):
        """Shows the parameters that are required for a causal search with the TETRAD program.

        Args:
            algo_name: A string indicating the name of the algorithm of interest.
            test_name: Optional; A string indicating the independence test that the algorithm uses.
                Use show_algo_info to find out if this is a necessary input. Defaults to None.
            score_name: Optional; A string indicating the search score that the algorithm uses.
                Use show_algo_info to find out if this is a necessary input. Defaults to None.
        """
        self._plain_searcher.show_algo_params(algo_name, test_name, score_name)

    def run_quick_search(self, verbose=True, keep_vm=True, show_graph=True, save_graph=True):
        """Infers the causal graph from the data and domain knowledge with preset parameters.

        Args:
            verbose: Optional; A boolean indicating if we want verbose output. Defaults to True.
            keep_vm: A boolean indicating if we want to keep the Java VM (used by TETRAD) alive
                after the search. This is required to use TETRAD objects afterwards. Defaults to
                True.
            show_graph: A boolean indicating if the resulting graph should be shown. Defaults to
                True.
            show_graph: A boolean indicating if the resulting graph should be saved. Defaults to
                True.
        """
        self.run_search(algo='fges',
                        use_knowledge=True,
                        verbose=verbose,
                        keep_vm=keep_vm,
                        show_graph=show_graph,
                        save_graph=save_graph,
                        scoreId='cg-bic-score',
                        faithfulnessAssumed=True,
                        symmetricFirstStep=True)

    def run_search(self,
                   algo,
                   use_knowledge=True,
                   verbose=True,
                   keep_vm=True,
                   show_graph=True,
                   save_graph=True,
                   **kwargs):
        """Infers the causal graph from the data and domain knowledge.

        This is where the causal discovery algorithms are invoked. Currently only algorithms from
        the TETRAD program are available. The algorithms are called via pycausal, which is a Python
        wrapper around the TETRAD program provided by the creators of the original software. It
        seems that superfluous arguments are ignored, meaning e.g. that passing a score does not
        cause problems when invoking constraint based algorithms like PC. Note that you do not need
        to specify a threshold for distinguish between discrete and continuous variables, since
        this is taken care of internally by the cause2e.searcher.

        Args:
            algo: A string indicating the search algorithm.
            use_knowledge: Optional; A boolean indicating if we want to use our domain knowledge
                (some TETRAD algorithms cannot use it). Defaults to True.
            verbose: Optional; A boolean indicating if we want verbose output. Defaults to True.
            keep_vm: A boolean indicating if we want to keep the Java VM (used by TETRAD) alive
                after the search. This is required to use TETRAD objects afterwards. Defaults to
                True.
            show_graph: A boolean indicating if the resulting graph should be shown. Defaults to
                True.
            show_graph: A boolean indicating if the resulting graph should be saved. Defaults to
                True.
            **kwargs: Arguments that are used to further specify parameters for the search. Use
                show_algo_params to find out which ones need to be passed.
        """
        self._searcher = self._plain_searcher
        self._searcher.run_search(algo=algo,
                                  use_knowledge=use_knowledge,
                                  verbose=verbose,
                                  keep_vm=keep_vm,
                                  **kwargs
                                  )
        self.graph = self._searcher.graph_output
        if show_graph:
            self.display_graph()
        if save_graph:
            self.save_graphs()

    def display_graph(self, edge_analysis=True):
        """Shows the causal graph.

        Args:
            edge_analysis: Optional; A boolean indicating if an analysis about the influence of
                domain knowledge on the resulting graph should be shown.
        """
        self.graph.show()
        self.print_edge_analysis()

    def print_edge_analysis(self):
        """Analyzes which part of the edges were forced by domain knowledge."""
        self.graph.print_edge_analysis()

    def add_edge(self, source, destination, directed=True, show=True):
        """Adds an edge to the causal graph.

        Consider adding the desired edge to the domain knowledge and rerunning the search if you
        are sure that it belongs in the graph.

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
        self.graph.add_edge(source, destination, directed, show)

    def remove_edge(self, source, destination, directed=True, show=True):
        """Removes an edge from the causal graph.

        Consider adding the desired edge to the domain knowledge and rerunning the search if you
        are sure that it does not belong in the graph.

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
        self.graph.remove_edge(source, destination, directed, show)

    def reverse_edge(self, source, destination, direction_strict=False, show=True):
        """Reverses an edge in the causal graph.

        Consider adding the desired edge to the domain knowledge and rerunning the search if you
        are sure that it belongs in the graph in the desired orientation.

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
        self.graph.reverse_edge(source, destination, direction_strict, show)

    def has_edge(self, source, destination, directed=True):
        """Checks if the causal graph contains a specific edge.

        Args:
            source: A string indicating the name of the source variable.
                For and edge 'a -> b', a is the source.
            destination: A string indicating the name of the destination variable.
                For and edge 'a -> b', b is the destination.
            directed: Optional; A boolean indicating if the edge should be a directed one. If not,
                the roles of source and destination can be exchanged. Defaults to True.
        """
        return self.graph.has_edge(source, destination, directed)

    def has_node(self, name):
        """Checks if the causal graph contains a specific node.

        Args:
            name: A string indicating the name of the node in question.
        """
        return self.graph.has_node(name)

    def is_acyclic(self):
        """Checks if the causal graph is acyclic.

        The graph is considered acyclic if it has no undirected edges and does not
        contain any directed cycles.

        Returns:
            A boolean that is True if and only if the graph is acyclic.

        Raises:
            AssertionError: At least one edge is undirected.
        """
        return self.graph.is_acyclic()

    def has_undirected_edges(self):
        """Checks if the causal graph has undirected edges.

        Returns:
            A boolean that is True if and only if the graph has at least one undirected edge.
        """
        return self.graph.has_undirected_edges()

    def respects_knowledge(self):
        """Checks if the causal graph respects the domain knowledge.

        This means that it contains all the edges that were required in the domain knowledge
        and none of the edges that were forbidden in the domain knowledge.

        Returns:
            A boolean that is True if and only if the graph respects the domain knowledge.
        """
        return self.graph.respects_knowledge(self.knowledge)

    def save_graphs(self, file_extensions=['dot', 'png', 'svg'], verbose=True, strict=True):
        """Saves the causal graph in various file formats.

        Args:
            file_extensions: Optional; A list of strings indicating the desired file extensions.
                Defaults to ['dot', 'png', 'svg'].
            verbose: Optional; A boolean indicating if confirmation messages should be printed.
                Defaults to True.
            strict: Optional; A boolean indicating if the graph must be acyclic and in accordance
                to the domain knowledge to allow saving. Defaults to True.
        """
        for ext in file_extensions:
            self.save_graph(ext, verbose, strict)

    def save_graph(self, file_extension, verbose=True, strict=True):
        """Saves the causal graph to a file.

        Args:
            file_extension: A string indicating the desired file extension.
            verbose: Optional; A boolean indicating if confirmation messages should be printed.
                Defaults to True.
            strict: Optional; A boolean indicating if the graph must be acyclic and in accordance
                to the domain knowledge to allow saving. Defaults to True.
        """
        name = self._get_graph_name(file_extension)
        self.graph.save(name, file_extension, verbose, strict, self.knowledge)
        if file_extension == 'png':
            name_analysis = self._get_analysis_name(name)
            self.graph.save_edge_analysis(name_analysis)

    def _get_graph_name(self, file_extension):
        """Returns the name for the files in which the causal graph is stored.

        Args:
            file_extension: A string indicating the desired file extension.
        """
        additions = self._get_graph_name_additions()
        name = self.paths.create_output_name(file_extension, additions)
        return name

    def _get_graph_name_additions(self):
        """Returns details about the experiment that should be added to the file names.

        Currently it only indicates if domain knowledge was used or not.
        """
        additions = ''
        if self.knowledge is None:
            additions += '_no_knowledge'
        return additions

    def _get_analysis_name(self, graph_name):
        return graph_name[:-4] + '_edge_analysis.png'

    def run_all_quick_analyses(self,
                               estimand_types=['nonparametric-ate',
                                               'nonparametric-nde',
                                               'nonparametric-nie'
                                               ],
                               verbose=False,
                               show_tables=True,
                               show_heatmaps=True,
                               show_validation=True,
                               show_largest_effects=True,
                               generate_pdf_report=True):
        """Performs all possible quick causal anlyses with preset parameters.

        Args:
            estimand_types: A list of strings indicating the types of causal effects.
            verbose: Optional; A boolean indicating if verbose output should be displayed for each
                analysis. Defaults to False.
            show_tables: Optional; A boolean indicating if the resulting causal estimates should be
                displayed in tabular form. Defaults to True.
            show_heatmaps: Optional; A boolean indicating if the resulting causal estimates should
                be displayed and saved in heatmap form. Defaults to True.
            show_validation: Optional; A boolean indicating if the resulting causal estimates
                should be compared to previous expectations. Defaults to True.
            show_largest_effects: Optional; A boolean indicating if the largest causal effects should
                be listed. Defaults to True.
            generate_pdf_report: Optional; A boolean indicating if the causal graph, heatmaps,
                validations and estimates should be written to files and combined into a pdf.
        """
        self._create_estimator()
        self._estimator.run_all_quick_analyses(estimand_types,
                                               verbose,
                                               show_tables,
                                               show_heatmaps,
                                               show_validation,
                                               show_largest_effects,
                                               generate_pdf_report,
                                               )

    def _create_estimator(self):
        self._estimator = estimator.Estimator.from_learner(self, same_data=True)


class StructureLearnerDatabricks(StructureLearner):
    """Main class for performing causal discovery on a Databricks cluster.

    Attributes:
        paths: A cause2e.PathManager managing paths and file names.
        data: A pandas.Dataframe containing the data.
        transformations: A list storing all the performed preprocessing transformations.
        variables: A set containing the names of all variables in the data.
        continuous: A set containing the names of all continuous variables in the data.
        discrete: A set containing the names of all discrete variables in the data.
        knowledge: A dictionary containing domain knowledge about required or forbidden edges in
            the causal graph. Known quantitative effects can be included for later validation.
        graph: A cause2e.Graph representing the causal graph.
        str_knowledge_graph: A string that is used to show the passed domain knowledge.
        str_graph: A string that is used to show the result of the graph search.
        str_report: A string that is used to show the pdf report.
    """

    def __init__(self, paths, spark, display_width=800, display_height=400):
        super().__init__(paths)
        self._spark = spark
        self._display_width = display_width
        self._display_height = display_height

    def show_knowledge(self):
        """Shows all domain knowledge that is used for causal discovery."""
        print("====================")
        if self.knowledge:
            print("Showing knowledge for graph search.\n")
            self.knowledge_graph.show()
            self._print_knowledge_display_instruction()
            print("Required edges:")
            for edge in self.knowledge['required']:
                print(edge)
            print("--------------------")
            print("Forbidden edges:")
            for edge in self.knowledge['forbidden']:
                print(edge)
        else:
            print("No knowledge has been provided.")
        print("====================")

    @staticmethod
    def _print_knowledge_display_instruction():
        """Prints an instruction for showing the knowledge on Databricks."""
        command = 'displayHTML(<name of StructureLearnerDatabricks>.str_knowledge_graph)'
        print(f"Run {command} to show the knowledge graph.")
        print("Save the knowledge and retry if no file is found.\n")

    @property
    def str_knowledge_graph(self):
        name = self.paths.create_knowledge_graph_name('png')
        return self._get_display_str(name)

    def _get_display_str(self, name):
        """Returns a string for 'display' on Databricks with suitable dimensions.

        Args:
            name: The absolute path to a file in the filestore.
        """
        src_str = self._get_src_str(name)
        return f"<img {src_str} {self._display_dimension_str}>"

    def _get_src_str(self, name):
        """Returns a path in the right format for 'display' on Databricks.

        Args:
            name: The absolute path to a file in the filestore.
        """
        modified_name = name.replace('/dbfs/FileStore', 'files')
        return f"src='{modified_name}'"

    @property
    def _display_dimension_str(self):
        return f"width='{self._display_width}' height='{self._display_height}'"

    def display_graph(self, edge_analysis=True):
        """Shows the causal graph.

        Args:
            edge_analysis: Optional; A boolean indicating if an analysis about the influence of
                domain knowledge on the resulting graph should be shown.
        """
        self.graph.show()
        self._print_graph_display_instruction()
        self.print_edge_analysis()

    @staticmethod
    def _print_graph_display_instruction():
        """Prints an instruction for showing the causal graph on Databricks."""
        command = 'displayHTML(<name of StructureLearnerDatabricks>.str_graph)'
        print(f"Run {command} to show the graph.")
        print("Save the graph with strict=False and retry if no file is found.\n")

    @property
    def str_graph(self):
        name = self._get_graph_name('png')
        return self._get_display_str(name)

    @property
    def str_report(self):
        return self._estimator.str_report

    def _create_estimator(self):
        self._estimator = estimator.EstimatorDatabricks.from_learner(self, same_data=True)
