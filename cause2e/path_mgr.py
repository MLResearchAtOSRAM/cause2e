"""
path_mgr.py
================================================================
This module implements the PathManager and PathManagerQuick classes.

It is used to handle all the navigation to the right directories
and files for reading inputs and writing outputs of a causal analysis.
"""


import os


class PathManager():
    """Main class for path management.

    Attributes:
        full_data_name: A string indicating the full path to the data_file.
        png_name: Name of the png file where the graph will be stored.
        svg_name: Name of the svg file where the graph will be stored.
        dot_name: Name of the dot file where the graph will be stored.
    """

    def __init__(self, experiment_name, data_name, data_dir, output_dir):
        """Inits PathManager."""
        self._experiment_name = experiment_name
        self._data_name = data_name
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._find_or_create_output_dir()

    def _find_or_create_output_dir(self):
        """Creates the output directory if it does not exist yet."""
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    @property
    def full_data_name(self):
        return os.path.join(self._data_dir, self._data_name)

    @property
    def png_name(self):
        return self.create_output_name('png')

    @property
    def svg_name(self):
        return self.create_output_name('svg')

    @property
    def dot_name(self):
        return self.create_output_name('dot')

    def create_output_name(self, file_extension, additions=''):
        """Returns the name for a file where some result will be stored.

        An example would be storing the causal graph.

        Args:
            file_extension: A string indicating the desired file extension.
            additions: Optional; A string indicating further information to be added to the name.
                Defaults to ''.
        """
        file_name = self._experiment_name + additions + '.' + file_extension
        return os.path.join(self._output_dir, file_name)

    def create_reporting_paths(self):
        """Returns the names of pngs needed for reporting."""
        output_name = self.create_output_name('pdf', '_report')
        knowledge_graph = self.create_knowledge_graph_name()
        graph = self.png_name  # TODO: What if additions were made to the default name?
        edge_analysis = self._create_edge_analysis_name()
        graphs = [knowledge_graph, graph, edge_analysis]
        estimand_types = ['ate', 'nde', 'nie']
        heatmaps = [self._create_heatmap_name(x) for x in estimand_types]
        validations = [self._create_validation_name(x) for x in ['True', 'False']]
        largest_effects = [self._create_largest_effects_name(x) for x in estimand_types]
        results = [self._create_result_name(x) for x in estimand_types]
        input_names = graphs + heatmaps + validations + largest_effects + results
        return output_name, input_names

    def create_knowledge_graph_name(self, file_extension='png'):
        return self.create_output_name(file_extension, '_knowledge_graph')

    def _create_edge_analysis_name(self):
        return self.create_output_name('png', '_edge_analysis')

    def _create_heatmap_name(self, estimand_type):
        return self.create_output_name('png', f'_heatmap_{estimand_type}')

    def _create_validation_name(self, valid):
        return self.create_output_name('png', f'_validation_{valid}')

    def _create_largest_effects_name(self, estimand_type):
        return self.create_output_name('png', f'_largest_effects_{estimand_type}')

    def _create_result_name(self, estimand_type):
        return self.create_output_name('png', f'_results_{estimand_type}')

    @property
    def output_name_stump(self):
        """Returns an output name without file extension."""
        return self.create_output_name("")[:-1]


class PathManagerQuick(PathManager):
    """A subclass of the PathManager that assumes that data directory and output directory
        reside in a common directory.

    Attributes:
        See PathManager.
    """

    def __init__(self, experiment_name, data_name, directory, nested_output=True):
        """Inits PathManagerQuick."""
        data_dir = os.path.join(directory, 'data')
        if nested_output:
            output_dir = os.path.join(directory, 'output', experiment_name)
        else:
            output_dir = os.path.join(directory, 'output')
        super().__init__(experiment_name, data_name, data_dir, output_dir)
