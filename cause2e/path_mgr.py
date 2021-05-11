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
        experiment_name: A string indicating the name of the current experiment.
            This could e.g. contain the name of the used causal discovery algorithm.
        data_name: A string indicating the name of the data file (without path).
        data_dir: A string indicating the path to the directory where the data is located.
        output_dir: A string indicating the path to the directory where the output is written.
        full_data_name: A string indicating the full path to the data_file.
        png_name: Name of the png file where the graph will be stored.
        svg_name: Name of the svg file where the graph will be stored.
        dot_name: Name of the dot file where the graph will be stored.
    """

    def __init__(self, experiment_name, data_name, data_dir, output_dir):
        """Inits PathManager."""
        self.experiment_name = experiment_name
        self.data_name = data_name
        self.data_dir = data_dir
        self.output_dir = output_dir

    @property
    def full_data_name(self):
        return os.path.join(self.data_dir, self.data_name)

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
        file_name = self.experiment_name + additions + '.' + file_extension
        return os.path.join(self.output_dir, file_name)


class PathManagerQuick(PathManager):
    """A subclass of the PathManager that assumes that data directory and output directory
        reside in a common directory.

    Args:
        See PathManager.
    """

    def __init__(self, experiment_name, data_name, directory):
        """Inits PathManagerQuick."""
        data_dir = os.path.join(directory, 'data')
        output_dir = os.path.join(directory, 'output')
        super().__init__(experiment_name, data_name, data_dir, output_dir)
