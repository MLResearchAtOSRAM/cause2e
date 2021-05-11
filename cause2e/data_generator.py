"""
data_generator.py
================================================================
This module implements the DataGenerator class.

It is used to generate synthetic data from a known causal model. This allows testing our
causal discovery and causal estimation algorithms before applying them to real data where we do
not necessarlily know the underlying model and therefore cannot reliably assess the performance.
Currently, only a simple linear model from the DoWhy library is supported.
"""


from dowhy import datasets
import pydot
from IPython.display import Image, display


class DataGenerator:
    """Main class for generating synthetic data.

    Attributes:
        paths: A cause2e.PathManager managing paths and file names.
        data: A pandas.Dataframe containing the data.
        ate: A float indicating the average treatment effect.
            This is what we want to estimate in the subsequent causal analysis.
    """
    def __init__(self, paths=None):
        """Inits the DataGenerator."""
        self.paths = paths

    def generate_linear_dataset(self, beta, n_common_causes, nrows, **kwargs):
        """Generates a linear dataset.

        Args:
            beta: A float describing the strength of the causal effect.
            n_common_causes: An integer indicating the number of common causes that treatment
                variable and outcome variable share.
            nrows: An integer indicating the number of samples to be generated.
        """
        dic = datasets.linear_dataset(beta, n_common_causes, nrows, **kwargs)
        self.data = dic['df']
        self._dot_str = dic['dot_graph']
        self.ate = dic['ate']

    def display_graph(self):
        """Displays the causal graph that was used to generate the data."""
        # TODO: should be handled by graph module
        graphs = pydot.graph_from_dot_data(self._dot_str)
        self._dot = graphs[0]
        self._png = self._dot.create_png(prog='dot')
        print('True causal graph:\n')
        display(Image(self._png))

    def write_csv(self):
        """Writes the generated data to a csv file."""
        self.data.to_csv(self.paths.full_data_name)
