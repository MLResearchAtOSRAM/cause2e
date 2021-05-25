"""
estimator.py
================================================================
This module implements the Estimator class.

It is used to estimate causal effects from the data and the causal graph.
The module contains a wrapper around the core functionality of the DoWhy library and some utiliy
methods for transitioning to the estimation phase after a causal discovery phase.
The proposed procedure is as follows:\n

1) Imitate the preprocessing steps that have been applied to the data before the causal discovery.\n
2) Create a causal model from the data, the causal graph and the desired cause-effect pair.\n
3) Use do-calculus to algebraically identify a statistical estimand for the desired causal effect.\n
4) Estimate the estimand from the data.\n
5) Check the robustness of the estimate.\n

For more information about steps 2-5, please refer to https://microsoft.github.io/dowhy/.
"""


import dowhy
from cause2e import preproc, reader
import numpy as np
from sklearn.linear_model import LinearRegression


class Estimator():
    """Main class for estimating causal effects.

    Attributes:
        paths: A cause2e.PathManager managing paths and file names.
        reader: A cause2e.Reader for reading the data.
        data: A pandas.Dataframe containing the data.
        transformations: A list storing all the performed preprocessing transformations. Ensures
            that the data fits the causal graph.
        variables: A set containing the names of all variables in the data.
        model: A dowhy.causal_model.CausalModel that can identify, estimate and refute causal
            effects.
        spark: Optional; A pyspark.sql.SparkSession in case you want to use spark. Defaults to
            None.
    """

    def __init__(self, paths, transformations, spark=None):
        """Inits CausalEstimator"""
        self.paths = paths
        self.transformations = transformations.copy()
        self.spark = spark

    @property
    def reader(self):
        return reader.Reader(self.paths.full_data_name,
                             self.spark
                             )

    def read_csv(self, **kwargs):
        """Reads data from a csv file."""
        self.data = self.reader.read_csv(**kwargs)

    def read_parquet(self, **kwargs):
        """Reads data rom a parquet file."""
        self.data = self.reader.read_parquet(**kwargs)

    @property
    def variables(self):
        return set(self.data.columns)

    def imitate_data_trafos(self):
        """Imitates all the preprocessing steps applied before causal discovery."""
        self._ensure_preprocessor()
        self.preprocessor.apply_stored_transformations(self.transformations)

    def _ensure_preprocessor(self):
        """Ensures that a preprocessor is initialized."""
        if not hasattr(self, 'preprocessor'):
            self.preprocessor = preproc.Preprocessor(self.data, self.transformations)

    @property
    def _dot_name(self):
        return self.paths.dot_name

    def initialize_model(self, treatment, outcome, **kwargs):
        """Initializes the causal model.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            **kwargs: Advanced parameters for the analysis. Please refer to
                https://microsoft.github.io/dowhy/ for more information.
        """
        # self._check_graph()
        self.model = dowhy.CausalModel(data=self.data,
                                       treatment=treatment,
                                       outcome=outcome,
                                       graph=self._dot_name,
                                       **kwargs
                                       )
        self.treatment = treatment
        self.outcome = outcome

    def identify_estimand(self, verbose=True, **kwargs):
        """Algebraically identifies a statistical estimand for the causal effect from the graph.

        Args:
            verbose: Optional; A boolean indicating if verbose output should be displayed. Defaults
                to True.
            **kwargs: Advanced parameters for the analysis. Please refer to
                https://microsoft.github.io/dowhy/ for more information.
        """
        self.estimand = self.model.identify_effect(**kwargs)
        if verbose:
            print(self.estimand)

    def estimate_effect(self, method_name, verbose=True, **kwargs):
        """Estimates the causal effect from the statistical estimand and the data.

        Args:
            method_name: The name of the estimation method to be used.
            verbose: Optional; A boolean indicating if verbose output should be displayed. Defaults
                to True.
            **kwargs: Advanced parameters for the analysis. Please refer to
                https://microsoft.github.io/dowhy/ for more information.
        """
        self.estimated_effect = self.model.estimate_effect(self.estimand,
                                                           method_name,
                                                           **kwargs
                                                           )
        if verbose:
            print(self.estimated_effect)

    def check_robustness(self, method_name, verbose=True, **kwargs):
        """Checks the robustness of the estimated causal effects.

        Args:
            method_name: The name of the robustness check to be used.
            verbose: Optional; A boolean indicating if verbose output should be displayed. Defaults
                to True.
            **kwargs: Advanced parameters for the analysis. Please refer to
                https://microsoft.github.io/dowhy/ for more information.
        """
        self.robustness_info = self.model.refute_estimate(self.estimand,
                                                          self.estimated_effect,
                                                          method_name,
                                                          **kwargs
                                                          )
        if verbose:
            print(self.robustness_info)

    def compare_to_noncausal_regression(self, input_cols, drop_cols=False):
        """Prints a comparison of the causal estimate to a noncausal linear regression estimate.

        Args:
            input_cols: A set of columns to be used in the linear regression.
            drop_cols: Optional; A boolean indicating if input_cols should indicate which columns
                to drop instead of which columns to use. Defaults to False.
        """
        X, y, col_names = self._get_regression_input(input_cols, drop_cols)
        reg = LinearRegression(normalize=True).fit(X, y)
        self._print_regression_results(X, y, col_names, reg)

    def _get_regression_input(self, input_cols, drop_cols):
        """Gets the data in the right format for the linear regression.

        Args:
            input_cols: A set of columns to be used in the linear regression.
            drop_cols: Optional; A boolean indicating if input_cols should indicate which columns
                to drop instead of which columns to use. Defaults to False.

        Returns:
            A numpy array containing the input data for the linear regression.
            A numpy array containing the target data for the linear regression.
            A list containing the names of the columns that are used in the linear regression.
        """
        input_df = self._get_input_df(input_cols, drop_cols)
        if input_df.shape[1] == 1:
            X = np.array(input_df).reshape(-1, 1)
        else:
            X = np.array(input_df)
        y = np.array(self.data[self.outcome]).reshape(-1, 1)
        return X, y, input_df.columns

    def _get_input_df(self, input_cols, drop_cols):
        """Returns a dataframe containing only specified columns.

        Args:
            input_cols: A set of columns to be used in the linear regression.
            drop_cols: Optional; A boolean indicating if input_cols should indicate which columns
                to drop instead of which columns to use. Defaults to False.
        """
        if drop_cols:
            return self.data.drop(columns=input_cols, axis=1)
        else:
            return self.data[input_cols]

    def _print_regression_results(self, X, y, col_names, reg):
        """Prints the results of comparing the causal estimate to the linear regression estimate.

        Args:
            X: A numpy array containing the input data for the linear regression.
            y: A numpy array containing the target data for the linear regression.
            col_names: A list containing the names of the columns that are used in the linear regression.
            reg: A fitted linear regression object from sklearn.
        """
        coefs = {name: coef[1] for name, coef in zip(col_names, np.ndenumerate(reg.coef_))}
        print(f"Regression score: {reg.score(X, y)}")
        print("--------------------------------")
        print("Regression coefficients:")
        for it in coefs.items():
            print(it)
        print(f"Intercept: {reg.intercept_}")
        print("--------------------------------")
        msg_part = f"estimate for effect of {self.treatment} on {self.outcome}:"
        print(f"Causal {msg_part} {self.estimated_effect.value}")
        print(f"Regression {msg_part} {coefs[self.treatment]}")
