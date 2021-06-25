"""
_regression_mgr.py
================================================================
This module implements the RegressionManager class.

It is used as a helper class for comparing the output of a causal estimation to a noncausal
regression analysis.
"""


import numpy as np
from sklearn.linear_model import LinearRegression


class RegressionManager:
    """Helper class for comparing the causal effect estimate to a noncausal regression estimate.

    Attributes:
        quick_results: A pandas.DataFrame storing the results of calls to the quick_analysis method.
    """

    def __init__(self, data, treatment, outcome, causal_effect):
        self._data = data
        self._treatment = treatment
        self._outcome = outcome
        self._causal_effect = causal_effect

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
        y = np.array(self._data[self._outcome]).reshape(-1, 1)
        return X, y, input_df.columns

    def _get_input_df(self, input_cols, drop_cols):
        """Returns a dataframe containing only specified columns.

        Args:
            input_cols: A set of columns to be used in the linear regression.
            drop_cols: Optional; A boolean indicating if input_cols should indicate which columns
                to drop instead of which columns to use. Defaults to False.
        """
        if drop_cols:
            return self._data.drop(columns=input_cols, axis=1)
        else:
            return self._data[input_cols]

    def _print_regression_results(self, X, y, col_names, reg):
        """Prints the results of comparing the causal estimate to the linear regression estimate.

        Args:
            X: A numpy array containing the input data for the linear regression.
            y: A numpy array containing the target data for the linear regression.
            col_names: A list containing the names of the columns that are used in the linear
                regression.
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
        msg_part = f"estimate for effect of {self._treatment} on {self._outcome}:"
        print(f"Causal {msg_part} {self._causal_effect}")
        print(f"Regression {msg_part} {coefs[self._treatment]}")
