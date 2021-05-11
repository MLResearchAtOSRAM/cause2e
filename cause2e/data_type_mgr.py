"""
data_type_mgr.py
================================================================
This module implements the DataTypeManager class.

It is used to communicate the data types of the variables (continuous or discrete) to a TETRAD
causal discovery algorithm. These algorithms require a hard threshold on the number of unique
values of a variable to determine its type, which can lead to problems if there are continuous
variables with few unique values. Therefore, we allow explicitly stating the desired data types
and add small Gaussian noise to ensure a sufficient number of unique values for all continuous
variables.
"""


import numpy as np


class DataTypeManager():
    """Main class for enforcing desired data types.

    Attributes:
        data: A pandas.Dataframe containing the data.
        variables: A set containing the names of all variables in the data.
        desired_continuous: A set containing the names of all variables that should be continuous.
        desired_discrete: A set containing the names of all variables that should be discrete.
        current_continuous: A set containing the names of all variables that are considered
            continuous according to the current threshold.
        current_discrete: A set containing the names of all variables that are considered
            discrete according to the current threshold.
        threshold: An integer such that all continuous variables have more unique values
            and all discrete variables have fewer unique values than the threshold.
    """

    def __init__(self, data, desired_continuous=set(), desired_discrete=set()):
        """Inits DataTypeManager."""
        self.data = data
        self.desired_continuous = desired_continuous
        self.desired_discrete = desired_discrete
        self.threshold = 0

    @property
    def variables(self):
        return set(self.data.columns)

    @property
    def current_continuous(self):
        return {var for var in self.variables if self.above_threshold(var)}

    def above_threshold(self, var):
        """Checks if a variable has more unique values than the current threshold.

        Args:
            var: A string indicating the name of the variable under consideration.

        Returns:
            A boolean indicating if the variable has more unique values than the current threshold.
        """
        return self.data[var].nunique() > self.threshold

    @property
    def current_discrete(self):
        return self.variables - self.current_continuous

    def enforce_desired_types(self, complete=True, verbose=False):
        """Ensures that all variables are typed correctly.

        Args:
            complete: Optional; A boolean indicating if all variables' desired types must be stated
                explicitly. Otherwise, subsets are ok.
            verbose: Optional; a boolean indicating if the resulting variable types and the
                threshold should be printed. Defaults to False.
        """
        self._check_type_input(complete)
        self._set_type_threshold()
        self._enforce_continuous()
        if verbose:
            print("Continuous:\n")
            print(self.current_continuous)
            print("Discrete:\n")
            print(self.current_discrete)
            print(f"Threshold: {self.threshold}")
        self._check_type_threshold()

    def _check_type_input(self, complete=True):
        """Checks if the user's input about the desired types is acceptable.

        Args:
            complete: Optional; A boolean indicating if all variables' desired types must be stated
                explicitly. Otherwise, subsets are ok.
        """
        assert(self.desired_discrete or self.desired_continuous)
        assert(self.desired_discrete.issubset(self.variables))
        assert(self.desired_continuous.issubset(self.variables))
        assert(self.desired_discrete.isdisjoint(self.desired_continuous))
        if complete:
            assert(self.desired_continuous | self.desired_discrete == self.variables)

    def _set_type_threshold(self):
        """Sets a threshold that ensures that all variables are typed correctly."""
        if self.desired_discrete:
            unique_vals_discrete = self.data[self.desired_discrete].nunique()
            most_distinct_values_discrete = max(unique_vals_discrete)
        else:
            most_distinct_values_discrete = 1
        self.threshold = most_distinct_values_discrete + 1

    def _enforce_continuous(self):
        """Ensures that all continuous variables have enough unique values."""
        for var in self.desired_continuous - self.current_continuous:
            self._add_noise(var)

    def _add_noise(self, var):
        """Adds very weak Gaussian noise to a variable.

        Args:
            var: A string indicating the name of the variable.
        """
        n_samples = len(self.data)
        variance = self.data[var].var() / 10000
        self.data[var] += np.random.normal(0, variance, n_samples)

    def _check_type_threshold(self):
        """Checks if the threshold leads to the desired variable types."""
        assert self.desired_continuous.issubset(self.current_continuous)
        assert self.desired_discrete.issubset(self.current_discrete)
