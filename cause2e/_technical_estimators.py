"""
_technical_estimators.py
================================================================
This module implements several technical estimator classes.

The technical estimators are needed for estimating causal effects in the presence of categorical
or mixed data. This requires reducing the estimation to several binary or continuous estimations
and aggregating them. Cases with categorical confounders or mediators are not fully supported yet.
"""


from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import dowhy
from IPython.display import display


class _Estimator(ABC):
    """Abstract base class for technical estimator classes.

    Attributes:
        data: A pandas.Dataframe containing the data.
        model: A dowhy.causal_model.CausalModel that can identify, estimate and refute causal
            effects.
        estimand: A dowhy.causal_identifier.IdentifiedEstimand indicating the most recent estimand.
    """

    def __init__(self, data, model, estimand):
        self.data = data
        self.model = model
        self.estimand = estimand
        self._treatment = estimand.treatment_variable[0]
        self._outcome = estimand.outcome_variable[0]
        self._estimand_type = estimand.estimand_type

    @classmethod
    def from_estimator(cls, estimator):
        return cls(estimator.data, estimator.model, estimator.estimand)

    def _is_categorical(self, var_name):
        """Returns whether a data column is categorical.

        Args:
            var_name: A string indicating the name of the column of interest.
        """
        probe_sample = self.data[var_name].iloc[0]
        try:
            float(probe_sample)
            return False
        except ValueError:
            return True

    @abstractmethod
    def estimate_effect(self, verbose):
        pass


class EstimatorVariableTypes(_Estimator):
    """Technical estimator class for treatment and outcome of variable data types.

    Attributes:
        data: A pandas.Dataframe containing the data.
        model: A dowhy.causal_model.CausalModel that can identify, estimate and refute causal
            effects.
        estimand: A dowhy.causal_identifier.IdentifiedEstimand indicating the most recent estimand.
        estimated_effect: A dowhy.causal_estimator.CausalEstimate indicating the most recent
            estimated effect.
    """

    @property
    def estimated_effect(self):
        return self._estimator.estimated_effect

    def estimate_effect(self, verbose=True):
        """Returns an estimate of the causal effect.

        Args:
            verbose: Optional; A boolean indicating if additional information should be displayed.
                Defaults to True.
        """
        self._estimator = self._choose_estimator()
        return self._estimator.estimate_effect(verbose)

    def _choose_estimator(self):
        """Chooses the right estimator based on treatment and outcome data types."""
        categorical_treatment = self._is_categorical(self._treatment)
        categorical_outcome = self._is_categorical(self._outcome)
        if not categorical_treatment and not categorical_outcome:
            return _EstimatorNumeric.from_estimator(self)
        elif categorical_treatment ^ categorical_outcome:
            return _EstimatorMixed.from_estimator(self)
        elif categorical_treatment and categorical_outcome:
            return _EstimatorCategorical.from_estimator(self)


class _EstimatorCategorical(_Estimator):
    """Technical estimator class for categorical treatment and outcome.

    Attributes:
        data: A pandas.Dataframe containing the data.
        model: A dowhy.causal_model.CausalModel that can identify, estimate and refute causal
            effects.
        estimand: A dowhy.causal_identifier.IdentifiedEstimand indicating the most recent estimand.
        estimated_effect: A dowhy.causal_estimator.CausalEstimate indicating the most recent
            estimated effect.
    """

    def __init__(self, data, model, estimand):
        super().__init__(data, model, estimand)
        self._prepare_levels()

    def _prepare_levels(self):
        """Gets the unique levels of treatment and outcome and their number."""
        self._treatment_levels = self.data[self._treatment].unique()
        self._n_treatments = len(self._treatment_levels)
        self._outcome_levels = self.data[self._outcome].unique()
        self._n_outcomes = len(self._outcome_levels)

    @classmethod
    def from_estimator(cls, estimator):
        return cls(estimator.data, estimator.model, estimator.estimand)

    @property
    def estimated_effect(self):
        return self._estimator_mixed.estimated_effect

    def estimate_effect(self, verbose=True):
        """Returns an estimate of the aggregated causal effect.

        Args:
            verbose: Optional; A boolean indicating if additional information should be displayed.
                Defaults to True.
        """
        self._estimate_effects(aggregation_methods=['maximum'], verbose=verbose)
        return self._aggregated_effects['maximum']

    def _estimate_effects(self, aggregation_methods=['mean', 'maximum', 'minimum'], verbose=True):
        """Calculates aggregated estimates of the causal effect.

        Args:
            aggregation_methods: A list of strings indicating the aggregation methods.
            verbose: Optional; A boolean indicating if additional information should be displayed.
                Defaults to True.
        """
        self._create_effects_array()
        self._aggregated_effects = {method: _calculate_aggregated_effect(self._effects_arr, method)
                                    for method in aggregation_methods}
        if verbose:
            self._print_aggregated_effect_info(aggregation_methods)

    def _create_effects_array(self):
        """Calculates estimates of the causal effect for all combinations of treatment and outcome value pairs."""
        self._effects_arr = np.zeros((self._n_treatments, self._n_treatments, self._n_outcomes, self._n_outcomes))
        for i, zero_val in enumerate(self._outcome_levels):
            for j, one_val in enumerate(self._outcome_levels):
                self._calculate_effects_array_submatrix(i, j, one_val, zero_val)

    def _calculate_effects_array_submatrix(self, i, j, one_val, zero_val):
        """Calculates estimates of the causal effect for all combinations of outcome value pairs.

        Args:
            i: An integer used for storing the result.
            j: An integer used for storing the result.
            one_val: The value of the outcome variable that should be treated as 1.
            zero_val: The value of the outcome variable that should be treated as 0.
        """
        if one_val == zero_val:
            arr = np.empty((self._n_treatments))
            arr[:] = np.nan
            self._effects_arr[:, :, i, j] = arr
        else:
            filtered_data = self._transform_categorical_column(one_val, zero_val)
            self._estimator_mixed = _EstimatorMixed(filtered_data, self.model, self.estimand)
            self._estimator_mixed.create_effects_array()
            self._effects_arr[:, :, i, j] = self._estimator_mixed.effects_arr

    def _transform_categorical_column(self, one_val, zero_val):
        """Transforms a categorical data column into a binary column.

        Args:
            one_val: A string indicating the value that should be translated to 1.
            zero_val: A string indicating the value that should be translated to 0.
        """
        translation_dict = {one_val: 1, zero_val: 0}
        filtered_data = self.data.copy()
        filtered_data[self._outcome] = filtered_data[self._outcome].map(translation_dict)
        filtered_data.dropna(axis=0, inplace=True)
        return filtered_data

    def _print_aggregated_effect_info(self, methods):
        """Prints an interpretation of the estimated effects.

        Args:
            methods: A list of strings indicating the used aggregation methods.
        """
        for method in methods:
            effect = '{:,.2f}'.format(self._aggregated_effects[method])
            print(f"The {method} absolute {self._estimand_type} of {self._treatment} on {self._outcome} over " +
                  f"all combinations of zero and one values is {effect}.")
        print("\n")


class _EstimatorMixed(_Estimator):
    """Technical estimator class for numeric treatment and categorical outcome (or the other way around).

    Attributes:
        data: A pandas.Dataframe containing the data.
        model: A dowhy.causal_model.CausalModel that can identify, estimate and refute causal
            effects.
        estimand: A dowhy.causal_identifier.IdentifiedEstimand indicating the most recent estimand.
        effects_arr: A numpy array containing estimates of the causal effect for all combinations
            of treatment or outcome value pairs.
        estimated_effect: A dowhy.causal_estimator.CausalEstimate indicating the most recent
            estimated effect.
    """

    def __init__(self, data, model, estimand):
        super().__init__(data, model, estimand)
        self._prepare_levels()

    def _prepare_levels(self):
        """Gets the unique levels of treatment or outcome and their number."""
        self._categorical_treatment = self._is_categorical(self._treatment)
        if self._categorical_treatment:
            self._levels = self.data[self._treatment].unique()
        else:
            self._levels = self.data[self._outcome].unique()

    @classmethod
    def from_estimator(cls, estimator):
        return cls(estimator.data, estimator.model, estimator.estimand)

    @property
    def estimated_effect(self):
        return self._estimator_numeric.estimated_effect

    def estimate_effect(self, verbose=True):
        """Returns an estimate of the aggregated causal effect.

        Args:
            verbose: Optional; A boolean indicating if additional information should be displayed.
                Defaults to True.
        """
        self._estimate_effects(aggregation_methods=['maximum'], verbose=verbose)
        return self._aggregated_effects['maximum']

    def _estimate_effects(self, aggregation_methods=['mean', 'maximum', 'minimum'], verbose=True):
        """Calculates aggregated estimates of the causal effect.

        Args:
            aggregation_methods: A list of strings indicating the aggregation methods.
            verbose: Optional; A boolean indicating if additional information should be displayed.
                Defaults to True.
        """
        self.create_effects_array()
        self._aggregated_effects = {method: _calculate_aggregated_effect(self.effects_arr, method)
                                    for method in aggregation_methods}
        self._create_effects_df()
        if verbose:
            self._print_pretty_effects_df()
            self._print_aggregated_effect_info(aggregation_methods)

    def create_effects_array(self):
        """Calculates estimates of the causal effect for all combinations of treatment or outcome value pairs."""
        n_levels = len(self._levels)
        self.effects_arr = np.zeros((n_levels, n_levels))
        for i, zero_val in enumerate(self._levels):
            for j, one_val in enumerate(self._levels):
                self._calculate_effects_array_entry(i, j, one_val, zero_val)

    def _calculate_effects_array_entry(self, i, j, one_val, zero_val):
        """Calculates an estimate of the causal effect for given treatment or outcome value pairs.

        Args:
            i: An integer used for storing the result.
            j: An integer used for storing the result.
            one_val: The value of the treatment or outcome variable that should be treated as 1.
            zero_val: The value of the treatment or outcome variable that should be treated as 0.
        """
        if i < j:
            if self._categorical_treatment:
                self._estimator_numeric = _EstimatorNumeric.from_estimator(self,
                                                                           treatment_val=one_val,
                                                                           control_val=zero_val,
                                                                           )
            else:
                self._estimator_numeric = _EstimatorNumeric.from_estimator(self,
                                                                           one_val=one_val,
                                                                           zero_val=zero_val,
                                                                           )
            self.effects_arr[i, j] = self._estimator_numeric.estimate_effect(verbose=False)
        elif i > j:
            self.effects_arr[i, j] = -1 * self.effects_arr[j, i]
        else:
            self.effects_arr[i, j] = float('NaN')

    def _create_effects_df(self):
        """Creates a dataframe containing causal estimates for all combinations of treatment or outcome value pairs."""
        effects_df = pd.DataFrame(self.effects_arr, columns=self._levels, index=self._levels)
        effects_df = effects_df.applymap('{:,.2f}'.format)
        if self._categorical_treatment:
            effects_df.columns.name = 'One Value'
            effects_df.index.name = 'Zero Value'
        else:
            effects_df.columns.name = 'Treatment Value'
            effects_df.index.name = 'Control Value'
        self._effects_df = effects_df

    def _print_pretty_effects_df(self):
        """Prints the causal estimates in a dataframe with appropriate title."""
        caption = f"{self._estimand_type} of {self._treatment} on {self._outcome}"
        effects_style = self._effects_df.style.set_caption(caption)
        display(effects_style)

    def _print_aggregated_effect_info(self, methods):
        """Prints an interpretation of the estimated effects.

        Args:
            methods: A list of strings indicating the used aggregation methods.
        """
        for method in methods:
            if self._categorical_treatment:
                levels_names = "control and treatment"
            else:
                levels_names = "zero and one"
            effect = '{:,.2f}'.format(self._aggregated_effects[method])
            print(f"The {method} absolute {self._estimand_type} of {self._treatment} on {self._outcome} over " +
                  f"all combinations of {levels_names} values is {effect}.")
        print("\n")


class _EstimatorNumeric(_Estimator):
    """Technical estimator class for numeric treatment and outcome.

    Attributes:
        data: A pandas.Dataframe containing the data.
        model: A dowhy.causal_model.CausalModel that can identify, estimate and refute causal
            effects.
        estimand: A dowhy.causal_identifier.IdentifiedEstimand indicating the most recent estimand.
        estimated_effect: A dowhy.causal_estimator.CausalEstimate indicating the most recent
            estimated effect.
    """
    def __init__(self,
                 data,
                 model,
                 estimand,
                 estimation_method=None,
                 treatment_val=None,
                 control_val=None,
                 one_val=None,
                 zero_val=None,
                 ):
        super().__init__(data, model, estimand)
        self._treatment_val = treatment_val
        self._control_val = control_val
        self._one_val = one_val
        self._zero_val = zero_val
        self._prepare_data(data)
        self._prepare_estimation_details(estimation_method)

    def _prepare_data(self, data):
        """Prepares data for causal estimation by filtering out undesirable values.

        Args:
            data: A pandas.DataFrame containing the data.
        """
        self.data = data.copy()
        self._backup_data = data.copy()
        if self._treatment_val:
            self._transform_categorical_column(self._treatment, self._treatment_val, self._control_val)
        if self._zero_val:
            self._transform_categorical_column(self._outcome, self._one_val, self._zero_val)
        self.model._data = self.data

    def _transform_categorical_column(self, var_name, one_val, zero_val):
        """Transforms a categorical data column into a binary column.

        Args:
            var_name: A string indicating the name of the column.
            one_val: A string indicating the value that should be translated to 1.
            zero_val: A string indicating the value that should be translated to 0.
        """
        translation_dict = {one_val: 1, zero_val: 0}
        self.data[var_name] = self.data[var_name].map(translation_dict)
        self.data.dropna(axis=0, inplace=True)

    def _prepare_estimation_details(self, estimation_method):
        """Prepares the methodic details for a linear causal estimation with DoWhy.

        Args:
            estimation_method: A string indicating the preferred estimation method.

        Raises:
            KeyError: 'estimand_type must be nonparametric-ate, nonparametric-nde
                or nonparametric-nie'
        """
        if estimation_method:
            self._estimation_method = estimation_method
            self._estimation_kwargs = {}
        elif self._estimand_type == 'nonparametric-ate':
            self._estimation_method = 'backdoor.linear_regression'
            self._estimation_kwargs = {}
        elif self._estimand_type in ['nonparametric-nde', 'nonparametric-nie']:
            self._estimation_method = 'mediation.two_stage_regression'
            fsm = dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
            ssm = dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
            method_params = {'first_stage_model': fsm,
                             'second_stage_model': ssm,
                             }
            self._estimation_kwargs = {'confidence_intervals': False,
                                       'test_significance': False,
                                       'method_params': method_params,
                                       }
        else:
            raise KeyError("estimand_type must be nonparametric-ate, nonparametric-nde or "
                           + "nonparametric-nie")

    @classmethod
    def from_estimator(cls, estimator, **kwargs):
        return cls(estimator.data, estimator.model, estimator.estimand, **kwargs)

    def estimate_effect(self, verbose=True):
        """Estimates the causal effect from the statistical estimand and the data.

        Args:
            method_name: The name of the estimation method to be used.
            verbose: Optional; A boolean indicating if verbose output should be displayed. Defaults
                to True.
            **kwargs: Advanced parameters for the analysis. Please refer to
                https://microsoft.github.io/dowhy/ for more information.
        """
        self.estimated_effect = self.model.estimate_effect(self.estimand,
                                                           self._estimation_method,
                                                           **self._estimation_kwargs,
                                                           )
        self.model._data = self._backup_data
        return self.estimated_effect.value

    def interprete_estimated_effect(self):
        """Prints a human-readable interpretation of the estimated causal effect."""
        start = f"{self._estimand_type} of "
        if self._treatment_val:
            treatment_str = f"changing {self._treatment} from {self._control_val} to {self._treatment_val} on "
        else:
            treatment_str = f"{self._treatment} on "
        if self._one_val:
            outcome_str = f"changing {self._outcome} from {self._zero_val} to {self._one_val}"
        else:
            outcome_str = f"{self._outcome}"
        end = f": {'{:,.2f}'.format(self.effect)}."
        print(start + treatment_str + outcome_str + end)


def _calculate_aggregated_effect(arr, method='maximum'):
    """Returns an aggregated value from a numerical array.

    Args:
        arr: A numpy array that should be aggregated into one value.
        method: Optional; A string indicating the aggregation method. Defaults to 'maximum'.
    """
    if method == 'mean':
        return np.nanmean(abs(arr))
    if method == 'maximum':
        return np.nanmax(abs(arr))
    if method == 'minimum':
        return np.nanmin(abs(arr))
