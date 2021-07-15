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
from cause2e import _preproc, _reader, _result_mgr, _regression_mgr


class Estimator():
    """Main class for estimating causal effects.

    Attributes:
        paths: A cause2e.PathManager managing paths and file names.
        data: A pandas.Dataframe containing the data.
        transformations: A list storing all the performed preprocessing transformations. Ensures
            that the data fits the causal graph.
        variables: A set containing the names of all variables in the data.
        model: A dowhy.causal_model.CausalModel that can identify, estimate and refute causal
            effects.
        treatment: A string indicating the most recent treatment variable.
        outcome: A string indicating the most recent outcome variable.
        estimand_type: A string indicating the most recent type of causal effect.
        estimand: A dowhy.causal_identifier.IdentifiedEstimand indicating the most recent estimand.
        estimated_effect: A dowhy.causal_estimator.CausalEstimate indicating the most recent
            estimated effect.
        robustness_info: A dowhy.causal_refuter.CausalRefutation indicating the results of the most
            recent robustness check.
        spark: Optional; A pyspark.sql.SparkSession in case you want to use spark. Defaults to
            None.
    """

    def __init__(self, paths, transformations=[], validation_dict={}, spark=None):
        """Inits CausalEstimator"""
        self.paths = paths
        self.transformations = transformations.copy()
        self.spark = spark
        self._quick_results_list = []
        self._validation_dict = validation_dict

    @classmethod
    def from_learner(cls, learner, same_data=False):
        if learner.knowledge:
            validation_dict = learner.knowledge['expected_effects']
        else:
            validation_dict = {}
        if same_data:
            estim = cls(paths=learner.paths,
                        validation_dict=validation_dict,
                        spark=learner.spark)
            estim.data = learner.data
            return estim
        else:
            return cls(paths=learner.paths,
                       transformations=learner.transformations,
                       validation_dict=validation_dict,
                       spark=learner.spark)


    @property
    def _reader(self):
        return _reader.Reader(self.paths.full_data_name,
                              self.spark
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

    def imitate_data_trafos(self, vals_list=None):
        """Imitates all the preprocessing steps applied before causal discovery.
        
        Args:
            vals_list: A list containing one column of values for each 'add_variable' step in the
                transformations. Defaults to None.
        """
        self._ensure_preprocessor()
        self._preprocessor.apply_stored_transformations(self.transformations, vals_list)
        
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
            self._preprocessor = _preproc.Preprocessor(self.data, self.transformations)

    @property
    def _dot_name(self):
        return self.paths.dot_name

    # TODO: Cache this?
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
            show_validation: Optional; A boolean indicating if the resulting causal estimates should
                be compared to previous expectations. Defaults to True.
            show_largest_effects: Optional; A boolean indicating if the largest causal effects should
                be listed. Defaults to True.
            generate_pdf_report: Optional; A boolean indicating if the causal graph, heatmaps,
                validations and estimates should be written to files and combined into a pdf.
        """
        vars = self.variables
        self.run_multiple_quick_analyses(vars, vars, estimand_types, verbose, show_tables, show_heatmaps,
                                         show_validation, generate_pdf_report)

    def run_multiple_quick_analyses(self,
                                    treatments,
                                    outcomes,
                                    estimand_types,
                                    verbose=False,
                                    show_tables=True,
                                    show_heatmaps=True,
                                    show_validation=True,
                                    show_largest_effects=True,
                                    generate_pdf_report=True):
        """Performs multiple quick causal analyses with preset parameters.

        Args:
            treatments: A list of strings indicating the names of the treatment variables.
            outcomes: A list of strings indicating the names of the outcome variables.
            estimand_types: A list of strings indicating the types of causal effects.
            verbose: Optional; A boolean indicating if verbose output should be displayed for each
                analysis. Defaults to False.
            show_tables: Optional; A boolean indicating if the resulting causal estimates should be
                displayed in tabular form. Defaults to True.
            show_heatmaps: Optional; A boolean indicating if the resulting causal estimates should
                be displayed and saved in heatmap form. Defaults to True.
            show_validation: Optional; A boolean indicating if the resulting causal estimates should
                be compared to previous expectations. Defaults to True.
            show_largest_effects: Optional; A boolean indicating if the largest causal effects should
                be listed. Defaults to True.
            generate_pdf_report: Optional; A boolean indicating if the causal graph, heatmaps,
                validations and estimates should be written to files and combined into a pdf.
        """
        for treatment in treatments:
            for outcome in outcomes:
                for estimand_type in estimand_types:
                    try:
                        self.run_quick_analysis(treatment, outcome, estimand_type, None, verbose)
                    except TypeError:
                        msg = "No mediation analysis possible. Look at the ATEs instead."
                        self._quick_results_list.append([treatment, outcome, estimand_type,
                                                         float("NaN"), msg, msg, msg])
                        effect = (treatment, outcome, estimand_type)
                        if effect in self._validation_dict:
                            self._validate_effect(effect)
                        continue
                    except AssertionError:
                        msg = "No causal path"
                        self._quick_results_list.append([treatment, outcome, estimand_type,
                                                         0.0, msg, msg, msg])
                        effect = (treatment, outcome, estimand_type)
                        if effect in self._validation_dict:
                            self._validate_effect(effect)
                        continue
        if show_heatmaps or generate_pdf_report:
            self.show_heatmaps(save=generate_pdf_report)
        if show_validation or generate_pdf_report:
            self.show_validation(save=generate_pdf_report)
        if show_largest_effects or generate_pdf_report:
            for estimand_type in estimand_types:
                self.show_largest_effects(estimand_type, save=generate_pdf_report)
        if show_tables or generate_pdf_report:
            self.show_quick_results(save=generate_pdf_report)
        if generate_pdf_report:
            self.generate_pdf_report()

    def run_quick_analysis(self,
                           treatment,
                           outcome,
                           estimand_type,
                           robustness_method=None,
                           verbose=True):
        """Performs a quick causal analysis with preset parameters.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
            robustness_method: Optional; A string indicating the robustness check to be used.
                Defaults to None.
            verbose: Optional; A boolean indicating if verbose output should be displayed. Defaults
                to True.

        Raises:
            KeyError: 'estimand_type must be nonparametric-ate, nonparametric-nde
                or nonparametric-nie'
        """
        #TODO: fix error when input is categorical (e.g. string-type season in sprinkler data)
        self.initialize_model(treatment, outcome, estimand_type)
        self.identify_estimand(verbose, proceed_when_unidentifiable=True)
        if estimand_type == 'nonparametric-ate':
            self.estimate_effect('backdoor.linear_regression', verbose)
        elif estimand_type in ['nonparametric-nde', 'nonparametric-nie']:
            fsm = dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
            ssm = dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
            self.estimate_effect('mediation.two_stage_regression',
                                 verbose,
                                 confidence_intervals=False,
                                 test_significance=False,
                                 method_params={'first_stage_model': fsm,
                                                'second_stage_model': ssm
                                                }
                                 )
        else:
            raise KeyError("estimand_type must be nonparametric-ate, nonparametric-nde or "
                           + "nonparametric-nie")
        if robustness_method:
            self.check_robustness(robustness_method, verbose)
        self._store_results(robustness_method)
        effect = (treatment, outcome, estimand_type)
        if effect in self._validation_dict:
            self._validate_effect(effect)
            
    def initialize_model(self, treatment, outcome, estimand_type, **kwargs):
        """Initializes the causal model.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
            **kwargs: Advanced parameters for the analysis. Please refer to
                https://microsoft.github.io/dowhy/ for more information.
        """
        self.model = dowhy.CausalModel(data=self.data,
                                       treatment=treatment,
                                       outcome=outcome,
                                       estimand_type=estimand_type,
                                       graph=self._dot_name,
                                       **kwargs
                                       )
        self.treatment = treatment
        self.outcome = outcome
        self.estimand_type = estimand_type

    def identify_estimand(self, verbose=True, **kwargs):
        """Algebraically identifies a statistical estimand for the causal effect from the graph.

        Args:
            verbose: Optional; A boolean indicating if verbose output should be displayed. Defaults
                to True.
            **kwargs: Advanced parameters for the analysis. Please refer to
                https://microsoft.github.io/dowhy/ for more information.
        """
        self._check_directed_paths()
        self.estimand = self.model.identify_effect(**kwargs)
        if verbose:
            print(self.estimand)

    def _check_directed_paths(self):
        """Checks if there is a directed path from treatment to outcome."""
        if self.treatment != self.outcome:
            msg = f"There is no directed path from {self.treatment} to {self.outcome}, so the "\
                  + "causal effect is zero!"
            assert self.model._graph.get_all_directed_paths([self.treatment], [self.outcome]), msg

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
            
    def _validate_effect(self, effect):#should this go into a new class?
        """Checks if an estimated effect matches previous expectations.
        
        Args:
            effect: A triple of treatment, outcome and estimand type.
        """
        self._add_estimated_effect(effect)
        val = self._validation_dict[effect]
        estimated = val['Estimated']
        expected = val['Expected']
        val['Valid'] = self._compare_estimated_to_expected_effect(estimated, expected)
        
    def _add_estimated_effect(self, effect):
        """Looks up the value of an estimated effect for validation.
        
        Args:
            effect: A triple of treatment, outcome and estimand type.
        """
        estimated_val = self.get_quick_result_estimate(*effect)
        self._validation_dict[effect]['Estimated'] = estimated_val

    def _compare_estimated_to_expected_effect(self, estimated, expected):
        """Compares estimated and expected values of an effect.

        Args:
            estimated: A float indicating the estimated value of the effect.
            expected: A tuple indicating expectations about the effect.

        Raises:
            AssertionError: [description]

        Returns:
            [type]: [description]
        """
        operator = expected[0]
        expected_val = expected[1]
        if operator == 'greater':
            return estimated > expected_val
        elif operator == 'less':
            return estimated < expected_val
        elif operator == 'equal':
            return estimated == expected_val
        elif operator == 'between':
            lower_bound = expected_val
            upper_bound = expected[2]
            return lower_bound < estimated < upper_bound
        else:
            raise AssertionError


    def _store_results(self, robustness_method):
        """Stores the results of an anlysis for later inspection.

        Growing dataframes dynamically is bad, therefore we use a list and convert it to a prettier
        form only when required.

        Args:
            robustness_method: A string indicating the used robustness check. None if no check was
                performed.
        """
        if robustness_method:
            robustness_info = self.robustness_info
        else:
            robustness_info = 'No robustness check was performed.'
        results = [self.treatment,
                   self.outcome,
                   self.estimand_type,
                   self.estimated_effect.value,
                   self.estimand,
                   self.estimated_effect,
                   robustness_info
                   ]
        self._quick_results_list.append(results)

    @property  # TODO: cache this?
    def _result_mgr(self):
        return _result_mgr.ResultManager(self._quick_results_list, self._validation_dict)

    def erase_quick_results(self):
        """Erases stored results from quick analyses."""
        self._quick_results_list = []

    def show_quick_results(self, save=True):
        """Shows all results from quick analyses in tabular form."""
        if save:
            name = self._get_results_csv_name()
            self._result_mgr.show_quick_results(name)
        else:
            self._result_mgr.show_quick_results()

    def _get_results_csv_name(self):
        return self.paths.create_output_name('csv', '_results')

    def show_heatmaps(self, save=True):
        """Shows heatmaps for strengths of causal effects.

        Args:
            save: Optional; A boolean indicating if the result should be saved to png.
                Defaults to True.
        """
        if save:
            name = self.paths.output_name_stump
            self._result_mgr.show_heatmaps(name)
        else:
            self._result_mgr.show_heatmaps()

    def get_quick_result_estimate(self, treatment, outcome, estimand_type):
        """Returns a stored estimated effect.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        return self._result_mgr.get_quick_result_estimate(treatment, outcome, estimand_type)

    def show_quick_result_methods(self, treatment, outcome, estimand_type):
        """Shows methodic information about the result of a quick analysis.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        self._result_mgr.show_quick_result_methods(treatment, outcome, estimand_type)
        
    def show_largest_effects(self, estimand_type, n_results=10, save=True):
        """Shows the largest causal effects in decreasing order.

        Args:
            estimand_type: A string indicating the type of causal effect.
            n_results: Optional; An integer indicating the number of effects to be shown.
                Defaults to 10.
            save: Optional; A boolean indicating if the result should be saved to png.
                Defaults to True.
        """
        if save:
            name = self.paths.output_name_stump
            self._result_mgr.show_largest_effects(estimand_type, n_results, name)
        else:
            self._result_mgr.show_largest_effects(estimand_type, n_results)
        
    def show_validation(self, save=True):
        """Shows if selected estimated effects match previous expectations.
        
        Args:
            save: Optional; A boolean indicating if the result should be saved to png.
                Defaults to True.
        """
        if save:
            name = self.paths.output_name_stump
            self._result_mgr.show_validation(name)
        else:
            self._result_mgr.show_validation()

    def generate_pdf_report(self, dpi=(300, 300)):
        """Generates a pdf report with the causal graph and all results.

        Args:
            dpi: Optional; A pair indicating the resolution. Defaults to (300, 300).
        """
        output_name, input_names = self.paths.create_reporting_paths()
        self._result_mgr.generate_pdf_report(output_name, input_names, dpi)

    def compare_to_noncausal_regression(self, input_cols, drop_cols=False):
        """Prints a comparison of the causal estimate to a noncausal linear regression estimate.

        Args:
            input_cols: A set of columns to be used in the linear regression.
            drop_cols: Optional; A boolean indicating if input_cols should indicate which columns
                to drop instead of which columns to use. Defaults to False.
        """
        self._regression_mgr.compare_to_noncausal_regression(input_cols, drop_cols)

    @property
    def _regression_mgr(self):
        return _regression_mgr.RegressionManager(self.data,
                                                 self.treatment,
                                                 self.outcome,
                                                 self.estimated_effect.value
                                                 )
