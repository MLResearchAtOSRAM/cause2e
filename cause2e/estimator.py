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
from cause2e import _preproc, _reader, _technical_estimators, _result_mgr, _regression_mgr


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
        self._spark = spark
        self._result_mgr = _result_mgr.ResultManager(quick_results_list=[],
                                                     validation_dict=validation_dict
                                                     )

    @classmethod
    def from_learner(cls, learner, same_data=False):
        if learner.knowledge:
            validation_dict = learner.knowledge['expected_effects']
        else:
            validation_dict = {}
        if same_data:
            estim = cls(paths=learner.paths,
                        validation_dict=validation_dict,
                        spark=learner._spark,
                        )
            estim.data = learner.data
            return estim
        else:
            return cls(paths=learner.paths,
                       transformations=learner.transformations,
                       validation_dict=validation_dict,
                       spark=learner._spark,
                       )

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
        self.erase_quick_results()
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
                Defaults to True.
        """
        for treatment in treatments:
            for outcome in outcomes:
                for estimand_type in estimand_types:
                    self.run_quick_analysis(treatment, outcome, estimand_type, None, verbose)
                    effect = (treatment, outcome, estimand_type)
                    self._result_mgr.validate_effect(effect)
        self.analyze_quick_results(estimand_types,
                                   show_tables,
                                   show_heatmaps,
                                   show_validation,
                                   show_largest_effects,
                                   generate_pdf_report)

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
        error_flag = "pass"
        if treatment == outcome:
            self.treatment = treatment
            self.outcome = outcome
            self.estimand_type = estimand_type
            error_flag = "trivial"
        else:
            try:
                self.initialize_model(treatment, outcome, estimand_type)
                self.identify_estimand(verbose, proceed_when_unidentifiable=True)
                technical_estimator = _technical_estimators.EstimatorVariableTypes.from_estimator(self)
                quantitative_effect = technical_estimator.estimate_effect(verbose)
                self.estimated_effect = technical_estimator.estimated_effect  # only for bookkeeping
                self.estimated_effect.value = quantitative_effect
                if robustness_method:
                    self.check_robustness(robustness_method, verbose)
            except TypeError:
                error_flag = "missing_mediators"
            except AssertionError:
                error_flag = "missing_causal_path"
            except ValueError:
                error_flag = "categorical_problem"
        storage_input = self._create_storage_input(error_flag)
        self._result_mgr.process_result(error_flag, storage_input)

    def _create_storage_input(self, error_flag):
        storage_input = [self.treatment, self.outcome, self.estimand_type]
        if error_flag == "pass":
            storage_input.extend([self.estimated_effect.value, self.estimand, self.estimated_effect])
        return storage_input

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
        if self.estimand_type == 'nonparametric-nde':
            self._check_directed_edge()
        else:
            self._check_directed_paths()
        self.estimand = self.model.identify_effect(**kwargs)
        if verbose:
            print(self.estimand)

    def _check_directed_edge(self):
        """Checks if there is a directed edge from treatment to outcome."""
        if self.treatment != self.outcome:
            msg = f"There is no directed edge from {self.treatment} to {self.outcome}, so the "\
                  + "causal effect is zero!"
            assert (self.treatment, self.outcome) in set(self.model._graph._graph.edges), msg

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

    def analyze_quick_results(self,
                              estimand_types,
                              show_tables,
                              show_heatmaps,
                              show_validation,
                              show_largest_effects,
                              generate_pdf_report):
        """Summarizes the result of quick analyses for further analysis.

        Args:
            estimand_types: A list of strings indicating the types of causal effects.
            show_tables: A boolean indicating if the resulting causal estimates should be
                displayed in tabular form.
            show_heatmaps: A boolean indicating if the resulting causal estimates should
                be displayed and saved in heatmap form.
            show_validation: A boolean indicating if the resulting causal estimates should
                be compared to previous expectations.
            show_largest_effects: A boolean indicating if the largest causal effects should
                be listed.
            generate_pdf_report: A boolean indicating if the causal graph, heatmaps,
                validations and estimates should be written to files and combined into a pdf.
        """
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

    def erase_quick_results(self):
        """Erases stored results from quick analyses."""
        self._result_mgr.erase_quick_results()

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


class EstimatorDatabricks(Estimator):
    """Main class for estimating causal effects on a Databricks cluster.

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
        str_report: A string that is used to show the pdf report.
    """

    def __init__(self, paths, spark, transformations=[], validation_dict={}):
        super().__init__(paths, transformations, validation_dict)
        self._spark = spark

    def generate_pdf_report(self, dpi=(300, 300)):
        """Generates a pdf report with the causal graph and all results.

        Args:
            dpi: Optional; A pair indicating the resolution. Defaults to (300, 300).
        """
        output_name, input_names = self.paths.create_reporting_paths()
        self._report_name = output_name
        self._print_result_display_instruction()
        self._result_mgr.generate_pdf_report(output_name, input_names, dpi)

    @staticmethod
    def _print_result_display_instruction():
        """Prints an instruction for showing the pdf report on Databricks."""
        command_estim = 'displayHTML(<name of EstimatorDatabricks>.str_report)'
        command_learner = 'displayHTML(<name of StructureLearnerDatabricks>.str_report)'
        print(f"Run {command_estim} to show the report.")
        print(f"Run {command_learner} to show the report if you have called the analysis directly from the learner.\n")

    @property
    def str_report(self):
        return self._get_src_str(self._report_name)

    def _get_src_str(self, name):
        modified_name = name.replace('/dbfs/FileStore', 'files')
        return f"<img src = '{modified_name}'>"
