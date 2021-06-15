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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
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
        treatment: A string indicating the most recent treatment variable.
        outcome: A string indicating the most recent outcome variable.
        estimand_type: A string indicating the most recent type of causal effect.
        estimand: A dowhy.causal_identifier.IdentifiedEstimand indicating the most recent estimand.
        estimated_effect: A dowhy.causal_estimator.CausalEstimate indicating the most recent
            estimated effect.
        robustness_info: A dowhy.causal_refuter.CausalRefutation indicating the results of the most
            recent robustness check.
        quick_results: A pandas.DataFrame storing the results of calls to the quick_analysis method.
        spark: Optional; A pyspark.sql.SparkSession in case you want to use spark. Defaults to
            None.
    """

    def __init__(self, paths, transformations=[], spark=None):
        """Inits CausalEstimator"""
        self.paths = paths
        self.transformations = transformations.copy()
        self.spark = spark
        self._quick_results_list = []

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

    # TODO: Cache this?
    def run_all_quick_analyses(self,
                               estimand_types=['nonparametric-ate',
                                               'nonparametric-nde',
                                               'nonparametric-nie'
                                               ],
                               verbose=False,
                               show_tables=True,
                               save_tables=True,
                               show_heatmaps=True):
        """Performs all possible quick causal anlyses with preset parameters.

        Args:
            estimand_types: A list of strings indicating the types of causal effects.
            verbose: Optional; A boolean indicating if verbose output should be displayed for each
                analysis. Defaults to False.
            show_tables: Optional; A boolean indicating if the resulting causal estimates should be
                displayed in tabular form. Defaults to True.
            save_tables: Optional; A boolean indicating if the resulting causal estimates should be
                written to a csv. Defaults to True.
            show_heatmaps: Optional; A boolean indicating if the resulting causal estimates should
                be displayed and saved in heatmap form. Defaults to True.
        """
        vars = self.variables
        self.run_multiple_quick_analyses(vars, vars, estimand_types, verbose, show_tables,
                                         save_tables, show_heatmaps)

    def run_multiple_quick_analyses(self,
                                    treatments,
                                    outcomes,
                                    estimand_types,
                                    verbose=False,
                                    show_tables=True,
                                    save_tables=True,
                                    show_heatmaps=True):
        """Performs multiple quick causal analyses with preset parameters.

        Args:
            treatments: A list of strings indicating the names of the treatment variables.
            outcomes: A list of strings indicating the names of the outcome variables.
            estimand_types: A list of strings indicating the types of causal effects.
            verbose: Optional; A boolean indicating if verbose output should be displayed for each
                analysis. Defaults to False.
            show_tables: Optional; A boolean indicating if the resulting causal estimates should be
                displayed in tabular form. Defaults to True.
            save_tables: Optional; A boolean indicating if the resulting causal estimates should be
                written to a csv. Defaults to True.
            show_heatmaps: Optional; A boolean indicating if the resulting causal estimates should
                be displayed and saved in heatmap form. Defaults to True.
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
                        continue
                    except AssertionError:
                        msg = "No causal path"
                        self._quick_results_list.append([treatment, outcome, estimand_type,
                                                         0.0, msg, msg, msg])
                        continue
        if show_heatmaps:
            self.show_heatmaps()
        if show_tables:
            self.show_quick_results(save=False)
        if save_tables:
            self.save_quick_results()

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

    @property
    def quick_results(self):
        return self._result_handler.quick_results

    @property
    def _result_handler(self):
        return _ResultHandler(self._quick_results_list)

    def erase_quick_results(self):
        """Erases stored results from quick analyses."""
        self._quick_results_list = []

    def show_quick_results(self, save=True):
        """Shows all results from quick analyses in tabular form."""
        self._result_handler.show_quick_results()
        if save:
            self.save_quick_results()

    def save_quick_results(self, line_terminator='\r'):
        """Saves all results from quick analyses in tabular form in a csv."""
        name = self._get_results_csv_name()
        self._result_handler.save_quick_results(name, line_terminator)

    def _get_results_csv_name(self):
        return self.paths.create_output_name('csv', '_results')

    def show_heatmaps(self, save=True):
        """Shows heatmaps for strengths of causal effects.

        Args:
            save: Optional; A boolean indicating if the result should be saved to png.
                Defaults to True.
        """
        if save:
            name = self._get_heatmap_name()
            self._result_handler.show_heatmaps(name)
        else:
            self._result_handler.show_heatmaps()

    def _get_heatmap_name(self):
        return self.paths.png_name[:-4]

    def get_quick_result_estimate(self, treatment, outcome, estimand_type):
        """Returns a stored estimated effect.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        return self._result_handler.get_quick_result(treatment, outcome, estimand_type)

    def show_quick_result_methods(self, treatment, outcome, estimand_type):
        """Shows methodic information about the result of a quick analysis.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        self._result_handler.show_quick_result_methods(treatment, outcome, estimand_type)

    # TODO: Include prior knowledge
    def generate_pdf_report(self, dpi=(300, 300)):
        """Generates a pdf report with the causal graph and all results.

        Args:
            dpi: Optional; A pair indicating the resolution. Defaults to (300, 300).
        """
        output_name = self.paths.create_output_name('pdf', '_report')
        graph_name = self.paths.png_name
        estimand_types = ['ate', 'nde', 'nie']
        heatmaps = [self._create_heatmap_name(x) for x in estimand_types]
        results = [self._create_result_name(x) for x in estimand_types]
        self._result_handler.generate_pdf_report(output_name, graph_name, heatmaps, results, dpi)

    def _create_heatmap_name(self, estimand_type):
        return self.paths.create_output_name('png', f'_heatmap_{estimand_type}')

    def _create_result_name(self, estimand_type):
        return self.paths.create_output_name('png', f'_results_{estimand_type}')

    def compare_to_noncausal_regression(self, input_cols, drop_cols=False):
        """Prints a comparison of the causal estimate to a noncausal linear regression estimate.

        Args:
            input_cols: A set of columns to be used in the linear regression.
            drop_cols: Optional; A boolean indicating if input_cols should indicate which columns
                to drop instead of which columns to use. Defaults to False.
        """
        self._regression_handler.compare_to_noncausal_regression(input_cols, drop_cols)

    @property
    def _regression_handler(self):
        return _RegressionHandler(self.data,
                                  self.treatment,
                                  self.outcome,
                                  self.estimated_effect.value
                                  )


class _ResultHandler:
    """Helper class for managing the output of analyses."""

    def __init__(self, quick_results_list):
        self._quick_results_list = quick_results_list

    @property
    def quick_results(self):
        return pd.DataFrame(self._quick_results_list,
                            columns=['Treatment',
                                     'Outcome',
                                     'Estimand_type',
                                     'Estimated_effect',
                                     'Estimand',
                                     'Estimation',
                                     'Robustness_info'
                                     ]
                            )

    def show_quick_result_methods(self, treatment, outcome, estimand_type):
        """Shows methodic information about the result of a quick analysis.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        row = self.get_quick_result(treatment, outcome, estimand_type)
        self._show_existing_quick_result_methods(row)

    def get_quick_result(self, treatment, outcome, estimand_type):
        """Returns the result of a quick analysis.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        filtered_df = self._get_matching_quick_results(treatment, outcome, estimand_type)
        if filtered_df.shape[0] > 1:
            print("More than one matching result found! Please query quick_results manually.\n")
        elif filtered_df.shape[0] == 0:
            print("No result has been stored for this query.\n")
        else:
            return filtered_df.iloc[0]

    def _get_matching_quick_results(self, treatment, outcome, estimand_type):
        mask = (self.quick_results['Treatment'] == treatment) &\
               (self.quick_results['Outcome'] == outcome) &\
               (self.quick_results['Estimand_type'] == estimand_type)
        return self.quick_results[mask]

    def _show_existing_quick_result_methods(self, row):
        print(row['Estimand'])
        # catches cases where no full analysis was performed.
        if row['Estimand'] != row['Estimation']:
            print(row['Estimation'])
            print(row['Robustness_info'])

    def get_quick_result_estimate(self, treatment, outcome, estimand_type):
        """Returns a stored estimated effect.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        row = self.get_quick_result(treatment, outcome, estimand_type)
        return row['Estimate_Effect']

    def show_quick_results(self, save_to_name=None):
        """Shows all results from quick analyses in tabular form."""
        print("Only quantitative estimates are shown. For methodic details, use "
              + "Estimator.show_quick_result_methods.\n")
        print("Average treatment effects from quick analyses:\n")
        print(self._quick_results_ate)
        print("\n================================\n")
        print("Natural direct effects from quick analyses:\n")
        print(self._quick_results_nde)
        print("\n================================\n")
        print("Natural indirect effects from quick analyses:\n")
        print(self._quick_results_nie)
        if save_to_name:
            self.save_quick_results(save_to_name)

    @property
    def _quick_results_ate(self):
        return self._get_pivot_df_from_quick_results(estimand_type='nonparametric-ate')

    @property
    def _quick_results_nde(self):
        pivot_df = self._get_pivot_df_from_quick_results(estimand_type='nonparametric-nde')
        return pivot_df.fillna(self._quick_results_ate)  # no mediation -> ate equals direct effect
        #fix error when printing results of mediation analysis from stored results
        #fix error when input is categorical (e.g. string-type season in sprinkler data)

    @property
    def _quick_results_nie(self):
        pivot_df = self._get_pivot_df_from_quick_results(estimand_type='nonparametric-nie')
        return pivot_df.fillna(0)  # no mediation -> no indirect effect

    def _get_pivot_df_from_quick_results(self, estimand_type):
        mask = self.quick_results['Estimand_type'] == estimand_type
        df = self.quick_results[mask][['Treatment', 'Outcome', 'Estimated_effect']]
        pivot_df = df.pivot_table(index='Treatment', columns='Outcome', dropna=False)
        return pivot_df

    def save_quick_results(self, name, line_terminator='\r'):
        """Saves all quantitative quick results to a csv.

        Args:
            name: A boolean indicating the name of the csv.
            line_terminator: Optional; A string indicating the line terminator for writing to csv.
                Defaults to '\r'.
        """
        print("Saving all causal estimates from quick analyses to csv.\n")
        with open(name, 'w') as f:
            f.write("SEP=,\n")
            f.write("Average treatment effects from quick analyses:\n")
            self._quick_results_ate.to_csv(f, line_terminator=line_terminator)
            f.write("\n")
            f.write("Natural direct effects from quick analyses:\n")
            self._quick_results_nde.to_csv(f, line_terminator=line_terminator)
            f.write("\n")
            f.write("Natural indirect effects from quick analyses:\n")
            self._quick_results_nie.to_csv(f, line_terminator=line_terminator)
            f.write("\n")

    def show_heatmaps(self, save_to_name=None):
        """Shows and possibly saves heatmaps and dataframes of the causal effect strengths.

        Args:
            save_to_name: Optional; A string indicating the beginning of the name of the png files
                where the heatmaps and dataframes should be saved. Defaults to None.
        """
        if save_to_name:
            print("Showing and saving heat matrices of the causal estimates.\n")
        else:
            print("Showing heat matrices of the causal estimates.\n")
        self.show_heatmap('nonparametric-ate', save_to_name)
        self.show_heatmap('nonparametric-nde', save_to_name)
        self.show_heatmap('nonparametric-nie', save_to_name)

    def show_heatmap(self, estimand_type, save_to_name=None):
        """Shows and possibly saves a heatmap and dataframe of the causal effect strengths.

        Args:
            estimand_type: A string indicating the type of causal effect.
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the heatmap and dataframe should be saved. Defaults to None.
        """
        df, title = self._select_heatmap_input(estimand_type)
        df.columns = [tup[1] for tup in df.columns.values]
        heatmap = sns.heatmap(df)
        plt.title(title, size=18)
        plt.ylabel("Treatment", size=15)
        plt.xlabel("Outcome", size=15)
        plt.show()
        if save_to_name:
            figure = heatmap.get_figure()
            name = save_to_name + '_heatmap_' + estimand_type[-3:] + '.png'
            figure.savefig(name, bbox_inches="tight", dpi=400)
            self._save_quick_results_df_as_png(estimand_type, save_to_name)

    def _save_quick_results_df_as_png(self, estimand_type, save_to_name):
        """Saves the quick results dataframe to a png file.

        Args:
            estimand_type: A string indicating the type of causal effect.
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the dataframe should be saved. Defaults to None.
        """
        df, title = self._select_heatmap_input(estimand_type)
        df.update(df.applymap('{:,.2f}'.format))
        plt.ioff()
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.set_title(title)
        t = ax.table(cellText=df.values, colLabels=[x[1] for x in df.columns], rowLabels=df.index)
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.auto_set_column_width(col=list(range(len(df.columns))))
        t.scale(1, 4)
        fig.tight_layout()
        name = save_to_name + '_results_' + estimand_type[-3:] + '.png'
        fig.savefig(name, bbox_inches="tight", dpi=400)
        plt.close(fig)

    def _select_heatmap_input(self, estimand_type):
        """Returns the right dataframe and title for creating a heatmap.

        Args:
            estimand_type: A string indicating the type of causal effect.

        Raises:
            KeyError: 'estimand_type must be nonparametric-ate, nonparametric-nde
                or nonparametric-nie'
        """
        if estimand_type == 'nonparametric-ate':
            df = self._quick_results_ate.copy()
            title = "Average Treatment Effects\n (direct + indirect influences)"
        elif estimand_type == 'nonparametric-nde':
            df = self._quick_results_nde.copy()
            title = "Natural Direct Effects\n (direct influences only)"
        elif estimand_type == 'nonparametric-nie':
            df = self._quick_results_nie.copy()
            title = "Natural Indirect Effects\n (indirect influences only)"
        else:
            raise KeyError("estimand_type must be nonparametric-ate, nonparametric-nde or "
                           + "nonparametric-nie")
        return df, title

    def generate_pdf_report(self, output_name, graph, heatmaps, results, dpi=(300, 300)):
        """Generates a pdf report with the causal graph and all results.

        Args:
            output_name: A string indicating the name of the output pdf.
            graph: A string indicating the name of the png where the causal graph is stored.
            heatmaps: A list of strings indicating the names of the pngs where the heatmaps are
                stored.
            heatmaps: A list of strings indicating the names of the pngs where the quantiative
                results are stored.
            dpi: Optional; A pair indicating the resolution. Defaults to (300, 300).
        """
        input_names = [graph] + heatmaps + results
        ims = [self._convert_rgba_to_rgb(filename) for filename in input_names]
        im = ims[0]
        im_list = ims[1:]
        im.save(output_name, "PDF", dpi=dpi, save_all=True, append_images=im_list)
        print(f"Successfully generated report in {output_name}.\n")

    def _convert_rgba_to_rgb(self, filename):
        """Returns an rgb version of an rgba png file.

        Args:
            filename: A string indicating the name of the png file.
        """
        rgba = Image.open(filename)
        rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # white background
        rgb.paste(rgba, mask=rgba.split()[3])
        return rgb


class _RegressionHandler:
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
