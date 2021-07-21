"""
_result_mgr.py
================================================================
This module implements the ResultManager class.

It is used as a helper class for managing the output of analyses after the estimation of causal
effects has been performed. It derives heatmaps, tables, selected validations and a pdf report
from the results. It makes use of several helper classes itself.
"""


import pandas as pd
from math import isnan
import seaborn as sns
import matplotlib.pyplot as plt
import PIL


class ResultManager:
    """Helper class for managing the output of analyses."""

    def __init__(self, quick_results_list, validation_dict):
        self._numeric_result_mgr = _NumericResultManager(quick_results_list)
        self._query_mgr = _QueryManager(self._numeric_result_mgr)
        self._heatmap_mgr = _HeatmapManager(self._numeric_result_mgr)
        self._validation_mgr = _ValidationManager(validation_dict)

    def show_quick_results(self, save_to_name=None):
        """Shows all results from quick analyses in tabular form."""
        self._numeric_result_mgr.show_quick_results(save_to_name)

    def get_quick_result_estimate(self, treatment, outcome, estimand_type):
        """Returns a stored estimated effect.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.

        Raises:
            KeyError: 'estimand_type must be nonparametric-ate, nonparametric-nde
                or nonparametric-nie'
        """
        return self._query_mgr.get_result_estimate(treatment, outcome, estimand_type)

    def show_quick_result_methods(self, treatment, outcome, estimand_type):
        """Shows methodic information about the result of a quick analysis.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        self._query_mgr.show_result_methods(treatment, outcome, estimand_type)

    def show_largest_effects(self, estimand_type, n_results, save_to_name=None):
        """Shows the largest causal effects in decreasing order.

        Args:
            estimand_type: A string indicating the type of causal effect.
            n_results: An integer indicating the number of effects to be shown.
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the dataframe should be saved. Defaults to None.
        """
        self._query_mgr.show_largest_effects(estimand_type, n_results, save_to_name)

    def show_heatmaps(self, save_to_name=None):
        """Shows and possibly saves heatmaps and dataframes of the causal effect strengths.

        Args:
            save_to_name: Optional; A string indicating the beginning of the name of the png files
                where the heatmaps and dataframes should be saved. Defaults to None.
        """
        self._heatmap_mgr.show_heatmaps(save_to_name)

    def show_validation(self, save_to_name=None):
        """Shows if selected estimated effects match previous expectations.

        Args:
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the validation report should be saved. Defaults to None.
        """
        self._validation_mgr.show_validation(save_to_name)

    def generate_pdf_report(self, output_name, input_names, dpi=(300, 300)):
        """Generates a pdf report with the causal graph and all results.

        Args:
            output_name: A string indicating the name of the output pdf.
            input_names: A list of strings indicating the names of the pngs used for creating the pdf.
            dpi: Optional; A pair indicating the resolution. Defaults to (300, 300).
        """
        _generate_pdf_report(output_name, input_names, dpi)


class _NumericResultManager:
    """Helper class for dealing with numeric result tables.

    Attributes:
        quick_results: A pandas DataFrame containing all stored results and methods from quick analyses.
        quick_results_ate: A pandas DataFrame containing all quantitative average treatment effects.
        quick_results_nde: A pandas DataFrame containing all quantitative natural direct effects.
        quick_results_nie: A pandas DataFrame containing all quantitative natural indirect effects.
    """

    def __init__(self, quick_results_list):
        self._quick_results_list = quick_results_list
        self.quick_results = self._get_quick_results()
        self.quick_results_ate = self._get_quick_results_ate()
        self.quick_results_nde = self._get_quick_results_nde()
        self.quick_results_nie = self._get_quick_results_nie()

    def _get_quick_results(self):
        return pd.DataFrame(self._quick_results_list,
                            columns=['Treatment',
                                     'Outcome',
                                     'Estimand_type',
                                     'Estimated_effect',
                                     'Estimand',
                                     'Estimation',
                                     'Robustness_info',
                                     ]
                            )

    def _get_quick_results_ate(self):
        return self._get_pivot_df_from_quick_results(estimand_type='nonparametric-ate')

    def _get_quick_results_nde(self):
        pivot_df = self._get_pivot_df_from_quick_results(estimand_type='nonparametric-nde')
        return pivot_df.fillna(self.quick_results_ate)  # no mediation -> ate equals direct effect
        # TODO: fix error when printing results of mediation analysis from stored results

    def _get_quick_results_nie(self):
        pivot_df = self._get_pivot_df_from_quick_results(estimand_type='nonparametric-nie')
        return pivot_df.fillna(0)  # no mediation -> no indirect effect

    def show_quick_results(self, save_to_name=None):
        """Shows all results from quick analyses in tabular form."""
        print("Only quantitative estimates are shown. For methodic details, use "
              + "Estimator.show_quick_result_methods.\n")
        print("Average treatment effects from quick analyses:\n")
        print(self.quick_results_ate)
        print("\n================================\n")
        print("Natural direct effects from quick analyses:\n")
        print(self.quick_results_nde)
        print("\n================================\n")
        print("Natural indirect effects from quick analyses:\n")
        print(self.quick_results_nie)
        if save_to_name:
            self._save_quick_results(save_to_name)

    def _get_pivot_df_from_quick_results(self, estimand_type):
        mask = self.quick_results['Estimand_type'] == estimand_type
        df = self.quick_results[mask][['Treatment', 'Outcome', 'Estimated_effect']]
        pivot_df = df.pivot_table(index='Treatment', columns='Outcome', dropna=False)
        return pivot_df

    def _save_quick_results(self, name):
        """Saves all quantitative quick results to a csv.

        Args:
            name: A boolean indicating the name of the csv.
        """
        print("Saving all causal estimates from quick analyses to csv.\n")
        with open(name, 'w') as f:
            f.write("SEP=,\n")
            f.write("Average treatment effects from quick analyses:\n")
            self.quick_results_ate.to_csv(f, line_terminator="\n")
            f.write("\n")
            f.write("Natural direct effects from quick analyses:\n")
            self.quick_results_nde.to_csv(f, line_terminator="\n")
            f.write("\n")
            f.write("Natural indirect effects from quick analyses:\n")
            self.quick_results_nie.to_csv(f, line_terminator="\n")
            f.write("\n")


class _QueryManager:
    """Helper class for querying the stored results."""

    def __init__(self, numeric_results_mgr):
        self._results_df = numeric_results_mgr.quick_results
        self._results_ate = numeric_results_mgr.quick_results_ate
        self._results_nde = numeric_results_mgr.quick_results_nde
        self._results_nie = numeric_results_mgr.quick_results_nie

    def show_result_methods(self, treatment, outcome, estimand_type):
        """Shows methodic information about the result of a quick analysis.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        row = self._get_result(treatment, outcome, estimand_type)
        self._show_existing_result_methods(row)

    def _get_result(self, treatment, outcome, estimand_type):
        """Returns the result of a quick analysis.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.
        """
        filtered_df = self._get_matching_results(treatment, outcome, estimand_type)
        if filtered_df.empty:
            print("No result has been stored for this query.\n")
        elif filtered_df.shape[0] > 1:
            print("More than one matching result found! Please query quick_results manually.\n")
        else:
            return filtered_df.iloc[0]

    def _get_matching_results(self, treatment, outcome, estimand_type):
        mask = (self._results_df['Treatment'] == treatment) &\
               (self._results_df['Outcome'] == outcome) &\
               (self._results_df['Estimand_type'] == estimand_type)
        return self._results_df[mask]

    def _show_existing_result_methods(self, row):
        print(row['Estimand'])
        # catches cases where no full analysis was performed.
        if row['Estimand'] != row['Estimation']:
            print(row['Estimation'])
            print(row['Robustness_info'])

    def get_result_estimate(self, treatment, outcome, estimand_type):
        """Returns a stored estimated effect.

        Args:
            treatment: A string indicating the name of the treatment variable.
            outcome: A string indicating the name of the outcome variable.
            estimand_type: A string indicating the type of causal effect.

        Raises:
            KeyError: 'estimand_type must be nonparametric-ate, nonparametric-nde
                or nonparametric-nie'
        """
        if estimand_type == 'nonparametric-ate':
            df = self._results_ate
        elif estimand_type == 'nonparametric-nde':
            df = self._results_nde
        elif estimand_type == 'nonparametric-nie':
            df = self._results_nie
        else:
            raise KeyError("estimand_type must be nonparametric-ate, nonparametric-nde or "
                           + "nonparametric-nie")
        return df.loc[treatment]['Estimated_effect'][outcome]

    def show_largest_effects(self, estimand_type, n_results, save_to_name=None):
        """Shows the largest causal effects in decreasing order.

        Args:
            estimand_type: A string indicating the type of causal effect.
            n_results: An integer indicating the number of effects to be shown.
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the dataframe should be saved. Defaults to None.
        """
        largest_effects = self._get_largest_effects(estimand_type, n_results)
        title = f"{n_results} Largest {_HeatmapManager._select_title(estimand_type)}:"
        print(title)
        print(largest_effects)
        print("\n================================\n")
        if save_to_name:
            self._save_largest_effects_as_png(largest_effects, title, estimand_type, save_to_name)

    def _get_largest_effects(self, estimand_type, n_results):
        """Returns the largest causal effects in decreasing order.

        Args:
            estimand_type: A string indicating the type of causal effect.
            n_results: An integer indicating the number of effects to be shown.
        """
        df = self._fill_mediation_estimates()
        df_filtered = self._get_matching_nontrivial_estimates(df, estimand_type)
        df_largest = df_filtered['Estimated_effect'].apply(abs).nlargest(n_results)
        df['Estimated_effect'] = df['Estimated_effect'].apply('{:,.2f}'.format)
        return df.iloc[df_largest.index][['Treatment', 'Outcome', 'Estimated_effect']]

    def _fill_mediation_estimates(self):
        """Fills NaNs from mediation with the correct ATEs or zeroes."""
        df = self._results_df.copy()
        df['Estimated_effect'] = df.apply(lambda x: self._replace_nans(x), axis=1)
        return df

    def _replace_nans(self, row):
        """Returns the correct effect strengths for mediation when there are only direct effects.

        Args:
            row: A row of the Pandas DataFrame where all results are stored.
        """
        if isnan(row['Estimated_effect']):
            return self.get_result_estimate(row['Treatment'], row['Outcome'], row['Estimand_type'])
        else:
            return row['Estimated_effect']

    def _get_matching_nontrivial_estimates(self, df, estimand_type):
        """Returns a dataframe containing only nontrivial causal estimates of a given effect type.

        Args:
            df: The Pandas DataFrame to be filtered.
            estimand_type: A string indicating the type of causal effect.
        """
        mask = (df['Treatment'] != df['Outcome']) & (df['Estimand_type'] == estimand_type)
        return df[mask]

    def _save_largest_effects_as_png(self, df, title, estimand_type, save_to_name):
        """Saves the largest causal effects of a given type to a png file.

        Args:
            df: A pandas DataFrame containing the largest causal effects.
            title: A string used for describing the saved dataframe.
            estimand_type: A string indicating the type of causal effects.
            save_to_name: A string indicating the beginning of the name of the png file
                where the dataframe should be saved.
        """
        filename = save_to_name + '_largest_effects_' + estimand_type[-3:] + '.png'
        save_df_as_png(df, title, filename, col_labels=df.columns)


class _HeatmapManager:
    """Helper class for generating and saving heatmaps."""

    def __init__(self, numeric_results_mgr):
        self._results_ate = numeric_results_mgr.quick_results_ate
        self._results_nde = numeric_results_mgr.quick_results_nde
        self._results_nie = numeric_results_mgr.quick_results_nie

    def show_heatmaps(self, save_to_name=None):
        """Shows and possibly saves heatmaps and dataframes of the causal effect strengths.

        Args:
            save_to_name: Optional; A string indicating the beginning of the name of the png files
                where the heatmaps and dataframes should be saved. Defaults to None.
        """
        if save_to_name:
            print("Showing and saving heatmaps of the causal estimates.\n")
        else:
            print("Showing heatmaps of the causal estimates.\n")
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
        plt.figure(figsize=(3, 3))
        heatmap = sns.heatmap(df)
        plt.title(title, size=18)
        plt.ylabel("Treatment", size=15)
        plt.xlabel("Outcome", size=15)
        plt.show()
        if save_to_name:
            figure = heatmap.get_figure()
            name = save_to_name + '_heatmap_' + estimand_type[-3:] + '.png'
            figure.savefig(name, bbox_inches="tight", dpi=300)
            self._save_quantitative_heatmap_as_png(estimand_type, save_to_name)

    def _save_quantitative_heatmap_as_png(self, estimand_type, save_to_name):
        """Saves the quick results dataframe to a png file.

        Args:
            estimand_type: A string indicating the type of causal effect.
            save_to_name: A string indicating the beginning of the name of the png file
                where the dataframe should be saved.
        """
        df, title = self._select_heatmap_input(estimand_type)
        df.update(df.applymap('{:,.2f}'.format))
        col_labels = [x[1] for x in df.columns]
        row_labels = df.index
        filename = save_to_name + '_results_' + estimand_type[-3:] + '.png'
        save_df_as_png(df, title, filename, col_labels, row_labels)

    def _select_heatmap_input(self, estimand_type):
        """Returns the right dataframe and title for creating a heatmap.

        Args:
            estimand_type: A string indicating the type of causal effect.
        """
        df = self._select_results_df(estimand_type)
        title = self._select_title(estimand_type)
        return df, title

    def _select_results_df(self, estimand_type):
        """Returns the right stored results.

        Args:
            estimand_type: A string indicating the type of causal effect.

        Raises:
            KeyError: 'estimand_type must be nonparametric-ate, nonparametric-nde
                or nonparametric-nie'
        """
        if estimand_type == 'nonparametric-ate':
            return self._results_ate.copy()
        elif estimand_type == 'nonparametric-nde':
            return self._results_nde.copy()
        elif estimand_type == 'nonparametric-nie':
            return self._results_nie.copy()
        else:
            raise KeyError("estimand_type must be nonparametric-ate, nonparametric-nde or "
                           + "nonparametric-nie")

    @staticmethod
    def _select_title(estimand_type):
        """Returns the right title for displaying results.

        Args:
            estimand_type: A string indicating the type of causal effect.

        Raises:
            KeyError: 'estimand_type must be nonparametric-ate, nonparametric-nde
                or nonparametric-nie'
        """
        if estimand_type == 'nonparametric-ate':
            return "Average Treatment Effects\n (direct + indirect influences)"
        elif estimand_type == 'nonparametric-nde':
            return "Natural Direct Effects\n (direct influences only)"
        elif estimand_type == 'nonparametric-nie':
            return "Natural Indirect Effects\n (indirect influences only)"
        else:
            raise KeyError("estimand_type must be nonparametric-ate, nonparametric-nde or "
                           + "nonparametric-nie")


class _ValidationManager:
    """Helper class for validating expected causal effects."""

    def __init__(self, validation_dict):
        self._validation_dict = validation_dict

    def show_validation(self, save_to_name=None):
        """Shows if selected estimated effects match previous expectations.

        Args:
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the validation report should be saved. Defaults to None.
        """
        passed_validations = {k for (k, v) in self._validation_dict.items() if v['Valid']}
        failed_validations = self._validation_dict.keys() - passed_validations
        result_str = ""
        if passed_validations:
            result_str += self._generate_validation_strings(passed_validations, 'Passed')
            result_str += " \n"
        if failed_validations:
            result_str += self._generate_validation_strings(failed_validations, 'Failed')
        print("================================")
        print(result_str)
        print("================================")
        if save_to_name:
            self._save_validations_as_png(result_str, save_to_name)

    def _generate_validation_strings(self, effect_names, description):
        """Returns a subset of validation results in pretty string format.

        Args:
            effect_names: A set indicating the subset of validated effects.
            description: A string describing the subset of validated effects.
        """
        n_validations = len(self._validation_dict)
        n_effects = len(effect_names)
        result_str = f"{description} validations ({n_effects}/{n_validations}): \n"
        for effect in effect_names:
            result_str += self._generate_single_validation_str(effect)
        return result_str

    def _generate_single_validation_str(self, effect):
        """Returns a single validation result in pretty string format.

        Args:
            effect: A triple indicating treatment, outcome and effect type.
        """
        entry = self._validation_dict[effect]
        expected_str = self._get_expected_str(entry['Expected'])
        estimated_str = "{:.2f}".format(entry['Estimated'])
        valid_str = entry['Valid']
        estimated = f"Estimated {effect[2][-3:]} of {effect[0]} on {effect[1]}: {estimated_str} "
        expected = f"(expected: {expected_str}) -> {valid_str} \n"
        return estimated + expected

    def _get_expected_str(self, expectation):
        """Returns an expected effect in pretty string form.

        Args:
            expectation: A tuple describing the expected effect type and size.
        """
        type_ = expectation[0]
        if type_ in {'less', 'greater', 'equal'}:
            if type_ == 'equal':
                return f"equal to {expectation[1]}"
            else:
                return f"{type_} than {expectation[1]}"
        elif type_ == 'between':
            return f"between {expectation[1]} and {expectation[2]}"

    def _save_validations_as_png(self, validation_str, save_to_name):
        """Saves the validation results to a png file for reporting purposes.

        Args:
            validation_str: A string containing the validation results to be saved.
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the validation report should be saved.
        """
        for valid in [True, False]:
            df, title = _ValidationManager._create_validation_material_for_saving(validation_str, valid)
            filename = save_to_name + '_validation_' + str(valid) + '.png'
            save_df_as_png(df, title, filename, col_labels=df.columns)

    @staticmethod
    def _create_validation_material_for_saving(validation_str, valid):
        """Returns title and dataframe for saving the validation results

        Args:
            validation_str: A string generated for printing the same information to the console.
            valid: A boolean indicating if we want passed (True) or failed (False) validations.
        """
        lines = validation_str.split('\n')
        descriptor = "Passed" if valid else "Failed"
        lines_valid = [line for line in lines if f"-> {valid}" in line]
        if lines_valid:
            title = [line for line in lines if f"{descriptor} validations" in line][0]
            df = _ValidationManager._create_df_from_strings(lines_valid)
        else:
            title = f"No {descriptor.lower()} validations."
            df = pd.DataFrame([title], columns=["Validations:"])
        return df, title

    @staticmethod
    def _create_df_from_strings(strings):
        """Returns a pandas DataFrame suitable for writing to a png.

        Args:
            strings: A list of strings.
        """
        delimiters = ['Estimated ', ' of ', ' on ', ' (expected: ', ': ', ') -> ']
        split_strings = [_ValidationManager._split_string(string, delimiters)[1:-1] for string in strings]
        df = pd.DataFrame(split_strings)
        col_names = ['Effect Type', 'Treatment', 'Outcome', 'Estimated', 'Expected']
        df.columns = col_names
        df['Effect Type'] = df['Effect Type'].apply(_ValidationManager._replace_effect_name)
        return df

    @staticmethod
    def _split_string(string, delimiters, replacement_char=','):
        """Splits a string on all specified delimiters.

        Args:
            string: The string to be split.
            delimiters: A list of strings that trigger splitting.
            replacement_char: Optional; A string that replaces all delimiters and is then used for
                splitting. Defaults to ','.

        Returns:
            A list containing the parts of the split string.
        """
        for x in delimiters:
            string = string.replace(x, replacement_char)
        return string.split(replacement_char)

    @staticmethod
    def _replace_effect_name(name):
        """Returns an easier to understand description of a causal effect type.

        Args:
            name: A string indicating the causal effect type.

        Raises:
            KeyError: 'Unknown effect name.'
        """
        if name in {'ate', 'nonparametric-ate'}:
            return "direct + indirect"
        elif name in {'nde', 'nonparametric-nde'}:
            return "direct"
        elif name in {'nie', 'nonparametric-nie'}:
            return "indirect"
        else:
            raise KeyError("Unknown effect name.")


def save_df_as_png(df, title, filename, col_labels=None, row_labels=None, loc='upper left'):
    """Saves a dataframe as png to include it in a pdf later on.

    Args:
        df: The pandas DataFrame to be saved.
        title: A string describing the dataframe.
        filename: The filename of the png to be created.
        col_labels: Optional; The column labels to be displayed in the png. Defaults to None.
        row_labels: Optional; The row labels to be displayed in the png. Defaults to None.
        loc: Optional; A string indicating the desired location of the dataframe in the png.
            Defaults to 'upper left'.
    """
    df, title = df, title
    plt.ioff()
    fig, ax = plt.subplots(figsize=(3, 5))
    fig.patch.set_visible(False)
    fig.tight_layout()
    ax.axis('off')
    ax.axis('tight')
    ax.set_title(title)
    t = ax.table(cellText=df.values, colLabels=col_labels, rowLabels=row_labels, loc=loc)
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.auto_set_column_width(col=list(range(len(df.columns))))
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _generate_pdf_report(output_name, input_names, dpi=(300, 300)):
    """Generates a pdf report with the causal graph and all results.

    Args:
        output_name: A string indicating the name of the output pdf.
        input_names: A list of strings indicating the names of the pngs used for creating the pdf.
        dpi: Optional; A pair indicating the resolution. Defaults to (300, 300).
    """
    ims = [_convert_rgba_to_rgb(filename) for filename in input_names]
    im = ims[0]
    im_list = ims[1:]
    im.save(output_name, "PDF", dpi=dpi, save_all=True, append_images=im_list)
    print(f"Successfully generated report in {output_name}.\n")


def _convert_rgba_to_rgb(filename):
    """Returns an rgb version of an rgba png file.

    Args:
        filename: A string indicating the name of the png file.
    """
    rgba = PIL.Image.open(filename)
    rgb = PIL.Image.new('RGB', rgba.size, (255, 255, 255))  # white background
    if rgba.mode == 'RGBA':
        rgb.paste(rgba, mask=rgba.split()[3])
    elif rgba.mode == 'RGB':
        rgb.paste(rgba)
    else:
        print("Unknown image type (not rgb or rgba)!\n")
    return rgb
