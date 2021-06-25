"""
_result_mgr.py
================================================================
This module implements the ResultManager class.

It is used as a helper class for managing the output of analyses after the estimation of causal 
effects has been performed. It derives heatmaps, tables, selected validations and a pdf report
from the results.
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class ResultManager:
    """Helper class for managing the output of analyses."""

    def __init__(self, quick_results_list, validation_dict):
        self._quick_results_list = quick_results_list
        self._validation_dict = validation_dict

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
            
        Raises:
            KeyError: 'estimand_type must be nonparametric-ate, nonparametric-nde
                or nonparametric-nie'
        """
        if estimand_type == 'nonparametric-ate':
            df = self._quick_results_ate
        elif estimand_type == 'nonparametric-nde':
            df = self._quick_results_nde
        elif estimand_type == 'nonparametric-nie':
            df = self._quick_results_nie
        else:
            raise KeyError("estimand_type must be nonparametric-ate, nonparametric-nde or "
                           + "nonparametric-nie")
        return df.loc[treatment]['Estimated_effect'][outcome]

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
    
    def show_validation(self, save_to_name=None, img_width=1000, img_height=500):
        """Shows if selected estimated effects match previous expectations.
        
        Args:
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the validation report should be saved. Defaults to None.
            img_width: Optional; The width of the saved png. Defaults to 1000.
            img_height: Optional; The height of the saved png. Defaults to 500.
        """
        passed_validations = {k for (k, v) in self._validation_dict.items() if v['Valid']}
        failed_validations = self._validation_dict.keys() - passed_validations
        result_str = ""
        if passed_validations:
            result_str += self._generate_validation_strings(passed_validations, 'Passed')
            result_str += " \n"
        if failed_validations:
            result_str += self._generate_validation_strings(failed_validations, 'Failed')
        if save_to_name:
            self._save_validations_as_png(result_str, save_to_name, img_width, img_height)
        print(result_str)
        
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
        expectation = entry['Expected']
        expected_str = self._get_expected_str(expectation)
        estimated_str = "{:.2f}".format(entry['Estimated'])
        valid_str = entry['Valid']
        return f"Estimated {effect[2]} of {effect[0]} on {effect[1]}: {estimated_str} (expected: {expected_str}) -> {valid_str} \n"
    
    def _get_expected_str(self, expectation):
        """Returns an expected effect in pretty string form.

        Args:
            expectation: A tuple describing the expected effect type and size.
        """
        type_ = expectation[0]
        if type_ in {'less', 'greater', 'equal'}:
            val = expectation[1]
            if type_ == 'equal':
                return f"equal to {val}"
            else:
                return f"{type_} than {val}"
        elif type_ == 'between':
            lower_bound = expectation[1]
            upper_bound = expectation[2]
            return f"between {lower_bound} and {upper_bound}"
        
    def _save_validations_as_png(self, validation_str, save_to_name, img_width=1000, img_height=500):
        """Saves the validation results to a png file for reporting purposes.

        Args:
            validation_str: A string containing the validation results to be saved.
            save_to_name: Optional; A string indicating the beginning of the name of the png file
                where the validation report should be saved.
            img_width: Optional; The width of the saved png. Defaults to 1000.
            img_height: Optional; The height of the saved png. Defaults to 500.
        """
        img = Image.new('RGBA', (img_width, img_height), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((100, 100), validation_str, fill=(0, 0, 0))
        img.save(save_to_name, 'png')
        
    def generate_pdf_report(self, output_name, graph, heatmaps, validations, results, dpi=(300, 300)):
        """Generates a pdf report with the causal graph and all results.

        Args:
            output_name: A string indicating the name of the output pdf.
            graph: A string indicating the name of the png where the causal graph is stored.
            heatmaps: A list of strings indicating the names of the pngs where the heatmaps are
                stored.
            validations: A string indicating the name of the png where the validation results
                are stored.
            results: A list of strings indicating the names of the pngs where the quantiative
                results are stored.
            dpi: Optional; A pair indicating the resolution. Defaults to (300, 300).
        """
        input_names = [graph] + heatmaps + [validations] + results
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
