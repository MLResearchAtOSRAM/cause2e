import os
import pandas as pd
import matplotlib.pyplot as plt


def perform_sensitivity_analysis(outcome, file_dir, experiment_names, effect_type, save):
    dfs = _get_effects_on_outcome(outcome, file_dir, experiment_names, effect_type)
    df_min_max = _get_df_min_max(dfs)
    file_name = None
    if save:
        effect_str = _transform_effect_type(effect_type)
        file_name = os.path.join(file_dir, f'sensitivity_analysis_{effect_str}.png')
    _plot_confidence_intervals(df_min_max, effect_type, file_name=file_name)


def _get_effects_on_outcome(outcome, file_dir, experiment_names, effect_type):
    return [_get_effect_on_outcome(
                                   outcome=outcome,
                                   filename=_create_result_name(file_dir, exp_name),
                                   effect_type=effect_type
                                   )
            for exp_name in experiment_names]


def _create_result_name(file_dir, experiment_name):
    return os.path.join(file_dir, experiment_name + '_results.csv')


def _get_effect_on_outcome(outcome, filename, effect_type):
    df = _get_result_df_from_csv(filename, effect_type, 10)
    df = df[outcome].drop('Treatment')
    df.name = _transform_effect_type(effect_type)
    df.rename_axis('Treatment')
    return df


def _get_result_df_from_csv(filename, effect_type, n_vars):
    if effect_type == 'nonparametric-ate':
        header = 3
    elif effect_type == 'nonparametric-nde':
        header = 3 + n_vars + 3
    elif effect_type == 'nonparametric-nie':
        header = 3 + 2*n_vars + 6
    return pd.read_csv(filename, header=header, nrows=n_vars, index_col='Outcome')


def _transform_effect_type(effect_type):
    if effect_type in {'nonparametric-ate', 'nonparametric-nde', 'nonparametric-nie'}:
        return effect_type[-3:].upper()
    else:
        raise KeyError('Unknown effect type!')


def _plot_confidence_intervals(df, effect_type, file_name=None):
    n_variables = len(df)
    for x1, x2, y in zip(df['Min'], df['Max'], range(n_variables)):
        plt.plot((x1, x2), (y, y), 'ro-', color='orange')
        plt.yticks(ticks=range(n_variables), labels=df.index)
        effect_str = _transform_effect_type(effect_type)
        plt.title(f'Estimated {effect_str}s on Income')
        if file_name:
            plt.savefig(file_name, bbox_inches='tight')
    plt.close('all')


def _get_df_min_max(dfs):
    df_combined = pd.concat(dfs, axis=1)
    df = pd.concat([df_combined.min(axis=1), df_combined.max(axis=1)], axis=1)
    df.columns = ['Min', 'Max']
    return df
