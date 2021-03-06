"""
_preproc.py
================================================================
This module implements the Preprocessor class.

It is used to handle all the data preprocessing and allows to replicate the preprocessing steps
applied before causal discovery before the causal estimation is started with a possibly different
(e.g. larger) dataset.
"""


class Preprocessor():
    """Main class for preprocessing.

    Attributes:
        data: A pandas Dataframe containing the data to be analyzed.
        transformations: A list of transformations to be applied. Leave blank when preprocessing is
            used before causal discovery. Use the transformations of another preprocessor if you
            want to imitate its steps e.g. before causal estimation.
    """

    def __init__(self, data, transformations=[]):
        """Inits Preprocessor."""
        self.data = data
        self.transformations = transformations

    def combine_variables(self, name, input_cols, func, keep_old=True, store=True):
        """Combines data from existing variables into a new variable.

        Args:
            name: A string indicating the name of the new variable.
            input_cols: A list containing the names of the variables that are used for generating
                the new variable.
            func: A function describing how the new variable is calculated from the input variables.
            keep_old: Optional; A boolean indicating if we want to keep the input variables in our
                data. Defaults to True.
            store: Optional; A boolean indicating if the transformation should be stored in the
                transformations attribute of the preprocessor. Defaults to True.
        """
        vals = func(self.data, *input_cols)
        self.add_variable(name, vals, store=False)
        if not keep_old:
            for name in input_cols:
                self.delete_variable(name, store=False)
        if store:
            trafo_type = 'combine_variables'
            kwargs = {'name': name,
                      'input_cols': input_cols,
                      'func': func,
                      'keep_old': keep_old
                      }
            self.transformations.append({'fun': trafo_type, 'kwargs': kwargs})

    def add_variable(self, name, vals, store=True):
        """Adds a new variable to the data.

        Args:
            name: A string indicating the name of the new variable.
            vals: A column of values for the new variable.
            store: Optional; A boolean indicating if the transformation should be stored in the
                transformations attribute of the preprocessor. Defaults to True.
        """
        assert(not vals.empty)
        self.data[name] = vals
        if store:
            trafo_type = 'add_variable'
            kwargs = {'name': name}
            self.transformations.append({'fun': trafo_type, 'kwargs': kwargs})

    def delete_variable(self, name, store=True):
        """Deletes a variable from the data.

        Args:
            name: A string indicating the name of the target variable.
            store: Optional; A boolean indicating if the transformation should be stored in the
                transformations attribute of the preprocessor. Defaults to True.
        """
        del self.data[name]
        if store:
            trafo_type = 'delete_variable'
            kwargs = {'name': name}
            self.transformations.append({'fun': trafo_type, 'kwargs': kwargs})

    def rename_variable(self, current_name, new_name, store=True):
        """Renames a variable in the data.

        Args:
            current_name: A string indicating the current name of the variable.
            new_name: A string indicating the desired new name of the variable.
            store: Optional; A boolean indicating if the transformation should be stored in the
                transformations attribute of the preprocessor. Defaults to True.
        """
        self.data.rename(columns={current_name: new_name}, inplace=True)
        if store:
            trafo_type = 'rename_variable'
            kwargs = {'current_name': current_name,
                      'new_name': new_name
                      }
            self.transformations.append({'fun': trafo_type, 'kwargs': kwargs})

    def binarize_variable(self, name, one_val, zero_val=None, store=True):
        """Transforms a variable to a binary variable.

        Args:
            name: A string indicating the name of the target variable.
            one_val: The value that should be translated to 1.
            zero_val: Optional; the value that should be translated to 0.
                Use None if everything except for one_val should be translated to 0. Defaults to None.
        """
        if zero_val:
            translation_dict = {zero_val: 0}
        else:
            levels = self.data[name].unique()
            translation_dict = {val: 0 for val in levels}
        translation_dict[one_val] = 1
        self.data[name] = self.data[name].map(translation_dict)
        self.data.dropna(axis=0, inplace=True)
        if store:
            trafo_type = 'binarize_variable'
            kwargs = {'name': name,
                      'one_val': one_val,
                      'zero_val': zero_val
                      }
            self.transformations.append({'fun': trafo_type, 'kwargs': kwargs})

    def normalize_variables(self, store=True):
        """Replaces all variables by their z-scores.

        Args:
            store: Optional; A boolean indicating if the transformations should be stored in the
                transformations attribute of the preprocessor. Defaults to True.
        """
        for name in self.data.columns:
            self.normalize_variable(name, store=store)

    def normalize_variable(self, name, store=True):
        """Replaces a variable by its z-scores.

        Args:
            name: A string indicating the name of the target variable.
            store: Optional; A boolean indicating if the transformation should be stored in the
                transformations attribute of the preprocessor. Defaults to True.
        """
        self.data[name] = (self.data[name] - self.data[name].mean())/self.data[name].std(ddof=0)
        if store:
            trafo_type = 'normalize_variable'
            kwargs = {'name': name}
            self.transformations.append({'fun': trafo_type, 'kwargs': kwargs})

    def apply_stored_transformations(self, transformations, vals_list=None):
        """Imitates a stored sequence of preprocessing steps.

        Args:
            transformations: A list of transformations to be applied.
            vals_list: A list containing one column of values for each 'add_variable' step in the
                transformations. Defaults to None.
        """
        for trafo in transformations:
            if trafo['fun'] == 'add_variable':
                vals = vals_list.pop(0)
            else:
                vals = None
            self._apply_stored_transformation(trafo, vals)

    def _apply_stored_transformation(self, trafo, vals=None):
        """Imitates a stored preprocessing step.

        Args:
            trafo: A transformation to be applied.
            vals_list: A column of values if trafo is an 'add_variable' step. Defaults to None.
        """
        kwargs = trafo['kwargs']
        fun = trafo['fun']
        if fun == 'combine_variables':
            self._apply_stored_combination(kwargs)
        elif fun == 'add_variable':
            self._apply_stored_addition(kwargs, vals)
        elif fun == 'delete_variable':
            self._apply_stored_deletion(kwargs)
        elif fun == 'rename_variable':
            self._apply_stored_renaming(kwargs)
        elif fun == 'binarize_variable':
            self._apply_stored_binarization(kwargs)
        elif fun == 'normalize_variable':
            self._apply_stored_normalization(kwargs)

    def _apply_stored_combination(self, kwargs):
        """Applies a stored combination of data columns.

        Args:
            kwargs: A dictionary containing all information about the transformation.
        """
        name = kwargs['name']
        input_cols = kwargs['input_cols']
        func = kwargs['func']
        keep_old = kwargs['keep_old']
        self.combine_variables(name, input_cols, func, keep_old, store=False)

    def _apply_stored_addition(self, kwargs, vals):
        """Applies a stored addition of a data column.

        Args:
            kwargs: A dictionary containing all information about the transformation.
            vals: A column of values that should be added.
        """
        name = kwargs['name']
        self.add_variable(name, vals, store=False)

    def _apply_stored_deletion(self, kwargs):
        """Applies a stored deletion of a data column.

        Args:
            kwargs: A dictionary containing all information about the transformation.
        """
        name = kwargs['name']
        self.delete_variable(name, store=False)

    def _apply_stored_renaming(self, kwargs):
        """Applies a stored renaming of a data column.

        Args:
            kwargs: A dictionary containing all information about the transformation.
        """
        current_name = kwargs['current_name']
        new_name = kwargs['new_name']
        self.rename_variable(current_name, new_name, store=False)

    def _apply_stored_binarization(self, kwargs):
        """Applies a stored binarization of a data column.

        Args:
            kwargs: A dictionary containing all information about the transformation.
        """
        name = kwargs['name']
        one_val = kwargs['one_val']
        zero_val = kwargs['zero_val']
        self.binarize_variable(name, one_val, zero_val, store=False)

    def _apply_stored_normalization(self, kwargs):
        """Applies a stored normalization of a data column.

        Args:
            kwargs: A dictionary containing all information about the transformation.
        """
        name = kwargs['name']
        self.normalize_variable(name, store=False)
