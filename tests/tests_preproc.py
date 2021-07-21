import unittest
from cause2e._preproc import Preprocessor
from pathlib import Path
import pandas as pd
import copy


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        data_path = Path('tests', 'fixtures', 'linear_test.csv')
        data = pd.read_csv(data_path)
        self.pproc = Preprocessor(data)
        self.good_variables = {'y', 'v0'}
        self.bad_variables = {'lala'}

    def test_delete_good_variables(self):
        n_deleted = len(self.good_variables)
        n_rows, n_cols = self.pproc.data.shape
        for var in self.good_variables:
            self.pproc.delete_variable(var)
        n_rows_new, n_cols_new = self.pproc.data.shape
        cols = set(self.pproc.data.columns)
        self.assertEquals(n_rows, n_rows_new)
        self.assertEquals(n_cols - n_deleted, n_cols_new)
        self.assertTrue(self.good_variables.isdisjoint(cols))

    def test_delete_bad_variables(self):
        with self.assertRaises(KeyError):
            for var in self.bad_variables:
                self.pproc.delete_variable(var)

    def combine_good_variables(self, keep_old):
        def func(data, col_1, col_2):
            return data[col_1] + 2*data[col_2]
        data = self.pproc.data
        new_col = 'W01_Delta'
        input_cols = ['W0', 'W1']
        desired_result = data[input_cols[0]] + 2*data[input_cols[1]]
        self.pproc.combine_variables(new_col, input_cols, func, keep_old)
        self.assertTrue(new_col in data)
        self.assertTrue(data[new_col].equals(desired_result))
        self.assertEqual(set(input_cols).issubset(data), keep_old)

    def test_combine_good_variables_delete_old(self):
        self.combine_good_variables(keep_old=False)

    def test_combine_good_variables_keep_old(self):
        self.combine_good_variables(keep_old=False)

    def test_add_variable_good(self):
        data = self.pproc.data
        vals = data['W0'] * 2
        name = 'DoubleW0'
        self.pproc.add_variable(name, vals)
        self.assertTrue(name in data)
        self.assertTrue(data[name].equals(vals))

    def test_rename_variable(self):
        data = self.pproc.data
        current_name = 'W0'
        new_name = 'W0_new'
        self.assertTrue(current_name in data)
        self.assertFalse(new_name in data)
        self.pproc.rename_variable('W0', 'W0_new')
        self.assertFalse(current_name in data)
        self.assertTrue(new_name in data)


class TestImitatePreprocessor(unittest.TestCase):
    def setUp(self):
        data_path = Path('tests', 'fixtures', 'linear_test.csv')
        data = pd.read_csv(data_path)
        self.pproc_1 = Preprocessor(data)
        self.pproc_2 = copy.deepcopy(self.pproc_1)

    def test_apply_stored_deletion(self):
        data = self.pproc_2.data
        self.assertEqual(len(self.pproc_2.transformations), 0)
        self.assertTrue('W0' in data)
        kwargs = {'name': 'W0'}
        trafo = {'fun': 'delete_variable', 'kwargs': kwargs}
        self.pproc_2._apply_stored_transformation(trafo)
        self.assertFalse('W0' in data)

    def test_apply_stored_combination(self):
        def func(data, col_1, col_2):
            return data[col_1] + 2*data[col_2]
        name = 'W01_Delta'
        input_cols = ['W0', 'W1']
        kwargs = {'name': name,
                  'input_cols': input_cols,
                  'func': func,
                  'keep_old': False
                  }
        trafo = {'fun': 'combine_variables', 'kwargs': kwargs}
        self.pproc_2._apply_stored_transformation(trafo)
        data = self.pproc_2.data
        self.assertTrue('W01_Delta' in data)

    def test_apply_stored_addition(self):
        data = self.pproc_2.data
        vals = data['W0'] * 2
        name = 'DoubleW0'
        kwargs = {'name': name}
        trafo = {'fun': 'add_variable', 'kwargs': kwargs}
        self.pproc_2._apply_stored_transformation(trafo, vals)
        self.assertTrue(name in data)
        self.assertTrue(data[name].equals(vals))

    def test_apply_stored_renaming(self):
        data = self.pproc_2.data
        current_name = 'W0'
        new_name = 'W0_new'
        kwargs = {'current_name': current_name, 'new_name': new_name}
        trafo = {'fun': 'rename_variable', 'kwargs': kwargs}
        self.assertTrue(current_name in data)
        self.assertFalse(new_name in data)
        self.pproc_2._apply_stored_transformation(trafo)
        self.assertFalse(current_name in data)
        self.assertTrue(new_name in data)

    def test_imitate_deletion(self):
        data_1 = self.pproc_1.data
        data_2 = self.pproc_2.data
        self.pproc_1.delete_variable('W0')
        trafo = self.pproc_1.transformations[0]
        kwargs = {'name': 'W0'}
        self.assertEqual(trafo, {'fun': 'delete_variable', 'kwargs': kwargs})
        self.pproc_2._apply_stored_transformation(trafo)
        self.assertFalse('W0' in data_2)
        self.assertTrue(data_1.equals(data_2))

    def test_imitate_combination(self):
        def func(data, col_1, col_2):
            return data[col_1] + 2*data[col_2]
        data_1 = self.pproc_1.data
        data_2 = self.pproc_2.data
        name = 'W01_Delta'
        input_cols = ['W0', 'W1']
        self.pproc_1.combine_variables(name, input_cols, func)
        trafo = self.pproc_1.transformations[0]
        kwargs = {'name': name,
                  'input_cols': input_cols,
                  'func': func,
                  'keep_old': True
                  }
        self.assertEqual(trafo, {'fun': 'combine_variables', 'kwargs': kwargs})
        self.pproc_2._apply_stored_transformation(trafo)
        self.assertTrue('W01_Delta' in data_2)
        self.assertTrue(data_1.equals(data_2))

    def test_imitate_addition(self):
        data_1 = self.pproc_1.data
        data_2 = self.pproc_2.data
        vals = data_1['W0'] * 2
        name = 'DoubleW0'
        self.pproc_1.add_variable(name, vals)
        trafo = self.pproc_1.transformations[0]
        kwargs = {'name': name}
        self.assertEqual(trafo, {'fun': 'add_variable', 'kwargs': kwargs})
        self.pproc_2._apply_stored_transformation(trafo, vals)
        self.assertTrue('DoubleW0' in data_2)
        self.assertTrue(data_1.equals(data_2))

    def test_imitate_renaming(self):
        data_1 = self.pproc_1.data
        data_2 = self.pproc_2.data
        self.pproc_1.rename_variable('W0', 'W0_new')
        trafo = self.pproc_1.transformations[0]
        kwargs = {'current_name': 'W0', 'new_name': 'W0_new'}
        self.assertEqual(trafo, {'fun': 'rename_variable', 'kwargs': kwargs})
        self.pproc_2._apply_stored_transformation(trafo)
        self.assertTrue('W0_new' in data_2)
        self.assertFalse('W0' in data_2)
        self.assertTrue(data_1.equals(data_2))

    def test_imitate_transformations(self):
        def func(data, col_1, col_2):
            return data[col_1] + 2*data[col_2]
        data_1 = self.pproc_1.data
        data_2 = self.pproc_2.data
        name_add = 'W01_Delta'
        input_cols = ['W0', 'W1']
        self.pproc_1.combine_variables(name_add, input_cols, func)
        vals = data_1['W0'] * 2
        name_comb = 'DoubleW0'
        self.pproc_1.add_variable(name_comb, vals)
        name_del = 'W0'
        self.pproc_1.delete_variable(name_del)
        name_ren_current = 'W2'
        name_ren_new = 'W2_new'
        self.pproc_1.rename_variable(name_ren_current, name_ren_new)
        transformations = self.pproc_1.transformations
        vals_list = [vals]
        self.pproc_2.apply_stored_transformations(transformations, vals_list)
        self.assertTrue(data_1.equals(data_2))

    def test_imitate_transformations_no_vals(self):
        def func(data, col_1, col_2):
            return data[col_1] + 2*data[col_2]
        data_1 = self.pproc_1.data
        data_2 = self.pproc_2.data
        name_add = 'W01_Delta'
        input_cols = ['W0', 'W1']
        self.pproc_1.combine_variables(name_add, input_cols, func)
        name_del = 'W0'
        self.pproc_1.delete_variable(name_del)
        transformations = self.pproc_1.transformations
        self.pproc_2.apply_stored_transformations(transformations)
        self.assertTrue(data_1.equals(data_2))


if __name__ == '__main__':
    unittest.main()
