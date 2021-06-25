import unittest
from cause2e._data_type_mgr import DataTypeManager
from pathlib import Path
import pandas as pd


class TestTypes(unittest.TestCase):
    def setUp(self):
        data_path = Path('tests', 'fixtures', 'linear_test.csv')
        self.data = pd.read_csv(data_path)

    def test_no_vars(self):
        type_mgr = DataTypeManager(self.data)
        with self.assertRaises(AssertionError):
            type_mgr._check_type_input()

    def test_nonexistent_vars(self):
        disc = {'lala'}
        type_mgr = DataTypeManager(self.data, desired_discrete=disc)
        with self.assertRaises(AssertionError):
            type_mgr._check_type_input()

        cont = {'haha'}
        type_mgr = DataTypeManager(self.data, desired_continuous=cont)
        with self.assertRaises(AssertionError):
            type_mgr._check_type_input()

    def test_conflicting_vars(self):
        disc = {'v0', 'y'}
        cont = {'W0', 'y'}
        type_mgr = DataTypeManager(self.data, cont, disc)
        with self.assertRaises(AssertionError):
            type_mgr._check_type_input(complete=False)

    def test_mixed_data(self):
        disc = {'v0'}
        cont = {'W0', 'W1', 'y'}
        type_mgr = DataTypeManager(self.data, cont, disc)
        with self.assertRaises(AssertionError):
            type_mgr.enforce_desired_types(complete=True)
        type_mgr = DataTypeManager(self.data, cont, disc)
        type_mgr.enforce_desired_types(complete=False, verbose=False)


if __name__ == '__main__':
    unittest.main()
