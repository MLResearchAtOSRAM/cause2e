import unittest
from cause2e.path_mgr import PathManager, PathManagerQuick
import os


class TestPathMgrQuick(unittest.TestCase):

    def setUp(self):
        self.mgr = PathManagerQuick('exp_name', 'data_name', 'pwd', nested_output=False)

    def test_new_exp_name(self):
        # other input args will work accordingly
        old_name = os.path.join('pwd', 'output', 'exp_name') + '.dot'
        new_name = os.path.join('pwd', 'output', 'new_exp_name') + '.dot'
        self.assertEqual(self.mgr.dot_name, old_name)
        self.mgr._experiment_name = 'new_exp_name'
        self.assertEqual(self.mgr.dot_name, new_name)

    def test_protection(self):
        # other protected attributes will work accordingly
        with self.assertRaises(AttributeError):
            self.mgr.dot_name = 'new_dot_name'


class TestPathMgrFull(unittest.TestCase):

    def setUp(self):
        self.mgr = PathManager('exp_name', 'data_name', 'data_dir', 'output_dir')

    def test_new_exp_name(self):
        # other input args will work accordingly
        old_name = os.path.join('output_dir', 'exp_name') + '.dot'
        new_name = os.path.join('output_dir', 'new_exp_name') + '.dot'
        self.assertEqual(self.mgr.dot_name, old_name)
        self.mgr._experiment_name = 'new_exp_name'
        self.assertEqual(self.mgr.dot_name, new_name)

    def test_protection(self):
        # other protected attributes will work accordingly
        with self.assertRaises(AttributeError):
            self.mgr.dot_name = 'new_dot_name'


if __name__ == '__main__':
    unittest.main()
