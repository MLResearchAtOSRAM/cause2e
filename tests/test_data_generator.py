import unittest
from cause2e.path_mgr import PathManager
from cause2e.data_generator import DataGenerator
import os


class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        pwd = os.getcwd()
        data_dir = os.path.join(pwd, 'tests', 'fixtures', 'data')
        paths = PathManager(experiment_name='foo',
                            data_name='test_generation',
                            data_dir=data_dir,
                            output_dir='bar')
        self.generator = DataGenerator(paths)

    def test_generate_linear_dataset(self):
        beta = 2
        n_common_causes = 3
        nrows = 1000
        self.generator.generate_linear_dataset(beta, n_common_causes, nrows)
        self.assertEqual(len(self.generator.data), nrows)
        self.assertEqual(len(self.generator.data.columns), n_common_causes + 2)
        self.generator.display_graph()
        self.generator.write_csv()
