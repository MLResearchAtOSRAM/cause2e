import unittest
from cause2e.path_mgr import PathManager
from cause2e.discovery import StructureLearner
import os
import pyspark


class LearnerForTesting(StructureLearner):
    def __init__(self, file_type, uses_spark=False, spark=None):
        self.file_type = file_type
        self.uses_spark = uses_spark
        self.spark = spark
        super().__init__(self.prepared_paths, self.prepared_spark)

    @property
    def prepared_paths(self):
        pwd = os.getcwd()
        data_dir = os.path.join(pwd, 'tests', 'fixtures')
        output_dir = 'whatever'
        return PathManager(experiment_name='bla',
                           data_name=self.data_name,
                           data_dir=data_dir,
                           output_dir=output_dir)

    @property
    def data_name(self):
        if self.file_type == 'csv':
            return 'linear_test.csv'
        elif self.file_type == 'parquet':
            return 'linear_test.parquet'

    @property
    def prepared_spark(self):
        if self.uses_spark:
            return self.spark
        else:
            return None


class PySparkTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        conf = pyspark.SparkConf().setMaster("local[*]").setAppName("testing")
        cls.sc = pyspark.SparkContext(conf=conf)
        cls.spark = pyspark.SQLContext(cls.sc)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


class TestDataTypes(unittest.TestCase):
    def setUp(self):
        self.learner = LearnerForTesting('csv')
        self.learner.read_csv()

    def test_threshold_mixed_data(self):
        cont = {'W0', 'W1', 'y'}
        disc = {'v0'}
        self.learner.continuous = cont
        self.learner.discrete = disc
        threshold = self.learner._plain_searcher._type_threshold_incomplete
        max_disc = max(self.learner.data[disc].nunique())
        min_cont = min(self.learner.data[cont].nunique())
        self.assertTrue(max_disc < threshold)
        self.assertTrue(threshold < min_cont)


class TestReaderPandasCsv(unittest.TestCase):
    def setUp(self):
        self.learner = LearnerForTesting('csv')

    def test_read(self):
        self.learner.read_csv()
        self.assertFalse(self.learner.data.empty)

    def test_read_restricted(self):
        n = 20
        self.learner.read_csv(nrows=n)
        self.assertEquals(len(self.learner.data), n)


class TestReaderPandasParquet(unittest.TestCase):
    def setUp(self):
        self.learner = LearnerForTesting('parquet')

    def test_read(self):
        self.learner.read_parquet()
        self.assertFalse(self.learner.data.empty)

    def test_read_restricted(self):
        n = 20
        self.learner.read_parquet(nrows=n)
        self.assertEquals(len(self.learner.data), n)


class TestReaderSparkCsv(PySparkTestCase):
    def setUp(self):
        self.learner = LearnerForTesting('csv', uses_spark=True, spark=self.spark)

    def test_read(self):
        self.learner.read_csv()
        self.assertFalse(self.learner.data.empty)

    def test_read_restricted(self):
        n = 20
        self.learner.read_csv(nrows=n)
        self.assertEquals(len(self.learner.data), n)


# class TestReaderSparkParquet(PySparkTestCase):
#     def setUp(self):
#         self.learner = LearnerForTesting('parquet', uses_spark=True, spark=self.spark)

#     def test_read(self):
#         self.learner.read_parquet()
#         self.assertFalse(self.learner.data.empty)

#     def test_read_restricted(self):
#         n = 20
#         self.learner.read_parquet(nrows=n)
#         self.assertEquals(len(self.learner.data), n)


if __name__ == '__main__':
    unittest.main()
