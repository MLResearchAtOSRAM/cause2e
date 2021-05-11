import unittest
from cause2e.reader import Reader
import os
import pyspark


class TestPandasReaderCsv(unittest.TestCase):
    def setUp(self):
        pwd = os.getcwd()
        data_path = os.path.join(pwd, 'tests', 'fixtures', 'linear_test.csv')
        self.reader = Reader(data_path, spark=None)

    def test_read_csv(self):
        data = self.reader.read_csv()
        self.assertFalse(data.empty)

    def test_read_csv_restricted(self):
        n = 20
        data = self.reader.read_csv(nrows=n)
        self.assertEquals(len(data), n)


class TestPandasReaderParquet(unittest.TestCase):
    def setUp(self):
        pwd = os.getcwd()
        data_path = os.path.join(pwd, 'tests', 'fixtures', 'linear_test.parquet')
        self.reader = Reader(data_path, spark=None)

    def test_read_parquet(self):
        data = self.reader.read_parquet()
        self.assertFalse(data.empty)

    def test_read_parquet_restricted(self):
        n = 20
        data = self.reader.read_parquet(nrows=n)
        self.assertEquals(len(data), n)


class PySparkTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conf = pyspark.SparkConf().setMaster("local[2]").setAppName("testing")
        cls.sc = pyspark.SparkContext(conf=conf)
        cls.spark = pyspark.SQLContext(cls.sc)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


class TestSparkReaderCsv(PySparkTestCase):
    def setUp(self):
        pwd = os.getcwd()
        data_path = os.path.join(pwd, 'tests', 'fixtures', 'linear_test.csv')
        self.reader = Reader(data_path, spark=self.spark)

    def test_read_csv(self):
        data = self.reader.read_csv()
        self.assertFalse(data.empty)

    def test_read_csv_restricted(self):
        n = 20
        data = self.reader.read_csv(nrows=n)
        self.assertEquals(len(data), n)


# class TestSparkReaderParquet(PySparkTestCase):
#     def setUp(self):
#         pwd = os.getcwd()
#         data_path = os.path.join(pwd, 'tests', 'fixtures', 'linear_test.parquet')
#         self.reader = Reader(data_path, spark=self.spark)

#     def test_read_parquet(self):
#         data = self.reader.read_parquet()
#         self.assertFalse(data.empty)

#     def test_read_parquet_restricted(self):
#         n = 20
#         data = self.reader.read_parquet(nrows=n)
#         self.assertEquals(len(data), n)


if __name__ == '__main__':
    unittest.main()
