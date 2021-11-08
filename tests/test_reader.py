import unittest
from cause2e._reader import Reader
import os


class TestPandasReaderCsv(unittest.TestCase):
    def setUp(self):
        pwd = os.getcwd()
        data_path = os.path.join(pwd, 'tests', 'fixtures', 'data', 'linear_test.csv')
        self.reader = Reader(data_path, spark=None)

    def test_read_csv(self):
        data = self.reader.read_csv()
        self.assertFalse(data.empty)

    def test_read_csv_restricted(self):
        n = 20
        data = self.reader.read_csv(nrows=n)
        self.assertEqual(len(data), n)


class TestPandasReaderParquet(unittest.TestCase):
    def setUp(self):
        pwd = os.getcwd()
        data_path = os.path.join(pwd, 'tests', 'fixtures', 'data', 'linear_test.parquet')
        self.reader = Reader(data_path, spark=None)

    def test_read_parquet(self):
        data = self.reader.read_parquet()
        self.assertFalse(data.empty)

    def test_read_parquet_restricted(self):
        n = 20
        data = self.reader.read_parquet(nrows=n)
        self.assertEqual(len(data), n)


if __name__ == '__main__':
    unittest.main()
