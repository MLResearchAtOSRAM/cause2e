"""
reader.py
================================================================
This module implements multiple Reader classes.

It is used to read data from files. Currently reading .csv and parquet files using pandas or spark
is supported.
"""


import pandas as pd


class Reader:
    """Main class for reading data.

    Attributes:
        technical: A cause2e Reader class based on either Pandas or Spark (only if the calling
        instance uses Spark, otherwise Pandas).
    """

    def __init__(self, data_path, spark):
        """Inits the Reader."""
        if spark:
            self.technical = _ReaderSpark(data_path, spark)
        else:
            self.technical = _ReaderPandas(data_path)

    def read_csv(self, **kwargs):
        """Returns the data from a csv as Pandas Dataframe.

        Args:
            **kwargs: Optional parameters for reading, e.g. nrows to limit the number of samples.
        """
        return self.technical.read_csv(**kwargs)

    def read_parquet(self, **kwargs):
        """Returns the data from a parquet as Pandas Dataframe.

        Args:
            **kwargs: Optional parameters for reading, e.g. nrows to limit the number of samples.
        """
        return self.technical.read_parquet(**kwargs)


class _ReaderPandas:
    """Technical class employing Pandas, used by the Reader class.

    Attributes:
        path: A string containing the full path to where the data is stored.
    """

    def __init__(self, data_path):
        """Inits _ReaderPandas."""
        self.path = data_path

    def read_csv(self, **kwargs):
        """Returns the data from a csv as Pandas Dataframe.

        Args:
            **kwargs: Optional parameters for reading, e.g. nrows to limit the number of samples.
        """
        return pd.read_csv(self.path, **kwargs)

    def read_parquet(self, **kwargs):
        """Returns the data from a parquet as Pandas Dataframe.

        Args:
            **kwargs: Optional parameters for reading, e.g. nrows to limit the number of samples.
        """
        if 'nrows' in kwargs:
            print('Pandas does not support partial reading of parquets.')
            print('Reading full data into RAM and restricting afterwards.')
            nrows = kwargs.pop('nrows')
            return pd.read_parquet(self.path, **kwargs).head(nrows)
        else:
            return pd.read_parquet(self.path, **kwargs)


class _ReaderSpark:
    """Technical class employing Spark, used by the Reader class.

    Attributes:
        path: A string containing the full path to where the data is stored.
        spark: A spark session.
    """

    def __init__(self, data_path, spark):
        """Inits _ReaderSpark."""
        self.path = data_path
        self.spark = spark

    def read_csv(self, **kwargs):
        """Returns the data from a csv as Pandas Dataframe.

        Args:
            **kwargs: Optional parameters for reading, e.g. nrows to limit the number of samples.
        """
        return self._read(format='csv', **kwargs)

    def read_parquet(self, **kwargs):
        """Returns the data from a parquet as Pandas Dataframe.

        Args:
            **kwargs: Optional parameters for reading, e.g. nrows to limit the number of samples.
        """
        return self._read(format='parquet', **kwargs)

    def _read(self, format, **kwargs):
        """Returns the data from a file as Pandas Dataframe.

        Args:
            format: A string indicating the file format.
            **kwargs: Optional parameters for reading, e.g. nrows to limit the number of samples.
        """
        if 'nrows' in kwargs:
            nrows = kwargs.pop('nrows')
            return self.spark.read.format(format)\
                                  .load(self.path,
                                        header=True,
                                        **kwargs
                                        )\
                                  .limit(nrows)\
                                  .toPandas()
        else:
            return self.spark.read.format(format)\
                                  .load(self.path,
                                        header=True,
                                        **kwargs
                                        )\
                                  .toPandas()
