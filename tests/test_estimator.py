import unittest
from cause2e.path_mgr import PathManager
from cause2e.estimation import Estimator
from cause2e import knowledge
import os


class EstimatorForTesting(Estimator):
    def __init__(self, file_type='csv', dataset='linear_test', uses_spark=False, spark=None):
        self.file_type = file_type
        self.dataset = dataset
        self.transformations = []
        self.validation_dict = {}
        self.uses_spark = uses_spark
        self.spark = spark
        super().__init__(self.prepared_paths, self.transformations, self.validation_dict, self.prepared_spark)

    @property
    def prepared_paths(self):
        pwd = os.getcwd()
        data_dir = os.path.join(pwd, 'tests', 'fixtures')
        output_dir = os.path.join(pwd, 'tests', 'output')
        return PathManager(experiment_name='sprinkler',
                           data_name=self.data_name,
                           data_dir=data_dir,
                           output_dir=output_dir)

    @property
    def data_name(self):
        return f'{self.dataset}.{self.file_type}'

    @property
    def validation_dict(self):
        if self.dataset == 'sprinkler':
            return self._sprinkler_validation_dict
        else:
            return {}

    @property
    def _sprinkler_validation_dict(self):
        validation_creator = knowledge.ValidationCreator()
        validation_creator.add_expected_effect(('Sprinkler', 'Wet', 'nonparametric-ate'), ('greater', 0))
        validation_creator.add_expected_effect(('Wet', 'Slippery', 'nonparametric-ate'), ('greater', 0))
        validation_creator.add_expected_effect(('Sprinkler', 'Rain', 'nonparametric-nde'), ('less', 0))
        validation_creator.add_expected_effect(('Slippery', 'Season', 'nonparametric-nie'), ('between', 0.2, 0.4))
        return validation_creator.expected_effects

    @property
    def prepared_spark(self):
        if self.uses_spark:
            return self.spark
        else:
            return None


class TestManualEstimation(unittest.TestCase):
    def setUp(self):
        self.estimator = EstimatorForTesting(dataset='sprinkler')

    def test_manual_estimation(self):
        variables = {'Season', 'Rain', 'Sprinkler', 'Wet', 'Slippery'}
        self._read_data(variables)
        self.estimator.binarize_variable('Season', one_val='Spring', zero_val='Winter')
        self.estimator.initialize_model('Rain', 'Slippery', 'nonparametric-ate')
        self.estimator.identify_estimand(verbose=True)
        self.estimator.estimate_effect(verbose=True, method_name="backdoor.linear_regression")
        self.estimator.check_robustness(method_name="random_common_cause", verbose=True)
        self.estimator.compare_to_noncausal_regression(input_cols={'Rain', 'Sprinkler'})

    def _read_data(self, variables):
        self.estimator.read_csv(index_col=0)
        self.assertFalse(self.estimator.data.empty)
        self.assertEqual(self.estimator.variables, variables)
        self.estimator.discrete = self.estimator.variables
        self.estimator.continuous = set()


if __name__ == '__main__':
    unittest.main()
