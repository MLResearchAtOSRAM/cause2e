import unittest
from cause2e.path_mgr import PathManager
from cause2e.estimator import Estimator, EstimatorDatabricks
from cause2e import knowledge
import os


class EstimatorForTesting(Estimator):
    def __init__(self, file_type='csv', dataset='linear_test', uses_spark=False, spark=None):
        parameters = ParameterPreparator(file_type='csv', dataset='linear_test', uses_spark=False, spark=None)
        super().__init__(parameters.prepared_paths,
                         parameters.transformations,
                         parameters.validation_dict,
                         parameters.prepared_spark
                         )


class EstimatorDatabricksForTesting(EstimatorDatabricks):
    def __init__(self, file_type='csv', dataset='linear_test', uses_spark=False, spark=None):
        parameters = ParameterPreparator(file_type='csv', dataset='linear_test', uses_spark=False, spark=None)
        super().__init__(parameters.prepared_paths,
                         parameters.prepared_spark,
                         parameters.transformations,
                         parameters.validation_dict,
                         )


class ParameterPreparator():
    def __init__(self, file_type='csv', dataset='linear_test', uses_spark=False, spark=None):
        self.file_type = file_type
        self.dataset = dataset
        self.transformations = []
        self.uses_spark = uses_spark
        self.spark = spark

    @property
    def prepared_paths(self):
        pwd = os.getcwd()
        data_dir = os.path.join(pwd, 'tests', 'fixtures', 'data')
        output_dir = os.path.join(pwd, 'tests', 'fixtures', 'output')
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


class TestEstimation(unittest.TestCase):
    def setUp(self):
        self.estimator = EstimatorForTesting(dataset='sprinkler')
        self.variables = {'Season', 'Rain', 'Sprinkler', 'Wet', 'Slippery'}
        self._read_data(self.variables)

    def test_manual_estimation(self):
        self._read_data(self.variables)
        self.estimator.binarize_variable('Season', one_val='Spring', zero_val='Winter')
        self.estimator.initialize_model('Rain', 'Slippery', 'nonparametric-ate')
        self.estimator.identify_estimand(verbose=True)
        self.estimator.estimate_effect(verbose=True, method_name="backdoor.linear_regression")
        self.estimator.check_robustness(method_name="random_common_cause", verbose=True)
        self.estimator.compare_to_noncausal_regression(input_cols={'Rain', 'Sprinkler'})

    def test_quick_estimation(self):
        self.estimator.binarize_variable('Season', one_val='Spring', zero_val='Winter')
        self.estimator.run_quick_analysis(treatment='Rain',
                                          outcome='Slippery',
                                          estimand_type='nonparametric-ate',
                                          robustness_method="random_common_cause",
                                          verbose=True
                                          )
        self.estimator.compare_to_noncausal_regression(input_cols={'Rain', 'Sprinkler'})

    def test_all_quick_estimations(self):
        self.estimator.binarize_variable('Season', one_val='Spring', zero_val='Winter')
        self.estimator.run_all_quick_analyses(estimand_types=['nonparametric-ate',
                                                              'nonparametric-nde',
                                                              'nonparametric-nie'
                                                              ],
                                              verbose=True,
                                              show_tables=True,
                                              show_heatmaps=True,
                                              show_validation=True,
                                              show_largest_effects=True,
                                              generate_pdf_report=True
                                              )

    def test_all_quick_estimations_no_reporting(self):
        self.estimator.binarize_variable('Season', one_val='Spring', zero_val='Winter')
        self.estimator.run_all_quick_analyses(estimand_types=['nonparametric-ate',
                                                              'nonparametric-nde',
                                                              'nonparametric-nie'
                                                              ],
                                              verbose=False,
                                              show_tables=False,
                                              show_heatmaps=False,
                                              show_validation=False,
                                              show_largest_effects=False,
                                              generate_pdf_report=False
                                              )

    def _read_data(self, variables):
        self.estimator.read_csv(index_col=0)
        self.assertFalse(self.estimator.data.empty)
        self.assertEqual(self.estimator.variables, variables)
        self.estimator.discrete = self.estimator.variables
        self.estimator.continuous = set()


class TestEstimationDatabricks(unittest.TestCase):
    def setUp(self):
        self.estimator = EstimatorDatabricksForTesting(dataset='sprinkler')
        self.variables = {'Season', 'Rain', 'Sprinkler', 'Wet', 'Slippery'}
        self._read_data(self.variables)

    def test_all_quick_estimations(self):
        self.estimator.binarize_variable('Season', one_val='Spring', zero_val='Winter')
        self.estimator.run_all_quick_analyses(estimand_types=['nonparametric-ate',
                                                              'nonparametric-nde',
                                                              'nonparametric-nie'
                                                              ],
                                              verbose=True,
                                              show_tables=True,
                                              show_heatmaps=True,
                                              show_validation=True,
                                              show_largest_effects=True,
                                              generate_pdf_report=True
                                              )
        self.estimator.str_report

    def _read_data(self, variables):
        self.estimator.read_csv(index_col=0)
        self.assertFalse(self.estimator.data.empty)
        self.assertEqual(self.estimator.variables, variables)
        self.estimator.discrete = self.estimator.variables
        self.estimator.continuous = set()


if __name__ == '__main__':
    unittest.main()
