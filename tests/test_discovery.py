import unittest
from cause2e.path_mgr import PathManager
from cause2e.discovery import StructureLearner
from cause2e import knowledge
import os
# from pycausal.pycausal import pycausal as pc


class LearnerForTesting(StructureLearner):
    def __init__(self, file_type='csv', dataset='linear_test', uses_spark=False, spark=None):
        self.file_type = file_type
        self.dataset = dataset
        self.uses_spark = uses_spark
        self.spark = spark
        super().__init__(self.prepared_paths, self.prepared_spark)

    @property
    def prepared_paths(self):
        pwd = os.getcwd()
        data_dir = os.path.join(pwd, 'tests', 'fixtures')
        output_dir = os.path.join(pwd, 'tests', 'output')
        return PathManager(experiment_name='test_discovery',
                           data_name=self.data_name,
                           data_dir=data_dir,
                           output_dir=output_dir)

    @property
    def data_name(self):
        return f'{self.dataset}.{self.file_type}'

    @property
    def prepared_spark(self):
        if self.uses_spark:
            return self.spark
        else:
            return None


class TestReaderPandasCsv(unittest.TestCase):
    def setUp(self):
        self.learner = LearnerForTesting('csv')

    def test_read(self):
        self.learner.read_csv()
        self.assertFalse(self.learner.data.empty)

    def test_read_restricted(self):
        n = 20
        self.learner.read_csv(nrows=n)
        self.assertEqual(len(self.learner.data), n)


class TestReaderPandasParquet(unittest.TestCase):
    def setUp(self):
        self.learner = LearnerForTesting('parquet')

    def test_read(self):
        self.learner.read_parquet()
        self.assertFalse(self.learner.data.empty)

    def test_read_restricted(self):
        n = 20
        self.learner.read_parquet(nrows=n)
        self.assertEqual(len(self.learner.data), n)


# class TestJavaVM(unittest.TestCase):
#     def test_vm(self):
#         pc().start_vm()
#         pc().stop_vm()

# TODO: Develop strategy for multiple tests that require JavaVM
# class TestDataTypes(unittest.TestCase):
#     def setUp(self):
#         self.learner = LearnerForTesting('csv')
#         self.learner.read_csv()

#     def test_threshold_mixed_data(self):
#         cont = {'W0', 'W1', 'y'}
#         disc = {'v0'}
#         self.learner.continuous = cont
#         self.learner.discrete = disc
#         searcher =
#         threshold = self.learner._plain_searcher._type_threshold_incomplete
#         max_disc = max(self.learner.data[disc].nunique())
#         min_cont = min(self.learner.data[cont].nunique())
#         self.assertTrue(max_disc < threshold)
#         self.assertTrue(threshold < min_cont)

# class TestSearch(unittest.TestCase):
#     def setUp(self):
#         self.learner = LearnerForTesting('csv')
#         self.learner.read_csv()

#     def test_quick_search(self):
#         self.learner.run_quick_search(verbose=False, keep_vm=False, show_graph=False, save_graph=False)
#         self.assertTrue(self.learner.graph)

#     def test_manual_quick_search(self):
#         pc().start_vm()
#         tetrad = s.tetradrunner()
#         tetrad.run(algoId='fges',
#                    dfs=self.learner.data,
#                    dataType='mixed',
#                    numCategoriesToDiscretize=10,
#                    # priorKnowledge=knowledge,
#                    verbose=False,
#                    )
#         pc().stop_vm()


class TestFullAnalysis(unittest.TestCase):
    def setUp(self):
        self.learner = LearnerForTesting(dataset='sprinkler')

    def test_full_analysis(self):
        variables = {'Season', 'Rain', 'Sprinkler', 'Wet', 'Slippery'}
        self._read_data(variables)
        edge_creator = self._create_graph_knowledge()
        validation_creator = self._create_validation_knowledge()
        self.learner.set_knowledge(edge_creator=edge_creator, validation_creator=validation_creator, show=True)
        self.learner.run_quick_search(verbose=False, keep_vm=False, show_graph=True)
        self.learner.binarize_variable('Season', one_val='Spring', zero_val='Winter')
        self.learner._create_estimator()
        self.learner.run_all_quick_analyses()

    def _read_data(self, variables):
        self.learner.read_csv(index_col=0)
        self.assertFalse(self.learner.data.empty)
        self.assertEqual(self.learner.variables, variables)
        self.learner.discrete = self.learner.variables
        self.learner.continuous = set()

    def _create_graph_knowledge(self):
        edge_creator = knowledge.EdgeCreator()
        edge_creator.forbid_edges_from_groups({'Season'}, incoming=self.learner.variables)
        edge_creator.forbid_edges_from_groups({'Slippery'}, outgoing=self.learner.variables)
        edge_creator.require_edge('Sprinkler', 'Wet')
        edge_creator.forbid_edge('Sprinkler', 'Rain')
        return edge_creator

    def _create_validation_knowledge(self):
        validation_creator = knowledge.ValidationCreator()
        validation_creator.add_expected_effect(('Sprinkler', 'Wet', 'nonparametric-ate'), ('greater', 0))
        validation_creator.add_expected_effect(('Wet', 'Slippery', 'nonparametric-ate'), ('greater', 0))
        validation_creator.add_expected_effect(('Sprinkler', 'Rain', 'nonparametric-nde'), ('less', 0))
        validation_creator.add_expected_effect(('Slippery', 'Season', 'nonparametric-nie'), ('between', 0.2, 0.4))
        return validation_creator


if __name__ == '__main__':
    unittest.main()
