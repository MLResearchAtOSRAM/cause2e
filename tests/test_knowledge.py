import unittest
import itertools
import pandas as pd
from cause2e.knowledge import (_set_product,
                               _set_product_multiple,
                               EdgeCreator,
                               ValidationCreator,
                               KnowledgeChecker,
                               Spellchecker,
                               SpellingError,
                               )
from cause2e.discovery import StructureLearner


class TestKnowledgeGeneration(unittest.TestCase):
    def setUp(self):
        self.order = [frozenset({'a', 'b'}),
                      frozenset({1, 2, 3}),
                      frozenset({'A', 'B'})
                      ]

    def test_set_product(self):
        result_manual = {('a', 1), ('a', 2), ('a', 3),
                         ('b', 1), ('b', 2), ('b', 3)
                         }
        result = _set_product({'a', 'b'}, {1, 2, 3})
        self.assertEqual(result, result_manual)

    def test_set_product_multiple(self):
        result_manual = {('a', 1), ('a', 2), ('a', 3),
                         ('b', 1), ('b', 2), ('b', 3),
                         (1, 'A'), (1, 'B'),
                         (2, 'A'), (2, 'B'),
                         (3, 'A'), (3, 'B')
                         }
        set_pairs = {(self.order[0], self.order[1]),
                     (self.order[1], self.order[2])
                     }
        result = _set_product_multiple(set_pairs)
        self.assertEqual(result, result_manual)

    def test_edge_creator(self):
        edge_creator = EdgeCreator()
        result_manual = {(self.order[1], self.order[0]),
                         (self.order[2], self.order[0]),
                         (self.order[2], self.order[1])
                         }
        result = edge_creator._forbid_set_pairs_from_temporal(self.order)
        self.assertEqual(result, result_manual)


class TestKnowledgeChecker(unittest.TestCase):
    def setUp(self):
        edges = {frozenset({'A', 'B'}),
                 ('A', 'C'),
                 ('B', 'D'),
                 ('C', 'D'),
                 ('D', 'E')
                 }
        forbidden = {('A', 'C'),
                     ('B', 'F')}
        required = {('C', 'D'),
                    ('B', 'D')}
        temporal = [frozenset({'A'}),
                    frozenset({'B', 'C'}),
                    frozenset({'E'})
                    ]
        edge_creator = EdgeCreator()
        edge_creator.forbid_edges(forbidden)
        edge_creator.require_edges(required)
        edge_creator.forbid_edges_from_temporal(temporal)
        knowledge_dict = {'required': edge_creator.required_edges,
                          'forbidden': edge_creator.forbidden_edges
                          }
        self.checker = KnowledgeChecker(edges, knowledge_dict)

    def test_respects_forbidden(self):
        with self.assertRaises(AssertionError):
            self.checker.respects_forbidden()
        self.checker.forbidden.discard(('A', 'C'))
        self.checker.respects_forbidden()

    def test_respects_required(self):
        self.checker.respects_required()
        self.checker.required.add(('A', 'B'))
        with self.assertRaises(AssertionError):
            self.checker.respects_required()

    # def test_respects_temporal(self):
    #     self.checker.respects_temporal()
    #     self.checker.temporal.append(frozenset({'A'}))
    #     with self.assertRaises(AssertionError):
    #         self.checker.respects_temporal()

    def test_respects_knowledge(self):
        with self.assertRaises(AssertionError):
            self.checker.respects_knowledge()
        self.checker.forbidden.discard(('A', 'C'))
        self.checker.respects_knowledge()


class TestNoKnowledgeChecker(unittest.TestCase):
    def setUp(self):
        edges = {frozenset({'A', 'B'}),
                 ('A', 'C'),
                 ('B', 'D'),
                 ('C', 'D'),
                 ('D', 'E')
                 }
        self.checker = KnowledgeChecker(edges)

    def test_respects_no_knowledge(self):
        self.checker.respects_knowledge()


class TestSpellchecker(unittest.TestCase):
    def setUp(self):
        self.variables = {'A', 'B', 'C'}
        self.edges = {
            ('A', 'B'),
        }
        self.expected_effects = {
            ('A', 'B', 'nonparametric-ate'): {'Expected': ('greater', 0)},
        }
        self.checker = Spellchecker(
            variables=self.variables,
            edges=self.edges,
            expected_effects=self.expected_effects,
        )

    def test_no_typo(self):
        self.checker.check_names()

    def test_typo_in_edges(self):
        self.checker._edges.add(('E', 'A'))
        with self.assertRaises(SpellingError):
            self.checker.check_names()

    def test_typo_in_expected_effects(self):
        self.checker._expected_effects[('A', 'Y', 'nonparametric-ate')] = {'Expected': ('greater', 0)},
        with self.assertRaises(SpellingError):
            self.checker.check_names()

    def test_init_without_validation(self):
        checker = Spellchecker(
            variables=self.variables,
            edges=self.edges,
            expected_effects={},
        )
        checker.check_names()

    def test_init_without_edges(self):
        checker = Spellchecker(
            variables=self.variables,
            edges=set(),
            expected_effects=self.expected_effects,
        )
        checker.check_names()

    def test_init_from_high_level(self):
        options = [True, False]
        checkers = []
        for use_edge_creator, use_validation_creator in itertools.product(options, options):
            checkers.append(self._init_from_high_level(use_edge_creator, use_validation_creator))
        checker = checkers[0]
        self.assertEqual(checker._variables, self.checker._variables)
        self.assertEqual(checker._edges, self.checker._edges)
        self.assertEqual(checker._expected_effects, self.checker._expected_effects)

    def _init_from_high_level(self, use_edge_creator, use_validation_creator):
        self._set_up_higher_level_input()
        if use_edge_creator:
            edge_creator = self.edge_creator
        else:
            edge_creator = None
        if use_validation_creator:
            validation_creator = self.validation_creator
        else:
            validation_creator = None
        return Spellchecker.from_high_level(
            learner=self.learner,
            edge_creator=edge_creator,
            validation_creator=validation_creator,
        )

    def _set_up_higher_level_input(self):
        self.learner = StructureLearner(paths=None)
        self.learner.data = pd.DataFrame(
            {var: [None] for var in self.variables}
        )
        self.edge_creator = EdgeCreator()
        self.edge_creator.require_edges(self.edges)
        self.validation_creator = ValidationCreator()
        self.validation_creator.expected_effects = self.expected_effects


if __name__ == '__main__':
    unittest.main()
