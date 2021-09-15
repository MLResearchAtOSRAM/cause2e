import unittest
from cause2e.knowledge import (_set_product,
                               _set_product_multiple,
                               EdgeCreator,
                               KnowledgeChecker
                               )


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


if __name__ == '__main__':
    unittest.main()
