"""
knowledge.py
================================================================
This module handles domain knowledge representation and verification.

It transforms knowledge about the data generating process into constraints on the edges of the
causal graph. It also verifies if a given causal graph respects a set of domain knowledge
constraints.
"""

import itertools


class ForbiddenEdgeCreator():
    """Main class for creating forbidden edges from a temporal order of the variables.

    Attributes:
        temporal_order: A list of variable sets indicating the temporal order in which the
            variables were generated. This is used to infer forbidden edges since the future
            cannot cause the past.
    """
    def __init__(self, temporal_order):
        """Inits ForbiddenEdgeCreator."""
        self.temporal_order = temporal_order

    def forbidden_edges_from_temporal(self):
        """Returns all pairs of variables such that the first variable cannot causally affect the second
        variable for temporal reasons.
        """
        set_pairs = self._forbidden_set_pairs_from_temporal()
        edges = _set_product_multiple(set_pairs)
        return edges

    def _forbidden_set_pairs_from_temporal(self):
        """
        Returns all pairs of sets such that variables in
        the first set cannot causally affect variables in the
        second set.
        """
        set_pairs = set()
        for i, later in enumerate(self.temporal_order):
            for j in range(i):
                set_pairs.add((frozenset(later), frozenset(self.temporal_order[j])))
        return set_pairs


class KnowledgeChecker:
    """Main class for checking that a causal graph respects constraints from domain knowledge.

    Attributes:
        existing: A set of all edges that exist in the causal graph under consideration.
        forbidden: A set of all edges that contradict domain knowledge.
        required: A set of all edges that must exist according to domain knowledge.
        temporal: A list containing sets of variables in the temporal order of their generation.
    """

    def __init__(self, edges, knowledge=None):
        """Inits KnowledgeChecker."""
        self.existing = edges
        self._format_knowledge(knowledge)

    def _format_knowledge(self, knowledge):
        """Allows specifying a partial knowledge dictionary."""
        if knowledge:
            self.forbidden = knowledge['forbidden']
            self.required = knowledge['required']
            self.temporal = knowledge['temporal']
        else:
            self.forbidden = set()
            self.required = set()
            self.temporal = []

    def respects_knowledge(self):
        """Returns a boolean indicating if all domain knowledge is respected."""
        self.respects_forbidden()
        self.respects_required()
        self.respects_temporal()
        print('Knowledge is respected!')
        return True

    def respects_forbidden(self):
        """Returns True if no forbidden edges are present, else raises Assertion error."""
        existing_but_forbidden = self.existing & self.forbidden
        msg = f'Forbidden edges: {existing_but_forbidden}'
        assert not existing_but_forbidden, msg
        return True

    def respects_required(self):
        """Returns True if all required edges are present, else raises Assertion error."""
        absent_but_required = self.required - self.existing
        msg = f'Missing edges: {absent_but_required}'
        assert not absent_but_required, msg
        return True

    def respects_temporal(self):
        """Returns True if no temporal knowledge is violated, else raises Assertion error."""
        creator = ForbiddenEdgeCreator(self.temporal)
        self.forbidden = creator.forbidden_edges_from_temporal()
        self.respects_forbidden()
        return True


def _set_product_multiple(set_pairs):
    """Helper function.

    Args:
        set_pairs: A set of set pairs

    Returns:
        set of all pairs (x,y) for an element x of the first entry of a set pair and an element y
        of the second entry of the same set pair. Remember to use frozensets for sets of sets.
    """
    se = {frozenset(_set_product(*sets)) for sets in set_pairs}
    return set(itertools.chain.from_iterable(se))  # flatten


def _set_product(set_1, set_2):
    """Returns the cartesian product of two sets."""
    return set(itertools.product(set_1, set_2))
