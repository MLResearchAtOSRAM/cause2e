"""
knowledge.py
================================================================
This module handles domain knowledge representation and verification.

It transforms knowledge about the data generating process into constraints on the edges of the
causal graph. It also verifies if a given causal graph respects a set of domain knowledge
constraints.
"""

import itertools


class ForbiddenEdgeCreator:
    """Main class for creating forbidden edges from a temporal order of the variables.

    Attributes:
        temporal_order: A list of variable sets indicating the temporal order in which the
            variables were generated. This is used to infer forbidden edges since the future
            cannot cause the past.
    """
    def __init__(self, temporal_order=None):
        """Inits ForbiddenEdgeCreator."""
        self.temporal_order = temporal_order
        self.forbidden_edges = set()

    def forbid_edges_from_temporal(self):
        """Finds all pairs of variables such that the first variable cannot causally affect the second
        variable for temporal reasons.
        """
        set_pairs = self._forbid_set_pairs_from_temporal()
        self.forbidden_edges |= _set_product_multiple(set_pairs)

    def _forbid_set_pairs_from_temporal(self):
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

    def forbid_edges_within_group(self, group):
        """Forbids edges within one group of variables.

        Args:
            group: A set containing variables that cannot affect each other.
        """
        self.forbidden_edges |= _set_product(group, group)

    def forbid_edges_from_groups(self,
                                 group,
                                 no_inf_on_group=set(),
                                 not_infd_by_group=set(),
                                 exceptions=set()
                                 ):
        """Forbids edges between groups of variables.

        Args:
            group: A set containing variables.
            no_inf_on_group: Optional; a set containing all variables that cannot affect variables
                in 'group'. Defaults to None.
            not_infd_by_group: Optional; a set containing all variables that cannot be affected by
                variables in 'group'. Defaults to None.
            exceptions: Optional; a set of edges that should not be forbidden even if the group
                structure entails it. Defaults to None.
        """
        edges = self._forbid_incoming_edges(group, no_inf_on_group)
        edges |= self._forbid_outgoing_edges(group, not_infd_by_group)
        edges -= exceptions
        self.forbidden_edges |= edges

    def _forbid_incoming_edges(self, group, no_inf_on_group):
        return _set_product(no_inf_on_group, group)

    def _forbid_outgoing_edges(self, group, not_infd_by_group):
        return _set_product(group, not_infd_by_group)


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
        print('Knowledge is respected!')
        return True

    def respects_forbidden(self):
        """Returns True if no forbidden edges are present, else raises Assertion error."""
        edge_creator = ForbiddenEdgeCreator(self.temporal)
        edge_creator.forbid_edges_from_temporal()
        self.forbidden |= edge_creator.forbidden_edges
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
