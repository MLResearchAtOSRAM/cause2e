"""
knowledge.py
================================================================
This module handles domain knowledge representation and verification.

It transforms knowledge about the data generating process into constraints on the edges of the
causal graph. It also verifies if a given causal graph respects a set of domain knowledge
constraints.
"""

import itertools


class EdgeCreator:
    """Main class for creating required and forbidden edges from domain knowledge.

    Attributes:
        forbidden_edges: A set of edges that must not appear in the causal graph.
        required_edges: A set of edges that must appear in the causal graph.
    """
    def __init__(self):
        """Inits EdgeCreator."""
        self.forbidden_edges = set()
        self.required_edges = set()

    def forbid_edges(self, edges):
        """Forbids multiple edges.

        Args:
            edges: A set of edges.
        """
        for edge in edges:
            self.forbid_edge(*edge)

    def forbid_edge(self, source, destination):
        """Forbids an edge between two nodes.

        Args:
            source: A string indicating the source node of the forbidden edge.
            destination: A string indicating the destination node of the forbidden edge.
        """
        self.forbidden_edges.add((source, destination))

    def forbid_edges_from_temporal(self, temporal_order):
        """Finds all pairs of variables such that the first variable cannot causally affect the second
        variable for temporal reasons.

        Args:
            temporal_order: A list of variable sets indicating the temporal order in which the
                variables were generated. This is used to infer forbidden edges since the future
                cannot cause the past.
        """
        set_pairs = self._forbid_set_pairs_from_temporal(temporal_order)
        self.forbidden_edges |= _set_product_multiple(set_pairs)

    def _forbid_set_pairs_from_temporal(self, temporal_order):
        """
        Returns all pairs of sets such that variables in
        the first set cannot causally affect variables in the
        second set.

        Args:
            temporal_order: A list of variable sets indicating the temporal order in which the
                variables were generated. This is used to infer forbidden edges since the future
                cannot cause the past.
        """
        set_pairs = set()
        for i, later in enumerate(temporal_order):
            for j in range(i):
                set_pairs.add((frozenset(later), frozenset(temporal_order[j])))
        return set_pairs

    def forbid_edges_within_group(self, group):
        """Forbids edges within one group of variables.

        Args:
            group: A set containing variables that cannot affect each other.
        """
        self.forbidden_edges |= _set_product(group, group)

    def forbid_edges_from_groups(self,
                                 group,
                                 incoming=set(),
                                 outgoing=set(),
                                 exceptions=set()
                                 ):
        """Forbids edges between groups of variables.

        Args:
            group: A set containing variables.
            incoming: Optional; a set containing all variables that cannot affect variables
                in 'group'. Defaults to None.
            outgoing: Optional; a set containing all variables that cannot be affected by
                variables in 'group'. Defaults to None.
            exceptions: Optional; a set of edges that should not be forbidden even if the group
                structure entails it. Defaults to None.
        """
        edges = self._create_edges_from_groups(group, incoming, outgoing, exceptions)
        self.forbidden_edges |= edges

    def require_edges(self, edges):
        """Requires multiple edges.

        Args:
            edges: A set of edges.
        """
        for edge in edges:
            self.require_edge(*edge)

    def require_edge(self, source, destination):
        """Requires an edge between two nodes.

        Args:
            source: A string indicating the source node of the required edge.
            destination: A string indicating the destination node of the required edge.
        """
        self.required_edges.add((source, destination))

    def require_edges_from_groups(self,
                                  group,
                                  incoming=set(),
                                  outgoing=set(),
                                  exceptions=set()
                                  ):
        """Requires edges between groups of variables.

        Args:
            group: A set containing variables.
            incoming: Optional; a set containing all variables that must affect variables
                in 'group'. Defaults to None.
            outgoing: Optional; a set containing all variables that must be affected by
                variables in 'group'. Defaults to None.
            exceptions: Optional; a set of edges that should not be required even if the group
                structure entails it. Defaults to None.
        """
        edges = self._create_edges_from_groups(group, incoming, outgoing, exceptions)
        self.required_edges |= edges

    def _create_edges_from_groups(self,
                                  group,
                                  incoming=set(),
                                  outgoing=set(),
                                  exceptions=set()
                                  ):
        """Creates edges between groups of variables.

        Args:
            group: A set containing variables.
            incoming: Optional; a set containing all variables that must (not) affect variables
                in 'group'. Defaults to None.
            outgoing: Optional; a set containing all variables that must (not) be affected by
                variables in 'group'. Defaults to None.
            exceptions: Optional; a set of edges that should not be required/forbidden even if
                the group structure entails it. Defaults to None.
        """
        edges = self._create_incoming_edges(group, incoming)
        edges |= self._create_outgoing_edges(group, outgoing)
        edges -= exceptions
        return edges

    def _create_incoming_edges(self, group, incoming):
        return _set_product(incoming, group)

    def _create_outgoing_edges(self, group, outgoing):
        return _set_product(group, outgoing)

    def forget_edges(self):
        """Forgets all the previously created edges to allow a new start."""
        self.forbidden_edges = set()
        self.required_edges = set()

    def show_edges(self):
        """Shows all currently required/forbidden edges."""
        print("-------------------")
        print("Required edges:")
        for edge in self.required_edges:
            print(edge)
        print("-------------------")
        print("Forbidden edges:")
        for edge in self.forbidden_edges:
            print(edge)
        print("-------------------")


class KnowledgeChecker:
    """Main class for checking that a causal graph respects constraints from domain knowledge.

    Attributes:
        existing: A set of all edges that exist in the causal graph under consideration.
        forbidden: A set of all edges that contradict domain knowledge.
        required: A set of all edges that must exist according to domain knowledge.
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
        else:
            self.forbidden = set()
            self.required = set()

    def respects_knowledge(self):
        """Returns a boolean indicating if all domain knowledge is respected."""
        self.respects_forbidden()
        self.respects_required()
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


class ValidationCreator:
    """Main class for recording expectations about causal effects that are validated after estimation.

    Attributes:
        expected_effects: A dictionary containing expected quantitative causal effects. This is
            evaluated after estimation of the effects.
    """
    def __init__(self):
        self.expected_effects = {}

    def add_expected_effect(self, effect, expected_val):
        self.expected_effects[effect] = {'Expected': expected_val}


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
