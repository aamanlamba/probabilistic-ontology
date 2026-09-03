"""Exact inference over small discrete Bayesian networks (Section 3.4 of the paper).

A Bayesian network is a pair B = (G, Phi) where G is a DAG over Boolean random
variables and Phi contains one conditional probability table per node, given
its parents. The joint distribution factorises as

    P(X) = product over nodes n of P(n | parents(n))

Because the networks used to annotate probabilistic ontologies are expected
to be small (Section 6: "the Bayesian network ... is assumed to be the
smallest component of the whole ontology"), exact inference here is done by
brute-force enumeration of all 2^|nodes| worlds. This mirrors the paper's own
complexity argument: the combinatorial blow-up is confined to the BN, while
everything built on top of it (the ontology's data) is handled separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterator, Mapping, Sequence, Tuple


@dataclass
class BayesianNetwork:
    #: node -> tuple of parent node names (the tuple's order matches the
    #: order of values used as keys into ``cpt[node]``).
    parents: Dict[str, Tuple[str, ...]]
    #: node -> {parent_value_tuple: P(node = True | parents = parent_value_tuple)}
    #: a node with no parents uses the single key ``()``.
    cpt: Dict[str, Dict[Tuple[bool, ...], float]]

    def nodes(self) -> Tuple[str, ...]:
        return tuple(self.parents)

    def topological_order(self) -> Sequence[str]:
        order = []
        visited = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in self.parents[node]:
                visit(parent)
            order.append(node)

        for node in self.parents:
            visit(node)
        return order

    def joint_probability(self, assignment: Mapping[str, bool]) -> float:
        """P(X) = prod_{n} P(n | parents(n)) for a full assignment X."""
        prob = 1.0
        for node, parents in self.parents.items():
            parent_values = tuple(assignment[p] for p in parents)
            p_true = self.cpt[node][parent_values]
            prob *= p_true if assignment[node] else (1.0 - p_true)
        return prob

    def enumerate_worlds(self) -> Iterator[Tuple[Dict[str, bool], float]]:
        """Yield every full truth assignment together with its joint probability."""
        nodes = self.nodes()
        for bits in product((False, True), repeat=len(nodes)):
            assignment = dict(zip(nodes, bits))
            yield assignment, self.joint_probability(assignment)

    def marginal(self, event: Mapping[str, bool]) -> float:
        """P(event), where ``event`` fixes a subset of variables and the rest
        are free (summed out). Used to compute P_B(context) for a context
        (a set of variables that must all be True), by calling
        ``marginal({v: True for v in context})``.
        """
        return sum(
            prob
            for assignment, prob in self.enumerate_worlds()
            if all(assignment[var] == value for var, value in event.items())
        )
