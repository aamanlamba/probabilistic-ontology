"""HL: the classical hypergraph-with-data ontology language (Section 2).

An HL ontology combines:
  * a hypergraph of *inclusion axioms* S -> T, where S is a finite set of
    concept names and T is a single concept name ("hyperedges"), and
  * a set of *facts* A(a), stating that individual a has concept A.

A concept T is *reachable* for an individual a (i.e. the ontology entails
A(a) -> ... -> T(a)) if T can be derived from a's known concepts by
repeatedly applying hyperedges whose whole source set is already known.
This is a directed-hypergraph reachability problem, computed here as a
simple forward-chaining fixpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Set, Tuple

Concept = str
Individual = str
Fact = Tuple[Concept, Individual]


@dataclass(frozen=True)
class Hyperedge:
    sources: FrozenSet[Concept]
    target: Concept

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", frozenset(self.sources))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        sources = ", ".join(sorted(self.sources)) or "TOP"
        return f"({{{sources}}} -> {self.target})"


class HLOntology:
    def __init__(self, hyperedges: Iterable[Hyperedge] = (), facts: Iterable[Fact] = ()):
        self.hyperedges = list(hyperedges)
        self.facts = set(facts)

    def close(self) -> Dict[Individual, Set[Concept]]:
        """Forward-chain all facts through the hypergraph until a fixpoint."""
        known: Dict[Individual, Set[Concept]] = {}
        for concept, individual in self.facts:
            known.setdefault(individual, set()).add(concept)

        changed = True
        while changed:
            changed = False
            for concepts in known.values():
                for edge in self.hyperedges:
                    if edge.target not in concepts and edge.sources <= concepts:
                        concepts.add(edge.target)
                        changed = True
        return known

    def entails(self, concept: Concept, individual: Individual) -> bool:
        """O |= concept(individual)?"""
        return concept in self.close().get(individual, set())
