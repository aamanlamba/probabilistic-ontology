"""The probabilistic extension of HL from Section 6 of the paper.

Definition (probabilistic HL ontology). A tuple (O, P, B, gamma) where:
  * O is an HL ontology (hyperedges + facts),
  * P assigns every fact a probability in [0, 1] (tuple independence:
    facts are mutually independent Bernoulli random variables - Golden
    Rule 5's "independence assumption", applied only to the data),
  * B = (G, Phi) is a Bayesian network, and
  * gamma (the "context function") maps every hyperedge to a subset of the
    BN's variables. A hyperedge fires only in Bayesian worlds where every
    variable in its context is True.

Consistency requirement from the paper: P(alpha) = P_B(gamma(alpha)) for
every hyperedge alpha, i.e. a hyperedge's stated probability must equal the
BN's marginal probability that its whole context is true. This lets
hyperedges be *dependent* on each other (through shared or overlapping BN
variables) instead of assuming them all independent, which is exactly the
"be careful with independence" golden rule (Section 5.5) in action: the
weather example (Cold -> Fog, Cold -> Smog) would double-count Cold's
probability if Fog and Smog were derived independently; here they instead
share BN variables and their true joint probability is respected.

Semantics (the "naive" Definition, directly before Section 6's discussion
of a smarter, data-polynomial algorithm): a *possible world* is a pair
(F, X) of a subset F of the facts and a full truth assignment X to the BN
variables. Its probability is the product of the included facts'
probabilities, the excluded facts' complement probabilities (tuple
independence), and the BN's joint probability of X. Each possible world
induces a classical HL ontology (the facts in F, plus every hyperedge whose
context is true in X), and

    P(c) = sum over worlds w with O(w) |= c of P(w).

This module implements exactly that definition by brute-force enumeration
(``entailment_probability``). It is the reference/"Algorithm 1" semantics:
correct and simple, but exponential in the number of facts and BN
variables - adequate for the small examples this tutorial-demo ships with.
The paper goes on (end of Section 6) to describe a smarter algorithm that
is only exponential in the (assumed small) Bayesian network and polynomial
in the data, by reducing each fixed Bayesian world to a union-of-conjunctive
-queries probability computation over a tuple-independent probabilistic
database; that optimisation is *not* implemented here, but is described in
the README for readers who want to take this further.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, FrozenSet, Iterable, List, Mapping, Set, Tuple

from .bayesnet import BayesianNetwork
from .hl import Concept, Fact, HLOntology, Hyperedge, Individual


@dataclass(frozen=True)
class ProbabilisticHyperedge:
    sources: FrozenSet[Concept]
    target: Concept
    #: names of Bayesian-network variables that must ALL be True for this
    #: hyperedge to be active (the empty set means "always active").
    context: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        # Accept plain sets/iterables at construction time (convenient for
        # callers) but store frozensets so instances stay hashable.
        object.__setattr__(self, "sources", frozenset(self.sources))
        object.__setattr__(self, "context", frozenset(self.context))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        sources = ", ".join(sorted(self.sources)) or "TOP"
        ctx = ", ".join(sorted(self.context)) or "always"
        return f"({{{sources}}} -> {self.target})  [context: {ctx}]"


class ProbabilisticHLOntology:
    def __init__(
        self,
        hyperedges: Iterable[ProbabilisticHyperedge],
        facts: Mapping[Fact, float],
        bn: BayesianNetwork,
    ):
        self.hyperedges: List[ProbabilisticHyperedge] = list(hyperedges)
        self.facts: Dict[Fact, float] = dict(facts)
        self.bn = bn

    # -- consistency check: P(alpha) must equal P_B(context(alpha)) -------
    def context_probability(self, context: Iterable[str]) -> float:
        """P_B(context): the BN's marginal probability that every variable
        in ``context`` is True."""
        return self.bn.marginal({var: True for var in context})

    def check_consistency(self, hyperedge_probabilities: Mapping[ProbabilisticHyperedge, float], tol: float = 1e-9) -> None:
        """Raise AssertionError if a stated P(alpha) does not match
        P_B(gamma(alpha)), as required by the paper's Definition."""
        for edge, expected in hyperedge_probabilities.items():
            actual = self.context_probability(edge.context)
            assert abs(actual - expected) < tol, (
                f"{edge}: expected P_B(context) = {expected}, got {actual}"
            )

    # -- possible-world construction ---------------------------------------
    def _classical_ontology(self, active_facts: Set[Fact], bn_world: Mapping[str, bool]) -> HLOntology:
        true_vars = {var for var, value in bn_world.items() if value}
        active_edges = [
            Hyperedge(edge.sources, edge.target)
            for edge in self.hyperedges
            if edge.context <= true_vars
        ]
        return HLOntology(active_edges, active_facts)

    # -- probability of a consequence ---------------------------------------
    def entailment_probability(self, concept: Concept, individual: Individual) -> float:
        """P(concept(individual)) = sum over possible worlds w = (F, X)
        with O(w) |= concept(individual) of P(w)."""
        fact_items: List[Tuple[Fact, float]] = list(self.facts.items())
        total = 0.0

        for bn_world, bn_prob in self.bn.enumerate_worlds():
            if bn_prob == 0.0:
                continue
            for bits in product((False, True), repeat=len(fact_items)):
                fact_prob = 1.0
                active_facts: Set[Fact] = set()
                for (fact, p), included in zip(fact_items, bits):
                    fact_prob *= p if included else (1.0 - p)
                    if fact_prob == 0.0:
                        break
                    if included:
                        active_facts.add(fact)
                if fact_prob == 0.0:
                    continue

                world_prob = fact_prob * bn_prob
                ontology = self._classical_ontology(active_facts, bn_world)
                if ontology.entails(concept, individual):
                    total += world_prob

        return total
