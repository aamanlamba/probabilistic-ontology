"""probontology: a small library demonstrating Rafael Peñaloza's tutorial
"Introduction to Probabilistic Ontologies" (Reasoning Web 2020).

It implements:
  * HL     - the classical (non-probabilistic) hypergraph-with-data ontology
             language used as the paper's running example (Section 2).
  * pHL    - the probabilistic extension of HL from Section 6, where
             hyperedges are annotated with a *context* (a set of Bayesian
             network variables) and facts carry independent probabilities.

See the top-level README.md for the paper's definitions and worked example,
and examples/ for runnable scripts.
"""

from .bayesnet import BayesianNetwork
from .hl import HLOntology, Hyperedge
from .probabilistic_hl import ProbabilisticHLOntology, ProbabilisticHyperedge

__all__ = [
    "BayesianNetwork",
    "HLOntology",
    "Hyperedge",
    "ProbabilisticHLOntology",
    "ProbabilisticHyperedge",
]
