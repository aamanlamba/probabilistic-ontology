"""Reproduces the running example from Section 6 of the paper
("Introduction to Probabilistic Ontologies", Peñaloza, RW 2020):
the Bayesian network of Figure 1, and the probabilistic HL ontology
(O_exa, P_exa, B_exa, gamma_exa) of Table 2 / Figure 3.

Run with:  python examples/paper_example.py
"""

from probontology import BayesianNetwork, ProbabilisticHLOntology, ProbabilisticHyperedge


def build_bayesian_network() -> BayesianNetwork:
    """Figure 1: X is a root; Y and W depend on X; Z depends on X and Y."""
    return BayesianNetwork(
        parents={
            "X": (),
            "Y": ("X",),
            "W": ("X",),
            "Z": ("X", "Y"),
        },
        cpt={
            "X": {(): 0.7},
            "Y": {(True,): 0.1, (False,): 0.7},
            "W": {(True,): 0.4, (False,): 0.8},
            "Z": {
                (True, True): 0.0,
                (True, False): 0.0,
                (False, True): 0.0,
                (False, False): 0.6,
            },
        },
    )


def build_ontology(bn: BayesianNetwork) -> ProbabilisticHLOntology:
    """Table 2 / Figure 3: the hypergraph (with contexts) and the facts."""
    hyperedges = [
        ProbabilisticHyperedge({"Pain"}, "Tachycardia", frozenset({"Y"})),
        ProbabilisticHyperedge({"Tachycardia", "Nausea"}, "Observation", frozenset({"W", "Y"})),
        ProbabilisticHyperedge({"Nausea"}, "Halucination", frozenset({"W", "X"})),
        ProbabilisticHyperedge({"Halucination"}, "Nausea", frozenset({"X"})),
        ProbabilisticHyperedge({"Halucination", "MemoryLoss"}, "Dementia", frozenset({"W", "Z"})),
        ProbabilisticHyperedge({"Dementia"}, "Observation", frozenset()),  # always active
    ]
    facts = {
        ("Dementia", "p1"): 0.5,
        ("Pain", "p1"): 0.8,
        ("Halucination", "p2"): 0.4,
        ("Pain", "p2"): 1.0,
        ("Nausea", "p3"): 0.9,
        ("MemoryLoss", "p3"): 0.7,
    }
    return ProbabilisticHLOntology(hyperedges, facts, bn)


def main() -> None:
    bn = build_bayesian_network()
    onto = build_ontology(bn)

    print("Sanity check: P(W, X, not Y, not Z) should be 0.252 (paper, Section 3.4)")
    joint = bn.joint_probability({"W": True, "X": True, "Y": False, "Z": False})
    print(f"  computed = {joint}\n")

    print("Context probabilities P_B(context), compared against Table 2's P_exa column:")
    expected = {
        frozenset({"Y"}): 0.28,
        frozenset({"W", "Y"}): 0.196,
        frozenset({"W", "X"}): 0.28,
        frozenset({"X"}): 0.7,
        frozenset({"W", "Z"}): 0.0432,
        frozenset(): 1.0,
    }
    for edge in onto.hyperedges:
        p = onto.context_probability(edge.context)
        print(f"  {edge!r:70s} P_B = {p:.4f}  (expected {expected[edge.context]:.4f})")

    print("\nEntailment probabilities for Observation(p_i):")
    for individual in ("p1", "p2", "p3"):
        p = onto.entailment_probability("Observation", individual)
        print(f"  P(Observation({individual})) = {p:.6f}")
    print(
        "\nThe paper states P(Observation(p1)) = 0.5 and P(Observation(p3)) = 0 directly,"
        "\nand leaves P(Observation(p2)) as an exercise for the reader - this script computes it (0.0112)."
    )


if __name__ == "__main__":
    main()
