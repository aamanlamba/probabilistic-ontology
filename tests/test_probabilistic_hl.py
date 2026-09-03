import math

import pytest

from examples.paper_example import build_bayesian_network, build_ontology


@pytest.fixture
def onto():
    bn = build_bayesian_network()
    return build_ontology(bn)


def test_context_probabilities_match_table_2(onto):
    expected = {
        frozenset({"Y"}): 0.28,
        frozenset({"W", "Y"}): 0.196,
        frozenset({"W", "X"}): 0.28,
        frozenset({"X"}): 0.7,
        frozenset({"W", "Z"}): 0.0432,
        frozenset(): 1.0,
    }
    for edge in onto.hyperedges:
        assert math.isclose(onto.context_probability(edge.context), expected[edge.context], abs_tol=1e-9)
    # Definition 6's consistency requirement: P(alpha) == P_B(gamma(alpha)).
    onto.check_consistency({edge: expected[edge.context] for edge in onto.hyperedges})


def test_observation_p1_matches_paper(onto):
    # Paper: "P(Observation(p1)) = 0.5 because ({Dementia}, Observation)
    # holds in all possible worlds, but Dementia(p1) only in worlds with
    # probability 0.5."
    assert math.isclose(onto.entailment_probability("Observation", "p1"), 0.5)


def test_observation_p3_matches_paper(onto):
    # Paper: "P(Observation(p3)) = 0 because for nausea and memory loss to
    # cause dementia, both X and Z should hold which ... can only happen
    # with probability 0."
    assert math.isclose(onto.entailment_probability("Observation", "p3"), 0.0, abs_tol=1e-12)


def test_observation_p2_solves_the_papers_exercise(onto):
    # The paper leaves P(Observation(p2)) "as an exercise to the reader".
    # Derivation: Pain(p2)=1 (certain) --{Y}--> Tachycardia(p2); and
    # Halucination(p2)=0.4 --{X}--> Nausea(p2); then Tachycardia+Nausea
    # --{W,Y}--> Observation, requiring the Bayesian world {W,X,Y} (Z free).
    # P(W,X,Y) = P(W|X)P(X)P(Y|X) = 0.4*0.7*0.1 = 0.028, so
    # P(Observation(p2)) = P(Halucination(p2)) * P(W,X,Y) = 0.4 * 0.028 = 0.0112.
    assert math.isclose(onto.entailment_probability("Observation", "p2"), 0.0112, abs_tol=1e-9)
