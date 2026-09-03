import math

from probontology import BayesianNetwork


def figure1_network() -> BayesianNetwork:
    return BayesianNetwork(
        parents={"X": (), "Y": ("X",), "W": ("X",), "Z": ("X", "Y")},
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


def test_joint_probability_matches_paper_worked_example():
    bn = figure1_network()
    # Paper, Section 3.4: P(W, X, not Y, not Z) = 0.4 * 0.7 * 0.9 * 1 = 0.252
    assert math.isclose(bn.joint_probability({"W": True, "X": True, "Y": False, "Z": False}), 0.252)


def test_worlds_sum_to_one():
    bn = figure1_network()
    total = sum(prob for _, prob in bn.enumerate_worlds())
    assert math.isclose(total, 1.0)


def test_context_marginals_match_table_2():
    bn = figure1_network()
    assert math.isclose(bn.marginal({"Y": True}), 0.28)
    assert math.isclose(bn.marginal({"W": True, "Y": True}), 0.196)
    assert math.isclose(bn.marginal({"W": True, "X": True}), 0.28)
    assert math.isclose(bn.marginal({"X": True}), 0.7)
    assert math.isclose(bn.marginal({"W": True, "Z": True}), 0.0432)
    assert math.isclose(bn.marginal({}), 1.0)
