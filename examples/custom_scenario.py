"""A small original scenario, built with the same library, showing that the
paper's language is a general tool and not just a vehicle for replaying its
own numbers.

Setting: a factory monitors a production line. Two upstream conditions are
correlated (a single "SupplierBatchDefect" event tends to cause both
"CalibrationDrift" and "MaterialFlaw" downstream, so they are NOT modelled
as independent - Golden Rule 5, "be careful with independence"), and two
sensor readings for a specific unit are noisy (independent Bernoulli
facts - Golden Rule 5 also flags this as a deliberate, different choice:
tuple-independence *is* reasonable for high-volume, per-unit sensor data).
"""

from probontology import BayesianNetwork, ProbabilisticHLOntology, ProbabilisticHyperedge


def build_bayesian_network() -> BayesianNetwork:
    # SupplierBatchDefect (root) independently affects CalibrationDrift and
    # MaterialFlaw; both feed into UnitAtRisk.
    return BayesianNetwork(
        parents={
            "SupplierBatchDefect": (),
            "CalibrationDrift": ("SupplierBatchDefect",),
            "MaterialFlaw": ("SupplierBatchDefect",),
        },
        cpt={
            "SupplierBatchDefect": {(): 0.05},
            "CalibrationDrift": {(True,): 0.6, (False,): 0.02},
            "MaterialFlaw": {(True,): 0.5, (False,): 0.01},
        },
    )


def build_ontology(bn: BayesianNetwork) -> ProbabilisticHLOntology:
    hyperedges = [
        # Always true: a unit with both a sensor vibration alarm and a
        # temperature alarm is flagged for inspection, regardless of cause.
        ProbabilisticHyperedge({"VibrationAlarm", "TempAlarm"}, "FlaggedForInspection", frozenset()),
        # Only meaningful in worlds where calibration has drifted.
        ProbabilisticHyperedge({"VibrationAlarm"}, "LikelyMiscalibrated", frozenset({"CalibrationDrift"})),
        # Only meaningful in worlds with a material flaw.
        ProbabilisticHyperedge({"TempAlarm"}, "LikelyMaterialDefect", frozenset({"MaterialFlaw"})),
        # A unit that is both likely miscalibrated and has a likely material
        # defect is scrapped - this can only fire when BOTH upstream BN
        # variables hold, capturing their (shared-cause) dependence.
        ProbabilisticHyperedge(
            {"LikelyMiscalibrated", "LikelyMaterialDefect"},
            "Scrapped",
            frozenset({"CalibrationDrift", "MaterialFlaw"}),
        ),
    ]
    facts = {
        ("VibrationAlarm", "unit42"): 0.9,
        ("TempAlarm", "unit42"): 0.3,
        ("VibrationAlarm", "unit99"): 0.1,
        ("TempAlarm", "unit99"): 0.1,
    }
    return ProbabilisticHLOntology(hyperedges, facts, bn)


def main() -> None:
    bn = build_bayesian_network()
    onto = build_ontology(bn)

    print("P(SupplierBatchDefect):", bn.marginal({"SupplierBatchDefect": True}))
    print("P(CalibrationDrift, MaterialFlaw) [shared cause -> correlated]:")
    print("  ", bn.marginal({"CalibrationDrift": True, "MaterialFlaw": True}))
    naive_independent = bn.marginal({"CalibrationDrift": True}) * bn.marginal({"MaterialFlaw": True})
    print("  (a wrongly-independent estimate would give:", naive_independent, ")")

    print()
    for unit in ("unit42", "unit99"):
        flagged = onto.entailment_probability("FlaggedForInspection", unit)
        scrapped = onto.entailment_probability("Scrapped", unit)
        print(f"{unit}: P(FlaggedForInspection) = {flagged:.4f}   P(Scrapped) = {scrapped:.6f}")


if __name__ == "__main__":
    main()
