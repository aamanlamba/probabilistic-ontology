from probontology import HLOntology, Hyperedge


def test_direct_fact_entailed():
    onto = HLOntology(facts=[("Nausea", "a")])
    assert onto.entails("Nausea", "a")
    assert not onto.entails("Halucination", "a")


def test_single_step_hyperedge():
    onto = HLOntology(
        hyperedges=[Hyperedge(frozenset({"Nausea"}), "Halucination")],
        facts=[("Nausea", "a")],
    )
    assert onto.entails("Halucination", "a")


def test_multi_source_hyperedge_from_paper_section_2():
    # "if an ontology contains the facts Nausea(a) and MemoryLoss(a), we can
    # derive the consequence Observation(a)" (Section 2, HL).
    onto = HLOntology(
        hyperedges=[
            Hyperedge(frozenset({"Nausea"}), "Halucination"),
            Hyperedge(frozenset({"Halucination", "MemoryLoss"}), "Dementia"),
            Hyperedge(frozenset({"Dementia"}), "Observation"),
        ],
        facts=[("Nausea", "a"), ("MemoryLoss", "a")],
    )
    assert onto.entails("Observation", "a")


def test_multi_source_hyperedge_requires_all_sources():
    onto = HLOntology(
        hyperedges=[Hyperedge(frozenset({"Halucination", "MemoryLoss"}), "Dementia")],
        facts=[("Halucination", "a")],  # MemoryLoss is missing
    )
    assert not onto.entails("Dementia", "a")


def test_entailment_does_not_leak_between_individuals():
    onto = HLOntology(
        hyperedges=[Hyperedge(frozenset({"Nausea"}), "Halucination")],
        facts=[("Nausea", "a")],
    )
    assert not onto.entails("Halucination", "b")
