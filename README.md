# Probabilistic Ontologies Demo

> A Bayesian-network-annotated hypergraph ontology language, implemented and validated against its source paper's own worked example.

A small Python library that demonstrates the ideas in:

> Rafael Peñaloza, **"Introduction to Probabilistic Ontologies"**, tutorial
> lecture, *Reasoning Web 2020: Declarative Artificial Intelligence*, LNCS
> vol. 12258, Springer, 2020. [DOI: 10.1007/978-3-030-60067-9_1](https://doi.org/10.1007/978-3-030-60067-9_1)

The tutorial is a conceptual/design-choices paper rather than a single
algorithm paper: it (1) gives an abstract, language-agnostic definition of
what an "ontology" and a "probabilistic ontology" even are, (2) gives a
primer on the probability theory needed (conditioning, independence,
Bayesian networks), (3) lays out **Five Golden Rules** for building a sound
probabilistic ontology language, and (4) works through **one concrete
language end-to-end** — a probabilistic extension of a tiny description
logic called **HL** — including a fully worked numeric example. This repo
implements that concrete language faithfully and reproduces the paper's own
example, including solving the exercise the paper leaves to the reader.

## The paper, in logical terms

**1. What an ontology *is*, abstractly.** The paper defines an *ontology
language* as a tuple `L = (A, O, C, ⊨)`:

- `A` — the set of well-formed **axioms** (the vocabulary you can write
  statements in),
- `O ⊆ P(A)` — the **acceptable ontologies** (which subsets of axioms are
  legal; must be closed under taking subsets),
- `C` — the set of possible **consequences** (questions you can ask), and
- `⊨ ⊆ O × C` — **entailment**, required to be *monotonic*: adding axioms
  can only add consequences, never remove them.

Graphs (`⊨` = reachability), Prolog, relational databases, and every
classical description logic are all instances of this one schema. The
paper's own running example, **HL**, is:

- axioms are either *inclusions* `S → T` (`S` a finite set of concept
  names, `T` a single concept name — a directed hyperedge) or *facts*
  `A(a)` (individual `a` has concept `A`),
- a consequence `A(a)` follows from an ontology iff `A` is **reachable**
  from `a`'s known concepts by repeatedly firing hyperedges whose entire
  source set is already known (`hl.py` in this repo).

**2. Uncertainty is added *on top* of a classical language, not baked in.**
`Definition 1`: a probabilistic ontology is a pair `(O, P)` where
`O ∈ O` is a classical ontology and `P : O ⇀ [0,1]` is a *partial* function
assigning some axioms a probability. Crucially, `P` is only allowed to
label whole axioms — it cannot appear *inside* the logical language itself
(that would make it a different kind of ontology language entirely, e.g.
one with built-in conditional statements). This is a clean separation of
concerns: the logic stays classical and monotonic; probability is a
separate annotation layer.

**3. The Five Golden Rules** (Section 5) — the design checklist the paper
argues every probabilistic ontology language must be checked against:

1. **Use probabilities** — only if you're really modelling *uncertainty*
   (an event that holds or doesn't, you just don't know which), not
   *vagueness* (no probabilities for "tall" — that's fuzzy logic, which is
   truth-functional; probability is not).
2. **Use the right probabilities** — *statistical* ("30% of flights are
   delayed") vs. *subjective* ("this flight has a 30% chance of delay")
   readings need different semantics (population partitioning vs.
   possible-worlds).
3. **To count or not to count** — should re-deriving the same consequence
   through *n* applications of a probabilistic rule multiply its
   probability by itself *n* times, or not? Depends on whether the
   repeated applications are genuinely independent evidence or the same
   fact reused.
4. **Understand the numbers** — a confidence score from an NLP extractor is
   not the same thing as the probability that the extracted fact is true;
   know where your numbers came from before trusting them.
5. **Be careful with independence** — assuming axioms are probabilistically
   independent is what makes reasoning tractable, but it silently breaks
   when axioms share a hidden common cause. `examples/custom_scenario.py`
   in this repo makes this concrete: two "independent-looking" downstream
   effects of a shared root cause have almost **9×** the joint probability
   that a naive independence assumption would predict.

**4. The one language built out to completion: probabilistic HL.**
The paper's central worked construction (Section 6) makes two independence
choices deliberately, in *different* directions, exactly to illustrate rule
5:

- **Facts** (the *data* — expected to be large, e.g. sensor/test results
  per individual) are assumed **tuple-independent**: each `A(a)` gets its
  own probability, and facts are treated as independent Bernoulli
  variables — the standard assumption from probabilistic databases.
- **Hyperedges** (the *schema* — expected to be small) are **not** assumed
  independent of each other. Instead, every hyperedge is mapped by a
  *context function* `γ` to a subset of the variables of a **Bayesian
  network** `B`. A hyperedge only "fires" in a possible world where every
  variable in its context is `True`, and its own stated probability must
  equal `P_B(γ(α))` — the BN's marginal probability that its whole context
  holds. Because the BN can encode arbitrary joint dependencies between
  contexts (shared parents, chains, etc.), two hyperedges that both depend
  on, say, `Cold`, are correctly *correlated* instead of independently
  multiplied — which is precisely the flaw rule 5 warns about (the paper's
  own worked counter-example: computing "chance of low visibility" by
  independently multiplying "chance of fog" and "chance of smog"
  double-counts a shared `Cold` fact).

  Formally, a **possible world** is a pair `(F, X)`: a subset `F` of the
  facts (chosen true; the rest false, each with its own/complement
  probability — tuple independence) together with a full truth assignment
  `X` to the BN's variables (weighted by the BN's joint distribution). Each
  world induces one *classical* HL ontology — `F`'s facts plus every
  hyperedge whose context is `⊆ X` — and

  ```
  P(c) = Σ { P(w) : w a possible world, O(w) ⊨ c }
  ```

  i.e. the probability of a consequence is the total probability mass of
  every possible world that classically entails it. This is an **open-world**
  semantics: a fact or hyperedge not explicitly included in a world can
  still hold there if it's *entailed* by what is, so `P` is really only a
  lower bound on the true probability of an axiom holding.

**5. Complexity.** Read literally, this needs to enumerate every subset of
facts *and* every truth assignment of the BN — exponential in both. The
paper's key algorithmic insight is that this can be re-organized: fix a BN
world `X` first (there are only as many as the — assumed small — Bayesian
network allows), which pins down one classical hypergraph, and reduce
computing `P_X(c)` to a probability query over a *tuple-independent
probabilistic database* (a union-of-conjunctive-queries problem, built by
tracing the fixed hypergraph backwards from `c`). Since that query problem
has data complexity that's merely polynomial (Dalvi & Suciu's dichotomy for
UCQs), and the BN enumeration only scales with the ontology's (small)
schema rather than its (large) data, `P(c) = Σ_X P_B(X)·P_X(c)` is
computable in time polynomial in the size of the data. This repo
implements the direct, literal definition (see "What's implemented" below)
rather than this optimized algorithm.

## What's implemented

| File | What it is |
|---|---|
| `probontology/bayesnet.py` | Exact inference (by full enumeration) over small discrete Bayesian networks: joint probability, and `marginal(event)` for `P_B(context)`. |
| `probontology/hl.py` | The classical `HL` ontology language: hyperedges + facts, forward-chaining reachability/entailment. |
| `probontology/probabilistic_hl.py` | The probabilistic extension: hyperedges annotated with a BN context, tuple-independent facts, and `entailment_probability(concept, individual)` implementing `P(c) = Σ_w P(w)` exactly as defined (the "naive" reference semantics — exponential, but exact and simple; adequate for the small examples here). |
| `examples/paper_example.py` | Reproduces the paper's own Figure 1 Bayesian network and Table 2/Figure 3 probabilistic ontology, verifies every `P_B(context)` against the paper's `P_exa` column, confirms the paper's own stated results `P(Observation(p1)) = 0.5` and `P(Observation(p3)) = 0`, and **computes `P(Observation(p2)) = 0.0112`**, the value the paper leaves as an exercise. |
| `examples/custom_scenario.py` | An original example (factory quality control) using the same library, demonstrating Golden Rule 5 with a concrete numeric comparison against a naive independence assumption. |
| `tests/` | pytest suite checking all of the above against hand-derived values from the paper. |

**Not implemented**: the paper's optimized, data-polynomial algorithm
(reducing each Bayesian world to a union-of-conjunctive-queries probability
computation over a probabilistic database). That's a substantial piece of
machinery in its own right (safe query plans / lineage-based probability
computation); this repo sticks to the direct semantics, which is exact and
easy to verify, and is fine for the ontology sizes a demo needs.

## Running it

```bash
pip install -r requirements.txt   # only pytest, for the test suite

# Reproduce the paper's own worked example:
PYTHONPATH=. python3 examples/paper_example.py

# An original example built with the same library:
PYTHONPATH=. python3 examples/custom_scenario.py

# Run the tests:
python3 -m pytest
```

### Quick taste of the API

```python
from probontology import BayesianNetwork, ProbabilisticHLOntology, ProbabilisticHyperedge

bn = BayesianNetwork(
    parents={"Flu": ()},
    cpt={"Flu": {(): 0.1}},
)

onto = ProbabilisticHLOntology(
    hyperedges=[
        ProbabilisticHyperedge({"Flu"}, "RunNose", context={"Flu"}),
    ],
    facts={("Flu", "alice"): 0.8},
    bn=bn,
)

print(onto.entailment_probability("RunNose", "alice"))  # 0.8 * P_B({Flu}) = 0.08
```

## Repository layout

```
probontology/            the library
  bayesnet.py             Bayesian networks + exact inference
  hl.py                   classical HL ontology language
  probabilistic_hl.py     probabilistic HL (Section 6 of the paper)
examples/
  paper_example.py         reproduces the paper's Table 2 / Figures 1,3,4
  custom_scenario.py        an original example
tests/                    pytest suite validating against the paper's numbers
```
