"""Corpus coverage validation tests (M28).

Validates the expanded 10-procedure corpus with 50 test cases per procedure.
For each procedure:
  - Ground truth = oracle decision with the full corpus
  - Evaluated at FixedK k=3, k=5, k=7 via semantic retrieval
  - Every procedure must be solvable (>0% accuracy at k=7)
  - No procedure should be trivially solved at k=3 across all runs

Run with:
    pytest tests/test_corpus_coverage.py -v
    pytest tests/test_corpus_coverage.py -v -k "structure"   # fast checks only
    pytest tests/test_corpus_coverage.py -v -k "fixedk"      # retrieval accuracy
"""

from __future__ import annotations

import json
import os

import pytest

from simulator.pa_simulator import PASimulator

# Number of test episodes per procedure
EPISODES_PER_PROC = 50
# Random seed
SEED = 99
# Minimum acceptable per-procedure accuracy at k=7 to confirm solvability
MIN_ACCURACY_K7 = 0.10
# Minimum acceptable overall accuracy at k=5
MIN_OVERALL_K5 = 0.40

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TEMPLATES_PATH = os.path.join(DATA_DIR, "templates.json")
CORPUS_STATS_PATH = os.path.join(DATA_DIR, "corpus_stats.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim():
    return PASimulator(seed=SEED)


@pytest.fixture(scope="module")
def templates():
    with open(TEMPLATES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def corpus_stats():
    with open(CORPUS_STATS_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_procedures(templates):
    return sorted(templates.keys())


# ---------------------------------------------------------------------------
# M28 part 1: Corpus structure
# ---------------------------------------------------------------------------

def test_all_procedures_have_min_coverage_chunks(templates):
    """Every procedure template must declare min_coverage_chunks."""
    for pc, tmpl in templates.items():
        assert "min_coverage_chunks" in tmpl, (
            f"Procedure {pc} ({tmpl['name']}) missing min_coverage_chunks"
        )
        assert tmpl["min_coverage_chunks"] >= 1, (
            f"Procedure {pc} has min_coverage_chunks < 1"
        )


def test_min_coverage_chunks_feasible(corpus_stats):
    """min_coverage_chunks must not exceed available coverage_criteria chunks."""
    for pc, proc in corpus_stats["procedures"].items():
        mc = proc["min_coverage_chunks"]
        avail = proc["coverage_criteria"]
        assert mc <= avail, (
            f"Procedure {pc} ({proc['name']}): min_coverage_chunks={mc} "
            f"exceeds available coverage chunks={avail}"
        )


def test_no_procedure_dominates_corpus(corpus_stats):
    """No single procedure should exceed 25% of the total corpus."""
    for pc, proc in corpus_stats["procedures"].items():
        assert proc["corpus_pct"] <= 25.0, (
            f"Procedure {pc} ({proc['name']}) dominates corpus: "
            f"{proc['corpus_pct']:.1f}% (limit 25%)"
        )


def test_every_procedure_has_coverage_chunks(corpus_stats):
    """Every procedure must have at least 1 coverage_criteria chunk."""
    for pc, proc in corpus_stats["procedures"].items():
        assert proc["coverage_criteria"] >= 1, (
            f"Procedure {pc} ({proc['name']}) has 0 coverage_criteria chunks"
        )


def test_corpus_stats_covers_all_template_procedures(templates, corpus_stats):
    """corpus_stats.json must include an entry for every procedure in templates.json."""
    template_procs = set(templates.keys())
    stats_procs = set(corpus_stats["procedures"].keys())
    missing = template_procs - stats_procs
    assert not missing, f"Procedures in templates but not corpus_stats: {missing}"


def test_procedure_code_populated_on_all_chunks(sim):
    """Every corpus chunk must have a non-empty procedure_code."""
    corpus = sim.get_corpus()
    empty = [c.chunk_id for c in corpus if not c.procedure_code]
    assert not empty, (
        f"{len(empty)} chunks have empty procedure_code: {empty[:5]}"
    )


# ---------------------------------------------------------------------------
# M28 part 2: Per-procedure oracle solvability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("proc_code", [
    "22633", "29881", "43239", "71260", "72148",
    "77067", "90837", "92507", "93306", "95810",
])
def test_procedure_solvable_at_k7(sim, proc_code):
    """Each procedure must have >0% accuracy at k=7 (procedure is retrievable).

    Ground truth = oracle with full corpus.
    Test accuracy = oracle with top-7 retrieved chunks.
    """
    corpus = sim.get_corpus()
    correct = 0

    for _ in range(EPISODES_PER_PROC):
        request = sim.generate_request(procedure_code=proc_code)
        gt = sim.oracle_decision(request, corpus)
        query_emb = sim.encode(request.to_text())
        candidates = sim.get_top_k_candidates(query_emb, k=7)
        retrieved = [sim.get_chunk(idx) for idx in candidates]
        decision = sim.oracle_decision(request, retrieved)
        correct += int(decision == gt)

    accuracy = correct / EPISODES_PER_PROC
    assert accuracy > MIN_ACCURACY_K7, (
        f"Procedure {proc_code}: {correct}/{EPISODES_PER_PROC} = {accuracy:.0%} "
        f"at k=7, below minimum {MIN_ACCURACY_K7:.0%}. "
        "Oracle cannot surface this procedure's chunks -- check source list in templates.json."
    )


# ---------------------------------------------------------------------------
# M28 part 3: FixedK accuracy across all procedures
# ---------------------------------------------------------------------------

def test_fixedk_accuracy(sim, all_procedures):
    """FixedK accuracy must be monotonically non-decreasing: k=3 <= k=5 <= k=7.

    Also checks:
    - Overall k=5 accuracy >= MIN_OVERALL_K5
    - No procedure at exactly 0% across ALL three k values
    - k=7 does not achieve 100% across all procedures (corpus isn't trivial)
    """
    corpus = sim.get_corpus()

    # Collect results[k][proc] = (correct, total)
    results: dict[int, dict[str, list[int]]] = {k: {} for k in (3, 5, 7)}

    for _ in range(EPISODES_PER_PROC):
        for proc_code in all_procedures:
            request = sim.generate_request(procedure_code=proc_code)
            gt = sim.oracle_decision(request, corpus)
            query_emb = sim.encode(request.to_text())

            for k in (3, 5, 7):
                candidates = sim.get_top_k_candidates(query_emb, k=k)
                retrieved = [sim.get_chunk(idx) for idx in candidates]
                decision = sim.oracle_decision(request, retrieved)
                if proc_code not in results[k]:
                    results[k][proc_code] = [0, 0]
                results[k][proc_code][1] += 1
                results[k][proc_code][0] += int(decision == gt)

    # Per-procedure accuracy
    acc: dict[int, dict[str, float]] = {}
    for k in (3, 5, 7):
        acc[k] = {
            pc: results[k][pc][0] / results[k][pc][1]
            for pc in all_procedures
        }

    # Overall accuracy at each k
    overall = {
        k: sum(results[k][pc][0] for pc in all_procedures) /
           sum(results[k][pc][1] for pc in all_procedures)
        for k in (3, 5, 7)
    }

    # 1. Monotonic accuracy overall
    assert overall[3] <= overall[5] + 0.05, (
        f"Overall accuracy did not improve from k=3 ({overall[3]:.0%}) to k=5 ({overall[5]:.0%})"
    )
    assert overall[5] <= overall[7] + 0.05, (
        f"Overall accuracy did not improve from k=5 ({overall[5]:.0%}) to k=7 ({overall[7]:.0%})"
    )

    # 2. k=5 overall >= minimum threshold
    assert overall[5] >= MIN_OVERALL_K5, (
        f"Overall FixedK(k=5) accuracy {overall[5]:.0%} < {MIN_OVERALL_K5:.0%} minimum. "
        "Corpus may be too difficult or oracle filtering is broken."
    )

    # 3. No procedure at 0% across all k values
    for pc in all_procedures:
        max_acc = max(acc[k][pc] for k in (3, 5, 7))
        assert max_acc > 0.0, (
            f"Procedure {pc} has 0% accuracy at ALL k values (k=3,5,7). "
            "This procedure's chunks are never retrieved -- check corpus sources."
        )

    # 4. k=7 should not be 100% on every procedure (corpus is non-trivial)
    all_perfect = all(acc[7][pc] >= 1.0 for pc in all_procedures)
    assert not all_perfect, (
        "All procedures hit 100% accuracy at k=7. "
        "Corpus may be too easy -- consider raising min_coverage_chunks."
    )


# ---------------------------------------------------------------------------
# M28 part 4: Oracle produces all three decision types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("proc_code", [
    "22633", "29881", "43239", "71260", "72148",
    "77067", "90837", "92507", "93306", "95810",
])
def test_oracle_produces_approve_and_deny(sim, proc_code):
    """With full corpus, oracle must produce both 'approve' and 'deny' for each procedure.

    A procedure that only produces 'pend' has an impossible min_coverage_chunks.
    A procedure that only produces 'approve' or only 'deny' has broken clinical rules.
    """
    corpus = sim.get_corpus()
    decisions: set[str] = set()

    for _ in range(EPISODES_PER_PROC):
        request = sim.generate_request(procedure_code=proc_code)
        decision = sim.oracle_decision(request, corpus)
        decisions.add(decision)

    assert "approve" in decisions, (
        f"Procedure {proc_code}: oracle never returned 'approve' with full corpus "
        f"over {EPISODES_PER_PROC} requests. Check requires_diagnosis_prefix and requires_prior_treatments."
    )
    assert "deny" in decisions, (
        f"Procedure {proc_code}: oracle never returned 'deny' with full corpus "
        f"over {EPISODES_PER_PROC} requests. Check exclusions chunks or coverage logic."
    )
    assert "pend" not in decisions or len(decisions) >= 2, (
        f"Procedure {proc_code}: oracle only returned 'pend' with full corpus. "
        "min_coverage_chunks may exceed available coverage chunks."
    )
