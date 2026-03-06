"""Validation tests for the PA simulator (M5 + M9).

Updated for the 10-procedure expanded corpus (M23–M28).
Tests cover request generation, oracle approve/deny/pend logic,
and retriever functionality.

Key changes from the PoC tests:
- Uses current procedure set (see templates.json)
- Filters chunks via `c.procedure_codes` (list), not metadata
- Oracle is order-independent: pends without enough coverage evidence
"""

import pytest

from simulator.pa_simulator import PASimulator
from simulator.types import PARequest


@pytest.fixture(scope="module")
def sim():
    return PASimulator(seed=42)


@pytest.fixture(scope="module")
def procedures(sim):
    """Return sorted list of procedure codes in the current corpus."""
    return sorted(sim._oracle.templates.keys())


# ---- M5: Request generation tests ----

def test_generate_specific_procedure(sim):
    req = sim.generate_request("72148")
    assert req.procedure_code == "72148"
    assert len(req.diagnosis_codes) >= 1
    assert 18 <= req.patient_age <= 85
    assert len(req.prior_treatments) >= 1
    assert req.urgency in ("routine", "urgent")


def test_generate_random_procedure(sim, procedures):
    reqs = [sim.generate_request() for _ in range(20)]
    procs = {r.procedure_code for r in reqs}
    assert procs.issubset(set(procedures)), (
        f"Generated procedures {procs} not subset of {set(procedures)}"
    )
    for r in reqs:
        assert r.request_id
        assert len(r.diagnosis_codes) >= 1
        assert r.patient_age > 0


# ---- M9: Oracle approve tests ----

def test_approve_72148_with_matching_dx_and_tx(sim):
    """72148 MRI Lumbar: matching diagnosis + required treatment → approve."""
    req = PARequest("t1", "72148", ["M54.5"], 45, ["physical_therapy"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and "72148" in c.procedure_codes]
    assert len(coverage) >= 2, "Need >= min_coverage_chunks coverage for 72148"
    assert sim.oracle_decision(req, coverage[:2]) == "approve"


def test_approve_72148_disc_degeneration(sim):
    req = PARequest("t2", "72148", ["M51.26"], 60, ["chiropractic"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and "72148" in c.procedure_codes]
    assert sim.oracle_decision(req, coverage[:2]) == "approve"


def test_approve_45378_colonoscopy(sim):
    """45378 Colonoscopy: matching diagnosis + prior treatment → approve."""
    req = PARequest("t3", "45378", ["K57.3"], 55, ["FOBT"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and "45378" in c.procedure_codes]
    assert len(coverage) >= 1, "Need >= min_coverage_chunks coverage for 45378"
    assert sim.oracle_decision(req, coverage[:1]) == "approve"


def test_approve_77067_mammography(sim):
    """77067 Screening Mammography: requires only 1 coverage chunk."""
    req = PARequest("t4", "77067", ["Z12.31"], 50, ["clinical_breast_exam"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and "77067" in c.procedure_codes]
    assert len(coverage) >= 1, "Need >= 1 coverage chunk for 77067"
    assert sim.oracle_decision(req, coverage[:1]) == "approve"


# ---- Oracle deny tests ----

def test_deny_wrong_diagnosis(sim):
    """Correct procedure, enough coverage, but wrong diagnosis → deny."""
    req = PARequest("t5", "72148", ["Z99.99"], 45, ["physical_therapy"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and "72148" in c.procedure_codes]
    assert sim.oracle_decision(req, coverage[:2]) == "deny"


def test_deny_wrong_treatment(sim):
    """Correct diagnosis but no matching prior treatment → deny."""
    req = PARequest("t6", "72148", ["M54.5"], 45, ["NSAIDs"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and "72148" in c.procedure_codes]
    assert sim.oracle_decision(req, coverage[:2]) == "deny"


def test_deny_clinical_mismatch_full_corpus(sim):
    """With full corpus, wrong diagnosis → deny (not pend)."""
    req = PARequest("t7", "72148", ["Z99.99"], 45, ["physical_therapy"], "routine")
    all_chunks = sim.get_corpus()
    assert sim.oracle_decision(req, all_chunks) == "deny"


# ---- Oracle pend tests ----

def test_pend_empty_retrieval(sim):
    req = PARequest("t8", "72148", ["M54.5"], 45, ["physical_therapy"], "routine")
    assert sim.oracle_decision(req, []) == "pend"


def test_pend_insufficient_coverage(sim):
    """With 0 coverage chunks but mc=1, oracle should pend (no coverage evidence)."""
    req = PARequest("t9", "92507", ["R13.1"], 45, ["speech_evaluation"], "routine")
    billing = [c for c in sim.get_corpus() if c.section_type == "billing"
               and "92507" in c.procedure_codes]
    # Give only billing chunks — 0 coverage chunks, oracle needs at least 1
    assert sim.oracle_decision(req, billing[:2]) == "pend"


def test_pend_exclusion_only_no_coverage(sim):
    """Exclusion chunks WITHOUT enough coverage → pend (order-independent).

    This is a key behavioral change: the old oracle would deny on
    exclusion-only evidence, but the new oracle correctly pends because
    it doesn't have enough coverage chunks to make a definitive decision.
    """
    req = PARequest("t10", "72148", ["M54.5"], 45, ["physical_therapy"], "routine")
    excl = [c for c in sim.get_corpus() if c.section_type == "exclusions"
            and "72148" in c.procedure_codes]
    if excl:
        assert sim.oracle_decision(req, excl[:1]) == "pend"


def test_pend_billing_only(sim):
    req = PARequest("t11", "45378", ["K57.3"], 40, ["colonoscopy_prep"], "routine")
    billing = [c for c in sim.get_corpus() if c.section_type == "billing"
               and "45378" in c.procedure_codes]
    if billing:
        assert sim.oracle_decision(req, billing[:1]) == "pend"
    else:
        pytest.skip("No billing chunks for 45378")


def test_pend_unknown_procedure(sim):
    req = PARequest("t12", "99999", ["Z99.99"], 50, ["rest"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"]
    assert sim.oracle_decision(req, coverage[:1]) == "pend"


# ---- Oracle order-independence guarantee ----

def test_oracle_order_independent(sim):
    """Same chunks, different order → same decision."""
    req = PARequest("t13", "72148", ["M54.5"], 45, ["physical_therapy"], "routine")
    corpus = sim.get_corpus()

    cov = [c for c in corpus if c.section_type == "coverage_criteria"
           and "72148" in c.procedure_codes][:2]
    excl = [c for c in corpus if c.section_type == "exclusions"
            and "72148" in c.procedure_codes][:1]

    if excl:
        decision_a = sim.oracle_decision(req, excl + cov)
        decision_b = sim.oracle_decision(req, cov + excl)
        assert decision_a == decision_b, (
            f"Order-dependent: excl+cov={decision_a}, cov+excl={decision_b}"
        )
    else:
        pytest.skip("No exclusion chunks for 72148")


# ---- Retriever tests ----

def test_retriever_returns_valid_indices(sim):
    query = sim.encode("lumbar MRI back pain")
    candidates = sim.get_top_k_candidates(query, k=5)
    assert len(candidates) <= 5
    corpus_size = len(sim.get_corpus())
    for idx in candidates:
        assert 0 <= idx < corpus_size


def test_retriever_exclude_works(sim):
    query = sim.encode("colonoscopy diagnostic screening")
    all_candidates = sim.get_top_k_candidates(query, k=5)
    if len(all_candidates) >= 2:
        excluded = {all_candidates[0]}
        filtered = sim.get_top_k_candidates(query, k=5, exclude=excluded)
        assert all_candidates[0] not in filtered
