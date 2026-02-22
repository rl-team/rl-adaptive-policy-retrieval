"""Validation tests for the PA simulator (M5 + M9)."""

import pytest

from simulator.pa_simulator import PASimulator
from simulator.types import PARequest


@pytest.fixture(scope="module")
def sim():
    return PASimulator(seed=42)


# ---- M5: Request generation tests ----

def test_generate_specific_procedure(sim):
    req = sim.generate_request("72148")
    assert req.procedure_code == "72148"
    assert len(req.diagnosis_codes) >= 1
    assert 18 <= req.patient_age <= 85
    assert len(req.prior_treatments) >= 1
    assert req.urgency in ("routine", "urgent")


def test_generate_random_procedure(sim):
    reqs = [sim.generate_request() for _ in range(10)]
    procs = {r.procedure_code for r in reqs}
    assert procs.issubset({"72148", "29881"})
    for r in reqs:
        assert r.request_id
        assert len(r.diagnosis_codes) >= 1
        assert r.patient_age > 0


# ---- M9: Oracle approve tests ----

def test_approve_72148_with_matching_dx_and_tx(sim):
    req = PARequest("t1", "72148", ["M54.5"], 45, ["physical_therapy"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and c.metadata.get("procedure_code") == "72148"]
    assert sim.oracle_decision(req, coverage[:1]) == "approve"


def test_approve_72148_disc_degeneration(sim):
    req = PARequest("t2", "72148", ["M51.26"], 60, ["chiropractic"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and c.metadata.get("procedure_code") == "72148"]
    assert sim.oracle_decision(req, coverage[:1]) == "approve"


def test_approve_29881_meniscus(sim):
    req = PARequest("t3", "29881", ["M23.21"], 35, ["physical_therapy"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and c.metadata.get("procedure_code") == "29881"]
    assert sim.oracle_decision(req, coverage[:1]) == "approve"


def test_approve_29881_acl(sim):
    req = PARequest("t4", "29881", ["S83.511A"], 28, ["corticosteroid_injection"], "urgent")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and c.metadata.get("procedure_code") == "29881"]
    assert sim.oracle_decision(req, coverage[:1]) == "approve"


def test_approve_29881_osteoarthritis(sim):
    req = PARequest("t5", "29881", ["M17.11"], 55, ["physical_therapy", "corticosteroid_injection"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and c.metadata.get("procedure_code") == "29881"]
    assert sim.oracle_decision(req, coverage[:2]) == "approve"


# ---- M9: Oracle deny tests ----

def test_deny_72148_exclusion_chunk(sim):
    req = PARequest("t6", "72148", ["M54.5"], 45, ["physical_therapy"], "routine")
    excl = [c for c in sim.get_corpus() if c.section_type == "exclusions"]
    assert len(excl) > 0
    assert sim.oracle_decision(req, excl[:1]) == "deny"


def test_deny_29881_exclusion_chunk(sim):
    req = PARequest("t7", "29881", ["M23.21"], 40, ["physical_therapy"], "routine")
    excl = [c for c in sim.get_corpus() if c.section_type == "exclusions"]
    assert sim.oracle_decision(req, excl[:1]) == "deny"


def test_deny_mixed_with_exclusion(sim):
    req = PARequest("t8", "72148", ["M54.5"], 50, ["physical_therapy"], "routine")
    corpus = sim.get_corpus()
    coverage = [c for c in corpus if c.section_type == "coverage_criteria"][:1]
    excl = [c for c in corpus if c.section_type == "exclusions"][:1]
    assert sim.oracle_decision(req, coverage + excl) == "deny"


def test_deny_exclusion_overrides_coverage(sim):
    req = PARequest("t9", "29881", ["M23.21"], 30, ["corticosteroid_injection"], "routine")
    all_chunks = sim.get_corpus()
    assert sim.oracle_decision(req, all_chunks) == "deny"


def test_deny_multiple_exclusions(sim):
    req = PARequest("t10", "72148", ["M51.26"], 55, ["chiropractic"], "routine")
    excl = [c for c in sim.get_corpus() if c.section_type == "exclusions"]
    assert sim.oracle_decision(req, excl) == "deny"


# ---- M9: Oracle pend tests ----

def test_pend_empty_retrieval(sim):
    req = PARequest("t11", "72148", ["M54.5"], 45, ["physical_therapy"], "routine")
    assert sim.oracle_decision(req, []) == "pend"


def test_pend_wrong_diagnosis(sim):
    req = PARequest("t12", "72148", ["Z99.99"], 45, ["physical_therapy"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and c.metadata.get("procedure_code") == "72148"]
    assert sim.oracle_decision(req, coverage[:1]) == "pend"


def test_pend_no_required_treatment(sim):
    req = PARequest("t13", "72148", ["M54.5"], 45, ["NSAIDs"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"
                and c.metadata.get("procedure_code") == "72148"]
    assert sim.oracle_decision(req, coverage[:1]) == "pend"


def test_pend_billing_only(sim):
    req = PARequest("t14", "29881", ["M23.21"], 40, ["physical_therapy"], "routine")
    billing = [c for c in sim.get_corpus() if c.section_type == "billing"]
    if billing:
        assert sim.oracle_decision(req, billing[:1]) == "pend"
    else:
        pytest.skip("No billing chunks in corpus")


def test_pend_unknown_procedure(sim):
    req = PARequest("t15", "99999", ["Z99.99"], 50, ["rest"], "routine")
    coverage = [c for c in sim.get_corpus() if c.section_type == "coverage_criteria"]
    assert sim.oracle_decision(req, coverage[:1]) == "pend"


# ---- M9: Retriever tests ----

def test_retriever_returns_valid_indices(sim):
    query = sim.encode("lumbar MRI back pain")
    candidates = sim.get_top_k_candidates(query, k=5)
    assert len(candidates) <= 5
    corpus_size = len(sim.get_corpus())
    for idx in candidates:
        assert 0 <= idx < corpus_size


def test_retriever_exclude_works(sim):
    query = sim.encode("knee arthroscopy meniscus")
    all_candidates = sim.get_top_k_candidates(query, k=5)
    if len(all_candidates) >= 2:
        excluded = {all_candidates[0]}
        filtered = sim.get_top_k_candidates(query, k=5, exclude=excluded)
        assert all_candidates[0] not in filtered
