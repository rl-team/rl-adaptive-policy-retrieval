"""
Mock PA Simulator -- Fake prior authorization simulator for RL development.

Self-contained module that replicates the public API of the real PASimulator
(simulator/oracle.py + simulator/request_generator.py + simulator/retriever.py)
so that the RL environment and agent can be built and tested before the real
data pipeline is ready.

This mock will be swapped out when the real simulator is integrated.

Reference: EDD 7.1 Simulator API, Use Cases 1-3.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Mock Data Classes (temporary -- will be replaced by the real simulator's types)
# ---------------------------------------------------------------------------

@dataclass
class MockPARequest:
    """Synthetic prior authorization request.

    Mirrors the real PARequest (EDD 6.1) but lives only in the mock.
    """
    request_id: str
    procedure_code: str          # CPT/HCPCS code, e.g. "72148"
    diagnosis_codes: List[str]   # ICD-10 codes, e.g. ["M54.5"]
    patient_age: int
    prior_treatments: List[str]  # e.g. ["physical_therapy", "NSAIDs"]
    urgency: str                 # "routine" | "urgent"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockPolicyChunk:
    """Synthetic policy chunk from the CMS corpus.

    Mirrors the real PolicyChunk (EDD 6.1) but with fake content.
    """
    chunk_id: str
    policy_id: str               # Parent LCD/NCD ID
    text: str                    # Human-readable chunk text
    embedding: np.ndarray        # 768-dim sentence-transformer embedding
    section_type: str            # "coverage_criteria" | "exclusions" | "billing"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Procedure Templates (matches EDD Use Case 1 template structure)
#
# CPT codes used below are real codes from the AMA CPT code set, verifiable at:
#   https://www.cms.gov/medicare/regulations-guidance/physician-self-referral/list-cpt-hcpcs-codes
#
# - CPT 72148: "MRI lumbar spine w/o dye"
#   -> ncd_trkg: NCD_id=177, NCD_mnl_sect="220.2", title "Magnetic Resonance Imaging"
# - CPT 29881: "Knee arthroscopy/surgery"
#   -> ncd_trkg: NCD_id=285, NCD_mnl_sect="150.9", title "Arthroscopic Lavage
#      and Arthroscopic Debridement for the Osteoarthritic Knee"
#
# ICD-10 codes (M54.5, M51.26, M23.21, etc.) are diagnosis codes
# from the WHO ICD-10-CM classification.
# ---------------------------------------------------------------------------

PROCEDURE_TEMPLATES = {
    "72148": {
        "name": "MRI Lumbar Spine",
        "common_diagnoses": [
            {"code": "M54.5", "name": "Low back pain", "weight": 0.6},
            {"code": "M51.26", "name": "Lumbar disc degeneration", "weight": 0.3},
            {"code": "G89.29", "name": "Other chronic pain", "weight": 0.1},
        ],
        "age_range": [18, 85],
        "typical_prior_treatments": [
            "physical_therapy", "NSAIDs", "chiropractic",
        ],
        "requires_diagnosis_prefix": ["M54", "M51"],
        "requires_prior_treatments": ["physical_therapy", "chiropractic"],
    },
    "29881": {
        "name": "Knee Arthroscopy",
        "common_diagnoses": [
            {"code": "M23.21", "name": "Derangement of meniscus", "weight": 0.5},
            {"code": "M17.11", "name": "Primary osteoarthritis, right knee", "weight": 0.3},
            {"code": "S83.511A", "name": "Sprain of ACL, right knee", "weight": 0.2},
        ],
        "age_range": [16, 75],
        "typical_prior_treatments": [
            "physical_therapy", "corticosteroid_injection", "bracing",
        ],
        "requires_diagnosis_prefix": ["M23", "M17", "S83"],
        "requires_prior_treatments": ["physical_therapy", "corticosteroid_injection"],
    },
}


# ---------------------------------------------------------------------------
# Chunk Templates -- Synthetic policy chunks for each procedure
#
# The text below is synthetic but modeled after real coverage language from
# the CMS Medicare Coverage Database (https://www.cms.gov/medicare-coverage-database).
#
# Specifically, the language and structure mirrors:
#   - ncd_trkg.itm_srvc_desc    (service descriptions)
#   - ncd_trkg.indctn_lmtn      (indications and limitations)
#   - lcd.indication             (covered indications)
#   - lcd.diagnoses_support      (ICD codes that support medical necessity)
#   - lcd.coding_guidelines      (billing/coding instructions)
#
# These tables come from the CMS publicly downloadable LCD and NCD databases
# (MS Access .mdb format) at:
#   https://www.cms.gov/medicare-coverage-database/downloads/downloadable-databases.aspx
# ---------------------------------------------------------------------------

def _build_chunk_templates() -> List[dict]:
    """Return a list of chunk template dicts (text + section_type + policy_id).

    Creates 10 chunks per procedure (20 total for 2 procedures), following
    the distribution suggested in the EDD:
    - coverage_criteria (4 per procedure)
    - exclusions         (3 per procedure)
    - billing            (3 per procedure)
    """
    templates: List[dict] = []

    # ---- MRI Lumbar Spine (72148) ----
    templates.extend([
        {
            "policy_id": "LCD_72148",
            "section_type": "coverage_criteria",
            "text": (
                "Coverage criteria for lumbar MRI: Patient must present with "
                "low back pain (M54.x) or disc degeneration (M51.x) that has "
                "not responded to conservative therapy for at least 6 weeks."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "coverage_criteria",
            "text": (
                "Prior treatment requirement: Documentation of failed "
                "physical therapy or chiropractic treatment is required "
                "before lumbar MRI authorization."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "coverage_criteria",
            "text": (
                "Clinical indications: Radiculopathy, progressive neurological "
                "deficit, or cauda equina syndrome are covered indications "
                "for lumbar spine MRI without prior conservative therapy."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "coverage_criteria",
            "text": (
                "Age considerations: Patients over 50 with new onset back pain "
                "and red flag symptoms (fever, weight loss, history of cancer) "
                "may be approved without prior therapy documentation."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "exclusions",
            "text": (
                "Exclusion: MRI lumbar spine is not covered if a similar MRI "
                "was performed within the last 90 days unless new symptoms or "
                "contraindication to prior imaging modality exists."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "exclusions",
            "text": (
                "Contraindication screening: Patients with ferromagnetic "
                "implants, certain cardiac pacemakers, or cochlear implants "
                "may not be eligible for MRI. Alternative imaging required."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "exclusions",
            "text": (
                "Non-covered: Screening MRI for asymptomatic patients or for "
                "pre-employment or insurance eligibility purposes is excluded "
                "from coverage under this LCD."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "billing",
            "text": (
                "Billing guidance for CPT 72148: Use modifier -59 when "
                "performed with other spinal imaging on the same date. "
                "Include ICD-10 code supporting medical necessity."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "billing",
            "text": (
                "Documentation requirements: The ordering provider must "
                "include clinical notes indicating the duration and severity "
                "of symptoms, prior treatments attempted, and clinical findings."
            ),
        },
        {
            "policy_id": "LCD_72148",
            "section_type": "billing",
            "text": (
                "Reimbursement: CPT 72148 is reimbursed at the facility rate "
                "when performed in a hospital outpatient setting. Professional "
                "component (modifier -26) billed separately by radiologist."
            ),
        },
    ])

    # ---- Knee Arthroscopy (29881) ----
    templates.extend([
        {
            "policy_id": "LCD_29881",
            "section_type": "coverage_criteria",
            "text": (
                "Coverage criteria for knee arthroscopy: Patient must have "
                "a documented meniscal tear (M23.x) or ligament injury (S83.x) "
                "confirmed by clinical examination or prior imaging."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "coverage_criteria",
            "text": (
                "Conservative therapy requirement: At least 6 weeks of "
                "physical therapy or corticosteroid injection must have been "
                "attempted before arthroscopic surgery authorization."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "coverage_criteria",
            "text": (
                "Functional limitation: Patient must demonstrate functional "
                "limitation (e.g., mechanical locking, giving way, inability "
                "to perform activities of daily living) due to knee pathology."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "coverage_criteria",
            "text": (
                "Imaging confirmation: MRI or other advanced imaging showing "
                "meniscal tear, loose body, or chondral damage supports "
                "medical necessity for arthroscopic intervention."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "exclusions",
            "text": (
                "Exclusion: Arthroscopic debridement for isolated knee "
                "osteoarthritis (M17.x) without mechanical symptoms is "
                "contraindicated and not covered per current evidence."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "exclusions",
            "text": (
                "Age limitation: Meniscal debridement in patients over 65 "
                "with degenerative tears requires additional documentation "
                "of mechanical symptoms to distinguish from osteoarthritis."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "exclusions",
            "text": (
                "Contraindication: Active joint infection, severe peripheral "
                "vascular disease, or inability to undergo anesthesia are "
                "contraindications to arthroscopic surgery."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "billing",
            "text": (
                "Billing guidance for CPT 29881: Includes diagnostic "
                "arthroscopy when performed with surgical arthroscopy. "
                "Do not separately report CPT 29870."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "billing",
            "text": (
                "Documentation for 29881: Operative report must include "
                "findings, procedures performed, and description of "
                "meniscal pathology addressed during intervention."
            ),
        },
        {
            "policy_id": "LCD_29881",
            "section_type": "billing",
            "text": (
                "Prior authorization timeline: Requests must be submitted "
                "at least 5 business days before scheduled procedure date. "
                "Expedited review available for acute traumatic injuries."
            ),
        },
    ])

    return templates


# ---------------------------------------------------------------------------
# MockPASimulator
# ---------------------------------------------------------------------------

class MockPASimulator:
    """Fake PA simulator with the same public API as the real PASimulator.

    Generates synthetic PA requests, maintains a fake policy chunk corpus
    with 768-dim embeddings, and provides a deterministic oracle that
    exercises all three decision paths (approve / deny / pend).

    Usage::

        sim = MockPASimulator(num_chunks=20, seed=42)
        request = sim.generate_request(procedure_code="72148")
        candidates = sim.get_top_k_candidates(query_embedding, k=10)
        chunk = sim.get_chunk(candidates[0])
        decision = sim.oracle_decision(request, [chunk])

    Parameters
    ----------
    num_chunks : int
        Number of synthetic policy chunks to generate (default 20).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, num_chunks: int = 20, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._request_counter = 0
        self._corpus = self._build_corpus(num_chunks)

    # ---- Corpus Construction ----

    def _build_corpus(self, num_chunks: int) -> List[MockPolicyChunk]:
        """Create synthetic policy chunks with random 768-dim embeddings."""
        chunk_templates = _build_chunk_templates()

        # Trim or repeat templates to match requested num_chunks
        if num_chunks <= len(chunk_templates):
            selected = chunk_templates[:num_chunks]
        else:
            # Repeat templates cyclically
            selected = []
            for i in range(num_chunks):
                selected.append(chunk_templates[i % len(chunk_templates)])

        corpus: List[MockPolicyChunk] = []
        for i, tmpl in enumerate(selected):
            embedding = self._rng.standard_normal(768).astype(np.float32)
            # Normalize to unit vector (cosine similarity is meaningful)
            embedding = embedding / np.linalg.norm(embedding)
            corpus.append(MockPolicyChunk(
                chunk_id=f"mock_chunk_{i:03d}",
                policy_id=tmpl["policy_id"],
                text=tmpl["text"],
                embedding=embedding,
                section_type=tmpl["section_type"],
                metadata={"source": "mock", "index": i},
            ))
        return corpus

    # ---- Public API (matches real PASimulator, EDD 7.1) ----

    def generate_request(
        self, procedure_code: Optional[str] = None,
    ) -> MockPARequest:
        """Generate a synthetic PA request.

        If *procedure_code* is not specified, one is randomly sampled from
        the available procedure templates.

        Parameters
        ----------
        procedure_code : str or None
            CPT/HCPCS code (e.g. "72148"). Random if None.

        Returns
        -------
        MockPARequest

        Raises
        ------
        KeyError
            If *procedure_code* is not in PROCEDURE_TEMPLATES.
        """
        if procedure_code is None:
            procedure_code = self._rng.choice(
                list(PROCEDURE_TEMPLATES.keys()),
            )
        if procedure_code not in PROCEDURE_TEMPLATES:
            raise KeyError(
                f"Unknown procedure code '{procedure_code}'. "
                f"Available: {list(PROCEDURE_TEMPLATES.keys())}"
            )

        tmpl = PROCEDURE_TEMPLATES[procedure_code]

        # Sample diagnosis (weighted)
        diagnoses = tmpl["common_diagnoses"]
        weights = np.array([d["weight"] for d in diagnoses])
        weights = weights / weights.sum()  # Normalize
        chosen_dx = self._rng.choice(diagnoses, p=weights)

        # Sample patient context
        age_lo, age_hi = tmpl["age_range"]
        age = int(self._rng.integers(age_lo, age_hi + 1))

        # Sample 1–3 prior treatments
        all_treatments = tmpl["typical_prior_treatments"]
        n_treatments = int(self._rng.integers(1, min(4, len(all_treatments) + 1)))
        prior_treatments = list(
            self._rng.choice(all_treatments, size=n_treatments, replace=False),
        )

        # Urgency: 90% routine, 10% urgent
        urgency = "urgent" if self._rng.random() < 0.1 else "routine"

        self._request_counter += 1
        return MockPARequest(
            request_id=f"mock_req_{self._request_counter:04d}",
            procedure_code=procedure_code,
            diagnosis_codes=[chosen_dx["code"]],
            patient_age=age,
            prior_treatments=prior_treatments,
            urgency=urgency,
            metadata={"diagnosis_name": chosen_dx["name"]},
        )

    def get_corpus(self) -> List[MockPolicyChunk]:
        """Return the full list of policy chunks."""
        return list(self._corpus)

    def get_chunk(self, chunk_idx: int) -> MockPolicyChunk:
        """Return a single chunk by its corpus index.

        Parameters
        ----------
        chunk_idx : int
            Index into the corpus (0-based).

        Returns
        -------
        MockPolicyChunk

        Raises
        ------
        IndexError
            If *chunk_idx* is out of range.
        """
        if chunk_idx < 0 or chunk_idx >= len(self._corpus):
            raise IndexError(
                f"Chunk index {chunk_idx} out of range "
                f"[0, {len(self._corpus)})"
            )
        return self._corpus[chunk_idx]

    def get_top_k_candidates(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude: Optional[set] = None,
    ) -> List[int]:
        """Return indices of the top-K most similar chunks to *query_embedding*.

        Uses cosine similarity over the corpus embeddings (matching EDD
        Use Case 2, Steps 3–8).

        Parameters
        ----------
        query_embedding : np.ndarray
            768-dim query vector.
        k : int
            Number of candidates to return.
        exclude : set of int or None
            Indices of already-retrieved chunks to skip.

        Returns
        -------
        list of int
            Chunk indices sorted by descending similarity.
        """
        if exclude is None:
            exclude = set()

        # Stack all corpus embeddings into a matrix
        corpus_embeddings = np.stack(
            [c.embedding for c in self._corpus], axis=0,
        )  # (num_chunks, 768)

        # Normalize query
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding) + 1e-8
        )

        # Cosine similarity (corpus embeddings are already unit-normalized)
        similarities = corpus_embeddings @ query_norm  # (num_chunks,)

        # Mask excluded indices
        for idx in exclude:
            similarities[idx] = -np.inf

        # Top-K
        top_indices = np.argsort(similarities)[::-1][:k]
        return [int(i) for i in top_indices if similarities[i] > -np.inf]

    def oracle_decision(
        self,
        request: MockPARequest,
        retrieved_chunks: List[MockPolicyChunk],
    ) -> str:
        """Deterministic oracle decision based on coverage rules.

        Logic (matching EDD Use Case 3, Steps 5–13):
        1. Check if any retrieved chunk is an "exclusions" chunk -> deny
        2. Check coverage criteria:
           a. At least one "coverage_criteria" chunk retrieved
           b. Diagnosis matches the procedure's required prefix
           c. At least one required prior treatment attempted
        3. If all coverage criteria met -> approve
        4. Otherwise -> pend (insufficient evidence)

        Parameters
        ----------
        request : MockPARequest
            The PA request being evaluated.
        retrieved_chunks : list of MockPolicyChunk
            Policy chunks the agent retrieved.

        Returns
        -------
        str
            "approve", "deny", or "pend"
        """
        if not retrieved_chunks:
            return "pend"

        proc = request.procedure_code
        if proc not in PROCEDURE_TEMPLATES:
            return "pend"

        tmpl = PROCEDURE_TEMPLATES[proc]

        # --- Step 1: Check exclusions ---
        # If any exclusion chunk from the correct policy is retrieved,
        # the oracle denies (simulates finding a contraindication).
        has_exclusion_chunk = any(
            c.section_type == "exclusions"
            and c.policy_id == f"LCD_{proc}"
            for c in retrieved_chunks
        )

        # --- Step 2: Check coverage criteria ---
        has_coverage_chunk = any(
            c.section_type == "coverage_criteria"
            and c.policy_id == f"LCD_{proc}"
            for c in retrieved_chunks
        )

        # Diagnosis prefix matching (e.g., "M54.5" starts with "M54")
        has_matching_diagnosis = any(
            any(
                dx_code.startswith(prefix)
                for prefix in tmpl["requires_diagnosis_prefix"]
            )
            for dx_code in request.diagnosis_codes
        )

        # Prior treatment check
        has_required_treatment = any(
            tx in tmpl["requires_prior_treatments"]
            for tx in request.prior_treatments
        )

        # --- Decision logic ---
        if has_exclusion_chunk:
            return "deny"
        elif has_coverage_chunk and has_matching_diagnosis and has_required_treatment:
            return "approve"
        else:
            return "pend"


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Run a quick sanity check exercising all oracle decision paths."""

    print("=" * 70)
    print("  MockPASimulator -- Self-Test")
    print("=" * 70)

    sim = MockPASimulator(num_chunks=20, seed=42)

    # --- 1. Show corpus summary ---
    corpus = sim.get_corpus()
    print(f"\nCorpus: {len(corpus)} chunks")
    section_counts: Dict[str, int] = {}
    for c in corpus:
        section_counts[c.section_type] = section_counts.get(c.section_type, 0) + 1
    for stype, count in sorted(section_counts.items()):
        print(f"  {stype}: {count} chunks")

    # --- 2. Generate requests and test all 3 decision paths ---
    print("\n" + "-" * 70)
    print("  Testing oracle decision paths")
    print("-" * 70)

    # Path A: APPROVE -- coverage_criteria chunks + matching diagnosis + treatment
    print("\n[Test A] Expected: APPROVE")
    req_a = MockPARequest(
        request_id="test_approve",
        procedure_code="72148",
        diagnosis_codes=["M54.5"],
        patient_age=45,
        prior_treatments=["physical_therapy"],
        urgency="routine",
    )
    # Give it a coverage_criteria chunk from LCD_72148
    coverage_chunks = [c for c in corpus
                       if c.section_type == "coverage_criteria"
                       and c.policy_id == "LCD_72148"]
    decision_a = sim.oracle_decision(req_a, coverage_chunks[:1])
    print(f"  Request: {req_a.procedure_code}, dx={req_a.diagnosis_codes}, "
          f"tx={req_a.prior_treatments}")
    print(f"  Retrieved: {len(coverage_chunks[:1])} coverage_criteria chunks")
    print(f"  Decision: {decision_a}")
    assert decision_a == "approve", f"Expected 'approve', got '{decision_a}'"
    print("  PASSED")

    # Path B: DENY -- exclusion chunk present
    print("\n[Test B] Expected: DENY")
    req_b = MockPARequest(
        request_id="test_deny",
        procedure_code="72148",
        diagnosis_codes=["M54.5"],
        patient_age=45,
        prior_treatments=["physical_therapy"],
        urgency="routine",
    )
    exclusion_chunks = [c for c in corpus
                        if c.section_type == "exclusions"
                        and c.policy_id == "LCD_72148"]
    decision_b = sim.oracle_decision(req_b, exclusion_chunks[:1])
    print(f"  Request: {req_b.procedure_code}, dx={req_b.diagnosis_codes}")
    print(f"  Retrieved: {len(exclusion_chunks[:1])} exclusions chunks")
    print(f"  Decision: {decision_b}")
    assert decision_b == "deny", f"Expected 'deny', got '{decision_b}'"
    print("  PASSED")

    # Path C: PEND -- no relevant chunks retrieved
    print("\n[Test C] Expected: PEND")
    req_c = MockPARequest(
        request_id="test_pend",
        procedure_code="72148",
        diagnosis_codes=["M54.5"],
        patient_age=45,
        prior_treatments=["physical_therapy"],
        urgency="routine",
    )
    # Give billing chunks only -- insufficient evidence for coverage
    billing_chunks = [c for c in corpus
                      if c.section_type == "billing"
                      and c.policy_id == "LCD_72148"]
    decision_c = sim.oracle_decision(req_c, billing_chunks[:1])
    print(f"  Request: {req_c.procedure_code}, dx={req_c.diagnosis_codes}")
    print(f"  Retrieved: {len(billing_chunks[:1])} billing chunks (no coverage criteria)")
    print(f"  Decision: {decision_c}")
    assert decision_c == "pend", f"Expected 'pend', got '{decision_c}'"
    print("  PASSED")

    # Path D: PEND -- empty retrieval
    print("\n[Test D] Expected: PEND (empty retrieval)")
    decision_d = sim.oracle_decision(req_c, [])
    print(f"  Retrieved: 0 chunks")
    print(f"  Decision: {decision_d}")
    assert decision_d == "pend", f"Expected 'pend', got '{decision_d}'"
    print("  PASSED")

    # --- 3. Test request generation and retrieval flow ---
    print("\n" + "-" * 70)
    print("  Full episode trace (request -> retrieval -> decision)")
    print("-" * 70)

    request = sim.generate_request(procedure_code="72148")
    print(f"\nGenerated request:")
    print(f"  ID:        {request.request_id}")
    print(f"  Procedure: {request.procedure_code}")
    print(f"  Diagnosis: {request.diagnosis_codes}")
    print(f"  Age:       {request.patient_age}")
    print(f"  Treatment: {request.prior_treatments}")
    print(f"  Urgency:   {request.urgency}")

    # Simulate a query embedding (use first chunk's embedding as proxy)
    query_emb = sim._rng.standard_normal(768).astype(np.float32)
    excluded: set = set()
    retrieved: List[MockPolicyChunk] = []

    print("\nRetrieval loop (max 5 steps):")
    for step in range(5):
        candidates = sim.get_top_k_candidates(query_emb, k=10, exclude=excluded)
        if not candidates:
            print(f"  Step {step + 1}: No more candidates available -- stopping")
            break

        chosen_idx = candidates[0]  # Greedy: pick most similar
        chunk = sim.get_chunk(chosen_idx)
        retrieved.append(chunk)
        excluded.add(chosen_idx)

        decision = sim.oracle_decision(request, retrieved)
        print(f"  Step {step + 1}: Retrieved chunk {chosen_idx:3d} "
              f"({chunk.section_type:20s}) | "
              f"Oracle: {decision}")

    # Final decision
    final_decision = sim.oracle_decision(request, retrieved)
    print(f"\nFinal state:")
    print(f"  Chunks retrieved: {len(retrieved)}")
    print(f"  Section types:    {[c.section_type for c in retrieved]}")
    print(f"  Oracle decision:  {final_decision}")

    # --- 4. Test second procedure ---
    print("\n" + "-" * 70)
    print("  Second procedure (29881 -- Knee Arthroscopy)")
    print("-" * 70)

    request2 = sim.generate_request(procedure_code="29881")
    print(f"\nGenerated request:")
    print(f"  ID:        {request2.request_id}")
    print(f"  Procedure: {request2.procedure_code}")
    print(f"  Diagnosis: {request2.diagnosis_codes}")
    print(f"  Treatment: {request2.prior_treatments}")

    # Retrieve coverage_criteria chunk from 29881
    coverage_29881 = [c for c in corpus
                      if c.section_type == "coverage_criteria"
                      and c.policy_id == "LCD_29881"]
    decision_2 = sim.oracle_decision(request2, coverage_29881[:1])
    print(f"  Decision with 1 coverage chunk: {decision_2}")

    # --- 5. Test error handling ---
    print("\n" + "-" * 70)
    print("  Error handling tests")
    print("-" * 70)

    # Invalid procedure code
    try:
        sim.generate_request(procedure_code="99999")
        print("  FAILED -- should have raised KeyError")
    except KeyError as e:
        print(f"  PASSED -- KeyError on invalid procedure: {e}")

    # Out-of-range chunk index
    try:
        sim.get_chunk(999)
        print("  FAILED -- should have raised IndexError")
    except IndexError as e:
        print(f"  PASSED -- IndexError on bad index: {e}")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_test()
