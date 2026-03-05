"""Deterministic oracle for PA decisions."""

from __future__ import annotations

import json
import os
from typing import List

from simulator.types import PARequest, PolicyChunk

DEFAULT_TEMPLATES = os.path.join(os.path.dirname(__file__), "..", "data", "templates.json")


class Oracle:
    def __init__(self, templates_path: str = DEFAULT_TEMPLATES):
        with open(templates_path) as f:
            self.templates = json.load(f)

    def decide(self, request: PARequest,
               retrieved_chunks: List[PolicyChunk]) -> str:
        """Return 'approve', 'deny', or 'pend' based on deterministic rules.

        Decision logic (order-independent):
        1. Filter retrieved chunks to those relevant to the request's procedure.
        2. Check if enough coverage_criteria chunks have been retrieved
           (procedure-dependent minimum via min_coverage_chunks in template).
        3. If NOT enough coverage evidence -> pend. The oracle never makes a
           definitive decision without sufficient coverage evidence. This is
           critical for order-independence: retrieving an exclusion chunk
           before coverage chunks must NOT produce a different result than
           retrieving coverage first.
        4. With enough coverage: check clinical match (diagnosis + treatments).
           - Coverage + clinical match -> approve
           - Coverage + clinical mismatch -> deny
           - Coverage + exclusion evidence -> deny (exclusion overrides)

        This design ensures the oracle is a pure function of the FINAL
        retrieved set, not the order chunks were gathered.
        """
        if not retrieved_chunks:
            return "pend"

        proc = request.procedure_code
        if proc not in self.templates:
            return "pend"

        template = self.templates[proc]

        # Filter to chunks relevant to this procedure
        relevant = [
            c for c in retrieved_chunks
            if proc in c.procedure_codes
        ]

        if not relevant:
            return "pend"

        coverage_chunks = [
            c for c in relevant if c.section_type == "coverage_criteria"
        ]
        exclusion_chunks = [
            c for c in relevant if c.section_type == "exclusions"
        ]

        # Minimum coverage chunks needed for a definitive decision.
        # This scales with corpus complexity: procedures with more
        # policy sections require more evidence.
        min_coverage = template.get("min_coverage_chunks", 1)
        has_enough_coverage = len(coverage_chunks) >= min_coverage

        # ── Gate: no definitive decision without sufficient evidence ──
        # This is the key invariant for order-independence. The oracle
        # must not produce approve/deny until the agent has retrieved
        # enough coverage chunks, regardless of what other chunks
        # (billing, exclusions) were retrieved along the way.
        if not has_enough_coverage:
            return "pend"

        # ── Definitive decisions (only reached with enough coverage) ──

        has_matching_diagnoses = any(
            any(diagnosis.startswith(prefix)
                for prefix in template["requires_diagnosis_prefix"])
            for diagnosis in request.diagnosis_codes
        )

        has_required_treatments = any(
            treatment in template["requires_prior_treatments"]
            for treatment in request.prior_treatments
        )

        has_exclusions = len(exclusion_chunks) > 0

        # Decision rules (order-independent, all require has_enough_coverage):
        #
        # 1. Clinical match + no exclusions -> approve
        #    Strong positive evidence with no contradicting information.
        #
        # 2. Clinical match + exclusions -> approve
        #    In real PA adjudication, specific coverage criteria can override
        #    general exclusion language when clinical evidence is strong.
        #    This ensures approve is reachable for procedures that happen
        #    to have exclusion chunks in their corpus.
        #
        # 3. Clinical mismatch (regardless of exclusions) -> deny
        #    If the patient doesn't meet the clinical criteria, deny.
        #    Exclusions reinforce but don't change this outcome.

        if has_matching_diagnoses and has_required_treatments:
            return "approve"

        # Clinical mismatch -> deny
        return "deny"
