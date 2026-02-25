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
        """Return 'approve', 'deny', or 'pend' based on deterministic rules / heuristics.

        Decision logic:
        1. Filter retrieved chunks to those relevant to the request's procedure.
        2. Count coverage_criteria chunks. The oracle needs a minimum number
           of coverage chunks to have enough evidence for a definitive decision.
           This minimum is procedure-dependent, creating varied difficulty.
        3. If enough coverage + matching diagnosis + prior treatments -> approve.
        4. If exclusion chunks outnumber coverage chunks -> deny.
        5. Otherwise -> pend (insufficient evidence).
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
            if c.procedure_code == proc
        ]

        if not relevant:
            return "pend"

        coverage_chunks = [
            c for c in relevant if c.section_type == "coverage_criteria"
        ]
        exclusion_chunks = [
            c for c in relevant if c.section_type == "exclusions"
        ]

        has_matching_diagnoses = any(
            any(diagnosis.startswith(prefix)
                for prefix in template["requires_diagnosis_prefix"])
            for diagnosis in request.diagnosis_codes
        )

        has_required_treatments = any(
            treatment in template["requires_prior_treatments"]
            for treatment in request.prior_treatments
        )

        # Minimum coverage chunks needed for a definitive decision.
        # This scales with corpus complexity: procedures with more
        # policy sections require more evidence.
        min_coverage = template.get("min_coverage_chunks", 1)

        has_enough_coverage = len(coverage_chunks) >= min_coverage

        # Decision rules:
        # 1. Enough exclusion evidence without coverage -> deny
        if len(exclusion_chunks) > 0 and not has_enough_coverage:
            return "deny"
        # 2. Enough coverage + clinical match -> approve
        if has_enough_coverage and has_matching_diagnoses and has_required_treatments:
            return "approve"
        # 3. Enough coverage but clinical mismatch -> deny
        if has_enough_coverage and (not has_matching_diagnoses or not has_required_treatments):
            return "deny"
        # 4. Not enough evidence either way -> pend
        return "pend"
