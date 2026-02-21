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
        """Return 'approve', 'deny', or 'pend' based on deterministic rules / heuristics."""
        if not retrieved_chunks:
            return "pend"

        proc = request.procedure_code
        if proc not in self.templates:
            return "pend"

        template = self.templates[proc]

        has_exclusion = any(
            c.section_type == "exclusions"
            for c in retrieved_chunks
        )

        has_coverage = any(
            c.section_type == "coverage_criteria"
            for c in retrieved_chunks
        )

        has_matching_diagnoses = any(
            any(diagnosis.startswith(prefix)
                for prefix in template["requires_diagnosis_prefix"])
            for diagnosis in request.diagnosis_codes
        )

        has_required_treatments = any(
            treatment in template["requires_prior_treatments"]
            for treatment in request.prior_treatments
        )

        if has_exclusion:
            return "deny"
        elif has_coverage and has_matching_diagnoses and has_required_treatments:
            return "approve"
        else:
            return "pend"
