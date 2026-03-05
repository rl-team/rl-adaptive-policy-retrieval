"""Shared data classes for the PA simulator."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class PARequest:
    request_id: str
    procedure_code: str
    diagnosis_codes: List[str]
    patient_age: int
    prior_treatments: List[str]
    urgency: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Human-readable summary for embedding (EDD 5.2)."""
        return (
            f"Procedure: {self.procedure_code}. "
            f"Diagnoses: {', '.join(self.diagnosis_codes)}. "
            f"Age: {self.patient_age}. "
            f"Prior treatments: {', '.join(self.prior_treatments)}."
        )


@dataclass
class PolicyChunk:
    chunk_id: str
    policy_id: str
    text: str
    embedding: np.ndarray
    section_type: str
    procedure_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
