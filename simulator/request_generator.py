"""Generate synthetic PA requests from procedure templates."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from simulator.types import PARequest

DEFAULT_TEMPLATES = os.path.join(os.path.dirname(__file__), "..", "data", "templates.json")


class PARequestGenerator:
    def __init__(self, templates_path: str = DEFAULT_TEMPLATES, seed: int = 42):
        with open(templates_path) as f:
            self.templates = json.load(f)
        self._rng = np.random.default_rng(seed)
        self._counter = 0

    def generate(self, procedure_code: Optional[str] = None) -> PARequest:
        if procedure_code is None:
            procedure_code = self._rng.choice(list(self.templates.keys()))
        if procedure_code not in self.templates:
            raise KeyError(f"Unknown procedure code '{procedure_code}'. "
                           f"Available: {list(self.templates.keys())}")

        template = self.templates[procedure_code]

        # weighted diagnosis sampling
        diagnoses = template["common_diagnoses"]
        weights = np.array([d["weight"] for d in diagnoses])
        weights = weights / weights.sum()
        chosen_diagnoses = self._rng.choice(diagnoses, p=weights)

        age_low, age_high = template["age_range"]
        age = int(self._rng.integers(age_low, age_high + 1))

        all_treatments = template["typical_prior_treatments"]
        prior_treatment_size = int(self._rng.integers(1, min(4, len(all_treatments) + 1)))
        prior_treatments = [str(x) for x in self._rng.choice(all_treatments, size=prior_treatment_size, replace=False)]

        urgency = "urgent" if self._rng.random() < 0.1 else "routine"

        self._counter += 1
        return PARequest(
            request_id=f"req_{self._counter:04d}",
            procedure_code=procedure_code,
            diagnosis_codes=[chosen_diagnoses["code"]],
            patient_age=age,
            prior_treatments=prior_treatments,
            urgency=urgency,
            metadata={"diagnosis_name": chosen_diagnoses["name"]},
        )
