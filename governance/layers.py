"""Four-layer governance model used by the capstone.

The project requirement explicitly specifies Application and Runtime controls;
this module records all four governance layers without inventing extra controls
as if they were mandated by the guideline.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class GovernanceLayer:
    name: str
    control: str

FOUR_LAYERS = (
    GovernanceLayer("Application", "Least autonomy: only Lookup Agent can call check_order_status."),
    GovernanceLayer("Model", "Deterministic MOCK_LLM is used for graded execution; no external model is required."),
    GovernanceLayer("Data", "Policy answers use the authored KB and order lookups use the synthetic dataset."),
    GovernanceLayer("Runtime", "Every request is checked against explicit token and estimated-cost caps."),
)
