"""System risk classification required by Task 15."""
from dataclasses import dataclass

SYSTEM_RISK_LEVEL = "Medium"

@dataclass(frozen=True)
class RiskAssessment:
    level: str
    rationale: str
    mitigations: tuple[str, ...]


_ASSESSMENT = RiskAssessment(
    level=SYSTEM_RISK_LEVEL,
    rationale=(
        "The Nykaa system is a customer-support agent, so it fits the project's "
        "Medium-risk category. It answers retail policy questions and looks up "
        "synthetic order records, rather than making high-risk medical, hiring, "
        "or financial decisions. Its outputs can still affect customer actions, "
        "so responses must remain grounded in the authored policy context and "
        "order data, while autonomy and runtime resources are explicitly bounded."
    ),
    mitigations=(
        "least-autonomy tool authorization",
        "grounded generation and review stage",
        "structured outputs and guardrails",
        "per-request token/cost budget",
    ),
)


def get_risk_assessment() -> RiskAssessment:
    return _ASSESSMENT
