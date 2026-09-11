"""Prompt-injection detection for Task 10."""
import re
_PATTERNS = {
    "ignore_previous_instructions": re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions\b", re.I),
    "ignore_system_prompt": re.compile(r"\bignore\s+(?:the\s+)?system\s+(?:prompt|message)\b", re.I),
    "reveal_system_prompt": re.compile(r"\b(?:show|reveal|print|give)\s+(?:me\s+)?(?:the\s+)?system\s+prompt\b", re.I),
    "developer_override": re.compile(r"\b(?:override|disregard)\s+(?:the\s+)?(?:developer|system)\s+(?:instructions?|rules?)\b", re.I),
}
def detect_prompt_injection(text: str) -> dict:
    matched = [name for name, pat in _PATTERNS.items() if pat.search(text)]
    return {"detected": bool(matched), "matched_patterns": matched}
def enforce_prompt_injection_guardrail(text: str) -> str:
    result = detect_prompt_injection(text)
    if result["detected"]:
        raise ValueError("Prompt injection detected: " + ", ".join(result["matched_patterns"]))
    return text
