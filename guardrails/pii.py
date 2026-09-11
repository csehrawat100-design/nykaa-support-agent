"""Fixed-format PII masking required by Task 10."""
import re
_PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?(?:[6-9]\d{4}[\s-]?\d{5})(?!\d)")
_CARD = re.compile(r"(?i)\b(?:payment\s+)?card\s+(?:ending|ends\s+in|last\s*4)\s*[:\-]?\s*(\d{4})\b")

def mask_pii(text: str) -> dict:
    masked = _PHONE.sub("[PHONE_MASKED]", text)
    phone_masked = masked != text
    masked2 = _CARD.sub("card ending [CARD_LAST4_MASKED]", masked)
    card_last4_masked = masked2 != masked
    return {"text": masked2, "phone_masked": phone_masked, "card_last4_masked": card_last4_masked}

def mask_pii_text(text: str) -> str:
    return mask_pii(text)["text"]