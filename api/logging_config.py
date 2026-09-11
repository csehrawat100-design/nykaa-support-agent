import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from guardrails.pii import mask_pii_text

DEFAULT_LOG_PATH = Path("logs/requests.jsonl")
_lock = Lock()

class JsonLinesFormatter(logging.Formatter):
    def format(self, record):
        payload = getattr(record, "json_payload", {"message": record.getMessage()})
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

def configure_logging(log_path=DEFAULT_LOG_PATH):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nykaa_api")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    target = str(path.resolve())
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target
               for h in logger.handlers):
        h = logging.FileHandler(path, encoding="utf-8")
        h.setFormatter(JsonLinesFormatter())
        logger.addHandler(h)
    logger.nykaa_log_path = target
    return logger

def request_log_payload(trace_id, method, path, duration_ms, request_text, status_code):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "method": method,
        "path": path,
        "duration_ms": round(float(duration_ms), 3),
        "request_text": mask_pii_text(request_text),
        "status_code": int(status_code),
    }

def write_request_log(logger, payload):
    with _lock:
        logger.info("", extra={"json_payload": payload})

def read_jsonl(path=DEFAULT_LOG_PATH):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
