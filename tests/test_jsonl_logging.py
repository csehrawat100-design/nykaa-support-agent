import json
import pytest
from fastapi.testclient import TestClient
import api.main as main
from api.logging_config import configure_logging, read_jsonl

@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(main, "logger", configure_logging(path))
    return path

@pytest.fixture
def client(isolated_log):
    return TestClient(main.app)

def test_every_http_request_creates_one_jsonl_record(client, isolated_log):
    assert client.get("/health").status_code == 200
    records = read_jsonl(isolated_log)
    assert len(records) == 1
    r = records[0]
    assert r["method"] == "GET"
    assert r["path"] == "/health"
    assert r["status_code"] == 200
    assert r["trace_id"]
    assert isinstance(r["duration_ms"], float)
    assert r["timestamp"]

def test_raw_pii_never_reaches_log(client, isolated_log, monkeypatch):
    class Result:
        response = "Supported answer."
        source_type = "policy"
        grounded = True
    monkeypatch.setattr("agents.crew.run_crew", lambda q: Result())
    phone = "+91 98765-43210"
    response = client.post("/ask", json={"query":f"Call me at {phone}; card ending 4821.",
                                          "session_id":"log-test"})
    assert response.status_code == 200
    logged = read_jsonl(isolated_log)[0]["request_text"]
    assert phone not in logged
    assert "98765-43210" not in logged
    assert "4821" not in logged
    assert "[PHONE_MASKED]" in logged
    assert "[CARD_LAST4_MASKED]" in logged

def test_one_json_object_per_line(client, isolated_log):
    client.get("/health")
    client.get("/health")
    client.get("/health")
    lines = isolated_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        assert isinstance(json.loads(line), dict)

def test_websocket_request_is_logged_and_masked(client, isolated_log, monkeypatch):
    monkeypatch.setattr("agents.memory.ask_with_memory",
                        lambda sid, text: {"crew_result":"Memory response."})
    with client.websocket_connect("/ws/chat?session_id=ws-log") as ws:
        ws.send_json({"query":"My phone is +91 98765-43210. What is the return policy?"})
        assert ws.receive_json()["session_id"] == "ws-log"
    records = read_jsonl(isolated_log)
    assert len(records) == 1
    assert records[0]["method"] == "WEBSOCKET"
    assert "98765-43210" not in records[0]["request_text"]
    assert "[PHONE_MASKED]" in records[0]["request_text"]
