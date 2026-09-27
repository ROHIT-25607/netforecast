"""API and WebSocket tests against the real app via TestClient."""
from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def session(client):
    """A fully-loaded synthetic session, shared across tests."""
    r = client.post("/api/session", json={"dataset": "synthetic"})
    if r.status_code == 409:
        pytest.skip("no trained checkpoint / sample capture available")
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]

    for _ in range(600):
        st = client.get(f"/api/session/{sid}/status").json()
        if st["done"]:
            break
        time.sleep(0.1)
    assert st["done"] and not st["error"], st
    return sid


# ------------------------------------------------------------------- meta

def test_health(client):
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and j["version"]


def test_datasets_declare_window_unit(client):
    ds = {d["id"]: d for d in client.get("/api/datasets").json()}
    assert "synthetic" in ds
    # CIC-IDS2017's "window_seconds" is really a flow count; the API must say so
    # rather than letting the UI print "200 seconds".
    assert ds["synthetic"]["window_unit"] == "seconds"
    if "cicids2017" in ds:
        assert ds["cicids2017"]["window_unit"] == "flows"


def test_mitre_reference(client):
    j = client.get("/api/mitre").json()
    assert "Exfiltration" in j["stages"]
    assert j["tactics"]["Exfiltration"]["tactic_id"] == "TA0010"


def test_dashboard_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "NetForecast" in r.text
    # the page must not reach out to a CDN -- this tool runs air-gapped
    assert "cdnjs" not in r.text and "jsdelivr" not in r.text


# --------------------------------------------------------------- sessions

def test_unknown_dataset_rejected(client):
    r = client.post("/api/session", json={"dataset": "nope"})
    assert r.status_code == 400 and "unknown dataset" in r.json()["detail"]


def test_unknown_session_404(client):
    assert client.get("/api/session/deadbeef/timeline").status_code == 404


def test_session_summary(client, session):
    j = client.get(f"/api/session/{session}").json()
    assert j["ready"] and j["n_rows"] > 0 and j["n_windows"] > 0
    assert j["window_unit"] == "seconds"


def test_timeline(client, session):
    j = client.get(f"/api/session/{session}/timeline").json()
    n = len(j["probability"])
    assert n > 0
    assert len(j["t"]) == len(j["stage"]) == len(j["window_index"]) == n
    assert all(0.0 <= p <= 1.0 for p in j["probability"])
    assert j["window_index"] == sorted(j["window_index"])


def test_forecast_respects_k(client, session):
    for k in (1, 3, 9):
        j = client.get(f"/api/session/{session}/forecast?k={k}").json()
        assert len(j["forecast"]) == k
        assert j["risk_band"] in ("LOW", "MEDIUM", "HIGH")
        assert 0.0 <= j["overall_infiltration_risk"] <= 1.0


def test_forecast_rejects_absurd_k(client, session):
    assert client.get(f"/api/session/{session}/forecast?k=999").status_code == 422


def test_explain(client, session):
    j = client.get(f"/api/session/{session}/explain").json()
    assert len(j["attention"]) == j["context_len"]
    assert sum(a["weight"] for a in j["attention"]) == pytest.approx(1.0, abs=1e-4)
    assert 0 < len(j["top_features"]) <= 8


def test_flows_and_state(client, session):
    f = client.get(f"/api/session/{session}/flows?limit=5").json()
    assert f["returned"] <= 5 and "src_ip" in f["columns"]
    s = client.get(f"/api/session/{session}/state").json()
    assert len(s["features"]) == 33
    assert s["features"][0]["name"] == "n_flows"


# ---------------------------------------------------------------- uploads

def test_upload_with_missing_columns_gives_clean_422(client):
    """A malformed upload must name the missing columns, not raise a traceback."""
    csv = io.BytesIO(b"timestamp,src_ip\n1,10.0.0.1\n2,10.0.0.2\n")
    r = client.post("/api/session/upload?dataset=synthetic",
                    files={"file": ("bad.csv", csv, "text/csv")})
    assert r.status_code == 202          # accepted, then fails during load
    sid = r.json()["session_id"]
    for _ in range(100):
        st = client.get(f"/api/session/{sid}/status").json()
        if st["done"]:
            break
        time.sleep(0.05)
    assert st["error"] and "missing required column" in st["error"]
    # and the analytics endpoints refuse rather than 500
    assert client.get(f"/api/session/{sid}/timeline").status_code == 409


def test_upload_rejects_non_csv(client):
    r = client.post("/api/session/upload?dataset=synthetic",
                    files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")})
    assert r.status_code == 415


# ----------------------------------------------------------------- ingest

def test_ingest_scores_windows_and_validates(client, synthetic_df):
    bad = client.post("/api/ingest", json={"source_id": "t-bad", "flows": [{"foo": 1}]})
    assert bad.status_code == 422 and "missing required column" in bad.json()["detail"]

    flows = synthetic_df.head(600).to_dict(orient="records")
    r = client.post("/api/ingest", json={"source_id": "t-ok", "flows": flows,
                                         "mode": "time", "window_size": 30})
    assert r.status_code == 200
    j = r.json()
    assert j["accepted"] == len(flows)
    assert j["windows"], "600 flows over a 30s window should close several windows"
    for w in j["windows"]:
        assert 0.0 <= w["probability"] <= 1.0
        assert w["stage"] in __import__("netforecast.mitre_mapping",
                                        fromlist=["STAGES"]).STAGES

    st = client.get("/api/ingest/t-ok").json()
    assert st["windows_completed"] == len(j["windows"])
    assert st["mode"] == "time"
    assert client.delete("/api/ingest/t-ok").status_code == 200


def test_ingest_unknown_source_404(client):
    assert client.get("/api/ingest/never-seen").status_code == 404


# -------------------------------------------------------------- websocket

def test_replay_websocket(client, session):
    with client.websocket_connect(f"/ws/replay/{session}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["n_frames"] > 0

        ws.send_json({"action": "seek", "frame": 5})
        f = ws.receive_json()
        assert f["type"] == "frame" and f["frame"] == 5
        assert f["band"] in ("LOW", "MEDIUM", "HIGH")
        assert 0.0 <= f["probability"] <= 1.0

        ws.send_json({"action": "speed", "value": 60})
        assert ws.receive_json() == {"type": "speed", "value": 60.0}

        ws.send_json({"action": "play"})
        frames = [ws.receive_json()["frame"] for _ in range(3)]
        assert frames == sorted(frames), "replay must advance monotonically"
        ws.send_json({"action": "pause"})


def test_replay_websocket_rejects_unknown_session(client):
    with client.websocket_connect("/ws/replay/deadbeef") as ws:
        assert ws.receive_json()["type"] == "error"
