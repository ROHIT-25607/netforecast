# Demo

The Streamlit prototype that used to live here (`demo/app.py`) has been replaced by a
FastAPI service and a SOC-console dashboard.

```bash
pip install -r requirements.txt
python tools/fetch_vendor.py          # one-time, needs connectivity; after this everything is offline
uvicorn server.main:app --reload
```

Then open <http://127.0.0.1:8000>.

| What | Where |
|---|---|
| Dashboard | <http://127.0.0.1:8000> |
| Interactive API docs | <http://127.0.0.1:8000/docs> |
| Live-flow feeder | `python tools/feeder.py --csv data/synthetic_flows.csv` |

Why the change: the Streamlit build recomputed and deep-copied the whole capture on every
widget interaction, and its replay loop blocked the page for minutes. The FastAPI version
scores a capture once into a session, streams the replay over a WebSocket, and exposes a
`POST /api/ingest` endpoint so a live collector can drive the same pipeline.
