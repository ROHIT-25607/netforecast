"""Replay a flow CSV into a running NetForecast server over HTTP.

This is the stand-in for a real collector. It proves the live path end to end:
flows go in over `POST /api/ingest`, the server windows and scores them, and the
dashboard's Live ingest panel updates over its WebSocket. Swap this script for a
NetFlow/IPFIX collector, a Zeek `conn.log` tail or a Kafka consumer and nothing
downstream changes.

    python tools/feeder.py --csv data/synthetic_flows.csv --rate 400
    python tools/feeder.py --csv data/synthetic_flows.csv --mode count --window-size 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import pandas as pd


def post(url: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/synthetic_flows.csv", help="flow records to replay")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--source-id", default="default", help="logical sensor identifier")
    ap.add_argument("--rate", type=float, default=400.0, help="flows per second")
    ap.add_argument("--batch", type=int, default=50, help="flows per HTTP request")
    ap.add_argument("--window-size", type=int, default=None,
                    help="override the window size (defaults to the model's own setting)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N flows")
    ap.add_argument("--mode", choices=["time", "count"], default=None,
                    help="how a window closes; defaults to the server-side setting")
    ap.add_argument("--reset", action="store_true", help="clear the source before feeding")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, nrows=args.limit)
    records = df.to_dict(orient="records")
    total = len(records)
    print(f"Feeding {total:,} flows from {args.csv} -> {args.server} "
          f"(source={args.source_id}, {args.rate:g} flows/s)")

    if args.reset:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{args.server}/api/ingest/{args.source_id}", method="DELETE"), timeout=10)
            print("  reset existing source")
        except urllib.error.HTTPError:
            pass

    sent = windows = 0
    delay = args.batch / args.rate if args.rate > 0 else 0.0
    t0 = time.perf_counter()
    try:
        for i in range(0, total, args.batch):
            chunk = records[i:i + args.batch]
            payload = {"source_id": args.source_id, "flows": chunk}
            if args.mode:
                payload["mode"] = args.mode
            if args.window_size:
                payload["window_size"] = args.window_size
            try:
                res = post(f"{args.server}/api/ingest", payload)
            except urllib.error.HTTPError as e:
                print(f"\n  server rejected the batch ({e.code}): {e.read().decode()[:300]}")
                return 1
            except urllib.error.URLError as e:
                print(f"\n  cannot reach {args.server}: {e.reason}")
                return 1

            sent += res["accepted"]
            for w in res["windows"]:
                windows += 1
                print(f"  window {w['seq']:>4}  flows={w['n_flows']:>4}  "
                      f"risk={w['probability']:.3f}  stage={w['stage']:<20}"
                      + (f"truth={w['truth_stage']}" if w.get("truth_stage") else ""))
            print(f"\r  sent {sent:,}/{total:,} flows, {windows} windows scored", end="")
            if delay:
                time.sleep(delay)

        post(f"{args.server}/api/ingest/{args.source_id}/flush", {})
    except KeyboardInterrupt:
        print("\n  interrupted")

    el = time.perf_counter() - t0
    print(f"\nDone: {sent:,} flows, {windows} windows in {el:.1f}s "
          f"({sent/max(el,1e-9):,.0f} flows/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
