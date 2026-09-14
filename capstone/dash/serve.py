#!/usr/bin/env python3
"""
Run dashboard — a viewer over the records the pipeline already writes.

    cd capstone
    python3 -m dash.serve                      # http://127.0.0.1:8765
    python3 -m dash.serve --port 9000 --runs ../data/runs

Standard library only: no framework, no CDN, no build step, works offline.

It adds NO new logging. Everything it shows comes from `data/runs/*.json`, which
run_episode.py writes BEFORE any predicate executes (design §10.5, A14), and every
score is computed by `verify/verifier.py` — the same scorer the CLI uses, imported,
not reimplemented. If the dashboard and the CLI ever disagree, that is a bug.

Endpoints
    GET /                 the page
    GET /api/runs         one summary row per record
    GET /api/run/<id>     the full record + its EpisodeScore
    GET /api/summary      per-family aggregate + gates, over all records
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import yaml  # noqa: E402
from verify.verifier import score_episode, aggregate, gates  # noqa: E402

CAPSTONE = os.path.dirname(HERE)
CONTRACT = json.load(open(os.path.join(CAPSTONE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(CAPSTONE, "contract", "cost_matrix.yaml")))

RUNS_DIR = os.path.join(CAPSTONE, "..", "data", "runs")


def load_records(runs_dir: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        try:
            r = json.load(open(p))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {os.path.basename(p)}: {e}", file=sys.stderr)
            continue
        if "episode_log" not in r or "ground_truth" not in r:
            continue
        r["_path"] = p
        out[r.get("run_id") or os.path.basename(p)[:-5]] = r
    return out


def score(rec: dict) -> dict:
    """The same call run_episode.py makes. One scorer, not two."""
    return asdict(score_episode(rec["episode_log"], rec["ground_truth"],
                                COSTCFG, CONTRACT, rec.get("family", "?")))


def summarise(recs: dict[str, dict]) -> dict:
    scored = [score_episode(r["episode_log"], r["ground_truth"], COSTCFG,
                            CONTRACT, r.get("family", "?")) for r in recs.values()]
    if not scored:
        return {"n": 0, "families": {}, "overall": {}, "gates": {}}
    by_family: dict[str, list] = {}
    for s in scored:
        by_family.setdefault(s.family, []).append(s)
    overall = aggregate(scored)
    return {
        "n": len(scored),
        "overall": overall,
        "gates": gates(overall, {"fp_max": 0.05}),
        "families": {f: aggregate(v) for f, v in sorted(by_family.items())},
        "contract_version": CONTRACT["contract_version"],
    }


class Handler(BaseHTTPRequestHandler):
    runs_dir = RUNS_DIR

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)

        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "index.html"), "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(500, b"index.html missing next to serve.py", "text/plain")
            return

        if path == "/api/runs":
            recs = load_records(self.runs_dir)
            rows = []
            for rid, r in recs.items():
                s = score(r)
                rows.append({
                    "run_id": rid,
                    "family": r.get("family"),
                    "agent": r.get("agent"),
                    "seed": r.get("config", {}).get("seed"),
                    "true_cause": s["true_cause"],
                    "declared_cause": s["declared_cause"],
                    "classification_ok": s["classification_ok"],
                    "detection_latency_s": s["detection_latency_s"],
                    "censored": s["censored"],
                    "false_positive_acted": s["false_positive_acted"],
                    "survived": s["survived"],
                    "expected_cost": s["expected_cost"],
                    "actions_consumed": s["actions_consumed"],
                    "refusal_ok": s["refusal_ok"],
                })
            self._json({"runs_dir": os.path.abspath(self.runs_dir), "rows": rows})
            return

        if path.startswith("/api/run/"):
            rid = path[len("/api/run/"):]
            rec = load_records(self.runs_dir).get(rid)
            if rec is None:
                self._json({"error": f"no record {rid!r}"}, 404)
                return
            self._json({"record": rec, "score": score(rec)})
            return

        if path == "/api/summary":
            self._json(summarise(load_records(self.runs_dir)))
            return

        self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):  # quieter
        if "/api/" not in (args[0] if args else ""):
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs", default=RUNS_DIR, help="directory of run records")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    Handler.runs_dir = a.runs
    n = len(load_records(a.runs))
    print(f"records : {n} in {os.path.abspath(a.runs)}")
    if n == 0:
        print("  (none yet — run:  python3 run_episode.py --corpus --out ../data/runs)")
    print(f"serving : http://{a.host}:{a.port}   (Ctrl-C to stop)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
