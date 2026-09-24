"""Live tail dashboard for harness decision traces.

Reads files the harness already writes -- no harness changes, no new
instrumentation. Two views:

  - single episode: tails one DecisionRecord JSONL file live (harness/trace.py)
  - batch: reads a run_llm.py --out summary JSON (already has accuracy,
    oracle agreement, by-family breakdown -- this just displays it) plus a
    live grid of in-progress episodes while a batch has no summary yet.

Polls files for new lines/entries rather than using inotify /
ReadDirectoryChangesW so it behaves identically whether the watched directory
is native Windows, a WSL path, or a mounted drive.

    python dash/live.py --trace-dir ../data/traces/llm --runs-dir ../data --port 8787
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRACE_DIR = os.path.join(HERE, "..", "..", "data", "traces", "llm")
DEFAULT_RUNS_DIR = os.path.join(HERE, "..", "..", "data")

app = FastAPI()
TRACE_DIR = DEFAULT_TRACE_DIR  # reset from --trace-dir in main()
RUNS_DIR = DEFAULT_RUNS_DIR    # reset from --runs-dir in main()


def _episodes() -> list[dict]:
    if not os.path.isdir(TRACE_DIR):
        return []
    out = []
    for name in os.listdir(TRACE_DIR):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(TRACE_DIR, name)
        try:
            st = os.stat(path)
            last = None
            records = 0
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    records += 1
                    last = line
        except OSError:
            continue
        # STATUS (capstone): cheap last-line peek so the in-progress grid can show
        # what the episode is doing right now, not just "N decisions so far" -- the
        # same leading-cause/confidence readout the batch summary shows once done.
        last_call = last_conf = leading_cause = last_error = None
        if last:
            try:
                rec = json.loads(last)
                last_call = rec.get("call")
                last_conf = rec.get("confidence")
                belief = rec.get("belief") or {}
                if belief:
                    leading_cause = max(belief, key=belief.get)
                last_error = rec.get("error")
            except json.JSONDecodeError:
                pass
        out.append({"episode": name[:-6], "size": st.st_size, "mtime": st.st_mtime,
                    "records": records, "last_call": last_call,
                    "confidence": last_conf, "leading_cause": leading_cause,
                    "last_error": last_error})
    out.sort(key=lambda r: -r["mtime"])
    return out


@app.get("/api/episodes")
def list_episodes():
    return JSONResponse(_episodes())


def _runs() -> list[dict]:
    """--out summary files (run_llm.py) sitting directly in RUNS_DIR. Not a
    recursive scan: data/traces/*.jsonl and data/llm_cache/*.json live in
    subdirectories and are a different shape, so a top-level listing plus a
    shape check is enough to avoid picking them up."""
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for name in os.listdir(RUNS_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(RUNS_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
            with open(path, encoding="utf-8") as fh:
                head = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not (isinstance(head, dict) and "summary" in head and "episodes" in head):
            continue
        out.append({"run": name[:-5], "mtime": st.st_mtime,
                    "episodes": head["summary"].get("episodes", len(head["episodes"]))})
    out.sort(key=lambda r: -r["mtime"])
    return out


@app.get("/api/runs")
def list_runs():
    return JSONResponse(_runs())


@app.get("/api/run/{name}")
def get_run(name: str):
    path = os.path.join(RUNS_DIR, name + ".json")
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(path, encoding="utf-8") as fh:
        return JSONResponse(json.load(fh))


@app.get("/api/stream/{name}")
async def stream(name: str):
    path = os.path.join(TRACE_DIR, name + ".jsonl")

    async def gen():
        pos = 0
        while not os.path.isfile(path):
            yield ": waiting for episode to start\n\n"
            await asyncio.sleep(1.0)
        while True:
            try:
                with open(path, encoding="utf-8") as fh:
                    fh.seek(pos)
                    lines = fh.readlines()
                    pos = fh.tell()
            except OSError:
                lines = []
            for line in lines:
                line = line.strip()
                if line:
                    yield f"data: {line}\n\n"
            yield ": ping\n\n"
            await asyncio.sleep(0.75)

    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as fh:
        return fh.read()


def main():
    global TRACE_DIR, RUNS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    TRACE_DIR = os.path.abspath(a.trace_dir)
    RUNS_DIR = os.path.abspath(a.runs_dir)
    os.makedirs(TRACE_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    print(f"watching episodes: {TRACE_DIR}")
    print(f"watching runs:     {RUNS_DIR}")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
