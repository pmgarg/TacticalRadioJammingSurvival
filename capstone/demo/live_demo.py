"""Live demo server: run ONE ns-3 episode and animate it in a browser as it happens.

WHAT THIS IS FOR
The dashboard replays traces after a run. For a demonstration you need the opposite:
the simulation, the agent's belief and the recovery all moving on screen at the same
time, so a viewer can see the link collapse, watch the agent buy evidence, and watch
delivery come back after it acts.

It drives sim/bridge_server.run() -- the SAME live ns-3 bridge the scored runs use --
with an on_tick observer that pushes each decision to the browser over SSE. Nothing
about the run is special-cased for the demo: the agent, the world and the verifier are
the ones that produce the numbers in the report.

    python3 demo/live_demo.py --port 8099                       # then open the page
    python3 demo/live_demo.py --port 8099 --agent teacher       # default is chosen in UI

Pick scenario + agent in the browser and press RUN.
"""
from __future__ import annotations
import argparse, json, os, queue, sys, threading, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer      # noqa: E402

NS3_BIN = os.environ.get("NS3_BIN", "")
SCEN_DIR = os.path.join(HERE, "..", "scenarios")
def _default_bundle() -> str:
    """Newest student that is actually present. v11 is the one calibrated on live-bridge
    rows (abstain 0.51); v10 shipped with abstain 0.0, meaning it never abstained."""
    for v in ("student_v12", "student_v11", "student_v10", "student_v9"):
        p_ = os.path.join(HERE, "..", "data", v, "student_bundle.json")
        if os.path.exists(p_):
            return p_
    return os.path.join(HERE, "..", "data", "student_v12", "student_bundle.json")


BUNDLE = os.environ.get("STUDENT_BUNDLE", _default_bundle())

# ONE QUEUE PER CLIENT, NOT ONE QUEUE.
#
# This was a shared queue.Queue consumed by every /api/stream connection, and a queue
# DISTRIBUTES: each event goes to exactly one consumer. With two browser tabs open -- or
# one stale connection that had not timed out yet -- the events were split between them.
# The visible symptom was a page that missed `start` but still received `phase` and the
# ticks, so the story panel never reset and narrated the previous episode on top of the
# current one. SSE is a broadcast, so fan out to every subscriber.
_subs: "set[queue.Queue[dict]]" = set()
_subs_lock = threading.Lock()
_running = threading.Event()


def _emit(kind: str, **kw) -> None:
    ev = {"kind": kind, **kw}
    with _subs_lock:
        targets = list(_subs)
    for q in targets:
        try:
            q.put_nowait(ev)
        except queue.Full:
            pass          # a wedged client must not stall the run


def scenarios() -> list[str]:
    try:
        return sorted(f for f in os.listdir(SCEN_DIR) if f.endswith((".yaml", ".yml")))
    except OSError:
        return []


def _ensure_dir(d: str) -> None:
    """makedirs, but survive a FILE sitting where the directory should be.

    `exist_ok=True` only forgives an existing DIRECTORY. An earlier version of this file
    passed a directory path to TraceStore, which opens its argument for append -- so it
    created a zero-byte FILE named `demo` where `demo/` belonged, and every later teacher
    run died with FileExistsError. Move the impostor aside rather than failing.
    """
    if os.path.exists(d) and not os.path.isdir(d):
        os.rename(d, d + ".not-a-directory.bak")
    os.makedirs(d, exist_ok=True)


def _make_teacher(scenario: str):
    """Build the LLM teacher exactly the way harness/run_llm.py does.

    Two things are easy to get wrong here and both are silent until you run it:
    the keyword is `trace=` (not `store=`), and TraceStore takes the path of ONE
    EPISODE FILE, not a directory -- it opens it for append.
    """
    from harness.loop import HarnessAgent
    from harness.registry import ToolRegistry
    from harness.trace import TraceStore
    from gateway.provider import make_provider
    ep = os.path.splitext(scenario)[0]
    path = os.path.join(HERE, "..", "data", "traces", "demo",
                        f"{ep}_{time.strftime('%H%M%S')}.jsonl")
    _ensure_dir(os.path.dirname(path))
    return HarnessAgent(provider=make_provider("claude"),
                        registry=ToolRegistry(),
                        trace=TraceStore(path),
                        episode=ep, family=ep.split("_")[0])


def run_episode(scenario: str, agent_kind: str) -> None:
    """Run one episode and stream it. Never raises into the HTTP thread.

    Phases are announced explicitly. With the LLM teacher the gap between "episode
    started" and "first decision" is ns-3 warm-up (5 s of simulated time) plus one real
    model call (~40 s of wall clock). Without a phase feed the page sat blank for a
    minute and looked broken -- which is exactly what it must not do on camera.
    """
    from sim.bridge_server import run as bridge_run
    path = os.path.join(SCEN_DIR, scenario)
    # The story used to read the jammer and the true cause from whatever the dropdown had
    # last fetched, which is not necessarily the episode now running. Send the truth with
    # the episode itself.
    truth = {}
    try:
        import yaml
        d = yaml.safe_load(open(path))
        j = (d.get("jammers") or [{}])[0]
        ev = (d.get("events") or [{}])[0]
        # Onset comes from `truth.onset_t`, which every family sets. Reading it from a
        # `jammer_on` event only worked for the jamming families: congestion_burst fires
        # `load_spike` and has no jammer at all, node_loss fires `node_down`, and fading
        # has no event -- so onset was None, the story never engaged, and the verdict
        # reported "agent said —" while the belief panel showed a confident diagnosis.
        onset = (d.get("truth") or {}).get("onset_t")
        if onset is None:
            onset = ev.get("t")
        truth = {"family": d.get("family"), "onset_t": onset,
                 "event_type": ev.get("type"),
                 "jammer_type": j.get("type"), "jammer_channels": j.get("channels") or [],
                 "has_jammer": bool(d.get("jammers")),
                 "n_channels": d.get("n_channels", 8)}
    except Exception:                                                # noqa: BLE001
        pass
    _emit("start", scenario=scenario, agent=agent_kind, ts=time.time(), truth=truth)
    _emit("phase", phase="building the teacher" if agent_kind == "teacher"
          else "loading the student")
    try:
        agent_obj = _make_teacher(scenario) if agent_kind == "teacher" else None
        if agent_obj is not None:
            _emit("phase", phase="checking the model is reachable")
            try:
                agent_obj.p.complete('Reply with ONLY: {"ok":true}')
            except Exception as e:                                    # noqa: BLE001
                raise RuntimeError(
                    "the teacher cannot reach the model, so it would abstain on every "
                    f"tick and produce a meaningless run. Fix this first: {str(e)[:220]}"
                ) from None
        name = "student" if agent_kind == "student" else "baseline"
        if not NS3_BIN or not os.path.exists(NS3_BIN):
            raise RuntimeError(
                f"ns-3 binary not found at {NS3_BIN!r}. Build it "
                "(./ns3 configure -- -DNS3_FAST_LINKERS=OFF && ./ns3 build jamming-sim) "
                "and restart this server with --ns3 pointing at it.")
        _emit("phase", phase="launching ns-3")
        # The state feed is 10 Hz; 2 Hz is plenty for the eye and keeps the SSE queue
        # from becoming the bottleneck on a slow browser.
        box = {"n": 0}

        def _state(e):
            box["n"] += 1
            if box["n"] % 5 == 0:
                _emit("state", **e)

        # Watch the provider's own error counter. A teacher whose CLI is not logged in
        # returns an error for EVERY call; the harness correctly abstains, so the run
        # looks normal -- 30 decisions, a flat 1/8 belief, no_op throughout -- and the
        # only clue is a uniform belief bar. That is far too subtle. Say it out loud.
        warned = {"n": -1}

        # A simulator that ABORTS mid-episode must not be scored as a result. ns-3 dies
        # with NS_FATAL and the bridge simply stops receiving state, so the run looked
        # "complete" at whatever second it happened to reach -- and the verdict was
        # computed on a truncated episode with no warning anywhere.
        _FATAL = ("NS_FATAL", "aborted. cond=", "terminating", "assert failed")

        def _log(ln):
            _emit("ns3log", line=ln[:400])
            if any(k in ln for k in _FATAL):
                _emit("simcrash", line=ln[:400])

        def _tick(e):
            _emit("tick", **e)
            if agent_obj is not None:
                prov = getattr(agent_obj, "p", None)
                _emit("meter", calls=getattr(prov, "calls", 0) or 0,
                      cache_hits=getattr(prov, "cache_hits", 0) or 0)
                n = getattr(prov, "errors", 0) or 0
                if n and n != warned["n"]:
                    warned["n"] = n
                    _emit("health", provider_errors=n,
                          message=f"{n} model call(s) FAILED — the teacher is not "
                                  f"reasoning, it is abstaining. Check `claude login`.")

        res = bridge_run(path, NS3_BIN, agent_name=name, verbose=False,
                         on_log=_log,
                         bundle=BUNDLE if agent_kind == "student" else None,
                         agent_obj=agent_obj,
                         on_state=_state,
                         on_tick=_tick)
        _emit("done", result={k: v for k, v in (res or {}).items()
                              if isinstance(v, (int, float, str, bool, type(None)))})
    except Exception as e:                                            # noqa: BLE001
        _emit("error", message=f"{type(e).__name__}: {e}")
    finally:
        _running.clear()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                                  # noqa: N802
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "demo.html"), "rb").read()
            return self._send(200, html, "text/html; charset=utf-8")
        if p == "/api/topology":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            sc = (q.get("scenario") or [""])[0]
            if sc not in scenarios():
                return self._send(400, b'{"error":"unknown scenario"}', "application/json")
            try:
                import yaml
                d = yaml.safe_load(open(os.path.join(SCEN_DIR, sc)))
            except Exception as e:                                    # noqa: BLE001
                return self._send(500, json.dumps({"error": str(e)}).encode(),
                                  "application/json")
            # Everything the topology view needs is already in the scenario recipe:
            # node positions and roles, the traffic flows, the jammer and when it fires.
            # No ns-3 change is required to draw the network.
            onset = None
            for ev in (d.get("events") or []):
                if ev.get("type") == "jammer_on":
                    onset = ev.get("t"); break
            return self._send(200, json.dumps({
                "name": d.get("name"), "family": d.get("family"),
                "nodes": [{"id": n["id"], "pos": n.get("pos") or [0, 0, 0],
                           "role": n.get("role"), "alive": n.get("alive", True)}
                          for n in (d.get("nodes") or [])],
                "flows": [{"src": f["src"], "dst": f["dst"],
                           "kbps": f.get("rate_kbps", 0)}
                          for f in (d.get("traffic") or [])],
                "jammers": [{"id": j["id"], "type": j.get("type"),
                             "pos": j.get("pos") or [0, 0, 0],
                             "channels": j.get("channels") or [],
                             "duty": j.get("duty", 1.0)}
                            for j in (d.get("jammers") or [])],
                "onset_t": onset, "channel": d.get("channel"),
                "n_channels": d.get("n_channels", 8),
                "duration_s": d.get("duration_s", 60),
            }).encode(), "application/json")
        if p == "/api/version":
            # So you can tell at a glance whether the RUNNING server is the current
            # code. Python does not reload modules: editing the file while the server
            # is up changes nothing until you restart it, and that has already cost
            # one confusing debugging session.
            import hashlib
            here = os.path.dirname(os.path.abspath(__file__))
            def h(fp):
                try:
                    return hashlib.sha256(open(fp, "rb").read()).hexdigest()[:10]
                except OSError:
                    return "?"
            return self._send(200, json.dumps({
                "features": ["phase", "state", "topology", "per-link"],
                "live_demo_py": h(os.path.join(here, "live_demo.py")),
                "demo_html": h(os.path.join(here, "demo.html")),
                "bridge_py": h(os.path.join(HERE, "sim", "bridge_server.py")),
                "ns3_bin": NS3_BIN,
                "ns3_ok": bool(NS3_BIN and os.path.exists(NS3_BIN)
                               and os.access(NS3_BIN, os.X_OK)),
                "student_bundle": os.path.normpath(BUNDLE),
                "student_ok": os.path.exists(BUNDLE),
            }).encode(), "application/json")
        if p == "/api/scenarios":
            return self._send(200, json.dumps({
                "scenarios": scenarios(), "ns3": bool(NS3_BIN and os.path.exists(NS3_BIN)),
                "bundle": os.path.exists(BUNDLE),
                "bundle_name": os.path.basename(os.path.dirname(BUNDLE))}).encode(),
                "application/json")
        if p == "/api/run":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            if _running.is_set():
                return self._send(409, b'{"error":"a run is already in progress"}',
                                  "application/json")
            sc = (q.get("scenario") or [""])[0]
            ag = (q.get("agent") or ["student"])[0]
            if sc not in scenarios():
                return self._send(400, b'{"error":"unknown scenario"}', "application/json")
            _running.set()
            threading.Thread(target=run_episode, args=(sc, ag), daemon=True).start()
            return self._send(200, b'{"ok":true}', "application/json")
        if p == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q: "queue.Queue[dict]" = queue.Queue(maxsize=4096)
            with _subs_lock:
                _subs.add(q)
            try:
                while True:
                    try:
                        ev = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    except queue.Empty:
                        # A REAL event, not an SSE comment. The browser's stall watchdog
                        # listens on onmessage, which comment lines do not trigger -- so a
                        # single 30-90 s teacher call looked exactly like a dead stream and
                        # the page declared "stream stalled" mid-run.
                        self.wfile.write(
                            b'data: {"kind":"ping"}\n\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                with _subs_lock:
                    _subs.discard(q)
        return self._send(404, b"not found", "text/plain")


def main() -> None:
    global NS3_BIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ns3", default=NS3_BIN)
    a = ap.parse_args()
    NS3_BIN = a.ns3
    if not NS3_BIN or not os.path.exists(NS3_BIN):
        print(f"WARNING: ns-3 binary not found at {NS3_BIN!r} -- set --ns3 or $NS3_BIN")
    print(f"\n  Live demo on  http://127.0.0.1:{a.port}\n")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
