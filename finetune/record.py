"""Records training conversations: the sites.py solvers drive LocalForge's real agent loop and browser.

    python finetune/record.py --app PATH_TO_LOCALFORGE_SOURCE --out data/train.jsonl --episodes 600

PATH_TO_LOCALFORGE_SOURCE is the folder that holds the localforge package (LocalForge-src or LocalForge-Autopilot).
Chrome/Edge/Chromium must be installed (the app finds it, or set LOCALFORGE_BROWSER).

Every model call the app makes becomes one example: exactly the messages the app sent (its system prompt,
tool results, progress notes, nudges and history cuts) plus the reply the solver wrote. Only episodes whose
check passes (the basket, form, inbox... really ended up right and the answer is correct) are kept.
"""
import argparse
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sites  # noqa: E402

CURRENT = {"task": None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _go(self, form):
        u = urllib.parse.urlparse(self.path)
        t = CURRENT["task"]
        fn = sites.route(t, u.path) if t else None
        if fn is None:
            body, code, ctype = sites.html(sites.page("Not found", "<p>Page not found.</p>"), 404)
        else:
            body, code, ctype = fn(urllib.parse.parse_qs(u.query), form, t.state)
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._go({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self._go(urllib.parse.parse_qs(self.rfile.read(n).decode()))


class ScriptedEngine:
    """Stands in for llama.cpp: answers each call from the solver and remembers what the app sent."""
    loaded = True
    model_path = "scripted.gguf"
    backend = "cpu"

    def __init__(self):
        self.last_finish, self.last_reasoning, self.last_tps, self.last_prompt_tps = "stop", "", 50.0, 500.0
        self.gen, self.calls, self.broken = None, [], False

    def gpu_summary(self):
        return ""

    def unload(self):
        pass

    def stop(self):
        pass

    def chat(self, messages, max_tokens, temperature, on_piece, think=False, think_cap=0):
        obs = messages[-1]["content"]
        if "Check before you choose" in obs and self.calls and not self.broken:
            # The app asks once, before a "cheapest/most..." pick, whether every option was checked. A good model
            # confirms from its notes (or the script that compared them all) and repeats its step.
            r = sites.Reply(confirm_choice(self.calls), "Did I check every option? Yes: see my notes / the script result.")
            return self._out(r, think, messages, on_piece)
        try:
            r = next(self.gen) if not self.calls else self.gen.send(obs)
        except StopIteration:
            self.broken = True          # the app wanted more than the solver had: drop the episode
            r = sites.Reply("Done.")
        except Exception as e:
            self.broken = True
            r = sites.Reply(f"(solver error {e!r})")
        if isinstance(r, str):
            r = sites.Reply(r)
        return self._out(r, think, messages, on_piece)

    def _out(self, r, think, messages, on_piece):
        self.last_reasoning = r.thought if think else ""
        self.last_finish = "stop"
        self.calls.append({"messages": messages, "think": bool(think), "thought": r.thought if think else "", "reply": r.text})
        on_piece(r.text)
        return r.text


def confirm_choice(calls) -> str:
    prev = calls[-1]["reply"]
    notes = [n.strip() for c in calls for n in re.findall(r"<note>([\s\S]*?)</note>", c["reply"])]
    if notes:
        why = "Yes, I checked every option first. My notes: " + "; ".join(notes) + "."
    else:
        why = "Yes: my script compared every item on the page (all of them, not just the first), so the result covers every option."
    action = re.sub(r"<note>[\s\S]*?</note>\s*", "", prev).strip()
    return why + "\n" + action


def pc_like_prompt(agent_mod, rng):
    """Makes the system prompt look like it does on a Windows PC (that's where the model will run)."""
    orig = agent_mod.tools.system_prompt
    user = rng.choice(["User", "Owner", "Admin", "PC"])

    def sp(home, work, langs, overview, os_name):
        h = f"C:\\Users\\{user}"
        return orig(h, f"{h}\\AppData\\Local\\LocalForge\\work", langs,
                    "PC overview: Documents, Downloads, Pictures, Desktop and Projects folders indexed.", "Windows 11")
    agent_mod.tools.system_prompt = sp
    return orig


def run_episode(app, engine, task, base, limit=240):
    engine.gen = task.solve(base)
    engine.calls, engine.broken = [], False
    app.clear_chat()
    app.send(task.prompt.replace("{base}", base))
    t0 = time.time()
    time.sleep(0.5)
    while app.generating:
        if time.time() - t0 > limit:
            app.stop()
            engine.broken = True
        time.sleep(0.2)
    answers = [m for m in app.messages if m["role"] == "assistant" and m["text"].strip()]
    final = answers[-1]["text"] if answers else ""
    try:
        ok = bool(task.check(task.state, final)) and not engine.broken
    except Exception:
        ok = False
    return ok, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, help="folder that holds the localforge package")
    ap.add_argument("--out", default="finetune/data/train.jsonl")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--families", default="", help="comma list (default: all)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    os.environ["LOCALFORGE_HOME"] = tempfile.mkdtemp(prefix="lf-record-")
    sys.path.insert(0, str(Path(a.app).resolve()))
    from localforge import agent  # noqa: E402

    rng = random.Random(a.seed)
    pc_like_prompt(agent, rng)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    app = agent.App()
    for k, v in {"autoApprove": True, "browserAutoApprove": True, "nCtx": 16384, "temperature": 0.3}.items():
        app.settings.set(k, v)
    engine = ScriptedEngine()
    app.engine = engine
    app.browser.ensure()

    fams = [f for f in (a.families.split(",") if a.families else sites.FAMILIES) if f]
    weights = [sites.WEIGHTS.get(f, 1) for f in fams]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {}
    with out.open("a", encoding="utf-8") as fh:
        for i in range(a.episodes):
            fam = rng.choices(fams, weights)[0]
            # Half the episodes plan with (short) thinking first, like the app's "think on website tasks" setting.
            app.settings.set("thinkBrowser", rng.random() < 0.5)
            task = sites.FAMILIES[fam](random.Random(rng.random()))
            CURRENT["task"] = task
            ok, final = run_episode(app, engine, task, base)
            s = stats.setdefault(fam, [0, 0])
            s[0] += ok
            s[1] += 1
            if ok:
                for n, c in enumerate(engine.calls):
                    fh.write(json.dumps({"family": fam, "episode": i, "step": n, **c}, ensure_ascii=False) + "\n")
            if a.verbose or not ok:
                print(f"[{i}] {fam}: {'ok' if ok else 'FAIL'} {len(engine.calls)} calls | {final[:160]!r}", flush=True)
                if not ok and a.verbose:
                    for c in engine.calls:
                        print("   OBS:", c["messages"][-1]["content"][:int(os.environ.get("OBSLEN", 600))].replace("\n", " | "))
                        print("   REPLY:", c["reply"][:300].replace("\n", " | "))
    print("\nkept / tried per family:", {k: f"{v[0]}/{v[1]}" for k, v in stats.items()})
    app.browser.close()


if __name__ == "__main__":
    main()
