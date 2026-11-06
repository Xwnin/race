"""Reads the public model catalogs of cloud LLM aggregators (no key needed for the list) and prints every model whose
name or description says uncensored/abliterated (plus Dolphin/Venice), marking which ones are free."""
import json, re, urllib.request
UA = {"User-Agent": "LocalForge-test/1.0"}
PAT = re.compile(r"uncensor|abliterat|dolphin|venice|unfilter|unrestrict", re.I)
def get(url):
    try:
        return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
    except Exception as e:
        print("  !", url, e); return None
def size(t):
    n = [float(x) for x in re.findall(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z])", t, re.I)]
    return max(n) if n else 0
CATALOGS = {
    "openrouter": "https://openrouter.ai/api/v1/models",
    "featherless": "https://api.featherless.ai/v1/models",
    "nanogpt": "https://nano-gpt.com/api/v1/models",
    "chutes": "https://llm.chutes.ai/v1/models",
    "venice": "https://api.venice.ai/api/v1/models",
    "together": "https://api.together.xyz/v1/models",
    "deepinfra": "https://api.deepinfra.com/v1/openai/models",
    "arliai": "https://api.arliai.com/v1/models",
    "infermatic": "https://api.totalgpt.ai/v1/models",
    "pollinations": "https://text.pollinations.ai/models",
    "kluster": "https://api.kluster.ai/v1/models",
    "novita": "https://api.novita.ai/v3/openai/models",
    "mancer": "https://neuro.mancer.tech/oai/v1/models",
}
for name, url in CATALOGS.items():
    d = get(url)
    if d is None: continue
    items = d.get("data", d) if isinstance(d, dict) else d
    if isinstance(items, dict): items = list(items.values())
    print(f"== {name}: {len(items)} models")
    hits = []
    for m in items:
        if isinstance(m, str): m = {"id": m}
        mid = str(m.get("id") or m.get("name") or m.get("model_name") or "")
        desc = str(m.get("description") or "")
        txt = mid + " " + str(m.get("name", "")) + " " + desc + " " + json.dumps(m.get("model_spec", {}))[:2000]
        if not PAT.search(txt): continue
        p = m.get("pricing") or {}
        free = mid.endswith(":free") or (str(p.get("prompt")) in ("0", "0.0") and str(p.get("completion")) in ("0", "0.0"))
        hits.append((size(mid + " " + desc[:200]), free, mid, desc[:140].replace("\n", " ")))
    for b, free, mid, desc in sorted(hits, key=lambda h: -h[0]):
        print(f"  {'FREE' if free else 'paid'} {b:>5.0f}B {mid} | {desc}")
    if name == "openrouter":
        print("  all :free ->", [m["id"] for m in items if m["id"].endswith(":free")])
