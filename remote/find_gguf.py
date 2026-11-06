"""Finds GGUF repos for the uncensored models we'd put on a Modal GPU, with file sizes."""
import json, urllib.request, urllib.parse
UA = {"User-Agent": "LocalForge-test/1.0"}
def get(u): return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60))
for q in ["Qwen3.6-35B-A3B-Uncensored-HauhauCS", "gemma-4-31B-it-abliterated GGUF", "Huihui-Qwen3.6-35B-A3B-abliterated GGUF"]:
    for m in get("https://huggingface.co/api/models?search=" + urllib.parse.quote(q) + "&sort=downloads&limit=8"):
        if "gguf" not in m["id"].lower() and "GGUF" not in str(m.get("tags", "")): continue
        try:
            files = get(f"https://huggingface.co/api/models/{m['id']}/tree/main")
            g = [(f["path"], round(f.get("size", 0) / 1e9, 1)) for f in files if f["path"].endswith(".gguf")]
        except Exception as e:
            g = str(e)
        print(m["id"], m.get("downloads"), g[:12] if isinstance(g, list) else g)
