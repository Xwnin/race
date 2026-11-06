"""Free video-generation Spaces (text/image to video) that are running on a GPU, with their API endpoints and inputs."""
import json, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
UA = {"User-Agent": "LocalForge-test/1.0"}
def get(u, t=40):
    try: return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t))
    except Exception: return None
TERMS = ["wan2.2", "wan2.1", "wan 2.2", "ltx-video", "ltx video", "ltxv", "hunyuanvideo", "cogvideox", "text-to-video", "image-to-video", "mochi", "framepack", "skyreels", "self-forcing", "fastwan", "video generation"]
ids = {}
for t in TERMS:
    for s in get(f"https://huggingface.co/api/spaces?search={urllib.parse.quote(t)}&sort=likes&limit=200") or []:
        ids[s["id"]] = s
print(len(ids), "candidate spaces")
def check(sid):
    info = get(f"https://huggingface.co/api/spaces/{sid}") or {}
    rt = info.get("runtime") or {}
    hw = (rt.get("hardware") or {}).get("current")
    if rt.get("stage") != "RUNNING" or not hw or hw.startswith("cpu") or info.get("sdk") != "gradio":
        return None
    host = info.get("host")
    api = get(host + "/gradio_api/info", 30) or {}
    eps = {k: [(p["parameter_name"], p.get("parameter_default")) for p in v.get("parameters", [])][:14] for k, v in (api.get("named_endpoints") or {}).items()}
    return info.get("likes", 0), sid, hw, host, eps
with ThreadPoolExecutor(12) as ex:
    rows = [r for r in ex.map(check, list(ids)) if r]
rows.sort(key=lambda r: -r[0])
for likes, sid, hw, host, eps in rows[:40]:
    print(f"\n{likes:>5} {sid} {hw} {host}")
    for k, v in list(eps.items())[:4]:
        print("   ", k, v)
