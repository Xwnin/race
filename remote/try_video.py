"""Makes one vertical (9:16) TikTok-style clip on a free video Space, anonymously, and reports time + output.
Usage: python remote/try_video.py OWNER/SPACE"""
import json, sys, time, urllib.request
sid = sys.argv[1]
UA = {"User-Agent": "LocalForge/1.0", "Content-Type": "application/json"}
def req(url, body=None, t=600):
    return urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, UA), timeout=t)
info = json.load(req(f"https://huggingface.co/api/spaces/{sid}"))
host = info["host"]
eps = json.load(req(host + "/gradio_api/info"))["named_endpoints"]
for k, v in eps.items():
    print(k, [(p["parameter_name"], (p.get("python_type") or {}).get("type"), p.get("parameter_default")) for p in v["parameters"]])
PROMPT = ("Vertical TikTok shot: a golden retriever puppy in sunglasses skateboarding down a sunny boardwalk, "
          "camera tracking alongside, vibrant colors, dynamic motion, sharp focus")
name = next((n for n in ["/text_to_video", "/generate_text_to_video", "/t2v_generation_async", "/generate", "/t2v"] if n in eps), None)
if not name:
    print("no text-to-video endpoint"); sys.exit()
data = []
for p in eps[name]["parameters"]:
    n, d = p["parameter_name"].lower(), p.get("parameter_default")
    if n in ("prompt",): data.append(PROMPT)
    elif "negative" in n: data.append(d if d is not None else "blurry, low quality, watermark, text")
    elif n in ("height", "height_ui"): data.append(832)
    elif n in ("width", "width_ui"): data.append(480)
    elif "duration" in n: data.append(5)
    elif "mode" in n and isinstance(d, str): data.append("text-to-video")
    elif n == "resolution_preset": data.append("9:16 - 480x864 (480p)")
    elif n == "size": data.append("720*1280")
    elif "watermark" in n: data.append(False)
    elif n == "num_frames": data.append(121)
    else: data.append(d)
print("calling", name, data)
t = time.time()
ev = json.load(req(f"{host}/gradio_api/call{name}", {"data": data}))["event_id"]
event = ""
for raw in req(f"{host}/gradio_api/call{name}/{ev}", t=900):
    line = raw.decode("utf-8", "replace").strip()
    if line.startswith("event:"): event = line[6:].strip()
    elif line.startswith("data:") and event in ("complete", "error"):
        print(f"{event} after {time.time()-t:.0f}s:", line[5:].strip()[:800])
        import re, os
        for u in re.findall(r'"url": "([^"]+\.mp4)"', line):
            os.makedirs("clips", exist_ok=True)
            fn = "clips/" + sid.replace("/", "_") + ".mp4"
            open(fn, "wb").write(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=120).read())
            print("saved", fn, os.path.getsize(fn), "bytes")
        break
    elif line.startswith("data:") and event == "generating":
        print("  generating:", line[5:].strip()[:200])
