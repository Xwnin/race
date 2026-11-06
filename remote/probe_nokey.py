"""Keyless endpoints: what models they serve right now."""
import json, urllib.request
UA = {"User-Agent": "LocalForge-test/1.0"}
for u in ["https://text.pollinations.ai/models", "https://gen.pollinations.ai/text/models", "https://chat.dphn.ai/api/models",
          "https://chat.dphn.ai/", "https://api.dphn.ai/v1/models", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/models"]:
    try:
        b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40).read(3000).decode("utf-8", "replace")
        print("==", u, "\n", b[:1500])
    except Exception as e:
        print("==", u, e)
