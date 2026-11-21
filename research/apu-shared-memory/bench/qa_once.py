import json, urllib.request
body = {"model": "b", "messages": [{"role": "user", "content": "Q: Who wrote Pride and Prejudice, and in what year was it published? Answer in one sentence. Then name three of her other works."}], "max_tokens": 110, "temperature": 0.0}
req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=300))
msg = r["choices"][0]["message"]
print("KEYS:", list(msg.keys()))
for k, v in msg.items():
    if isinstance(v, str) and v:
        print(f"[{k}]: {v[:400]}")
