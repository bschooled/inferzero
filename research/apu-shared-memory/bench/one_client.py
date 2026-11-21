import json, time, urllib.request
LONG = "In software engineering, " * 120
body = {"model": "bench", "messages": [{"role": "user", "content": LONG + " Argue the value of unit tests in 6 bullet points."}],
        "max_tokens": 300, "temperature": 0.0}
t0 = time.time()
req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=900))
dt = time.time() - t0
u = r["usage"]
print(f"single client: {u['completion_tokens']} tok, total {dt:.1f}s, e2e {u['completion_tokens']/dt:.2f} tok/s incl prompt")
