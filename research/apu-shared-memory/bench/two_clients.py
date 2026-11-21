import json, sys, time, urllib.request, threading

LONG = "In software engineering, " * 120
res = []

def call(tag):
    body = {"model": "bench", "messages": [{"role": "user", "content": LONG + f" Argue position {tag}: enumerate the value of unit tests in 6 bullet points."}],
            "max_tokens": 160, "temperature": 0.0}
    t0 = time.time()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    u = r["usage"]
    res.append((tag, u["completion_tokens"], dt))
    print(f"{tag}: {u['completion_tokens']} tok in {dt:.1f}s -> {u['completion_tokens']/dt:.2f} tok/s end-to-end")

th = [threading.Thread(target=call, args=(f"client{i}",)) for i in (1, 2)]
[t.start() for t in th]
[t.join() for t in th]
tot = sum(c for _, c, _ in res)
wall = max(d for _, _, d in res)
print(f"AGGREGATE: {tot} tokens in {wall:.1f}s wall -> {tot/wall:.2f} tok/s combined")
