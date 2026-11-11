"""Measure single-prediction latency against a RUNNING server (real HTTP, keep-alive).

    CHURN_API_KEYS=k uvicorn --factory app.main:get_app --port 8000 &
    python scripts/benchmark.py --url http://127.0.0.1:8000 --key k -n 1000
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "docs" / "example_request.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("-n", type=int, default=1000)
    a = ap.parse_args()
    times = []
    with httpx.Client(base_url=a.url, headers={"X-API-Key": a.key}) as c:
        for _ in range(50):
            c.post("/predict", json=SAMPLE)
        for _ in range(a.n):
            t = time.perf_counter()
            r = c.post("/predict", json=SAMPLE)
            times.append((time.perf_counter() - t) * 1000)
            r.raise_for_status()
    times.sort()
    q = lambda f: times[min(len(times) - 1, int(f * len(times)))]
    print(json.dumps({"requests": a.n, "median_ms": round(statistics.median(times), 2),
                      "p95_ms": round(q(.95), 2), "p99_ms": round(q(.99), 2),
                      "max_ms": round(times[-1], 2)}, indent=1))


if __name__ == "__main__":
    main()
