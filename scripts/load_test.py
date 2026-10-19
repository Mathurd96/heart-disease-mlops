#!/usr/bin/env python
"""Send random but valid traffic to the API (for demos / dashboard screenshots).

    python scripts/load_test.py --url http://localhost:8000 --requests 300 --rps 10

Roughly 10% of requests are deliberately invalid so the dashboard also shows
4xx responses. Uses only the standard library.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient(rng: random.Random) -> dict:
    return {
        "age": rng.randint(29, 77),
        "sex": rng.choice([0, 1]),
        "cp": rng.choice([1, 2, 3, 4]),
        "trestbps": rng.randint(94, 200),
        "chol": rng.randint(126, 564),
        "fbs": rng.choice([0, 1]),
        "restecg": rng.choice([0, 1, 2]),
        "thalach": rng.randint(71, 202),
        "exang": rng.choice([0, 1]),
        "oldpeak": round(rng.uniform(0, 6), 1),
        "slope": rng.choice([1, 2, 3]),
        "ca": rng.choice([0, 1, 2, 3]),
        "thal": rng.choice([3, 6, 7]),
    }


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--rps", type=float, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts: dict[int, int] = {}
    for _ in range(args.requests):
        payload = random_patient(rng)
        if rng.random() < 0.1:
            payload["age"] = -1  # invalid -> HTTP 422
        status = post(f"{args.url}/predict", payload)
        counts[status] = counts.get(status, 0) + 1
        time.sleep(1.0 / args.rps)
    print("Status code counts:", counts)


if __name__ == "__main__":
    main()
