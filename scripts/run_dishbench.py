#!/usr/bin/env python3
"""Run DishBench evaluation against a deployed model.

Usage:
    python scripts/run_dishbench.py --model-id dishspace-lora-v1
"""

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DishBench evaluation")
    parser.add_argument("--profile-name", required=True, help="Kitchen profile to evaluate")
    parser.add_argument("--benchmark", default="dishbench_v1")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="dev-key-change-me")
    args = parser.parse_args()

    resp = httpx.post(
        f"{args.api_url}/evaluate",
        json={
            "profile_name": args.profile_name,
            "benchmark": args.benchmark,
        },
        headers={"X-API-Key": args.api_key},
        timeout=120.0,
    )
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2))

    # Summary
    print(f"\n{'='*40}")
    print(f"DishBench Results — {args.profile_name}")
    print(f"{'='*40}")
    print(f"  Overall success rate: {result.get('overall_success_rate', 0):.1%}")
    for cat in result.get("categories", []):
        print(f"  {cat['category']:30s} {cat['success_rate']:.1%}")


if __name__ == "__main__":
    main()
