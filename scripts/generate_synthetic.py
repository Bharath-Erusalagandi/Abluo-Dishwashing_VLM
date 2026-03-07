#!/usr/bin/env python3
"""Generate synthetic grasp annotations and optionally upload to Supabase.

Usage:
    python scripts/generate_synthetic.py --count 5000 --seed 42
    python scripts/generate_synthetic.py --count 1000 --upload
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic grasp data")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic_annotations.json")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    from src.data.synthetic_generator import generate_batch

    samples = generate_batch(count=args.count, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([s.model_dump(mode="json") for s in samples], indent=2))
    print(f"✅ {len(samples)} samples → {out}")

    if args.upload:
        import asyncio
        from src.data.supabase_client import db

        count = asyncio.run(db.insert_annotations_batch(samples))
        print(f"✅ Uploaded {count} to Supabase")


if __name__ == "__main__":
    main()
