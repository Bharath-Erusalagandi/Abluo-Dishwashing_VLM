#!/usr/bin/env python3
"""Run DishBench evaluation — compare baseline vs fine-tuned model.

Usage:
    # Evaluate a fine-tuned adapter
    python scripts/evaluate.py --adapter models/dora/kitchen_v1/adapter

    # Compare baseline vs fine-tuned (the key validation)
    python scripts/evaluate.py --compare-baseline --adapter models/dora/kitchen_v1/adapter

    # Quick check (fewer scenarios)
    python scripts/evaluate.py --adapter models/dora/kitchen_v1/adapter --quick

    # Specific categories only
    python scripts/evaluate.py --adapter models/dora/kitchen_v1/adapter --categories wet_ceramics transparent_glass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    p = argparse.ArgumentParser(description="DishBench evaluation for DishSpace")
    p.add_argument("--adapter", type=str, default=None,
                   help="Path to saved DoRA/LoRA adapter directory")
    p.add_argument("--base-model", type=str, default="physical-intelligence/pi0-base")
    p.add_argument("--compare-baseline", action="store_true",
                   help="Run both baseline and fine-tuned, show comparison")
    p.add_argument("--categories", nargs="+", default=None,
                   help="Specific DishBench categories to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument("--quick", action="store_true",
                   help="Run fewer scenarios for quick check")

    args = p.parse_args()

    from src.evaluation.evaluator import (
        compare_baseline_vs_finetuned,
        evaluate_adapter,
        DISHBENCH_CATEGORIES,
    )

    # Override category counts for quick mode
    if args.quick:
        for cat in DISHBENCH_CATEGORIES.values():
            cat["count"] = min(cat["count"], 10)

    if args.compare_baseline:
        adapter = args.adapter or "models/dora/kitchen_v1/adapter"
        results = compare_baseline_vs_finetuned(
            adapter_path=adapter,
            base_model=args.base_model,
            categories=args.categories,
            seed=args.seed,
        )
    elif args.adapter:
        results = evaluate_adapter(
            adapter_path=args.adapter,
            base_model=args.base_model,
            categories=args.categories,
            seed=args.seed,
        )
        results = results.model_dump()
    else:
        # No adapter — evaluate baseline only
        from src.evaluation.evaluator import DishBenchEvaluator

        print("📊 Evaluating baseline model (no adapter)")
        evaluator = DishBenchEvaluator(seed=args.seed)
        try:
            evaluator.load_model(args.base_model)
        except Exception:
            pass
        eval_results = evaluator.run(categories=args.categories)
        results = eval_results.model_dump()
        print(f"\n   Overall: {eval_results.overall_success_rate:.1%}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, default=str))
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
