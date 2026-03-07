"""DishSpace CLI — entry point for all workflows.

Usage:  dishspace <command> [options]

Commands:
    serve          Start the FastAPI server
    generate       Generate synthetic grasp data
    scrape         Scrape YouTube dishwashing videos
    train          Fine-tune π₀ with DoRA (local or Modal)
    finetune       Launch a fine-tuning job on Modal
    evaluate       Run DishBench evaluation
    plan           Plan grasps from a local image
    deploy         Deploy to real robot for dishwashing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic grasp data."""
    from src.data.synthetic_generator import generate_batch

    samples = generate_batch(count=args.count, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([s.model_dump(mode="json") for s in samples], indent=2))
    print(f"✅ Generated {len(samples)} samples → {out}")

    if args.upload:
        import asyncio
        from src.data.supabase_client import db

        count = asyncio.run(db.insert_annotations_batch(samples))
        print(f"✅ Uploaded {count} annotations to Supabase")


def cmd_scrape(args: argparse.Namespace) -> None:
    """Scrape YouTube dishwashing videos."""
    from src.data.video_scraper import build_video_manifest

    manifest = build_video_manifest(max_per_query=args.max_videos)
    print(f"✅ Built manifest with {len(manifest)} unique videos")


def cmd_finetune(args: argparse.Namespace) -> None:
    """Launch a fine-tuning job on Modal."""
    from src.models.schemas import FineTuneStatus

    job = FineTuneStatus(
        status="queued",
        total_epochs=args.epochs,
    )
    print(f"✅ Fine-tune job created: {job.job_id}")
    print(f"   Base model: {args.base_model}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {args.epochs}, LoRA rank: {args.lora_rank}")

    if args.run:
        try:
            from src.inference.modal_worker import FineTuneWorker

            worker = FineTuneWorker()
            result = worker.run_finetune.remote(
                profile_name=args.dataset,
                base_model=args.base_model,
                training_data=[],  # placeholder — load from Supabase in production
                lora_config={
                    "rank": args.lora_rank,
                    "epochs": args.epochs,
                    "adapter_type": "dora",
                },
            )
            print(f"✅ Fine-tune complete: {json.dumps(result, indent=2)}")
        except ImportError:
            print("⚠️ Modal not installed. Install with: pip install modal")
            print("   Or use 'dishspace train' for local training.")


def cmd_train(args: argparse.Namespace) -> None:
    """Run local fine-tuning (without Modal)."""
    import subprocess

    cmd = [
        sys.executable, "scripts/train.py",
        "--base-model", args.base_model,
        "--samples", str(args.samples),
        "--epochs", str(args.epochs),
        "--rank", str(args.lora_rank),
        "--adapter-type", args.adapter_type,
        "--output-dir", args.output_dir,
    ]

    if args.annotations:
        cmd.extend(["--annotations", args.annotations])
    if args.quick:
        cmd.append("--quick")
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run DishBench evaluation."""
    if args.compare_baseline:
        from src.evaluation.evaluator import compare_baseline_vs_finetuned

        results = compare_baseline_vs_finetuned(
            adapter_path=args.adapter or "models/dora/kitchen_v1/adapter",
            categories=args.categories,
            seed=args.seed,
        )
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2, default=str))
            print(f"💾 Results saved to {args.output}")
    elif args.local:
        from src.evaluation.evaluator import DishBenchEvaluator, evaluate_adapter

        if args.adapter:
            results = evaluate_adapter(
                adapter_path=args.adapter,
                categories=args.categories,
                seed=args.seed,
            )
        else:
            evaluator = DishBenchEvaluator(seed=args.seed)
            results = evaluator.run(categories=args.categories)
            print(f"\n   Overall: {results.overall_success_rate:.1%}")
    else:
        import httpx

        resp = httpx.post(
            f"http://{args.host}:{args.port}/evaluate",
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


def cmd_plan(args: argparse.Namespace) -> None:
    """Plan grasps for a local image."""
    import base64

    img_bytes = Path(args.image).read_bytes()
    image_b64 = base64.b64encode(img_bytes).decode("utf-8")

    depth_b64 = None
    if args.depth:
        depth_b64 = base64.b64encode(Path(args.depth).read_bytes()).decode("utf-8")

    import httpx

    resp = httpx.post(
        f"http://{args.host}:{args.port}/grasp_plan",
        json={
            "image_base64": image_b64,
            "depth_base64": depth_b64,
            "kitchen_profile": args.profile,
            "robot": args.robot,
        },
        headers={"X-API-Key": args.api_key},
        timeout=30.0,
    )
    resp.raise_for_status()
    result = resp.json()

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"✅ Saved → {args.output}")
    else:
        print(json.dumps(result, indent=2))


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy to real robot for dishwashing."""
    import subprocess

    cmd = [
        sys.executable, "scripts/deploy_robot.py",
        "--ros-host", args.ros_host,
        "--ros-port", str(args.ros_port),
        "--max-cycles", str(args.max_cycles),
        "--api-key", args.api_key,
    ]

    if args.adapter:
        cmd.extend(["--adapter", args.adapter])
    if args.api_url:
        cmd.extend(["--api-url", args.api_url])
    if args.single_grasp:
        cmd.append("--single-grasp")
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"🤖 Deploying dishwashing robot...")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dishspace",
        description="DishSpace AI — kitchen robotics fine-tuning toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic data")
    p_gen.add_argument("--count", type=int, default=5000)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--output", default="data/synthetic_annotations.json")
    p_gen.add_argument("--upload", action="store_true", help="Upload to Supabase")

    # scrape
    p_scrape = sub.add_parser("scrape", help="Scrape YouTube videos")
    p_scrape.add_argument("--max-videos", type=int, default=20, help="Max results per query")

    # finetune (Modal)
    p_ft = sub.add_parser("finetune", help="Launch fine-tuning job on Modal")
    p_ft.add_argument("--base-model", default="physical-intelligence/pi0-base")
    p_ft.add_argument("--dataset", default="dishspace-v1")
    p_ft.add_argument("--epochs", type=int, default=3)
    p_ft.add_argument("--lora-rank", type=int, default=16)
    p_ft.add_argument("--run", action="store_true", help="Dispatch to Modal")

    # train (local)
    p_train = sub.add_parser("train", help="Fine-tune locally (no Modal)")
    p_train.add_argument("--base-model", default="physical-intelligence/pi0-base")
    p_train.add_argument("--annotations", default=None, help="Path to annotations JSON")
    p_train.add_argument("--samples", type=int, default=5000)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--adapter-type", default="dora", choices=["dora", "lora"])
    p_train.add_argument("--output-dir", default="models/dora/kitchen_v1")
    p_train.add_argument("--quick", action="store_true", help="Quick validation run")
    p_train.add_argument("--dry-run", action="store_true", help="Data pipeline only, no training")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run DishBench evaluation")
    p_eval.add_argument("--profile-name", default="default", help="Kitchen profile to evaluate")
    p_eval.add_argument("--benchmark", default="dishbench_v1")
    p_eval.add_argument("--adapter", default=None, help="Path to adapter for local eval")
    p_eval.add_argument("--compare-baseline", action="store_true", help="Compare baseline vs fine-tuned")
    p_eval.add_argument("--local", action="store_true", help="Run locally (not via API)")
    p_eval.add_argument("--categories", nargs="+", default=None)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--output", default=None, help="Save results to file")
    p_eval.add_argument("--host", default="localhost")
    p_eval.add_argument("--port", type=int, default=8000)
    p_eval.add_argument("--api-key", default="dev-key-change-me")

    # plan
    p_plan = sub.add_parser("plan", help="Plan grasps from local image")
    p_plan.add_argument("image", help="Path to RGB image")
    p_plan.add_argument("--depth", help="Path to depth image (16-bit PNG)")
    p_plan.add_argument("--profile", default="default")
    p_plan.add_argument("--robot", default="UR5_realsense")
    p_plan.add_argument("--output", help="Save results to file")
    p_plan.add_argument("--host", default="localhost")
    p_plan.add_argument("--port", type=int, default=8000)
    p_plan.add_argument("--api-key", default="dev-key-change-me")

    # deploy
    p_deploy = sub.add_parser("deploy", help="Deploy to real robot for dishwashing")
    p_deploy.add_argument("--ros-host", default="localhost")
    p_deploy.add_argument("--ros-port", type=int, default=9090)
    p_deploy.add_argument("--adapter", default=None, help="Path to DoRA adapter")
    p_deploy.add_argument("--api-url", default=None, help="Use API inference")
    p_deploy.add_argument("--api-key", default="dev-key-change-me")
    p_deploy.add_argument("--max-cycles", type=int, default=100)
    p_deploy.add_argument("--single-grasp", action="store_true")
    p_deploy.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    commands = {
        "serve": cmd_serve,
        "generate": cmd_generate,
        "scrape": cmd_scrape,
        "train": cmd_train,
        "finetune": cmd_finetune,
        "evaluate": cmd_evaluate,
        "plan": cmd_plan,
        "deploy": cmd_deploy,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
