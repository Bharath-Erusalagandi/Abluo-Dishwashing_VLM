"""FastAPI server for DishSpace grasp planning API.

Endpoints:
    POST /grasp_plan       — Primary grasp planning from RGB-D input
    POST /grasp_plan/batch — Batch processing for multiple frames
    POST /fine_tune        — Trigger LoRA fine-tune on customer data
    GET  /fine_tune/{id}/status — Check fine-tune job progress
    GET  /profiles         — List available kitchen profiles
    GET  /profiles/{id}    — Get profile details
    POST /evaluate         — Run DishBench evaluation
    GET  /health           — Service health
    GET  /usage            — API usage summary
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from src.config import settings
from src.inference.grasp_planner import planner
from src.models.schemas import (
    CategoryResult,
    CoordinateFrame,
    EvalRequest,
    EvalResponse,
    FineTuneRequest,
    FineTuneResponse,
    FineTuneStatus,
    GraspOptions,
    GraspRequest,
    GraspResponse,
    KitchenProfile,
)
from src.pipeline.ros_bridge import grasps_to_ros_trajectory, trajectory_to_moveit_json
from src.utils.logging import get_logger
from src.evaluation.evaluator import DishBenchEvaluator, resolve_benchmark_categories

log = get_logger(__name__)

# ── App ──

app = FastAPI(
    title="DishSpace AI",
    description="Fine-tuning layer for kitchen robotics grasp planning",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ──

# In-memory stores (replaced by Supabase in production)
_profiles: dict[str, KitchenProfile] = {
    "default": KitchenProfile(
        name="default",
        description="General kitchen profile — works for most commercial rack setups",
        training_samples=5500,
        eval_success_rate=0.85,
    ),
}

_finetune_jobs: dict[str, FineTuneStatus] = {}
_usage: dict[str, int] = defaultdict(int)

# ── Auth ──

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    """Validate API key. Returns the key if valid.

    In MVP, accepts the dev key from .env.
    Production: validate against Supabase users table.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key. Set X-API-Key header.")
    if api_key != settings.api.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return api_key


# ── Rate Limiting (simple in-memory) ──

_rate_windows: dict[str, list[float]] = defaultdict(list)


async def check_rate_limit(request: Request) -> None:
    """Simple sliding-window rate limiter."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _rate_windows[client_ip]

    # Remove old entries (older than 60s)
    _rate_windows[client_ip] = [t for t in window if now - t < 60]

    if len(_rate_windows[client_ip]) >= settings.api.rate_limit_per_min:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {settings.api.rate_limit_per_min} requests/min.",
        )
    _rate_windows[client_ip].append(now)


# ── Startup ──


@app.on_event("startup")
async def startup():
    log.info("api_starting", version="0.1.0")
    planner.load_model()
    log.info("api_ready")


# ── Health ──


@app.get("/health")
async def health():
    """Service health check."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "model_loaded": planner._model_loaded,
        "profiles_available": len(_profiles),
        "uptime_s": time.time(),
    }


# ── Grasp Planning ──


@app.post("/grasp_plan", response_model=GraspResponse)
async def grasp_plan(
    request: GraspRequest,
    api_key: str = Depends(verify_api_key),
    _rl: None = Depends(check_rate_limit),
):
    """Primary grasp planning endpoint.

    Accepts an RGB-D image and returns ROS-compatible grasp trajectories.
    """
    _usage[api_key] += 1

    try:
        result = planner.plan(
            image_b64=request.image_base64,
            depth_b64=request.depth_base64,
            kitchen_profile=request.kitchen_profile,
            robot=request.robot,
            options=request.options,
        )
        log.info(
            "grasp_plan_served",
            request_id=result.request_id,
            grasps=len(result.grasp_plan),
            latency_ms=result.latency_ms,
        )
        return result
    except Exception as e:
        log.error("grasp_plan_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


class BatchGraspRequest(BaseModel):
    """Wrapper for batch grasp planning."""
    requests: list[GraspRequest]


class BatchGraspResponse(BaseModel):
    """Wrapper for batch grasp planning results."""
    results: list[GraspResponse]


@app.post("/grasp_plan/batch", response_model=BatchGraspResponse)
async def grasp_plan_batch(
    body: BatchGraspRequest,
    api_key: str = Depends(verify_api_key),
    _rl: None = Depends(check_rate_limit),
):
    """Batch grasp planning for multiple frames."""
    if len(body.requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 frames per batch request.")

    _usage[api_key] += len(body.requests)
    results = []
    for req in body.requests:
        try:
            result = planner.plan(
                image_b64=req.image_base64,
                depth_b64=req.depth_base64,
                kitchen_profile=req.kitchen_profile,
                robot=req.robot,
                options=req.options,
            )
            results.append(result)
        except Exception as e:
            log.error("batch_frame_error", error=str(e))
            results.append(GraspResponse(
                grasp_plan=[],
                latency_ms=0,
                model_version="error",
            ))

    return BatchGraspResponse(results=results)


@app.post("/grasp_plan/ros_trajectory")
async def grasp_plan_ros(
    request: GraspRequest,
    frame: CoordinateFrame = CoordinateFrame.CAMERA,
    api_key: str = Depends(verify_api_key),
    _rl: None = Depends(check_rate_limit),
):
    """Grasp planning with direct MoveIt-formatted trajectory output."""
    _usage[api_key] += 1
    result = planner.plan(
        image_b64=request.image_base64,
        depth_b64=request.depth_base64,
        kitchen_profile=request.kitchen_profile,
        robot=request.robot,
        options=request.options,
    )

    trajectory = grasps_to_ros_trajectory(result.grasp_plan, frame)
    moveit_json = trajectory_to_moveit_json(trajectory)

    return {
        "grasp_response": result.model_dump(),
        "moveit_trajectory": moveit_json,
    }


# ── Fine-Tuning ──


@app.post("/fine_tune", response_model=FineTuneResponse)
async def trigger_fine_tune(
    request: FineTuneRequest,
    api_key: str = Depends(verify_api_key),
):
    """Trigger a DoRA fine-tune job for a customer kitchen profile."""
    job_id = f"ft_{uuid.uuid4().hex[:8]}"

    # Estimate cost: ~$5/hr on A10, LoRA usually 3-4 hours
    estimated_hours = max(1, request.sample_count / 1500)
    estimated_cost = estimated_hours * 5.0

    status = FineTuneStatus(
        job_id=job_id,
        status="queued",
        total_epochs=request.adapter_config.epochs,
    )
    _finetune_jobs[job_id] = status

    log.info(
        "finetune_triggered",
        job_id=job_id,
        profile=request.profile_name,
        samples=request.sample_count,
        estimated_cost=f"${estimated_cost:.2f}",
    )

    # TODO: Dispatch to Modal.com GPU worker
    # In production, this triggers an async Modal function

    return FineTuneResponse(
        job_id=job_id,
        status="queued",
        estimated_duration_min=int(estimated_hours * 60),
        estimated_cost_usd=round(estimated_cost, 2),
    )


@app.get("/fine_tune/{job_id}/status", response_model=FineTuneStatus)
async def fine_tune_status(
    job_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Check fine-tune job progress."""
    if job_id not in _finetune_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return _finetune_jobs[job_id]


# ── Profiles ──


@app.get("/profiles", response_model=list[KitchenProfile])
async def list_profiles(api_key: str = Depends(verify_api_key)):
    """List all available kitchen profiles."""
    return list(_profiles.values())


@app.get("/profiles/{profile_name}", response_model=KitchenProfile)
async def get_profile(
    profile_name: str,
    api_key: str = Depends(verify_api_key),
):
    """Get a specific kitchen profile."""
    if profile_name not in _profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")
    return _profiles[profile_name]


# ── Evaluation ──


@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(
    request: EvalRequest,
    api_key: str = Depends(verify_api_key),
):
    """Run DishBench evaluation on a kitchen profile.

    Generates test scenarios across requested categories and evaluates
    the grasp model. If the profile has a DoRA adapter, it is loaded
    and compared against the base model.
    """
    # Look up profile adapter
    adapter_path = None
    if request.profile_name in _profiles:
        profile = _profiles[request.profile_name]
        if profile.adapter_path:
            adapter_path = profile.adapter_path

    evaluator = DishBenchEvaluator(
        adapter_path=adapter_path,
        seed=42,
    )

    # Only load a real model when a local adapter exists. The API should not
    # block on remote weight downloads during local development or tests.
    if adapter_path and Path(adapter_path).exists():
        try:
            base_model = _profiles.get(request.profile_name, _profiles["default"]).base_model
            evaluator.load_model(base_model)
        except Exception:
            pass

    categories = list(resolve_benchmark_categories(request.benchmark, request.categories).keys())

    results = evaluator.run(
        benchmark=request.benchmark,
        categories=categories,
        verbose=False,
    )

    # Override profile name from request
    results.profile_name = request.profile_name
    results.benchmark = request.benchmark

    log.info(
        "evaluation_complete",
        profile=request.profile_name,
        overall=f"{results.overall_success_rate:.1%}",
        categories=len(results.categories),
    )

    return results


# ── Usage ──


@app.get("/usage")
async def usage(api_key: str = Depends(verify_api_key)):
    """API usage summary for the authenticated key."""
    return {
        "api_key": api_key[:8] + "...",
        "total_requests": _usage.get(api_key, 0),
        "rate_limit_per_min": settings.api.rate_limit_per_min,
    }
