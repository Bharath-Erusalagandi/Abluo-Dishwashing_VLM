"""Tests for the FastAPI server endpoints."""

import base64
import io

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from src.api.server import app

API_KEY = "dev-key-change-me"
HEADERS = {"X-API-Key": API_KEY}


def _make_test_image_b64() -> str:
    """Create a tiny valid PNG and return its base64 encoding."""
    img = Image.new("RGB", (64, 48), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.anyio
async def test_health_no_auth_required(client: AsyncClient):
    """Health endpoint should not require API key."""
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_grasp_plan_requires_auth(client: AsyncClient):
    """Grasp plan should require API key."""
    resp = await client.post("/grasp_plan", json={"image_base64": "test"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_grasp_plan_basic(client: AsyncClient):
    resp = await client.post(
        "/grasp_plan",
        json={"image_base64": _make_test_image_b64(), "kitchen_profile": "default"},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "grasp_plan" in data
    assert "request_id" in data
    assert "latency_ms" in data


@pytest.mark.anyio
async def test_grasp_plan_batch(client: AsyncClient):
    img = _make_test_image_b64()
    resp = await client.post(
        "/grasp_plan/batch",
        json={
            "requests": [
                {"image_base64": img},
                {"image_base64": img},
            ]
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2


@pytest.mark.anyio
async def test_profiles_list(client: AsyncClient):
    resp = await client.get("/profiles", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1  # default profile


@pytest.mark.anyio
async def test_profile_get_default(client: AsyncClient):
    resp = await client.get("/profiles/default", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "default"


@pytest.mark.anyio
async def test_profile_not_found(client: AsyncClient):
    resp = await client.get("/profiles/nonexistent", headers=HEADERS)
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_finetune_create(client: AsyncClient):
    resp = await client.post(
        "/fine_tune",
        json={
            "profile_name": "test-kitchen",
            "training_data_folder": "data/training/test-v1",
            "sample_count": 100,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.anyio
async def test_finetune_status(client: AsyncClient):
    # Create a job first
    resp = await client.post(
        "/fine_tune",
        json={
            "profile_name": "test-kitchen-2",
            "training_data_folder": "data/training/test-v1",
            "sample_count": 50,
        },
        headers=HEADERS,
    )
    job_id = resp.json()["job_id"]

    # Check status
    resp = await client.get(f"/fine_tune/{job_id}/status", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id


@pytest.mark.anyio
async def test_evaluate(client: AsyncClient):
    resp = await client.post(
        "/evaluate",
        json={
            "profile_name": "default",
            "benchmark": "dishbench_v1",
            "categories": ["wet_ceramics", "transparent_glass"],
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "categories" in data
    assert data["profile_name"] == "default"


@pytest.mark.anyio
async def test_usage(client: AsyncClient):
    resp = await client.get("/usage", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
