"""Supabase client for DishSpace data pipeline.

Handles:
- Storing / retrieving grasp annotations
- File uploads (RGB, depth, point clouds)
- Kitchen profile management
- Fine-tune job tracking
"""

from __future__ import annotations

from typing import Optional

from src.config import settings
from src.models.schemas import (
    FineTuneStatus,
    GraspAnnotation,
    KitchenProfile,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


class SupabaseClient:
    """Thin wrapper around the Supabase Python client."""

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not settings.supabase.is_configured:
                raise RuntimeError(
                    "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in .env"
                )
            from supabase import create_client

            self._client = create_client(settings.supabase.url, settings.supabase.key)
            log.info("supabase_connected", url=settings.supabase.url)
        return self._client

    # ── Grasp Annotations ──

    async def insert_annotation(self, annotation: GraspAnnotation) -> dict:
        """Insert a single grasp annotation."""
        data = annotation.model_dump()
        result = self.client.table("grasp_annotations").insert(data).execute()
        log.info("annotation_inserted", sample_id=annotation.sample_id, source=annotation.source)
        return result.data[0] if result.data else {}

    async def insert_annotations_batch(self, annotations: list[GraspAnnotation]) -> int:
        """Batch insert annotations. Returns count of inserted rows."""
        if not annotations:
            return 0
        data = [a.model_dump() for a in annotations]
        # Supabase supports batch insert
        result = self.client.table("grasp_annotations").insert(data).execute()
        count = len(result.data) if result.data else 0
        log.info("annotations_batch_inserted", count=count)
        return count

    async def get_annotations(
        self,
        source: Optional[str] = None,
        object_type: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict]:
        """Query annotations with optional filters."""
        query = self.client.table("grasp_annotations").select("*")
        if source:
            query = query.eq("source", source)
        if object_type:
            query = query.eq("object_type", object_type)
        if success is not None:
            query = query.eq("success", success)
        query = query.range(offset, offset + limit - 1)
        result = query.execute()
        return result.data or []

    async def get_annotation_count(self) -> int:
        """Get total number of annotations."""
        result = (
            self.client.table("grasp_annotations")
            .select("sample_id", count="exact")
            .execute()
        )
        return result.count or 0

    async def get_failure_distribution(self) -> dict[str, int]:
        """Get count of annotations per failure mode."""
        result = (
            self.client.table("grasp_annotations")
            .select("failure_mode")
            .execute()
        )
        dist: dict[str, int] = {}
        for row in result.data or []:
            mode = row.get("failure_mode", "none")
            dist[mode] = dist.get(mode, 0) + 1
        return dist

    # ── File Storage ──

    async def upload_file(self, bucket: str, path: str, file_bytes: bytes, content_type: str = "image/png") -> str:
        """Upload a file to Supabase storage. Returns the public URL."""
        self.client.storage.from_(bucket).upload(path, file_bytes, {"content-type": content_type})
        url = self.client.storage.from_(bucket).get_public_url(path)
        log.info("file_uploaded", bucket=bucket, path=path)
        return url

    async def download_file(self, bucket: str, path: str) -> bytes:
        """Download file bytes from Supabase storage."""
        return self.client.storage.from_(bucket).download(path)

    # ── Kitchen Profiles ──

    async def save_profile(self, profile: KitchenProfile) -> dict:
        """Upsert a kitchen profile."""
        data = profile.model_dump()
        result = (
            self.client.table("kitchen_profiles")
            .upsert(data, on_conflict="profile_id")
            .execute()
        )
        log.info("profile_saved", profile_id=profile.profile_id, name=profile.name)
        return result.data[0] if result.data else {}

    async def get_profile(self, profile_name: str) -> Optional[dict]:
        """Look up a profile by name."""
        result = (
            self.client.table("kitchen_profiles")
            .select("*")
            .eq("name", profile_name)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    async def list_profiles(self) -> list[dict]:
        """List all available kitchen profiles."""
        result = self.client.table("kitchen_profiles").select("*").execute()
        return result.data or []

    # ── Fine-Tune Jobs ──

    async def create_finetune_job(self, job: FineTuneStatus) -> dict:
        """Insert a fine-tune job record."""
        data = job.model_dump()
        result = self.client.table("finetune_jobs").insert(data).execute()
        log.info("finetune_job_created", job_id=job.job_id)
        return result.data[0] if result.data else {}

    async def update_finetune_status(self, job_id: str, **updates) -> dict:
        """Update a fine-tune job's status fields."""
        result = (
            self.client.table("finetune_jobs")
            .update(updates)
            .eq("job_id", job_id)
            .execute()
        )
        log.info("finetune_status_updated", job_id=job_id, **updates)
        return result.data[0] if result.data else {}

    async def get_finetune_status(self, job_id: str) -> Optional[dict]:
        """Get a fine-tune job's current status."""
        result = (
            self.client.table("finetune_jobs")
            .select("*")
            .eq("job_id", job_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None


# Module-level singleton
db = SupabaseClient()
