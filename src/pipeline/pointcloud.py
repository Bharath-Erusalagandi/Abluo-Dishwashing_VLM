"""Point cloud processing pipeline using Open3D.

Converts RGB-D input to segmented point clouds with per-object
feature extraction for grasp planning.

Supports two segmentation strategies:
  1. **Grounded SAM 2** (default) — uses foundation-model masks from
     ``src.pipeline.segmentation`` for accurate per-object point clouds.
  2. **DBSCAN clustering** (fallback) — geometric clustering when no
     masks are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


# RealSense D435 default intrinsics at 640x480
DEFAULT_INTRINSICS = {
    "fx": 615.0,
    "fy": 615.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480,
}


@dataclass
class SegmentedObject:
    """A single segmented object from the point cloud."""

    label: str = "unknown"
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rgb_crop: Optional[np.ndarray] = None
    pixel_bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # x1, y1, x2, y2
    estimated_wet: bool = False
    point_count: int = 0


@dataclass
class PointCloudResult:
    """Result of the full point cloud processing pipeline."""

    objects: list[SegmentedObject] = field(default_factory=list)
    table_plane: Optional[np.ndarray] = None  # [a, b, c, d] plane coefficients
    depth_quality: float = 1.0  # fraction of valid depth pixels
    total_points: int = 0


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: Optional[dict] = None,
) -> tuple:
    """Convert RGB-D images to an Open3D point cloud.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth: (H, W) uint16 depth map in millimetres.
        intrinsics: Camera intrinsic parameters.

    Returns:
        (open3d.geometry.PointCloud, depth_quality_score)
    """
    import open3d as o3d

    intrinsics = intrinsics or DEFAULT_INTRINSICS
    h, w = depth.shape[:2]

    # Measure depth quality
    valid_pixels = np.count_nonzero(depth)
    total_pixels = h * w
    depth_quality = valid_pixels / total_pixels

    # Create Open3D images
    rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

    # Build RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1000.0,  # mm to meters
        depth_trunc=2.0,  # max 2 meters
        convert_rgb_to_intensity=False,
    )

    # Camera intrinsics
    cam = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics["width"],
        height=intrinsics["height"],
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)
    log.info(
        "pointcloud_created",
        points=len(pcd.points),
        depth_quality=f"{depth_quality:.2%}",
    )
    return pcd, depth_quality


def segment_plane(pcd, distance_threshold: float = 0.01, ransac_n: int = 3, num_iterations: int = 1000):
    """Remove the dominant plane (table/rack surface) from point cloud.

    Returns:
        (plane_model, inlier_cloud, outlier_cloud)
    """
    import open3d as o3d

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    log.info(
        "plane_segmented",
        plane_points=len(inliers),
        remaining_points=len(outlier_cloud.points),
    )
    return plane_model, inlier_cloud, outlier_cloud


def cluster_objects(
    pcd,
    eps: float = 0.02,
    min_points: int = 50,
    max_clusters: int = 20,
) -> list[SegmentedObject]:
    """Cluster remaining points into individual objects using DBSCAN.

    Args:
        pcd: Open3D point cloud (table plane already removed).
        eps: DBSCAN neighbourhood radius in metres.
        min_points: Minimum cluster size.
        max_clusters: Maximum number of objects to return.

    Returns:
        List of SegmentedObject with point cloud data.
    """
    import open3d as o3d

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
    )

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if len(labels) == 0:
        return []

    unique_labels = set(labels)
    unique_labels.discard(-1)  # remove noise label

    objects: list[SegmentedObject] = []
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)

    for label in sorted(unique_labels):
        if len(objects) >= max_clusters:
            break

        mask = labels == label
        obj_points = points[mask]
        obj_colors = colors[mask]
        obj_normals = normals[mask]

        centroid = obj_points.mean(axis=0)
        bbox_min = obj_points.min(axis=0)
        bbox_max = obj_points.max(axis=0)

        # Estimate wetness from specular reflectance (high color variance)
        color_var = obj_colors.var(axis=0).mean() if len(obj_colors) > 5 else 0
        estimated_wet = color_var > 0.05  # heuristic threshold

        objects.append(SegmentedObject(
            label=f"object_{label}",
            points=obj_points,
            colors=obj_colors,
            normals=obj_normals,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            estimated_wet=estimated_wet,
            point_count=len(obj_points),
        ))

    log.info("objects_clustered", count=len(objects))
    return objects


def process_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: Optional[dict] = None,
    masks: Optional[list] = None,
    labels: Optional[list[str]] = None,
) -> PointCloudResult:
    """Full point cloud processing pipeline.

    1. Convert RGB-D to point cloud
    2. Remove table plane (RANSAC)
    3. Segment objects — via Grounded SAM 2 masks (preferred) or DBSCAN fallback
    4. Extract per-object features

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth: (H, W) uint16 depth map in mm.
        intrinsics: Camera parameters.
        masks: Optional list of (H, W) bool masks from Grounded SAM 2.
            When provided, each mask defines one object (no DBSCAN needed).
        labels: Optional list of labels corresponding to ``masks``.

    Returns:
        PointCloudResult with segmented objects.
    """
    # Step 1: Create point cloud
    pcd, depth_quality = rgbd_to_pointcloud(rgb, depth, intrinsics)

    if len(pcd.points) < 100:
        log.warning("insufficient_points", count=len(pcd.points))
        return PointCloudResult(depth_quality=depth_quality)

    # Step 2: Remove table surface
    plane_model, _table_pcd, objects_pcd = segment_plane(pcd)

    # Step 3: Segment objects
    if masks:
        objects = _objects_from_masks(pcd, rgb, depth, masks, labels, intrinsics)
        log.info("objects_from_masks", count=len(objects))
    else:
        objects = cluster_objects(objects_pcd)

    return PointCloudResult(
        objects=objects,
        table_plane=np.array(plane_model),
        depth_quality=depth_quality,
        total_points=len(pcd.points),
    )


def _objects_from_masks(
    pcd,
    rgb: np.ndarray,
    depth: np.ndarray,
    masks: list[np.ndarray],
    labels: Optional[list[str]],
    intrinsics: Optional[dict] = None,
) -> list[SegmentedObject]:
    """Extract per-object point clouds using Grounded SAM 2 masks.

    Each mask is a (H, W) bool array identifying one object. We back-
    project the masked pixels into 3D using the depth map and camera
    intrinsics.
    """
    import open3d as o3d

    intrinsics = intrinsics or DEFAULT_INTRINSICS
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    all_points = np.asarray(pcd.points) if len(pcd.points) > 0 else np.empty((0, 3))

    objects: list[SegmentedObject] = []
    h, w = depth.shape[:2]

    for i, mask in enumerate(masks):
        if mask.shape != (h, w):
            continue

        ys, xs = np.where(mask & (depth > 0))
        if len(xs) < 10:
            continue

        zs = depth[ys, xs].astype(np.float64) / 1000.0  # mm → m
        x3d = (xs.astype(np.float64) - cx) * zs / fx
        y3d = (ys.astype(np.float64) - cy) * zs / fy
        pts = np.stack([x3d, y3d, zs], axis=1)

        colors = rgb[ys, xs].astype(np.float64) / 255.0

        # Estimate normals
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(pts)
        obj_pcd.colors = o3d.utility.Vector3dVector(colors)
        obj_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )
        normals = np.asarray(obj_pcd.normals)

        centroid = pts.mean(axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)

        # Pixel bounding box
        px1, py1 = int(xs.min()), int(ys.min())
        px2, py2 = int(xs.max()), int(ys.max())

        # Wet estimate from colour variance
        color_var = colors.var(axis=0).mean() if len(colors) > 5 else 0
        estimated_wet = color_var > 0.05

        label = labels[i] if labels and i < len(labels) else f"object_{i}"

        objects.append(SegmentedObject(
            label=label,
            points=pts,
            colors=colors,
            normals=normals,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            rgb_crop=rgb[py1:py2, px1:px2].copy(),
            pixel_bbox=[px1, py1, px2, py2],
            estimated_wet=estimated_wet,
            point_count=len(pts),
        ))

    return objects
