"""DishSpace Demo UI — Streamlit app.

The cold-outreach weapon: upload a failure video/image, get grasp zones overlaid.
Shareable via a single link.

Run:  streamlit run demo/app.py
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

# ── Page Config ──

st.set_page_config(
    page_title="DishSpace AI — Fix Your Robot's Grasp",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ──

API_BASE = "http://localhost:8000"
DEMO_API_KEY = "dev-key-change-me"


# ── Helpers ──


def encode_image(image: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_grasp_api(
    image_b64: str,
    depth_b64: Optional[str] = None,
    profile: str = "default",
    robot: str = "UR5_realsense",
    min_confidence: float = 0.5,
    max_grasps: int = 5,
) -> dict:
    """Call the /grasp_plan API endpoint."""
    import httpx

    try:
        response = httpx.post(
            f"{API_BASE}/grasp_plan",
            json={
                "image_base64": image_b64,
                "depth_base64": depth_b64,
                "kitchen_profile": profile,
                "robot": robot,
                "options": {
                    "collision_check": True,
                    "max_grasps": max_grasps,
                    "min_confidence": min_confidence,
                    "coordinate_frame": "camera",
                },
            },
            headers={"X-API-Key": DEMO_API_KEY},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error("⚠️ API server not running. Start it with: `uvicorn src.api.server:app`")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def draw_grasp_overlay(
    image: np.ndarray,
    grasps: list[dict],
) -> np.ndarray:
    """Draw grasp zones on the image.

    Green = high confidence (safe grasp)
    Yellow = medium confidence (risky)
    Red = low confidence or collision risk
    """
    import cv2

    overlay = image.copy()

    for grasp in grasps:
        conf = grasp.get("confidence", 0.5)
        bbox = grasp.get("object_bbox", [0, 0, 50, 50])
        obj_name = grasp.get("object", "unknown")
        grasp_type = grasp.get("grasp_type", "unknown")

        # Color based on confidence
        if conf >= 0.85:
            color = (0, 255, 0)  # green
            status = "SAFE"
        elif conf >= 0.7:
            color = (0, 255, 255)  # yellow
            status = "RISKY"
        else:
            color = (0, 0, 255)  # red
            status = "DANGER"

        # Collision risk override
        collision_risk = grasp.get("failure_risk", {}).get("collision", 0)
        if collision_risk > 0.2:
            color = (0, 0, 255)
            status = "COLLISION"

        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw grasp point (center of bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(overlay, (cx, cy), 8, color, -1)
        cv2.circle(overlay, (cx, cy), 12, color, 2)

        # Draw approach arrow
        cv2.arrowedLine(overlay, (cx, y1 - 30), (cx, cy - 12), color, 2, tipLength=0.3)

        # Label
        label = f"{obj_name} ({conf:.0%}) [{status}]"
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Grasp type
        cv2.putText(overlay, grasp_type, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Semi-transparent overlay
    result = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
    return result


# ── UI ──


def main():
    # Header
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>🍽️ DishSpace AI</h1>
            <h3 style='color: #666;'>Fix Your Robot's Grasp in 30 Seconds</h3>
            <p style='color: #999;'>Upload a failure image → See the fix → Export to ROS</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")
        profile = st.selectbox(
            "Kitchen Profile",
            ["default", "commercial_rack", "home_dishwasher", "conveyor"],
            index=0,
        )
        robot = st.selectbox(
            "Robot Type",
            ["UR5_realsense", "UR10_realsense", "Franka_realsense", "Kinova_gen3"],
            index=0,
        )
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
        max_grasps = st.slider("Max Grasps", 1, 10, 5)

        st.divider()
        st.markdown("**API Status**")
        try:
            import httpx
            r = httpx.get(f"{API_BASE}/health", timeout=3.0)
            if r.status_code == 200:
                st.success("✅ API Connected")
                health = r.json()
                st.json(health)
            else:
                st.error("❌ API Error")
        except Exception:
            st.warning("⚠️ API Offline")

    # ── Main Content ──

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Input")

        upload_type = st.radio(
            "Input type",
            ["Image", "Video (extract frames)", "Sample image"],
            horizontal=True,
        )

        uploaded_image = None

        if upload_type == "Image":
            uploaded_file = st.file_uploader(
                "Upload RGB image from robot camera",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Drag and drop or click to upload. Max 100MB.",
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file).convert("RGB")

        elif upload_type == "Video (extract frames)":
            uploaded_file = st.file_uploader(
                "Upload video file",
                type=["mp4", "avi", "mov"],
            )
            if uploaded_file:
                st.info("Video frame extraction coming soon. Upload a single frame for now.")

        elif upload_type == "Sample image":
            if st.button("🎲 Generate Sample Image"):
                # Create a synthetic kitchen scene image for demo
                rng = np.random.default_rng(42)
                img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # light gray background

                # Draw some "objects" (colored rectangles simulating dishes)
                import cv2
                # Mug
                cv2.rectangle(img, (150, 200), (220, 320), (139, 90, 43), -1)
                cv2.ellipse(img, (185, 200), (35, 15), 0, 0, 360, (120, 80, 40), -1)
                cv2.putText(img, "mug", (155, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                # Plate
                cv2.ellipse(img, (400, 300), (80, 20), 0, 0, 360, (220, 220, 230), -1)
                cv2.ellipse(img, (400, 298), (78, 18), 0, 0, 360, (240, 240, 250), -1)
                cv2.putText(img, "plate", (370, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                # Wine glass
                cv2.line(img, (520, 350), (520, 250), (200, 200, 210), 2)
                cv2.ellipse(img, (520, 240), (25, 40), 0, 0, 360, (210, 210, 220), 1)
                cv2.putText(img, "glass", (495, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                uploaded_image = Image.fromarray(img)

        if uploaded_image:
            st.image(uploaded_image, caption="Input Image", use_container_width=True)

    with col2:
        st.subheader("🎯 Grasp Analysis")

        if uploaded_image:
            if st.button("🔍 Analyze Grasps", type="primary", use_container_width=True):
                with st.spinner("Running grasp planning..."):
                    start_time = time.time()

                    # Encode and send to API
                    image_b64 = encode_image(uploaded_image)
                    result = call_grasp_api(
                        image_b64,
                        profile=profile,
                        robot=robot,
                        min_confidence=min_confidence,
                        max_grasps=max_grasps,
                    )

                    elapsed = time.time() - start_time

                if result and "grasp_plan" in result:
                    grasps = result["grasp_plan"]

                    if grasps:
                        # Draw overlay
                        img_array = np.array(uploaded_image)
                        overlay = draw_grasp_overlay(img_array, grasps)
                        st.image(overlay, caption="Grasp Analysis Result", use_container_width=True)

                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Objects Found", len(grasps))
                        col_b.metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")
                        col_c.metric(
                            "Collision Free",
                            "✅ Yes" if result.get("collision_free") else "❌ No",
                        )

                        # Scene metadata
                        meta = result.get("scene_metadata", {})
                        if meta:
                            st.caption(
                                f"Depth quality: {meta.get('depth_quality', 1.0):.0%} | "
                                f"Wet: {'Yes' if meta.get('wet_surface_detected') else 'No'} | "
                                f"Soap: {'Yes' if meta.get('soap_presence') else 'No'}"
                            )

                        # Detailed results table
                        st.subheader("📋 Grasp Details")
                        for i, g in enumerate(grasps):
                            with st.expander(
                                f"Grasp {i + 1}: {g.get('object', 'unknown')} — {g.get('confidence', 0):.0%}"
                            ):
                                st.json(g)

                        # Export actions
                        st.divider()
                        col_dl, col_cp, col_sh = st.columns(3)

                        # Download JSON
                        json_str = json.dumps(result, indent=2)
                        col_dl.download_button(
                            "📥 Download ROS JSON",
                            data=json_str,
                            file_name="dishspace_grasp_plan.json",
                            mime="application/json",
                        )

                        # Copy to clipboard
                        col_cp.code(json_str[:200] + "...", language="json")

                        # Share link
                        request_id = result.get("request_id", "demo")
                        col_sh.info(f"🔗 Share: dishspace.ai/demo/{request_id}")

                    else:
                        st.warning("No objects detected in the image. Try a different image with kitchen objects.")
                else:
                    st.error("Failed to get results from the API.")
        else:
            st.info("👈 Upload an image or generate a sample to get started.")

    # ── Footer ──
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #999; font-size: 0.8em; padding: 1rem 0;'>
            DishSpace AI · Fine-tuning layer for kitchen robotics<br>
            <a href='/docs' target='_blank'>API Docs</a> ·
            <a href='https://github.com/dishspace' target='_blank'>GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
