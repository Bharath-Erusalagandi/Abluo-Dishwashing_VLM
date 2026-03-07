# DishSpace AI — MVP Build Plan
**Fine-Tuning Layer for Kitchen Robotics**

> Fine-tune foundation models. Own the kitchen robot data layer. Sell the integration, not the science.

**Goal:** Get from zero to 3 paying pilot customers in 6 weeks by building a domain-specific fine-tuning and integration layer on top of existing open-source spatial models — without building a foundation model from scratch.

**Core Insight:** Physical Intelligence builds general. You build deep. Kitchen-specific failure data + seamless ROS integration is the moat. The model is a commodity; the domain knowledge is not.

**One-liner pitch:** *"We make kitchen robots 40% better at picking up wet dishes — in 4 hours, not 4 months."*

---

## 0. Executive Summary

DishSpace AI is a B2B infrastructure company that sells domain-specific fine-tuning and ROS integration for kitchen robotics. We do not build foundation models. We make existing foundation models production-ready for the hardest unsolved problem in commercial kitchen automation: reliable grasp planning on wet, soapy, reflective, and irregularly shaped dishware.

**The problem:** Kitchen robot companies (Armstrong, Sunday, Memo, Dishcraft) spend 60–70% of their engineering time on grasp failures. Their general-purpose models fail on wet ceramics, stacked bowls, and transparent glassware because no one has built a kitchen-specific fine-tuning layer.

**The solution:** A fine-tuning-as-a-service API that takes a customer's failure videos, generates a kitchen-specific LoRA adapter in hours, and returns ROS-compatible grasp trajectories via a single REST endpoint.

**The business model:** $5K one-time fine-tune per kitchen profile + $99–$199/mo per robot for ongoing API access.

**The ask:** 6 weeks to prove the concept with 3 pilot customers. Total runway cost: ~$3K (GPU + infrastructure).

### Key Metrics to Track

| Metric | Week 2 | Week 4 | Week 6 | Month 3 |
|---|---|---|---|---|
| Annotated grasp samples | 5,500 | 6,000 | 6,500+ | 8,000+ |
| Grasp success rate (mugs) | 85% | 88% | 90%+ | 93%+ |
| API latency (p95) | — | <250ms | <200ms | <150ms |
| Pilot customers | 0 | 1 call | 3 | 10 |
| Paying customers | 0 | 0 | 1 | 5+ |
| MRR | $0 | $0 | $99–$5K | $10K+ |

---

## 1. What You're Actually Building

DishSpace is not a foundation model company. It is the fine-tuning and integration infrastructure that makes any foundation model production-ready for commercial kitchen robotics. You sit between the big model labs and the hardware companies that are failing on fragile 3D manipulation.

You are building the "last mile" — the layer that turns a general robot brain into one that reliably picks up a wet wine glass.

### The Problem in Detail

Kitchen manipulation is the hardest unsolved domain in robotics grasping for five compounding reasons:

1. **Surface conditions change constantly.** A dry mug and a soapy-wet mug require fundamentally different grip force profiles. Water film reduces friction coefficients by 30–60%.
2. **Reflective/transparent surfaces break depth sensors.** Wine glasses, wet porcelain, and stainless steel reflect IR structured light, causing RealSense depth holes or phantom geometry.
3. **Objects are stacked and nested.** Plates in a rack, bowls inside bowls, utensils tangled together — each configuration is a unique collision puzzle.
4. **Fragility demands precision.** A 2mm error on a wine glass stem = shattered glass. Industrial bin-picking tolerance is 5–10mm. Kitchen tolerance is <2mm.
5. **Environment is adversarial.** Steam, soap suds, variable lighting, splashing water, vibrating conveyor belts. No lab setting replicates this.

General-purpose models (OpenVLA, RT-2, Octo) achieve ~60% grasp success on standard kitchen objects because they were trained on lab environments with dry, isolated objects. The gap between lab and commercial kitchen is where DishSpace lives.

### The Three Assets You're Selling

| Asset | What It Is | Why It's Defensible |
|---|---|---|
| **Kitchen Domain Data** | Annotated grasp failures from real deployments. Soap occlusion, wet ceramic, stacked bowls, transparent glass depth artifacts. | Nobody else has this at kitchen scale. Every pilot adds 50–200 labeled failures. |
| **Fine-Tuning Pipeline** | LoRA adapters on top of OpenVLA or GraspNet. Cheap to run, fast to iterate. Per-customer kitchen profiles. | Kitchen profile = instant fine-tune for new customers. Compounding data advantage. |
| **ROS Integration Layer** | ROS-compatible JSON trajectories out of the box. MoveIt-ready pose arrays. Collision-checked. | Armstrong and Sunday don't want to build this bridge. You already have it. |

### The Fourth Hidden Asset: DishBench

DishBench is a standardized 50-scenario evaluation benchmark for kitchen grasp planning. It becomes the industry standard for measuring kitchen manipulation performance — like ImageNet for kitchen robotics. Publishing it open-source (while keeping the training data proprietary) establishes DishSpace as the domain authority.

**DishBench Categories (10 scenarios each):**
1. **Wet ceramics** — mugs, plates, bowls with water film
2. **Transparent glass** — wine glasses, tumblers, measuring cups with depth artifacts
3. **Stacked/nested** — plates in rack, bowls inside bowls, utensil bundles
4. **Reflective metal** — stainless steel pots, silverware, aluminum trays
5. **Adversarial conditions** — soap suds, steam occlusion, variable lighting, vibration

---

## 2. Competitive Landscape

### Direct Competitors

| Company | What They Do | Funding | Kitchen Focus | Why You Win |
|---|---|---|---|---|
| **Physical Intelligence (π)** | General-purpose foundation model for robotics | $400M+ Series A | None — optimizes for generality | They'll never go deep on kitchen. You own the vertical. |
| **Skild AI** | Scalable robot learning platform | $300M Series A | Minimal | Same generality trap. Kitchen is a rounding error for them. |
| **Covariant** | AI for warehouse picking | $222M total | None | Warehouse ≠ kitchen. Different physics, different objects. |
| **Dexterous Robotics** | Dishwashing robot hardware | $3M Seed | High — but builds hardware | They need YOUR software layer. Potential customer, not competitor. |

### Adjacent Players (Potential Customers)

| Company | Stage | Robot Type | Kitchen Problem | DishSpace Opportunity |
|---|---|---|---|---|
| **Armstrong Robotics** | 12 deployments | UR5-based arms | Wet dish grasp failures | Fine-tune their existing pipeline |
| **Sunday / Memo** | 50-home beta | Mobile manipulators | Fragile home dishware variety | LoRA adapters per home profile |
| **Dishcraft** | Commercial operations | Custom conveyor system | Stacked plate handling | Collision-aware trajectory planning |
| **Bear Robotics** | 1000+ restaurant deployments | Service robots | No manipulation yet | Future integration when they add arms |
| **Miso Robotics (Flippy)** | Fast food kitchens | Ceiling-mounted arm | Different domain but adjacent | Potential lateral expansion |

### Competitive Moat Analysis

```
             Defensibility Over Time
             
  High │                              ╱ DishSpace (data flywheel)
       │                           ╱
       │                        ╱
       │                     ╱
       │         ╱──────────── Physical Intelligence (general model)
       │      ╱
       │   ╱──────── Open-source baseline (GraspNet)
  Low  │╱
       └──────────────────────────────────────
        Month 1    Month 6    Month 12    Month 24
        
  DishSpace's data moat compounds. General models plateau on kitchen tasks.
```

**Key insight:** Physical Intelligence would need to deploy in 50+ commercial kitchens to match your failure dataset. That's not their business model. Your advantage grows every month.

---

## 3. MVP Product Specification

### 3.1 Core API

| Field | Value | Notes |
|---|---|---|
| Endpoint | `POST /grasp_plan` | REST, JSON body |
| Input | RGB-D image (640x480) | Intel RealSense D435/D455 or equivalent |
| Base Model | OpenVLA-7b or GraspNet | Fine-tuned, not built from scratch |
| Adapter | DishSpace kitchen LoRA | Your moat — proprietary fine-tune |
| Output | ROS-compatible trajectory JSON | `pose: [x,y,z,rx,ry,rz]` + confidence |
| Latency | <200ms on A10 GPU | Via Modal.com serverless |
| Accuracy Target | 90%+ grasp success | vs ~60% baseline on kitchen objects |

### 3.2 Full API Endpoint Specification

| Endpoint | Method | Purpose | Auth Required |
|---|---|---|---|
| `/grasp_plan` | POST | Primary grasp planning from RGB-D input | API key |
| `/grasp_plan/batch` | POST | Batch processing for multiple frames | API key |
| `/fine_tune` | POST | Trigger LoRA fine-tune on customer data | API key + admin |
| `/fine_tune/{job_id}/status` | GET | Check fine-tune job progress | API key |
| `/profiles` | GET | List available kitchen profiles | API key |
| `/profiles/{id}` | GET | Get profile details + performance stats | API key |
| `/evaluate` | POST | Run DishBench evaluation on a profile | API key |
| `/health` | GET | Service health + GPU availability | None |
| `/usage` | GET | API call count + billing summary | API key |

**Sample Request / Response:**

```json
// Request — Single Grasp Plan
POST /grasp_plan
{
  "image_base64": "<RGB-D frame, base64-encoded PNG>",
  "depth_base64": "<depth map, base64-encoded 16-bit PNG>",
  "kitchen_profile": "commercial_rack",
  "robot": "UR5_realsense",
  "base_model": "openVLA-7b",
  "options": {
    "collision_check": true,
    "max_grasps": 5,
    "min_confidence": 0.7,
    "coordinate_frame": "camera"    // "camera" | "world" | "robot_base"
  }
}

// Response
{
  "request_id": "req_a8f3b2c1",
  "grasp_plan": [
    {
      "pose": [0.42, -0.11, 0.38, 0.0, 1.57, 0.0],
      "confidence": 0.94,
      "object": "mug",
      "object_bbox": [120, 80, 210, 190],
      "grasp_type": "rim_pinch",
      "grip_force_n": 4.2,
      "failure_risk": {
        "slip": 0.03,
        "collision": 0.01,
        "occlusion": 0.02
      }
    },
    {
      "pose": [0.38, 0.05, 0.41, 0.0, 1.57, 0.3],
      "confidence": 0.87,
      "object": "plate",
      "object_bbox": [240, 100, 400, 280],
      "grasp_type": "edge_pinch",
      "grip_force_n": 6.8,
      "failure_risk": {
        "slip": 0.08,
        "collision": 0.03,
        "occlusion": 0.01
      }
    }
  ],
  "scene_metadata": {
    "objects_detected": 4,
    "depth_quality": 0.91,
    "wet_surface_detected": true,
    "soap_presence": false
  },
  "collision_free": true,
  "latency_ms": 142,
  "model_version": "dishspace-lora-v0.3.1",
  "profile_used": "commercial_rack"
}
```

```json
// Request — Fine-Tune Trigger
POST /fine_tune
{
  "profile_name": "joes_diner_rack_v2",
  "base_model": "openVLA-7b",
  "training_data": {
    "supabase_bucket": "pilot_data",
    "folder": "joes_diner/failures_week1",
    "sample_count": 312
  },
  "lora_config": {
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "epochs": 5,
    "learning_rate": 2e-4
  },
  "eval_holdout_pct": 0.15
}

// Response
{
  "job_id": "ft_9c2d4e1a",
  "status": "queued",
  "estimated_duration_min": 180,
  "estimated_cost_usd": 18.50,
  "webhook_url": "https://api.dishspace.ai/fine_tune/ft_9c2d4e1a/status"
}
```

### 3.3 Supported Robot Platforms (MVP)

| Robot Arm | Camera | ROS Version | MoveIt Support | Priority |
|---|---|---|---|---|
| Universal Robots UR5/UR5e | Intel RealSense D435 | ROS 2 Humble | Full | P0 — most common in kitchen pilots |
| Universal Robots UR10 | Intel RealSense D455 | ROS 2 Humble | Full | P0 — larger workspace for conveyor |
| Franka Emika Panda | RealSense D435 | ROS 2 Humble | Full | P1 — research labs, some pilots |
| Kinova Gen3 | Built-in depth | ROS 2 Humble | Partial | P2 — Sunday/home robots |
| Custom (URDF provided) | Any RGB-D | ROS 2 / ROS 1 | Via config | P2 — catch-all for pilots |

### 3.4 Demo UI (the door-opener)

This is your cold-outreach weapon. The web app does one thing: upload a failure video, get green grasp zones overlaid on the frame. It needs to be shareable via a single link.

**Core Features:**
- Upload RGB-D video or image from robot camera (drag-and-drop, up to 100MB)
- Calls `/grasp_plan`, overlays confidence-colored grasp zones (green = safe, yellow = risky, red = collision risk)
- 3D point cloud viewer — rotate/zoom the scene with grasp poses overlaid
- Side-by-side "before vs after" comparison — baseline model vs DishSpace fine-tuned
- Exportable JSON — customer can copy-paste into their ROS stack immediately
- Share link generated — one URL to send in cold DMs on robotics Discord
- No login required for first use (friction = death for cold outreach)

**Demo UI Flow:**
```
┌──────────────────────────────────────────────────────┐
│  DishSpace — Fix Your Robot's Grasp in 30 Seconds   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────────────────┐        │
│  │                                         │        │
│  │     [ Drag & drop your failure video ]  │        │
│  │     [ or paste a RealSense .bag URL  ]  │        │
│  │                                         │        │
│  └─────────────────────────────────────────┘        │
│                                                      │
│  Kitchen Profile: [ Commercial Rack  ▼ ]             │
│  Robot Type:      [ UR5 + RealSense  ▼ ]             │
│                                                      │
│  [ 🔍 Analyze Grasps ]                               │
│                                                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────┐  ┌──────────────────┐         │
│  │  Before (60%)     │  │  After (94%)     │         │
│  │  ● red zones      │  │  ● green zones   │         │
│  │  ✗ slip risk      │  │  ✓ stable grip   │         │
│  └──────────────────┘  └──────────────────┘         │
│                                                      │
│  Objects: mug (94%), plate (87%), glass (91%)        │
│  Wet surface detected: Yes                           │
│  Collision risk: None                                │
│                                                      │
│  [ Download ROS JSON ]  [ Copy to Clipboard ]        │
│  [ Share Link: dishspace.ai/demo/a8f3b2c1 ]         │
│                                                      │
└──────────────────────────────────────────────────────┘
```

> **The demo is not a toy. It IS the sales call.** If someone can upload their failure video and see a fix in 30 seconds, you've earned a pilot.

---

## 4. Technical Stack

| Layer | Tool | Why | Cost |
|---|---|---|---|
| Demo UI | Streamlit | Ship in hours, not days. Enough for pilots. | Free |
| API Server | FastAPI | Async, automatic OpenAPI docs, robotics-friendly | Free |
| Base Model | OpenVLA-7b (HuggingFace) | Open weights, fine-tuneable, strong spatial priors | Free |
| Backup Model | GraspNet (Contact-GraspNet) | Lighter weight, faster inference, good baseline | Free |
| Fine-tuning | LoRA via HuggingFace PEFT | Cheap adaptation without full retrain | ~$15/run |
| 3D Processing | Open3D + PyTorch3D | Point cloud ops, mesh processing | Free |
| Depth Completion | IP-Basic or DenseLiDAR | Fill RealSense depth holes on reflective surfaces | Free |
| Physics Sim | MuJoCo (free since DeepMind) | Grasp validation & training data gen | Free |
| GPU Inference | Modal.com | $0.50/hr A10, serverless, no DevOps overhead | ~$28/10K calls |
| Data Store | Supabase | Postgres + auth + file storage in one | Free tier |
| Frontend Host | Vercel | Free tier, instant deploy | Free |
| API Host | Railway | Simple, cheap, no K8s needed at MVP stage | $5/mo |
| ROS Bridge | roslibpy | Connect output JSON to robot arm directly | Free |
| Monitoring | Prometheus + Grafana Cloud | API latency, grasp success rate tracking | Free tier |
| Error Tracking | Sentry | Catch inference failures in production | Free tier |
| CI/CD | GitHub Actions | Auto-deploy on merge, run DishBench on PR | Free tier |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────┐  │
│  │  Streamlit Demo  │    │  Customer ROS   │    │  Python SDK │  │
│  │  (Vercel)        │    │  Nodes          │    │  pip install│  │
│  └────────┬─────────┘    └────────┬────────┘    └──────┬─────┘  │
└───────────┼───────────────────────┼─────────────────────┼────────┘
            │                       │                     │
            └───────────┬───────────┘                     │
                        │  HTTPS / REST                   │
┌───────────────────────▼─────────────────────────────────▼────────┐
│                       API LAYER (Railway)                         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  FastAPI Server                                           │    │
│  │  ├── POST /grasp_plan      ← primary inference            │    │
│  │  ├── POST /grasp_plan/batch ← multi-frame processing     │    │
│  │  ├── POST /fine_tune       ← trigger LoRA training        │    │
│  │  ├── GET  /profiles        ← list kitchen profiles        │    │
│  │  ├── POST /evaluate        ← run DishBench               │    │
│  │  ├── GET  /health          ← liveness + GPU status        │    │
│  │  └── GET  /usage           ← billing + rate limits        │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                              │                                    │
│  ┌──────────────────┐       │       ┌────────────────────┐      │
│  │  Auth Middleware  │───────┤       │  Rate Limiter      │      │
│  │  (API Key + JWT) │       │       │  (100 req/min free)│      │
│  └──────────────────┘       │       └────────────────────┘      │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                              │  gRPC / Modal API
┌─────────────────────────────▼────────────────────────────────────┐
│                     COMPUTE LAYER (Modal.com)                     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  GPU Worker (A10, 24GB VRAM)                             │     │
│  │                                                          │     │
│  │  1. Decode RGB-D input                                   │     │
│  │  2. Depth completion (fill RealSense holes)              │     │
│  │  3. Open3D point cloud extraction                        │     │
│  │     ├── Segment individual objects                       │     │
│  │     ├── Extract rim curves + handle primitives           │     │
│  │     └── Surface condition detection (wet/soap/steam)     │     │
│  │  4. OpenVLA-7b + DishSpace LoRA inference                │     │
│  │     ├── Generate candidate grasp poses                   │     │
│  │     └── Score with kitchen-specific confidence           │     │
│  │  5. MuJoCo collision check                               │     │
│  │     ├── Simulate top-3 grasp candidates                  │     │
│  │     └── Filter collision/instability                     │     │
│  │  6. ROS trajectory serialization                         │     │
│  │     └── MoveIt-compatible pose arrays                    │     │
│  │                                                          │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Training Worker (A10, on-demand)                        │     │
│  │  ├── LoRA fine-tune pipeline (PEFT)                      │     │
│  │  ├── DishBench evaluation suite                          │     │
│  │  └── Synthetic data generation (MuJoCo + ShapeNet)       │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                      DATA LAYER (Supabase)                        │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │  Auth / Users │  │  Failure DB   │  │  File Storage      │     │
│  │  (Postgres)   │  │  (Postgres)   │  │  (S3-compatible)   │     │
│  │               │  │               │  │                    │     │
│  │  - API keys   │  │  - Grasp logs │  │  - RGB-D frames    │     │
│  │  - Profiles   │  │  - Labels     │  │  - LoRA weights    │     │
│  │  - Billing    │  │  - DishBench  │  │  - Training data   │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

### Key Implementation Notes

**Why not full retrain — use LoRA:**
Full OpenVLA retrain = ~$2,000–$8,000 GPU cost. LoRA fine-tune = ~$15–$40 per kitchen profile. Customers can get a custom fine-tune for their specific rack/dish set in hours, not weeks.

| Approach | GPU Hours | Cost | Time | When to Use |
|---|---|---|---|---|
| Full retrain (OpenVLA-7b) | 40–80 hrs (A100) | $2,000–$8,000 | 2–4 days | Never at MVP stage |
| LoRA (rank=16) | 3–4 hrs (A10) | $15–$40 | 3–4 hours | Default for all customers |
| LoRA (rank=8, small data) | 1–2 hrs (A10) | $5–$15 | 1–2 hours | Quick profile updates |
| QLoRA (4-bit quantized) | 2–3 hrs (A10) | $10–$25 | 2–3 hours | If VRAM-constrained |

**3D backbone — Open3D pipeline:**
1. Receive RealSense depth frame (640×480, 16-bit)
2. Run depth completion to fill holes from reflective surfaces (IP-Basic algorithm)
3. Back-project to organized point cloud using camera intrinsics
4. Plane segmentation (RANSAC) to remove table/rack surface
5. DBSCAN clustering to isolate individual objects
6. Per-object: extract rim curves (Canny on depth edges), identify handle protrusions as cylindrical primitives
7. Compute surface normals + estimate wet/dry from specular reflectance patterns
8. Pass enriched point cloud + RGB crop per object to LoRA-adapted model

**Depth completion for transparent/reflective objects:**
RealSense D435 produces depth holes on wine glasses and wet porcelain. This is the #1 cause of baseline model failure. The fix:
- **IP-Basic** — fast morphological completion, <5ms per frame, good enough for opaque wet surfaces
- **TransCG** (Transparent Object Depth Completion) — neural approach for glass, ~20ms per frame, needed for wine glasses
- DishSpace runs both and merges: IP-Basic as default, TransCG triggered when transparent object detected in RGB

**Infra discipline — Modal.com serverless:**
Zero idle GPU cost. At $0.50/hr and <200ms per call, 10,000 calls costs roughly $28. Extremely cheap to pilot.

**Modal cold start mitigation:**
Modal serverless has a 10–30s cold start when no GPU is warm. For demo responsiveness:
- Keep one A10 warm during business hours (9am–6pm PT) — costs ~$4.50/day
- Use `modal.web_endpoint` with `keep_warm=1` for the primary inference function
- Queue fine-tune jobs to run on separate containers (no cold start sensitivity)

### Inference Pipeline Pseudocode

```python
# modal_worker.py — GPU inference on Modal.com

import modal
from peft import PeftModel
from transformers import AutoModelForCausalLM
import open3d as o3d
import numpy as np

app = modal.App("dishspace")
volume = modal.Volume.from_name("dishspace-models")

@app.cls(gpu="A10G", image=modal.Image.debian_slim().pip_install(
    "transformers", "peft", "open3d", "torch", "mujoco"
), volumes={"/models": volume}, keep_warm=1)
class GraspPlanner:
    
    @modal.enter()
    def load_model(self):
        """Load base model + LoRA adapter on container start."""
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "openvla/openvla-7b", device_map="auto", torch_dtype="float16"
        )
        self.default_adapter = PeftModel.from_pretrained(
            self.base_model, "/models/lora/kitchen_default_v0.3"
        )
    
    @modal.method()
    def plan_grasp(self, rgb_array, depth_array, profile, options):
        """Full pipeline: RGB-D → point cloud → inference → collision check → ROS JSON."""
        
        # 1. Depth completion
        depth_filled = self.complete_depth(depth_array)
        
        # 2. Point cloud extraction + segmentation
        pcd = self.rgbd_to_pointcloud(rgb_array, depth_filled)
        objects = self.segment_objects(pcd)
        
        # 3. Per-object grasp inference with LoRA adapter
        adapter = self.load_profile_adapter(profile)
        candidates = []
        for obj in objects:
            grasps = adapter.predict_grasps(obj.rgb_crop, obj.point_cloud)
            candidates.extend(grasps)
        
        # 4. MuJoCo collision check
        validated = self.collision_filter(candidates, pcd)
        
        # 5. Serialize to ROS-compatible JSON
        return self.to_ros_trajectory(validated, options.coordinate_frame)
```

---

## 5. 6-Week Build Plan

### Budget Estimate

| Item | Cost | When |
|---|---|---|
| Modal.com GPU (training + inference) | ~$50 | Weeks 1–4 |
| Railway API hosting | $5/mo | Week 3+ |
| Supabase (free tier) | $0 | Week 1+ |
| Vercel (free tier) | $0 | Week 4+ |
| Domain name (dishspace.ai) | $12/yr | Week 4 |
| Sentry (free tier) | $0 | Week 3+ |
| **Total MVP runway** | **~$70** | **6 weeks** |

### Week-by-Week Overview

| Week | Timeline | Success Metric | Deliverable | Status |
|---|---|---|---|---|
| Week 1 | Days 1–7 | Working data pipeline | 5,500 labeled grasp attempts in Supabase | ⬜ Not started |
| Week 2 | Days 8–14 | Baseline model fine-tuned | LoRA adapter with >85% on mugs | ⬜ Not started |
| Week 3 | Days 15–21 | API live on Railway | `/grasp_plan` returning <200ms responses | ⬜ Not started |
| Week 4 | Days 22–28 | Demo UI + 1st pilot call | Shareable URL + Armstrong/Sunday call | ⬜ Not started |
| Week 5 | Days 29–35 | 3 pilot integrations | 3 companies running DishSpace on their robots | ⬜ Not started |
| Week 6 | Days 36–42 | First paying customer | At least 1 signed contract ($99/mo or $5K) | ⬜ Not started |

### Week 1 — Data Pipeline (Days 1–7)
**Goal: 5,500+ labeled kitchen grasps by end of week (500 real + 5,000 synthetic).**

**Day 1–2: Real Data Collection**
- Scrape YouTube: search "dishwashing robot failure", "kitchen robot arm", "restaurant robot", "robot dropping dishes", "robotic dishwasher". Target 200+ videos.
- Extract failure frames using scene-change detection (PySceneDetect). Label each with failure mode.
- Download ROS bag datasets:
  - **UMD Robot Manipulation Dataset** — 1,200 manipulation sequences
  - **Columbia Grasp Dataset** — 238 objects with stable grasp labels
  - **YCB Object Set** — 77 household objects with 3D models + grasp annotations
  - **ACRONYM** — 17,000 objects, 1.8M grasp annotations (filter kitchen subset)
  - **TaskGrasp** — task-oriented grasps (relevant: pour, place, stack)

**Day 3–4: Synthetic Data Generation**
- Run MuJoCo simulation: generate synthetic grasp attempts on ShapeNet kitchen objects (plates, mugs, glasses, utensils). Target 5,000 labeled attempts.
- Randomize conditions per attempt:
  - Object material: ceramic (dry), ceramic (wet), glass, metal, plastic
  - Surface friction: μ = 0.2 (soapy) to μ = 0.8 (dry rubber grip)
  - Object orientation: upright, tilted (15°, 30°, 45°), inverted
  - Stacking: single, 2-stack, 3-stack, nested
  - Lighting: bright, dim, specular overhead (causes depth artifacts)
- Label each attempt: `{ success: bool, failure_mode, slip_distance_mm, contact_area_cm2 }`

**Day 5: Annotation Schema & Pipeline**
- Build annotation schema:
```json
{
  "sample_id": "syn_00142",
  "source": "mujoco_sim",          // mujoco_sim | youtube | ros_bag | pilot
  "object_type": "coffee_mug",
  "object_material": "ceramic_wet",
  "grasp_point_xyz": [0.042, -0.011, 0.038],
  "grasp_normal": [0.0, 0.0, 1.0],
  "approach_vector": [0.0, -1.0, 0.0],
  "grip_width_mm": 82,
  "grip_force_n": 4.2,
  "success": false,
  "failure_mode": "slip",           // slip | collision | occlusion | soap | depth_hole | fragile_break
  "failure_detail": "wet surface reduced friction, object slipped at 12mm/s",
  "environment": {
    "wet": true,
    "soap": false,
    "steam": false,
    "lighting": "overhead_bright",
    "rack_type": "commercial_grid"
  },
  "robot_config": {
    "arm": "UR5",
    "gripper": "Robotiq_2F85",
    "camera": "RealSense_D435",
    "camera_pose": [0.5, 0.0, 0.8, 0.0, 0.785, 0.0]
  },
  "rgb_path": "s3://dishspace/data/syn_00142_rgb.png",
  "depth_path": "s3://dishspace/data/syn_00142_depth.png",
  "pointcloud_path": "s3://dishspace/data/syn_00142_pcd.ply",
  "timestamp": "2026-03-05T14:23:11Z"
}
```

**Day 6–7: Upload + Validation**
- Write Supabase upload script (Python, async batch upload)
- Validate data integrity: check for missing depth frames, mislabeled failures, duplicate samples
- Build simple data dashboard (Streamlit, internal only): show distribution of object types, failure modes, data sources
- Final count target: 500 real + 5,000 synthetic = 5,500 labeled samples

> The `failure_mode` label is critical. It's the data that nobody else has at this depth. Log every failure category meticulously — this becomes your DishBench benchmark.

### Week 2 — Baseline Model (Days 8–14)
**Goal: Fine-tuned LoRA adapter that beats GraspNet baseline on coffee mugs.**

**Day 8–9: Model Setup**
- Download OpenVLA-7b weights from HuggingFace (`openvla/openvla-7b`).
- Set up LoRA training with PEFT library:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                                    # rank — 16 is the sweet spot for kitchen
    lora_alpha=32,                           # scaling factor
    target_modules=["q_proj", "v_proj"],     # attention projections
    lora_dropout=0.05,                       # light regularization
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
print(f"Trainable params: {model.print_trainable_parameters()}")
# Expected: ~0.1% of total params trainable (LoRA efficiency)
```

**Day 10–11: Training**
- Split data: 85% train (4,675 samples) / 15% eval (825 samples)
- Ensure eval set has balanced representation: 5 object types × 6 failure modes
- Train on Modal A10: `batch_size=8, lr=2e-4, epochs=5, warmup_steps=100`
- Training time: ~3–4 hours. Cost: ~$12 on Modal.
- Monitor loss curve — should plateau by epoch 3–4. If not converging, reduce lr to 1e-4.

**Day 12–13: Evaluation**
- Test on held-out set of 200 coffee mug grasps (dry, wet, in-rack, isolated)
- Metrics to track:

| Metric | Baseline (OpenVLA) | Target (DishSpace LoRA) | Stretch Goal |
|---|---|---|---|
| Grasp success (mugs, dry) | ~72% | >90% | >95% |
| Grasp success (mugs, wet) | ~48% | >80% | >88% |
| Grasp success (plates, stacked) | ~55% | >82% | >90% |
| Grasp success (wine glass) | ~35% | >70% | >80% |
| Mean confidence calibration | ±0.25 | ±0.10 | ±0.05 |
| Inference latency (p50) | 180ms | <150ms | <100ms |
| Inference latency (p95) | 320ms | <200ms | <150ms |

**Day 14: Iteration**
- If underperforming on any category: add 500 more synthetic samples for that class, retrain overnight
- If wet surface performance lags: check depth completion pipeline, possibly switch to TransCG for that subset
- Save best checkpoint to Modal Volume: `/models/lora/kitchen_default_v0.1`

*Don't chase perfection. 85% on one object type (mugs) is enough to demo. You'll improve with customer data.*

### Week 3 — API Wrapper (Days 15–21)
**Goal: Callable `/grasp_plan` endpoint that returns ROS-compatible JSON.**

**Day 15–16: FastAPI Scaffold**
- Scaffold FastAPI app with all endpoints from §3.2
- Implement request validation with Pydantic models:
```python
class GraspRequest(BaseModel):
    image_base64: str
    depth_base64: Optional[str] = None
    kitchen_profile: str = "default"
    robot: str = "UR5_realsense"
    base_model: str = "openVLA-7b"
    options: GraspOptions = GraspOptions()

class GraspOptions(BaseModel):
    collision_check: bool = True
    max_grasps: int = 5
    min_confidence: float = 0.7
    coordinate_frame: Literal["camera", "world", "robot_base"] = "camera"
```
- Add API key authentication middleware (check against Supabase users table)
- Add rate limiting: 100 req/min free tier, 1000 req/min paid

**Day 17–18: Modal Worker Integration**
- Build Modal.com worker (see pseudocode in §4): loads LoRA adapter on A10, processes RGB-D input, runs Open3D pipeline
- Implement warm container pool: `keep_warm=1` during business hours
- Add fallback: if primary LoRA adapter fails, fall back to GraspNet baseline (always return a result)
- Build roslibpy output serializer:
```python
def to_ros_trajectory(grasps, frame="camera"):
    """Convert DishSpace grasp poses to MoveIt-compatible trajectory."""
    return {
        "header": {"frame_id": frame, "stamp": time.time()},
        "poses": [
            {
                "position": {"x": g.pose[0], "y": g.pose[1], "z": g.pose[2]},
                "orientation": rpy_to_quaternion(g.pose[3], g.pose[4], g.pose[5])
            }
            for g in grasps
        ],
        "grip_commands": [
            {"width_mm": g.grip_width, "force_n": g.grip_force, "speed": 0.1}
            for g in grasps
        ]
    }
```

**Day 19–20: Testing + Deployment**
- Write 15 integration tests:

| Test | Input | Expected |
|---|---|---|
| Single mug (dry) | RGB-D of isolated mug | ≥1 grasp, confidence >0.85 |
| Single mug (wet) | RGB-D with water film visible | ≥1 grasp, wet_surface=true |
| Plate stack (3) | RGB-D of 3 stacked plates | ≥1 grasp per plate, collision_free=true |
| Wine glass (transparent) | RGB-D with depth holes | ≥1 grasp, depth_completed=true |
| Utensil cluster | RGB-D of tangled forks/knives | ≥2 grasps, objects separated |
| Empty scene | RGB-D with no objects | empty grasp_plan, latency <50ms |
| Bad image (corrupt) | Malformed base64 | 400 error with clear message |
| Rate limit exceeded | 200 rapid requests | 429 after limit |
| Invalid API key | Wrong key | 401 error |
| Batch (5 frames) | 5 sequential frames | 5 results, total latency <1s |
| Large image (4K) | 3840×2160 RGB-D | Auto-resize, normal response |
| MuJoCo collision detected | Grasp near obstacle | confidence reduced, collision flag |
| Profile switching | Same image, 2 profiles | Different grasp selections |
| Coordinate frame transform | camera vs robot_base | Correctly transformed poses |
| Concurrent requests (10) | 10 simultaneous | All succeed within 500ms |

- Deploy to Railway. Confirm <200ms p95 latency with real RealSense frames.
- Generate public Postman collection + OpenAPI spec (automatic from FastAPI).
- Write a 1-page integration guide: "From zero to grasp plans in 5 minutes."

**Day 21: Polish**
- Add structured logging (JSON format) with request_id for debugging
- Set up Sentry for error tracking
- Create Grafana dashboard: latency histogram, success rate, error rate, daily API calls
- Document known limitations in API docs

### Week 4 — Demo UI + First Pilot Call (Days 22–28)
**Goal: Shareable demo URL. First 30-minute call with Armstrong or Sunday.**

**Day 22–24: Build Streamlit App**
- Build Streamlit app with the UI flow from §3.4
- Core features to ship:
  1. **Video/image upload** — drag-and-drop, accepts .png, .jpg, .bag, .mp4 (extract frames from video)
  2. **Grasp visualization** — overlay confidence-colored bounding boxes + grasp arrows on the frame
  3. **3D point cloud viewer** — use Plotly 3D scatter for interactive rotation/zoom
  4. **Before/after comparison** — slider showing baseline model vs DishSpace model results
  5. **Export to ROS** button — downloads trajectory JSON, copy-to-clipboard for quick paste
  6. **Share link** — generates unique URL (dishspace.ai/demo/{hash}) — no login required to view
- Performance: demo page must load in <3 seconds, analysis in <5 seconds

**Day 25: Deploy + Domain**
- Deploy to Vercel. Configure dishspace.ai domain (or dishspace.vercel.app as fallback).
- Test on mobile (demo links will be opened on phones from Discord/LinkedIn DMs)
- Add analytics: track uploads, share link clicks, JSON downloads (Posthog free tier or Vercel analytics)

**Day 26–28: Outreach Sprint**
- Find engineering leads at target companies:
  - Armstrong Robotics — check LinkedIn for "robotics engineer" + "Armstrong"
  - Sunday/Memo — Hugging Face robotics channel, LeRobot Discord
  - Dexterous Robotics — recent YC batch, check Launch HN thread for founder
- Send personalized failure audit offers:

**Outreach Templates:**

*Cold DM (Discord/LinkedIn):*
> "Saw your robot struggling with [wet mugs / stacked plates / glass depth holes]. I built a tool that fixes this — upload your failure clip here → [demo URL] and see the fix in 30 seconds. No pitch, no call needed. Just the fix."

*Follow-up (if they engage):*
> "Nice — looks like your gripper is slipping on wet ceramics at the contact point. Our LoRA adapter increased grasp confidence from 48% → 87% on wet mugs. Want me to fine-tune a model specifically for your rack configuration? Takes about 4 hours, and I'll send you the ROS trajectory JSON you can test today."

*Cold email (to engineering lead):*
> Subject: Your robot's grasp success rate on wet dishes — before & after
>
> Hi [Name],
>
> I'm building DishSpace — a fine-tuning layer that makes foundation models production-ready for kitchen manipulation.
>
> I analyzed a video of [your robot type] handling [dish type] and found [specifically what's failing]. Here's a 30-second demo showing the fix: [demo URL with pre-loaded analysis]
>
> No pitch. If the fix works, happy to chat about running a free 2-week pilot on your actual failure data.
>
> [Your name]
> DishSpace AI

- Target: send 10 personalized messages by end of Day 28. Goal: 3 responses, 1 call scheduled.

### Weeks 5–6 — Pilot Integrations + Revenue (Days 29–42)
**Goal: 3 paid or unpaid pilots with real robots. First dollar by end of week 6.**

**Week 5 (Days 29–35): Onboard Pilots**
- For each pilot customer, run this playbook:

| Step | Day | Action | Time |
|---|---|---|---|
| 1 | Day 1 | Receive failure video logs (minimum 50 clips) | 30 min |
| 2 | Day 1 | Run through data pipeline, auto-label failure modes | 1 hour |
| 3 | Day 1 | Manual review + correct any mislabeled failures | 2 hours |
| 4 | Day 1–2 | Fine-tune LoRA adapter with customer data + existing dataset | 4 hours (GPU) |
| 5 | Day 2 | Run DishBench: evaluate before vs after on customer's object types | 1 hour |
| 6 | Day 2 | Send results report: "Your grasp success improved from X% → Y%" | 30 min |
| 7 | Day 3–4 | Help integrate ROS trajectory JSON into their MoveIt pipeline | 2–4 hours |
| 8 | Day 5 | Live test on their robot (remote via screen share or on-site) | 2 hours |

- Repeat for up to 3 pilots in parallel (stagger by 2 days each)

**Week 6 (Days 36–42): Convert to Paid**
- Send pilot results + pricing proposal:

**Pricing Tiers:**

| Tier | Price | Includes | Target Customer |
|---|---|---|---|
| **Starter** | $99/mo per robot | API access, default kitchen profile, 10K calls/mo | Startups testing MVP |
| **Pro** | $199/mo per robot | Custom LoRA fine-tune, 50K calls/mo, priority support | Production deployments |
| **Enterprise** | $5K one-time + $199/mo | Dedicated fine-tune, on-prem option, DishBench report | Large fleet operators |
| **Data Partnership** | Revenue share | Free API access in exchange for labeled failure data | Data-rich partners |

- The $5K enterprise fine-tune is the high-leverage play: 4 hours of GPU time ($12 cost) for $5K revenue. 417x margin.

**DishBench Report Template (send to each pilot):**
```
┌───────────────────────────────────────────────────┐
│  DishBench Results — [Customer Name]               │
│  Date: [Date]   Profile: [Profile Name]            │
├───────────────────────────────────────────────────┤
│                                                    │
│  OVERALL IMPROVEMENT                               │
│  Baseline:    62% grasp success (OpenVLA default)  │
│  DishSpace:   91% grasp success (+29% absolute)    │
│                                                    │
│  BY CATEGORY                                       │
│  ┌──────────────┬──────────┬──────────┬─────────┐ │
│  │ Object       │ Before   │ After    │ Δ       │ │
│  ├──────────────┼──────────┼──────────┼─────────┤ │
│  │ Mugs (dry)   │ 72%      │ 94%      │ +22%    │ │
│  │ Mugs (wet)   │ 48%      │ 87%      │ +39%    │ │
│  │ Plates       │ 68%      │ 93%      │ +25%    │ │
│  │ Wine glass   │ 35%      │ 78%      │ +43%    │ │
│  │ Utensils     │ 61%      │ 88%      │ +27%    │ │
│  └──────────────┴──────────┴──────────┴─────────┘ │
│                                                    │
│  FAILURE MODES REMAINING                           │
│  • Soap occlusion on glass: 2/10 failures          │
│  • Steam-heavy environment: needs more data        │
│                                                    │
│  RECOMMENDATION                                    │
│  Continue Pro tier. Collect 100+ more failures     │
│  for soap scenarios. Next retrain ETA: 2 weeks.   │
│                                                    │
└───────────────────────────────────────────────────┘
```

- Start YC S26 application. Key metrics to present:
  - **Traction:** 3 pilots, 1 paying customer
  - **Data moat:** 5,500+ labeled samples, 200+ real deployment failures
  - **Product:** Live API with <200ms latency, DishBench published
  - **Revenue path:** $5K enterprise fine-tune × 10 customers = $50K near-term

---

## 6. The Data Flywheel (Your Real Moat)

Every pilot generates more failure data. More failure data improves the fine-tuning. Better fine-tuning attracts more pilots. This is the loop that makes DishSpace defensible — not the model architecture.

```
  New customer pilot
        │
        ▼
  Failure logs collected  ──────────────────────────────┐
        │                                               │
        ▼                                               │
  Fine-tune improves  ──►  Better grasp success  ──►  More pilots
        │                                               │
        ▼                                               │
  DishBench score rises  ──►  Easier to sell  ──────────┘

  Physical Intelligence can't easily replicate this loop
  because they optimize for generality. You optimize for depth.
```

**What to collect from every pilot:**
- Video of every grasp failure (minimum 50 per pilot deployment)
- Failure mode labels: slip, collision, occlusion, incorrect pose estimate, depth artifact, fragile break
- Object types: plate sizes (6", 8", 10", 12"), glass shapes (tumbler, wine, champagne, pint), utensil variants (fork, knife, spoon, spatula, ladle)
- Environment conditions: lighting (lux reading if possible), soap/water presence, steam level, rack configuration (grid spacing, tilt angle)
- Robot telemetry: joint positions at failure moment, gripper force readings, camera exposure settings
- Temporal data: time-of-day patterns (steam increases during peak hours), degradation over shift

> This dataset is the thing a company like Physical Intelligence or Skild AI would pay $2M–$10M to acquire. It cannot be generated synthetically at the quality of real deployment data.

### Data Growth Projections

| Milestone | Real Samples | Synthetic Samples | Total | Source |
|---|---|---|---|---|
| Week 1 | 500 | 5,000 | 5,500 | YouTube + ROS bags + MuJoCo |
| Week 6 | 650 | 5,000 | 5,650 | + 3 pilot deployments |
| Month 3 | 2,000 | 8,000 | 10,000 | + 10 customers |
| Month 6 | 10,000 | 15,000 | 25,000 | Data partnership deals |
| Month 12 | 50,000 | 25,000 | 75,000 | Industry standard dataset |

**Data monetization paths:**
1. **Primary:** Improve DishSpace's own models (direct product improvement)
2. **Secondary:** License anonymized dataset to research labs ($50K–$200K deals)
3. **Strategic:** Dataset as acquisition leverage (Physical Intelligence, Skild, Google DeepMind)
4. **Community:** Publish DishBench subset open-source (marketing + credibility)

---

## 7. Go-to-Market

### Market Sizing

| Segment | TAM | SAM (Addressable Now) | SOM (6-month target) |
|---|---|---|---|
| Commercial kitchen robotics | $6.2B by 2028 | $180M (manipulation software layer) | $500K ARR |
| Restaurant automation | $3.4B by 2027 | $90M (grasp planning subsystem) | $200K ARR |
| Home robot dishwashing | $1.1B by 2029 | $40M (manipulation fine-tuning) | $50K ARR |
| **Total** | **$10.7B** | **$310M** | **$750K ARR** |

*Source: Grand View Research, Mordor Intelligence, internal estimates. SAM assumes 3–5% software attach rate on hardware deployments.*

### Target Customers (Weeks 1–6)

| Company | Why They Need You | Entry Point | Deal Size |
|---|---|---|---|
| Armstrong Robotics | 12 deployments, building a platform — need better grasp layer | Upload failure video → show fix | $99/mo + $5K fine-tune |
| Sunday / Memo | 50-home beta, fragile objects, home variety = hard | Discord cold DM with demo link | $99/mo per robot |
| Dexterous Robotics | Dishwashing-specific, need manipulation software | YC batch outreach | $5K fine-tune + $199/mo |
| Dishcraft | Commercial conveyor dishwashing, stacked plate handling | Contact ops manager | $5K setup + $199/mo |
| Restaurant chain pilot | Standardized racks = easy win, visible ROI | Contact ops manager, show labor cost math | $5K setup + $199/mo |

### Expanded Pipeline (Months 2–6)

| Company / Segment | Why | Entry Strategy | Estimated Deal |
|---|---|---|---|
| Bear Robotics | 1000+ restaurant deployments, adding manipulation | Partnership/integration | $10K+ license |
| Miso Robotics (Flippy) | Adjacent kitchen domain, could expand to dishware | Inbound from DishBench publication | $5K fine-tune |
| Samsung / LG (home robots) | R&D labs exploring kitchen manipulation | Conference demo (ICRA/CoRL) | $20K+ research license |
| Toyota Research Institute | Household robotics research program | Academic partnership | Data exchange + grant |
| Amazon (Sparrow team) | Warehouse → kitchen lateral expansion | Talent/data acquisition target | Acqui-hire conversation |
| Robot integrators (Universal Robots+, KUKA partners) | Sell to their end customers | Reseller partnership | Revenue share |

### Acquisition Playbook

**Step 1: The Failure Audit (Week 4)**
This is your outbound motion. One sentence: "Upload your robot failure video here. I'll show you a fix in 30 seconds." No pitch. No deck. Just the fix. If the demo works, they'll ask what it costs.

**Step 2: Discord / LinkedIn Scrape (Weeks 1–4)**
- Join: ROS Discourse, Hugging Face robotics channel, LeRobot Discord, r/robotics, Robotics Stack Exchange.
- Search for: "dishwashing failure", "grasp fail", "manipulation unstable", "glass knocked over", "depth holes RealSense", "wet object grasp".
- Respond with empathy + "I built something for this — want to try it?"
- Track every interaction in a simple CRM (Notion table or Airtable free tier)

**Step 2.5: Content Marketing (Weeks 2–6)**
Build credibility through technical content that attracts inbound leads:

| Content | Platform | Purpose | When |
|---|---|---|---|
| "Why kitchen robots fail at wet dishes" (blog) | Medium / personal blog | SEO + credibility | Week 2 |
| DishBench announcement + open-source eval suite | GitHub + Twitter/X | Community + authority | Week 4 |
| "LoRA fine-tuning for kitchen robotics" (tutorial) | HuggingFace blog | Developer inbound | Week 5 |
| Video: "Fixing your robot's grasp in 30 seconds" | YouTube + LinkedIn | Decision-maker eyeballs | Week 4 |
| Conference talk submission (ICRA / CoRL workshop) | Academic channels | Long-term credibility | Month 2 |
| Technical comparison: "DishSpace vs raw OpenVLA" | Twitter/X thread | Viral potential | Week 3 |

**Step 3: Conference Presence (Month 3+)**
- Submit DishBench paper to CoRL 2026 workshop or RSS 2026 workshop
- Demo booth at ROSCon 2026 (apply early)
- Attend ICRA 2026 — network with robotics decision-makers
- Goal: be the known name for "kitchen manipulation fine-tuning"

**Step 4: YC S26 App (Month 3)**
- The pitch: "We're the fine-tuning layer for kitchen robotics — the last mile between foundation models and production deployment."
- Longer version: "Kitchen robots fail 40% of the time on wet dishes. We built a LoRA fine-tuning pipeline + ROS integration that brings that to <10%. We have 3 pilots, 1 paying customer, and a proprietary dataset of 500+ real deployment failures. We're applying to YC S26."
- Required metrics by application: 3 pilots, 1 paying customer, proprietary dataset of 500+ real failure annotations.
- Acqui-hire story: Physical Intelligence and Skild AI need kitchen domain data. You're building the dataset they'd pay to own.

---

## 8. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation | Contingency |
|---|---|---|---|---|---|
| 1 | Hardware startups build manipulation in-house | Medium | High | Talk to them before building. If they're building it, pivot to the data service. | Become the data/benchmark provider instead of the API provider. |
| 2 | Real-world accuracy < synthetic accuracy | High | Medium | Never lead with accuracy numbers. Lead with "compare before/after on your failures." | Use sim-to-real transfer techniques (domain randomization, style transfer). |
| 3 | Physical Intelligence open-sources kitchen model | Low (18+ mo) | High | Your moat is the dataset and customer relationships, not the model. | Switch to data licensing model. DishBench becomes the standard. |
| 4 | 6-week timeline slips on model training | High | Medium | Week 2 target is 85% on mugs only. Don't generalize until you have a pilot. | Ship with GraspNet baseline + partial LoRA. Good enough for demo. |
| 5 | Customers won't share failure data | Medium | High | Offer anonymization + data ownership clauses. Make it easy to say yes. | Offer on-prem inference option where data never leaves their network. |
| 6 | RealSense depth quality too poor for transparent objects | Medium | Medium | Depth completion pipeline (TransCG). | Support alternative depth sensors (ZED 2, Azure Kinect). |
| 7 | Modal.com cold start latency hurts demo | Low | Medium | Keep 1 warm container during business hours. | Pre-compute results for common demo scenarios. |
| 8 | Base model (OpenVLA) gets discontinued or license changes | Low | High | Abstract model layer — support GraspNet, Octo, RT-2 as alternatives. | Multi-model architecture from day 1 (adapter pattern). |
| 9 | No PMF — customers don't care about grasp planning | Medium | Critical | **Validate before building.** Call Armstrong/Sunday in Week 0. | Pivot to adjacent problem (navigation, perception, or data labeling). |
| 10 | Competitor raises $50M+ and enters kitchen vertical | Low | High | Speed advantage — they're 12+ months behind on kitchen data. | Position for acquisition by the well-funded competitor. |

### Risk #9 is the Kill Shot

If kitchen robot companies don't consider grasp planning their #1 bottleneck, the entire thesis fails. This is why the first action item is **customer discovery, not code.**

**Validation checklist (complete before Day 8):**
- [ ] 3+ conversations with kitchen robotics engineers
- [ ] Confirmed: grasp failure rate >30% in production
- [ ] Confirmed: they don't have a dedicated team solving this
- [ ] Confirmed: they would use an external API/service for grasp planning
- [ ] Confirmed: they have RGB-D cameras on their robots
- [ ] Identified: what they're currently doing instead (heuristics? manual tuning? ignoring it?)

If 2+ of these fail, **pivot immediately** to one of:
- **Data service only** — sell labeled kitchen failure data to research labs
- **Simulation-as-a-service** — MuJoCo kitchen environments for robot companies to test in
- **Benchmark provider** — DishBench as a paid evaluation service

---

## 9. What Success Looks Like

| By When | What You Have | Why It Matters | Key Metric |
|---|---|---|---|
| Day 7 | 5,500 labeled grasps in Supabase | Data foundation for everything else | Dataset completeness |
| Week 2 | LoRA adapter beating baseline on mugs by 25%+ | Proof that fine-tuning works on kitchen data | Grasp success rate |
| Week 4 | Demo URL + API live + 1 pilot call scheduled | Product exists, someone wants to try it | Human interest |
| Week 6 | 3 pilots integrated, 1 paying customer | Proof of demand. Enough for YC app. | Revenue |
| Month 3 | 10 customers, 2,000+ real failure annotations, DishBench published | Proof of moat. Enough for seed round. | Data flywheel turning |
| Month 6 | $50K MRR or acquisition conversation | Exit or scale decision point | Unit economics |
| Month 12 | 50+ customers, 50K+ failure annotations, industry standard benchmark | Category leader in kitchen manipulation | Market position |

### Financial Projections (Conservative)

| Month | Customers | MRR | ARR Run Rate | Key Driver |
|---|---|---|---|---|
| 1 | 0 | $0 | $0 | Building |
| 2 | 1 | $5,099 | $61K | First enterprise fine-tune ($5K) + starter ($99) |
| 3 | 5 | $5,995 | $72K | 4 more starters + 1 more enterprise |
| 4 | 10 | $11,990 | $144K | Word of mouth from pilot results |
| 5 | 18 | $17,982 | $216K | DishBench publication drives inbound |
| 6 | 30 | $29,970 | $360K | Conference presence + partnerships |
| 9 | 60 | $59,940 | $719K | Seed round enables sales hire |
| 12 | 120 | $119,880 | $1.4M | Category leader position |

*Assumes average deal size of $999/mo (blended starter + pro + enterprise). Conservative churn: 5%/mo.*

---

## 10. Team & Hiring Plan

### Solo Founder Phase (Weeks 1–6)
You handle everything: model training, API development, outreach, pilot onboarding. This is the advantage — speed of iteration, zero coordination overhead, total context.

### First Hires (Month 3–6, Post-Seed)

| Role | Priority | Why | Comp Range |
|---|---|---|---|
| **ML Engineer (Robotics)** | P0 | Own the fine-tuning pipeline + model improvements | $150–$200K + equity |
| **Robotics Integration Engineer** | P0 | Own ROS bridge, MoveIt integration, on-site pilot support | $130–$170K + equity |
| **Developer Advocate / Sales Engineer** | P1 | Own outreach, demo calls, content marketing | $120–$150K + equity |
| **Data Engineer** | P2 | Own annotation pipeline, data quality, DishBench | $130–$160K + equity |

### Where to Find Them
- **ML engineers:** HuggingFace community, ICRA/CoRL attendees, LeRobot Discord
- **Robotics engineers:** ROS Discourse, Universal Robots partner network, CMU/MIT/Stanford robotics labs
- **Sales engineers:** Robotics startup alumni (Covariant, Berkshire Grey, Plus One Robotics)

---

## 11. IP & Legal Considerations

### Data Rights
- **Customer data:** Always anonymized. Customer retains ownership. DishSpace gets a license to use for model training (standard in API ToS).
- **Synthetic data:** 100% owned by DishSpace.
- **Public datasets (YCB, Columbia, etc.):** Check individual licenses. Most are CC-BY or Apache 2.0.
- **YouTube scraped data:** Use only for research/evaluation, not commercial training. Or extract metadata/descriptions only with link-back.

### Model Licensing
- **OpenVLA:** Apache 2.0 — commercial use allowed, LoRA derivatives are yours.
- **GraspNet / Contact-GraspNet:** MIT License — fully permissive.
- **MuJoCo:** Apache 2.0 (since DeepMind open-sourced it).

### Patent Landscape
- The kitchen-specific fine-tuning methodology is potentially patentable (provisional patent worth filing at Month 3).
- DishBench as a trademark — file TM application once established.
- Avoid claims about "AI accuracy" in marketing — no guaranteed performance, frame as "improvement over baseline."

---

## 12. Scaling Roadmap (Post-MVP)

### Phase 2: Multi-Domain Expansion (Month 6–12)

Once kitchen is locked down, the fine-tuning infrastructure generalizes to adjacent domains:

| Domain | Why It's Similar | Market Size | Difficulty |
|---|---|---|---|
| Healthcare instrument handling | Fragile, precise, wet surfaces | $2.1B | Medium — regulatory hurdles |
| Semiconductor wafer handling | Ultra-precise, reflective, clean room | $1.8B | High — extreme precision needed |
| Produce/food sorting | Organic shapes, variable textures, sorting speed | $3.2B | Low — lower precision requirement |
| Lab automation (pipettes, vials) | Small objects, glass, precise placement | $1.5B | Medium — established players |
| E-commerce returns processing | Variable objects, damaged packaging | $4.0B | Low — Amazon is the buyer |

### Phase 3: Platform Play (Month 12–24)

```
                    DishSpace Platform
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                  │
   Kitchen Profile   Healthcare Profile   Lab Profile
   (LoRA adapter)    (LoRA adapter)       (LoRA adapter)
        │                 │                  │
        ▼                 ▼                  ▼
   Base Model        Base Model          Base Model
   (OpenVLA)         (OpenVLA)           (OpenVLA)
```

The kitchen is the proof of concept. The platform is the unicorn.

### Phase 4: Foundation Model (Month 24+)
Once you have 100K+ real manipulation failures across domains, you have the data to train a domain-specific foundation model — not a general one, but a manipulation-specific one. This is a $100M+ company if the data flywheel works.

---

## 13. Key Decisions Log

Track every major decision to maintain velocity and avoid re-litigating:

| Decision | Date | Options Considered | Chosen | Rationale |
|---|---|---|---|---|
| Base model | TBD | OpenVLA-7b, Octo, RT-2, GraspNet | OpenVLA-7b | Best open weights, strong spatial priors, HuggingFace PEFT compatible |
| GPU provider | TBD | Modal, RunPod, Lambda, self-hosted | Modal.com | Serverless = zero idle cost, Python-native API, cheapest at low scale |
| Database | TBD | Supabase, PlanetScale, Firebase, self-hosted Postgres | Supabase | Postgres + auth + file storage in one, good free tier |
| API host | TBD | Railway, Fly.io, Render, self-hosted | Railway | Simplest deploy, auto-SSL, good enough for MVP |
| Fine-tuning approach | TBD | Full retrain, LoRA, QLoRA, prefix tuning | LoRA (rank=16) | Best quality/cost ratio, $15/run vs $2K+ for full retrain |
| Demo framework | TBD | Streamlit, Gradio, Next.js, custom React | Streamlit | Fastest to ship, adequate for pilot phase |
| Pricing | TBD | Per-call, per-robot/mo, enterprise one-time | Tiered (see §5) | Enterprise fine-tune ($5K) is the high-leverage play |

---

> **The single most important action: contact Armstrong or Sunday this week before writing a single line of code.** Confirm that grasp planning is actually their bottleneck. One 20-minute call de-risks the entire 6-week build.

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning method that trains small adapter weights instead of the full model |
| **RGB-D** | Color image + depth map (Red, Green, Blue + Depth) — typically from Intel RealSense or similar |
| **ROS** | Robot Operating System — middleware framework for robot software development |
| **MoveIt** | Motion planning framework for ROS — generates collision-free trajectories for robot arms |
| **PEFT** | Parameter-Efficient Fine-Tuning — HuggingFace library for LoRA and similar techniques |
| **OpenVLA** | Open Vision-Language-Action model — 7B parameter model for robot manipulation |
| **GraspNet** | Neural network for 6-DOF grasp detection in cluttered scenes |
| **MuJoCo** | Multi-Joint dynamics with Contact — physics simulator for robotics |
| **Open3D** | Library for 3D data processing (point clouds, meshes, visualization) |
| **Kitchen Profile** | A LoRA adapter fine-tuned on a specific customer's kitchen configuration |
| **DishBench** | DishSpace's proprietary 50-scenario evaluation benchmark for kitchen manipulation |
| **URDF** | Unified Robot Description Format — XML file describing robot geometry and joints |

## Appendix B: Reference Links

| Resource | URL | Purpose |
|---|---|---|
| OpenVLA | https://huggingface.co/openvla/openvla-7b | Base model weights |
| HuggingFace PEFT | https://github.com/huggingface/peft | LoRA fine-tuning library |
| Modal.com | https://modal.com | Serverless GPU inference |
| FastAPI | https://fastapi.tiangolo.com | API framework |
| Open3D | http://www.open3d.org | Point cloud processing |
| MuJoCo | https://mujoco.org | Physics simulation |
| Supabase | https://supabase.com | Database + auth + storage |
| roslibpy | https://github.com/gramaziokohler/roslibpy | Python ROS bridge |
| YCB Dataset | https://www.ycbbenchmarks.com | Object 3D models |
| Columbia Grasp | https://graspdataset.appspot.com | Grasp annotations |
| Contact-GraspNet | https://github.com/NVlabs/contact_graspnet | 6-DOF grasp generation |
| TransCG | https://github.com/PKU-EPIC/TransCG | Transparent object depth completion |

---
*DishSpace AI · MVP Build Plan · Confidential*