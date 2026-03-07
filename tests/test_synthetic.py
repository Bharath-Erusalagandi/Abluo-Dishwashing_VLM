"""Tests for synthetic data generator."""

import pytest
from src.data.synthetic_generator import generate_batch, generate_balanced_batch, generate_synthetic_sample, KITCHEN_OBJECTS


class TestSyntheticGenerator:
    """Synthetic data generation tests."""

    def test_single_sample(self):
        import numpy as np

        rng = np.random.default_rng(42)
        sample = generate_synthetic_sample(rng)
        assert sample is not None
        assert sample.object_type is not None
        assert sample.grasp_point_xyz is not None
        assert len(sample.grasp_point_xyz) == 3

    def test_batch_generation(self):
        samples = generate_batch(count=50, seed=42)
        assert len(samples) == 50

    def test_balanced_batch_covers_object_types(self):
        samples = generate_balanced_batch(count=len(KITCHEN_OBJECTS), seed=42)
        seen = {sample.object_type for sample in samples}
        assert seen == set(KITCHEN_OBJECTS.keys())

    def test_deterministic_seed(self):
        s1 = generate_batch(count=10, seed=123)
        s2 = generate_batch(count=10, seed=123)
        for a, b in zip(s1, s2):
            assert a.object_type == b.object_type
            assert a.success == b.success

    def test_has_success_and_failure(self):
        """A batch should contain both successes and failures."""
        samples = generate_batch(count=200, seed=42)
        successes = sum(1 for s in samples if s.success)
        failures = len(samples) - successes
        assert successes > 0
        assert failures > 0

    def test_kitchen_objects_valid(self):
        """Verify the kitchen objects config is well-formed."""
        for name, props in KITCHEN_OBJECTS.items():
            assert "size" in props
            assert "mass" in props
            assert "friction_dry" in props
            assert "grasp_types" in props
            assert len(props["grasp_types"]) > 0

    def test_sample_fields_populated(self):
        """Verify all critical fields are populated."""
        import numpy as np

        rng = np.random.default_rng(99)
        sample = generate_synthetic_sample(rng)
        assert sample.source.value == "mujoco_sim"
        assert sample.sample_id.startswith("syn_")
        assert sample.grip_width_mm > 0
        assert sample.grip_force_n > 0
        assert sample.environment is not None
        assert sample.robot_config is not None
        assert sample.environment.visible_object_count >= 1
        assert 0.0 <= sample.environment.occlusion_level <= 1.0
