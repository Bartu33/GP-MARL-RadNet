import numpy as np
import pytest

from src.core.target_classes import (CLASS_ORDER, RCS_ASPECTS, TARGET_CLASSES, VariantSpec,
                                     rcs_model, sample_instance, variant_spec)


def test_three_classes_with_distinct_labels():
    assert set(CLASS_ORDER) == {"uav", "aircraft", "missile"}
    assert sorted(TARGET_CLASSES[c]["label"] for c in CLASS_ORDER) == [0, 1, 2]


@pytest.mark.parametrize("cname", ["uav", "aircraft", "missile"])
def test_samples_stay_inside_hard_envelope(cname):
    rng = np.random.default_rng(1)
    for _ in range(500):
        inst = sample_instance(cname, rng)
        s = variant_spec(cname, inst["variant"])
        assert s.speed_range_mps[0] <= inst["speed_mps"] <= s.speed_range_mps[1]
        assert s.altitude_agl_range_m[0] <= inst["altitude_agl_m"] <= s.altitude_agl_range_m[1]
        assert abs(inst["climb_rate_mps"]) <= s.max_climb_rate_mps
        for a in RCS_ASPECTS:
            assert s.rcs_dbsm[a][0] <= inst["rcs_dbsm"][a] <= s.rcs_dbsm[a][1]


def test_rcs_ordering_broadside():
    """Broadside mean RCS: aircraft > missile > UAV (median over samples)."""
    rng = np.random.default_rng(2)
    side = np.array([0.0, 1.0, 0.0])
    med = {c: np.median([rcs_model(sample_instance(c, rng)).mean_rcs(side) for _ in range(200)])
           for c in CLASS_ORDER}
    assert med["aircraft"] > med["missile"] > med["uav"]


def test_inconsistent_turn_rate_rejected():
    base = variant_spec("missile", "subsonic_cruise")
    with pytest.raises(ValueError):
        VariantSpec(**{**base.__dict__, "max_turn_rate_rad_s": 1.0})
