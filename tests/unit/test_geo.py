"""Haversine distance: known values, and scalar and array forms agree."""

import math

import numpy as np
import pytest

from fraudlens.features.geo import EARTH_RADIUS_KM, haversine_km, haversine_km_array


def test_zero_distance_to_itself() -> None:
    assert haversine_km(40.0, -74.0, 40.0, -74.0) == 0.0


def test_one_degree_of_latitude() -> None:
    assert haversine_km(40.0, -74.0, 41.0, -74.0) == pytest.approx(EARTH_RADIUS_KM * math.pi / 180)


def test_new_york_to_los_angeles() -> None:
    assert haversine_km(40.7128, -74.0060, 34.0522, -118.2437) == pytest.approx(3_936, abs=2)


def test_antipodes_are_half_the_circumference() -> None:
    assert haversine_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(EARTH_RADIUS_KM * math.pi)


def test_distance_is_symmetric() -> None:
    there = haversine_km(36.08, -81.18, 48.89, -118.21)
    back = haversine_km(48.89, -118.21, 36.08, -81.18)

    assert there == pytest.approx(back, rel=1e-12)


def test_nan_coordinate_gives_nan() -> None:
    assert math.isnan(haversine_km(math.nan, -74.0, 41.0, -74.0))


def test_array_form_matches_scalar_form() -> None:
    rng = np.random.default_rng(0)
    lat1, lat2 = rng.uniform(-90, 90, (2, 500))
    long1, long2 = rng.uniform(-180, 180, (2, 500))
    lat1[0] = np.nan

    batch = haversine_km_array(lat1, long1, lat2, long2)
    scalar = [haversine_km(*point) for point in zip(lat1, long1, lat2, long2, strict=True)]

    np.testing.assert_allclose(batch, scalar, rtol=1e-12, equal_nan=True)
