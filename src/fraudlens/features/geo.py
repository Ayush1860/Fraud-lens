"""Great-circle distance on a spherical Earth."""

import math

import numpy as np
import numpy.typing as npt

EARTH_RADIUS_KM = 6371.0088
"""Mean Earth radius (IUGG), in kilometres."""

FloatArray = npt.NDArray[np.float64]


def haversine_km(lat1: float, long1: float, lat2: float, long2: float) -> float:
    """Return the great-circle distance in kilometres between two points.

    Args:
        lat1: Latitude of the first point, in degrees.
        long1: Longitude of the first point, in degrees.
        lat2: Latitude of the second point, in degrees.
        long2: Longitude of the second point, in degrees.

    Returns:
        Distance in kilometres, or NaN if any coordinate is NaN.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    half_dphi = (phi2 - phi1) / 2.0
    half_dlambda = math.radians(long2 - long1) / 2.0
    a = math.sin(half_dphi) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(half_dlambda) ** 2
    if math.isnan(a):
        return math.nan
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def haversine_km_array(
    lat1: FloatArray, long1: FloatArray, lat2: FloatArray, long2: FloatArray
) -> FloatArray:
    """Vectorised :func:`haversine_km`: the same formula, element by element.

    All inputs are in degrees and must have the same shape. NaN inputs give NaN outputs.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    half_dphi = (phi2 - phi1) / 2.0
    half_dlambda = np.radians(long2 - long1) / 2.0
    a = np.sin(half_dphi) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(half_dlambda) ** 2
    distance: FloatArray = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))
    return distance
