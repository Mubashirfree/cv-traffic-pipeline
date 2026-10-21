"""
Phase 2: Named polygon zones in world coordinates + point-in-polygon tagging.
"""

import json
from pathlib import Path
from shapely.geometry import Point, Polygon


# Zones defined in world coordinates (meters), consistent with the
# 0-3.5 (lane width) x 0-20 (distance) calibration rectangle used in homography.py
ZONES = {
    "Near_Zone":  Polygon([(0, 0), (3.5, 0), (3.5, 7), (0, 7)]),
    "Mid_Zone":   Polygon([(0, 7), (3.5, 7), (3.5, 14), (0, 14)]),
    "Far_Zone":   Polygon([(0, 14), (3.5, 14), (3.5, 20), (0, 20)]),
}


def get_zone(world_xy):
    """Returns zone name string, or None if outside all zones."""
    if world_xy is None:
        return None
    pt = Point(world_xy)
    for name, poly in ZONES.items():
        if poly.contains(pt):
            return name
    return None


if __name__ == "__main__":
    tests = [(1.5, 3), (2.0, 10), (0.5, 18), (10, 10)]
    for t in tests:
        print(t, "->", get_zone(t))


def get_zone_polygons():
    """Returns {zone_name: [(x,y), ...]} for storage/DB insertion."""
    return {name: list(poly.exterior.coords) for name, poly in ZONES.items()}
