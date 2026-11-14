"""Independent (from-first-principles) geometry for the geo-dedup verification.

Box = (lon_min, lat_min, lon_max, lat_max), edges are parallels (const lat) and meridians
(const lon). Gap between two boxes on the sphere = 0 if they overlap, else the minimum
over the 8 vertices of the exact point-to-other-box distance. Justification: the closest
pair lies on the boundaries; for parallel-parallel edges the minimum is at a shared or
nearest longitude which is a vertex of one of the boxes; for parallel-meridian the
minimum on the meridian side is at an endpoint unless the boxes overlap; for
meridian-meridian f(pa,pb)=sin pa sin pb + cos pa cos pb cos(dl) has no interior maximum
except at the equator (a minimum of f), so the maximum is on the segment endpoints.
Point-to-parallel-segment: closest longitude is the clamp (cos d is monotone in |dlon|).
Point-to-meridian-segment: cos d = A cos(phi - phi*) with phi* = atan2(sin pP, cos pP cos dl);
evaluate at the segment ends and at phi*, phi*+-180 when inside the segment.
"""

import math
import numpy as np

R = 6_371_008.8  # same IUGG mean radius as the code under test (so metres are comparable)


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def tile_bounds(z, x, y):
    """Standard OSM/XYZ Web-Mercator tile bounds (lon_min, lat_min, lon_max, lat_max)."""
    n = 2**z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bot = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_min, lat_bot, lon_max, lat_top


def _pt_parallel(plat, plon, lat0, lon1, lon2):
    """point -> parallel segment at lat0 spanning [lon1, lon2] (arrays broadcast)."""
    lon_c = np.clip(plon, lon1, lon2)
    return hav(plat, plon, lat0, lon_c)


def _pt_meridian(plat, plon, lon0, lat1, lat2):
    """point -> meridian segment at lon0 spanning [lat1, lat2]."""
    pP = np.radians(plat)
    dl = np.radians(plon - lon0)
    phi_s = np.degrees(np.arctan2(np.sin(pP), np.cos(pP) * np.cos(dl)))
    best = np.minimum(hav(plat, plon, lat1, lon0), hav(plat, plon, lat2, lon0))
    for cand in (phi_s, phi_s - 180.0, phi_s + 180.0):
        inside = (cand >= lat1) & (cand <= lat2)
        d = hav(plat, plon, np.clip(cand, lat1, lat2), lon0)
        best = np.where(inside, np.minimum(best, d), best)
    return best


def pt_box(plat, plon, b):
    """exact point -> box distance; 0 when inside. b = tuple of 4 broadcastable arrays."""
    lon1, lat1, lon2, lat2 = b
    inside = (plat >= lat1) & (plat <= lat2) & (plon >= lon1) & (plon <= lon2)
    d = np.minimum.reduce(
        [
            _pt_parallel(plat, plon, lat1, lon1, lon2),
            _pt_parallel(plat, plon, lat2, lon1, lon2),
            _pt_meridian(plat, plon, lon1, lat1, lat2),
            _pt_meridian(plat, plon, lon2, lat1, lat2),
        ]
    )
    return np.where(inside, 0.0, d)


def _gap_noshift(a, b):
    alon1, alat1, alon2, alat2 = a
    blon1, blat1, blon2, blat2 = b
    overlap = (alat1 <= blat2) & (blat1 <= alat2) & (alon1 <= blon2) & (blon1 <= alon2)
    best = np.inf
    for la, lo in ((alat1, alon1), (alat1, alon2), (alat2, alon1), (alat2, alon2)):
        best = np.minimum(best, pt_box(la, lo, b))
    for la, lo in ((blat1, blon1), (blat1, blon2), (blat2, blon1), (blat2, blon2)):
        best = np.minimum(best, pt_box(la, lo, a))
    return np.where(overlap, 0.0, best)


def box_gap(a, b):
    """great-circle gap in metres between boxes a and b (each 4 broadcastable arrays of
    lon_min, lat_min, lon_max, lat_max); b is also tried shifted by +-360 deg of longitude."""
    a = tuple(np.asarray(v, dtype=float) for v in a)
    b = tuple(np.asarray(v, dtype=float) for v in b)
    best = np.inf
    for k in (-360.0, 0.0, 360.0):
        bb = (b[0] + k, b[1], b[2] + k, b[3])
        best = np.minimum(best, _gap_noshift(a, bb))
    return best


def box_gap_bruteforce(a, b, n=400):
    """slow reference: dense boundary sampling (n points per edge) of both boxes."""

    def boundary(box):
        lon1, lat1, lon2, lat2 = box
        t = np.linspace(0, 1, n)
        lons = lon1 + (lon2 - lon1) * t
        lats = lat1 + (lat2 - lat1) * t
        pts = [
            (np.full(n, lat1), lons),
            (np.full(n, lat2), lons),
            (lats, np.full(n, lon1)),
            (lats, np.full(n, lon2)),
        ]
        return np.concatenate([p[0] for p in pts]), np.concatenate([p[1] for p in pts])

    la, lo = boundary(a)
    lb, lob = boundary(b)
    overlap = (a[1] <= b[3]) and (b[1] <= a[3]) and (a[0] <= b[2]) and (b[0] <= a[2])
    if overlap:
        return 0.0
    return float(hav(la[:, None], lo[:, None], lb[None, :], lob[None, :]).min())


if __name__ == "__main__":
    # self-test: random box pairs, exact vs dense sampling; and tile bounds vs known values
    rng = np.random.default_rng(0)
    worst = 0.0
    n_ok = 0
    for i in range(400):
        lat = rng.uniform(-80, 80)
        lon = rng.uniform(-179, 179)

        def mk(lat, lon, half_m):
            dlat = math.degrees(half_m / R)
            dlon = dlat / math.cos(math.radians(lat))
            return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

        a = mk(lat, lon, rng.choice([150, 320, 1320, 25000]))
        off_m = rng.uniform(0, 60000)
        ang = rng.uniform(0, 2 * math.pi)
        lat2 = lat + math.degrees(off_m * math.sin(ang) / R)
        lon2 = lon + math.degrees(off_m * math.cos(ang) / R) / math.cos(math.radians(lat))
        b = mk(lat2, lon2, rng.choice([12.8, 320, 640, 3840, 25000]))
        g = float(box_gap(a, b))
        ref = box_gap_bruteforce(a, b, n=600)
        # dense sampling overestimates by at most ~ s^2/(8d); tolerance scaled accordingly
        side = max(a[3] - a[1], b[3] - b[1]) * 111_000 * 2
        s = side / 600
        tol = 0.5 + (s * s / (8 * max(ref, 1.0)))
        if abs(g - ref) <= tol and g <= ref + 1e-6:
            n_ok += 1
        worst = max(worst, abs(g - ref))
    print(f"self-test exact-vs-dense: {n_ok}/400 within tolerance, worst |diff| = {worst:.3f} m")
    # tile bounds: z=0 whole world, z=1 x=0 y=0 is the NW quadrant
    print("tile z0:", tile_bounds(0, 0, 0))
    print("tile z1 0 0:", tile_bounds(1, 0, 0))
