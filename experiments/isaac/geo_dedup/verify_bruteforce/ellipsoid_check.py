"""How far the spherical great-circle gap (eval.geo_dedup.box_gap_m, IUGG mean radius) is from
the WGS84 geodesic at the 1,000 m decision boundary.

For latitudes 0..70 deg and both separations (north-south and east-west) two 640 m boxes are
placed so that box_gap_m reports exactly 1000 m, and the WGS84 geodesic distance (Vincenty
inverse, own implementation, converged to 1e-12) between the same facing edge points is
printed. A positive error means the ellipsoid distance is LARGER than the sphere said: a true
1000+x m gap reads 1000 m and is excluded (over-exclusion, conservative). A negative error
means the sphere OVER-reads: a true 1000-x m gap reads 1000 m, so a pair slightly inside the
buffer can be retained (the leak side). Run: python experiments/isaac/geo_dedup/verify_bruteforce/ellipsoid_check.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from eval.geo_dedup import EARTH_R, box_gap_m  # noqa: E402

A, F = 6378137.0, 1 / 298.257223563
B = (1 - F) * A


def vincenty_m(lat1, lon1, lat2, lon2):
    phi1, phi2, L = map(math.radians, (lat1, lat2, lon2 - lon1))
    U1, U2 = math.atan((1 - F) * math.tan(phi1)), math.atan((1 - F) * math.tan(phi2))
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(200):
        sl, cl = math.sin(lam), math.cos(lam)
        ss = math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
        if ss == 0:
            return 0.0
        cs = sU1 * sU2 + cU1 * cU2 * cl
        sig = math.atan2(ss, cs)
        sa = cU1 * cU2 * sl / ss
        c2a = 1 - sa * sa
        c2sm = cs - 2 * sU1 * sU2 / c2a if c2a else 0.0
        C = F / 16 * c2a * (4 + F * (4 - 3 * c2a))
        lam_new = L + (1 - C) * F * sa * (sig + C * ss * (c2sm + C * cs * (-1 + 2 * c2sm * c2sm)))
        if abs(lam_new - lam) < 1e-12:
            lam = lam_new
            break
        lam = lam_new
    else:
        raise RuntimeError("vincenty did not converge")
    u2 = c2a * (A * A - B * B) / (B * B)
    Ak = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    Bk = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    ds = (
        Bk
        * ss
        * (
            c2sm
            + Bk
            / 4
            * (cs * (-1 + 2 * c2sm * c2sm) - Bk / 6 * c2sm * (-3 + 4 * ss * ss) * (-3 + 4 * c2sm * c2sm))
        )
    )
    return B * Ak * (sig - ds)


def box(lat0, lon0, side_m=640.0):
    dlat = math.degrees(side_m / EARTH_R)
    dlon = math.degrees(side_m / (EARTH_R * math.cos(math.radians(lat0 + dlat / 2))))
    return {"lat_min": lat0, "lat_max": lat0 + dlat, "lon_min": lon0, "lon_max": lon0 + dlon}


def main():
    gap = 1000.0
    rows = []
    for lat in range(0, 71, 5):
        a = box(float(lat), 10.0)
        # north-south: b starts gap metres north of a's top edge (spherical arc)
        b = box(a["lat_max"] + math.degrees(gap / EARTH_R), 10.0)
        g = float(box_gap_m(a, {k: np.array([v]) for k, v in b.items()})[0])
        true = vincenty_m(a["lat_max"], a["lon_min"], b["lat_min"], a["lon_min"])
        rows.append(("N-S", lat, g, true))
        # east-west: b starts gap metres east of a's top (poleward) edge, where the gap is smallest
        dlon = math.degrees(gap / (EARTH_R * math.cos(math.radians(a["lat_max"]))))
        b = box(float(lat), a["lon_max"] + dlon)
        g = float(box_gap_m(a, {k: np.array([v]) for k, v in b.items()})[0])
        true = vincenty_m(a["lat_max"], a["lon_max"], b["lat_max"], b["lon_min"])
        rows.append(("E-W", lat, g, true))
    print("dir  lat  sphere_gap_m  wgs84_gap_m  error_m (wgs84 - sphere)")
    for d, lat, g, t in rows:
        print(f"{d:4s} {lat:3d}  {g:12.3f}  {t:11.3f}  {t - g:+8.3f}")
    errs = np.array([t - g for _, _, g, t in rows])
    assert all(abs(g - gap) < 1e-3 for _, _, g, _ in rows), "box_gap_m did not report 1000 m"
    print(
        f"max |error| {np.abs(errs).max():.3f} m; max leak (wgs84 < sphere) {min(errs.min(), 0):+.3f} m; "
        f"max over-exclude (wgs84 > sphere) {max(errs.max(), 0):+.3f} m"
    )
    print("ELLIPSOID_CHECK_DONE")


if __name__ == "__main__":
    main()
