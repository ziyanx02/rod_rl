import numpy as np
from typing import Tuple, List, Dict, Optional, Literal

PlaneAxis = Literal["x", "y"]

# ---------- existing helpers from earlier answer (kept for completeness) ----------

def _detect_plane_axis_and_value(ring_pts: np.ndarray, eps: float = 1e-9) -> Tuple[PlaneAxis, float]:
    ring = np.asarray(ring_pts, dtype=float)
    xspan = ring[:,0].max() - ring[:,0].min()
    yspan = ring[:,1].max() - ring[:,1].min()
    if xspan <= yspan and xspan <= eps:
        return "x", float(ring[:,0].mean())
    if yspan < xspan and yspan <= eps:
        return "y", float(ring[:,1].mean())
    if xspan < 1e-7 * max(1.0, yspan):
        return "x", float(ring[:,0].mean())
    if yspan < 1e-7 * max(1.0, xspan):
        return "y", float(ring[:,1].mean())
    raise ValueError("Ring vertices are not (numerically) constant in x or y.")

def _point_in_polygon_2d(p: np.ndarray, poly: np.ndarray, eps: float = 1e-12) -> bool:
    M = poly.shape[0]
    px, py = float(p[0]), float(p[1])
    inside = False
    for i in range(M):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % M]
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        cross = vx * wy - vy * wx
        dot = vx * wx + vy * wy
        seglen2 = vx*vx + vy*vy
        if abs(cross) <= eps * max(1.0, np.sqrt(seglen2)) and -eps <= dot <= seglen2 + eps:
            return True
        if (y1 > py) != (y2 > py):
            x_int = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x_int >= px - eps:
                inside = not inside
    return inside

def _project_ring_to_2d(ring_pts: np.ndarray, axis: PlaneAxis) -> np.ndarray:
    ring = np.asarray(ring_pts, dtype=float)
    return ring[:, [1, 2]] if axis == "x" else ring[:, [0, 2]]

def _segment_plane_intersection_param(a: float, b: float, c: float, eps: float = 1e-12) -> Optional[float]:
    da, db = a - c, b - c
    if abs(da) < eps and abs(db) < eps:
        return None
    if (da > eps and db > eps) or (da < -eps and db < -eps):
        return None
    denom = (a - b)
    if abs(denom) < eps:
        return None
    t = (a - c) / denom
    if t < -1e-10 or t > 1 + 1e-10:
        return None
    return float(np.clip(t, 0.0, 1.0))

def ring_crossing_count_axis_aligned(
    rope_pts: np.ndarray,
    ring_pts: np.ndarray,
    eps_plane: float = 1e-9
) -> Tuple[int, List[Tuple[int, float, np.ndarray]]]:
    rope = np.asarray(rope_pts, dtype=float)
    ring = np.asarray(ring_pts, dtype=float)
    axis, c = _detect_plane_axis_and_value(ring, eps_plane)
    ring2d = _project_ring_to_2d(ring, axis)

    count = 0
    hits: List[Tuple[int, float, np.ndarray]] = []
    for i in range(len(rope) - 1):
        p0, p1 = rope[i], rope[i+1]
        if axis == "x":
            t = _segment_plane_intersection_param(p0[0], p1[0], c, eps_plane)
            if t is None: continue
            x = p0 + t * (p1 - p0)
            x2d = np.array([x[1], x[2]])  # (y,z)
        else:
            t = _segment_plane_intersection_param(p0[1], p1[1], c, eps_plane)
            if t is None: continue
            x = p0 + t * (p1 - p0)
            x2d = np.array([x[0], x[2]])  # (x,z)
        if _point_in_polygon_2d(x2d, ring2d):
            count += 1
            hits.append((i, t, x))
    return count, hits

def rope_passes_through_ring_axis_aligned(rope_pts: np.ndarray, ring_pts: np.ndarray, eps_plane: float = 1e-9) -> bool:
    count, _ = ring_crossing_count_axis_aligned(rope_pts, ring_pts, eps_plane)
    return (count % 2) == 1

# ---------- NEW: closest distance from rope to ring center ----------

def _closest_point_on_segment_to_point(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Returns (distance, t, closest_point) from point P to segment AB,
    where closest_point = A + t*(B-A) with t in [0,1].
    """
    AB = B - A
    AP = P - A
    denom = AB.dot(AB)
    if denom == 0.0:
        # Degenerate segment; treat as a point
        return float(np.linalg.norm(AP)), 0.0, A.copy()
    t = float(np.clip(AP.dot(AB) / denom, 0.0, 1.0))
    Q = A + t * AB
    return float(np.linalg.norm(P - Q)), t, Q

def closest_distance_rope_to_point(
    rope_pts: np.ndarray,
    point: np.ndarray
) -> Tuple[float, int, float, np.ndarray]:
    """
    Minimum Euclidean distance from polyline 'rope_pts' to 'point'.
    Returns (min_dist, segment_index, t, closest_point_on_segment).
    """
    rope = np.asarray(rope_pts, dtype=float)
    P = np.asarray(point, dtype=float)
    if rope.ndim != 2 or rope.shape[1] != 3 or len(rope) < 1:
        raise ValueError("rope_pts must be (N,3)")
    if P.shape != (3,):
        raise ValueError("point must be shape (3,)")

    best = (float("inf"), -1, 0.0, rope[0])
    if len(rope) == 1:
        d = float(np.linalg.norm(P - rope[0]))
        return d, -1, 0.0, rope[0].copy()

    for i in range(len(rope) - 1):
        d, t, Q = _closest_point_on_segment_to_point(P, rope[i], rope[i+1])
        if d < best[0]:
            best = (d, i, t, Q)
    return best

def ring_center_from_axis_aligned_vertices(ring_pts: np.ndarray) -> np.ndarray:
    """
    For an axis-aligned ring (x=c or y=c), a robust center is the average of vertices.
    If your data guarantees a perfect circle, you can replace this with a circle fit.
    """
    return np.asarray(ring_pts, dtype=float).mean(axis=0)

def rope_ring_analysis_axis_aligned(
    rope_pts: np.ndarray,
    ring_pts: np.ndarray,
    eps_plane: float = 1e-9,
) -> Dict:
    """
    All-in-one helper:
      - whether rope passes through ring (odd piercings),
      - how many piercings and where,
      - closest distance from rope to ring center.
    """
    # Pass-through analysis
    crossing_count, hits = ring_crossing_count_axis_aligned(rope_pts, ring_pts, eps_plane)
    passes = (crossing_count % 2) == 1

    # Closest distance to ring center
    C = ring_center_from_axis_aligned_vertices(ring_pts)
    min_dist, seg_idx, t, closest_pt = closest_distance_rope_to_point(rope_pts, C)

    return {
        "passes": passes,
        "crossing_count": crossing_count,
        "hits": hits,  # list of (segment_index, t, 3D_point)
        "center": C,
        "min_distance_to_center": min_dist,
        "closest_point": {
            "segment_index": seg_idx,
            "t": t,
            "point": closest_pt,
        },
    }
