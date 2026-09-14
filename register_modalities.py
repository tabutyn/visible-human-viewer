"""Baked CT-to-color registration; output is metadata, never resampled pixels."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

ORIENTATIONS = ("identity", "flip_y", "flip_x", "rotate_180")
REGIONS = (("head", 0, .12), ("neck", .12, .18), ("chest", .18, .35), ("abdomen", .35, .52), ("pelvis", .52, .68), ("knee", .68, .84), ("ankle", .84, 1.0))
SCALE_X_LIMITS = (.35, 3.5)
SCALE_Y_LIMITS = (.25, 3.5)
REGION_THRESHOLDS = {
    "head": {"coverage": .70, "silhouette_distance": 8., "landmark_distance": 8., "worst_silhouette_distance": 12., "worst_landmark_distance": 12.},
    "neck": {"coverage": .65, "silhouette_distance": 12., "landmark_distance": 14., "worst_silhouette_distance": 18., "worst_landmark_distance": 21.},
    "chest": {"coverage": .70, "silhouette_distance": 13., "landmark_distance": 15., "worst_silhouette_distance": 20., "worst_landmark_distance": 23.},
    "abdomen": {"coverage": .75, "silhouette_distance": 10., "landmark_distance": 14., "worst_silhouette_distance": 15., "worst_landmark_distance": 21.},
    "pelvis": {"coverage": .75, "silhouette_distance": 10., "landmark_distance": 15., "worst_silhouette_distance": 15., "worst_landmark_distance": 23.},
    "knee": {"coverage": .55, "silhouette_distance": 15., "landmark_distance": 18., "worst_silhouette_distance": 23., "worst_landmark_distance": 27.},
    "ankle": {"coverage": .55, "silhouette_distance": 20., "landmark_distance": 20., "worst_silhouette_distance": 30., "worst_landmark_distance": 30.},
}


@dataclass
class AffineSolution:
    orientation: str; matrix: list[list[float]]; scale_x: float; scale_y: float; tx: float; ty: float; rotation_deg: float; overlap: float; residual: float; confidence: float


def _anatomy_components(mask, *, reject_support=False, seed=None):
    """Retain substantial body parts, including paired arms and legs."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, np.uint8)
    height, width = mask.shape
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < max(80, width * height * .001):
            continue
        if y > height * .82:
            continue
        if reject_support and w / max(1, h) > 4 and h < height * .28:
            continue
        if seed is not None and np.count_nonzero(seed[labels == label]) < max(20, area * .01):
            continue
        candidates.append((label, area))
    if not candidates:
        return np.zeros_like(mask, np.uint8)
    largest = max(area for _, area in candidates)
    keep = [label for label, area in candidates if area >= largest * .16]
    return np.isin(labels, keep).astype(np.uint8)


def color_anatomy_mask(image):
    rgb = image.astype(np.float32)
    if rgb.ndim != 3: return _anatomy_components((rgb > np.percentile(rgb, 35)).astype(np.uint8))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]; lum = .2126*r + .7152*g + .0722*b
    blue = (b > r * 1.08 + 5) & (b > g * 1.04 + 5)
    tissue_seed = ((r > 25) & (r > b * 1.08 + 4) & (g > b * .72)).astype(np.uint8)
    if np.count_nonzero(tissue_seed) >= 20:
        tissue = cv2.morphologyEx(tissue_seed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return _anatomy_components(tissue)
    candidate = cv2.morphologyEx(((lum > 18) & (lum < 248) & ~blue).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return _anatomy_components(candidate)


def ct_anatomy_mask(hu):
    body = cv2.morphologyEx((hu > -400).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return _anatomy_components(body, reject_support=True)


def orient(image, name):
    if name == "identity": return image
    if name == "flip_y": return cv2.flip(image, 0)
    if name == "flip_x": return cv2.flip(image, 1)
    if name == "rotate_180": return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(name)


def feature(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = np.maximum(cv2.Canny(gray, 35, 110), cv2.Canny(mask * 255, 20, 80))
    return np.exp(-cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3) / 3.5).astype(np.float32), edges


def descriptor(image, kind, orientation="identity"):
    image = orient(image, orientation)
    mask = color_anatomy_mask(image) if kind == "color" else ct_anatomy_mask(image)
    ys, xs = np.nonzero(mask); h, w = mask.shape
    if not len(xs): return np.full(14, 9., dtype=np.float32)
    cx, cy = xs.mean() / w, ys.mean() / h
    area = len(xs) / (w*h)
    radii = []
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        dx, dy = math.cos(angle), math.sin(angle); distance = []
        for step in np.linspace(0, 1.5, 80):
            x, y = int((cx + dx*step)*w), int((cy + dy*step)*h)
            if 0 <= x < w and 0 <= y < h and mask[y, x]: distance.append(step)
        radii.append(max(distance, default=0.))
    moments = cv2.HuMoments(cv2.moments(mask)).ravel()[:3]
    return np.asarray([area, cx, cy, *radii, *np.log1p(np.abs(moments))], dtype=np.float32)


def representative_indices(size, count):
    return sorted({round(i*(size-1)/max(1, count-1)) for i in range(min(size, count))}) if size else []


def spatial_sample_indices(size, count):
    """Uniform body coverage plus a denser independent head/neck fit."""
    if not size:
        return []
    whole_body = representative_indices(size, count)
    head_neck_end = min(size, max(1, math.ceil(size * .20)))
    local_count = max(8, math.ceil(count * .40))
    return sorted(set(whole_body) | set(representative_indices(head_neck_end, local_count)))


def constrained_dtw(color_desc, ct_desc, band=.05):
    """Banded, free-endpoint DTW; output pair order is strictly increasing."""
    n, m = len(color_desc), len(ct_desc); limit = max(1, math.ceil(max(n, m)*band))
    cost = np.full((n, m), np.inf); parent = np.full((n, m, 2), -1, dtype=int)
    for i in range(n):
        for j in range(m):
            if abs(i/max(1,n-1) - j/max(1,m-1)) > band: continue
            normalized_delta = abs(i/max(1,n-1) - j/max(1,m-1))
            local = float(np.linalg.norm(color_desc[i] - ct_desc[j])) + .30 * normalized_delta
            if i <= limit and j <= limit: cost[i,j] = local + .15 * (i + j)
            for pi, pj, penalty in ((i-1,j-1,0.), (i-1,j,.08), (i,j-1,.08)):
                if pi >= 0 and pj >= 0 and cost[pi,pj] + local + penalty < cost[i,j]:
                    cost[i,j] = cost[pi,pj] + local + penalty; parent[i,j] = (pi,pj)
    ends = [(cost[i,j] + .15 * ((n-1-i) + (m-1-j)), i,j) for i in range(n) for j in range(m) if i >= n-1-limit and j >= m-1-limit and np.isfinite(cost[i,j])]
    if not ends: raise ValueError("no path inside ±5% depth band")
    _, i, j = min(ends); path=[]
    while i >= 0 and j >= 0:
        path.append((i,j)); pi,pj=parent[i,j]
        if pi < 0: break
        i,j=int(pi),int(pj)
    path.reverse()
    # Compress DTW horizontal/vertical steps to one strictly-increasing pair.
    result=[]; last=(-1,-1)
    for i,j in path:
        if i > last[0] and j > last[1]: result.append((i,j)); last=(i,j)
    return result, float(min(ends)[0] / max(1, len(path)))


def _project(matrix):
    a,b,tx=map(float,matrix[0]); c,d,ty=map(float,matrix[1]); sx,sy=math.hypot(a,b),math.hypot(c,d)
    rotation=max(-12., min(12., math.degrees(math.atan2(b,a)))); r=math.radians(rotation)
    return np.array([[sx*math.cos(r),sx*math.sin(r),tx],[-sy*math.sin(r),sy*math.cos(r),ty]],np.float32),sx,sy,rotation


def _bounds_transform(fixed_mask, moving_mask):
    """Initialize fixed-pixel → moving-pixel sampling from anatomy bounds."""
    def bounds(mask):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None
        return float(xs.mean()), float(ys.mean()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
    fixed, moving = bounds(fixed_mask), bounds(moving_mask)
    if not fixed or not moving:
        return np.eye(2, 3, dtype=np.float32)
    fx, fy, fw, fh = fixed; mx, my, mw, mh = moving
    sx, sy = mw / max(1., fw), mh / max(1., fh)
    if not (SCALE_X_LIMITS[0] <= sx <= SCALE_X_LIMITS[1] and SCALE_Y_LIMITS[0] <= sy <= SCALE_Y_LIMITS[1]):
        return np.eye(2, 3, dtype=np.float32)
    return np.array([[sx, 0., mx - sx * fx], [0., sy, my - sy * fy]], dtype=np.float32)


def _evaluate_warp(fixed_feature, moving_feature, fixed_mask, moving_mask, warp):
    height, width = fixed_mask.shape
    aligned = cv2.warpAffine(moving_feature, warp, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    warped_mask = cv2.warpAffine(moving_mask, warp, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    union = np.logical_or(fixed_mask > 0, warped_mask > 0)
    intersection = np.logical_and(fixed_mask > 0, warped_mask > 0)
    iou = float(np.count_nonzero(intersection) / max(1, np.count_nonzero(union)))
    residual = float(np.mean(np.abs(fixed_feature[intersection] - aligned[intersection]))) if np.any(intersection) else 1.
    return iou, residual


def _alignment_score(overlap, residual):
    """Combine silhouette agreement and edge residual for candidate ranking."""
    return max(0., min(1., .72 * overlap + .28 * math.exp(-residual * 6.)))


def solve_affine(color, ct, orientation="identity"):
    moving = orient(ct, orientation)
    fixed_mask, moving_mask = color_anatomy_mask(color), ct_anatomy_mask(moving)
    fixed_feature, _ = feature(color, fixed_mask); moving_feature, _ = feature(moving, moving_mask)
    initial = _bounds_transform(fixed_mask, moving_mask)
    # findTransformECC mutates its input matrix even when the result is later
    # rejected. Keep the accepted fallback isolated from that scratch buffer.
    warp = initial.copy()
    try:
        candidate_input = initial.copy()
        _, candidate = cv2.findTransformECC(
            fixed_feature, moving_feature, candidate_input, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            inputMask=fixed_mask,
        )
        candidate, sx, sy, rotation = _project(candidate)
        initial_sx, initial_sy = initial[0, 0], initial[1, 1]
        plausible = (
            SCALE_X_LIMITS[0] <= sx <= SCALE_X_LIMITS[1]
            and SCALE_Y_LIMITS[0] <= sy <= SCALE_Y_LIMITS[1]
            and .65 * initial_sx <= sx <= 1.35 * initial_sx
            and .65 * initial_sy <= sy <= 1.35 * initial_sy
            and abs(rotation) <= 10
        )
        if plausible:
            candidate_iou, candidate_residual = _evaluate_warp(fixed_feature, moving_feature, fixed_mask, moving_mask, candidate)
            initial_iou, initial_residual = _evaluate_warp(fixed_feature, moving_feature, fixed_mask, moving_mask, initial)
            if _alignment_score(candidate_iou, candidate_residual) > _alignment_score(initial_iou, initial_residual) + 1e-6:
                warp = candidate.copy()
    except cv2.error:
        pass
    warp, sx, sy, rotation = _project(warp)
    overlap, residual = _evaluate_warp(fixed_feature, moving_feature, fixed_mask, moving_mask, warp)
    confidence = _alignment_score(overlap, residual)
    return AffineSolution(orientation, warp.astype(float).tolist(), sx, sy, float(warp[0, 2]), float(warp[1, 2]), rotation, overlap, residual, confidence)


def choose_orientation(color, ct):
    """Compatibility helper and single-pair orientation diagnostic."""
    return max((solve_affine(color, ct, name) for name in ORIENTATIONS), key=lambda item: item.confidence)


def _orientation_pixels(name,w,h):
    if name=="identity": return np.eye(3)
    if name=="flip_y": return np.array([[1,0,0],[0,-1,h-1],[0,0,1.]],float)
    if name=="flip_x": return np.array([[-1,0,w-1],[0,1,0],[0,0,1.]],float)
    return np.array([[-1,0,w-1],[0,-1,h-1],[0,0,1.]],float)


def inverse_uv(solution,w,h):
    """Exact OpenCV WARP_INVERSE_MAP destination(color)→source(CT) UV."""
    warp=np.eye(3); warp[:2]=np.asarray(solution.matrix,float)
    pixel=_orientation_pixels(solution.orientation,w,h) @ warp
    to_pixel=np.diag([w-1.,h-1.,1.]); to_uv=np.diag([1/max(1.,w-1.),1/max(1.,h-1.),1.])
    uv=to_uv@pixel@to_pixel
    return [float(uv[0,0]),float(uv[0,1]),float(uv[0,2]),float(uv[1,0]),float(uv[1,1]),float(uv[1,2])]


def _symmetric_edge_distance(edges_a, edges_b):
    distance_a=cv2.distanceTransform(255-edges_a,cv2.DIST_L2,3); distance_b=cv2.distanceTransform(255-edges_b,cv2.DIST_L2,3)
    return float((distance_a[edges_b>0].mean() if np.any(edges_b) else 99)+(distance_b[edges_a>0].mean() if np.any(edges_a) else 99))/2


def _warped_edge_distances(fixed_edges, moving_edges, matrix):
    before=_symmetric_edge_distance(fixed_edges,moving_edges)
    warped=cv2.warpAffine(moving_edges,np.asarray(matrix,np.float32),(fixed_edges.shape[1],fixed_edges.shape[0]),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)
    return before,_symmetric_edge_distance(fixed_edges,warped)


def _edge_distance(color,ct,solution):
    cm=color_anatomy_mask(color); mm=ct_anatomy_mask(orient(ct,solution.orientation))
    ce=cv2.Canny(cm*255,20,80); me=cv2.Canny(mm*255,20,80)
    before,after=_warped_edge_distances(ce,me,solution.matrix)
    return before,after


def _internal_landmark_edges(image, kind, mask):
    """Extract repeatable interior anatomy edges, excluding the silhouette."""
    interior=cv2.erode(mask.astype(np.uint8),np.ones((11,11),np.uint8))>0
    if kind=="color":
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image.astype(np.uint8)
        edges=cv2.Canny(gray,45,120)
    else:
        hu=image.astype(np.float32)
        gray=np.clip((hu+200.)*(255./1400.),0,255).astype(np.uint8)
        density_edges=cv2.Canny(gray,35,100)
        bone_edges=cv2.Canny((hu>200).astype(np.uint8)*255,20,80)
        edges=np.maximum(density_edges,bone_edges)
    edges[~interior]=0
    return edges


def _internal_landmark_distance(color,ct,solution):
    moving=orient(ct,solution.orientation)
    cm=color_anatomy_mask(color); mm=ct_anatomy_mask(moving)
    fixed_edges=_internal_landmark_edges(color,"color",cm)
    moving_edges=_internal_landmark_edges(moving,"density",mm)
    return _warped_edge_distances(fixed_edges,moving_edges,solution.matrix)


def _regional_qc(held, color_depths, cmax):
    regions={}
    for name,lo,hi in REGIONS:
        group=[item for item in held if lo <= color_depths[item["color_ordinal"]]/cmax <= hi]
        if not group:
            continue
        silhouette_before=float(np.mean([item["silhouette_before"] for item in group]))
        silhouette_after=float(np.mean([item["silhouette_after"] for item in group]))
        landmark_before=float(np.mean([item["landmark_before"] for item in group]))
        landmark_after=float(np.mean([item["landmark_after"] for item in group]))
        coverage=float(np.mean([item["coverage"] for item in group]))
        worst_silhouette=float(max(item["silhouette_after"] for item in group))
        worst_landmark=float(max(item["landmark_after"] for item in group))
        thresholds=REGION_THRESHOLDS[name]
        failures=[]
        if coverage < thresholds["coverage"]: failures.append("silhouette_coverage")
        if silhouette_after > thresholds["silhouette_distance"]: failures.append("silhouette_distance")
        if landmark_after > thresholds["landmark_distance"]: failures.append("internal_landmark_distance")
        if worst_silhouette > thresholds["worst_silhouette_distance"]: failures.append("silhouette_outlier")
        if worst_landmark > thresholds["worst_landmark_distance"]: failures.append("internal_landmark_outlier")
        if silhouette_after > silhouette_before: failures.append("silhouette_worsened")
        if landmark_after > landmark_before: failures.append("internal_landmarks_worsened")
        regions[name]={
            "levels":len(group),
            "before_symmetric_edge_distance":silhouette_before,
            "after_symmetric_edge_distance":silhouette_after,
            "improvement":(silhouette_before-silhouette_after)/silhouette_before if silhouette_before else 0.,
            "coverage":coverage,
            "internal_landmark_before_distance":landmark_before,
            "internal_landmark_after_distance":landmark_after,
            "internal_landmark_improvement":(landmark_before-landmark_after)/landmark_before if landmark_before else 0.,
            "worst_silhouette_distance":worst_silhouette,
            "worst_internal_landmark_distance":worst_landmark,
            "thresholds":thresholds,
            "accepted":not failures,
            "failures":failures,
        }
    return regions


def _depth_knots(matches,color_depths,ct_depths,confidence):
    cmax=max(color_depths) or 1
    return [{"color_depth": color_depths[i]/cmax,"ct_frame":float(ct_depths[j]),"confidence":float(confidence[k])} for k,(i,j) in enumerate(matches)]


def _depth_profile(matches, color_depths, ct_depths, confidence):
    """Build a monotonic physical-depth map over corroborated stack coverage."""
    knots = _depth_knots(matches, color_depths, ct_depths, confidence)
    x = np.asarray([color_depths[i] for i, _ in matches], dtype=float)
    y = np.asarray([ct_depths[j] for _, j in matches], dtype=float)
    slopes = np.diff(y) / np.maximum(1., np.diff(x))
    plausible = slopes[(slopes > .65) & (slopes < 1.35)]
    slope = float(np.median(plausible)) if plausible.size else 1.
    # Repeated structures can give a long DTW run the wrong offset.  The outer
    # anchors establish the global physical-frame offset; only spatially
    # corroborated matches may pull the smooth local map away from that line.
    edge_count = max(1, min(len(x) // 2, math.ceil(len(x) * .15)))
    edge_offsets = np.concatenate((y[:edge_count] - slope * x[:edge_count], y[-edge_count:] - slope * x[-edge_count:]))
    intercept = float(np.median(edge_offsets))
    cmax = max(color_depths) or 1
    expected = slope * x + intercept
    raw_adjustment = y - expected
    reliable = np.asarray(confidence, dtype=float) >= .25
    # The source frame numbers encode physical acquisition order. Repeated
    # pelvis/leg silhouettes can produce a confident DTW match roughly one
    # anatomy cycle away, so image evidence may bend but not replace that
    # physical trend. One percent covers gradual plane drift without allowing
    # the 3–5% false jumps observed in the male stack.
    max_local_warp = max(1., cmax * .01)
    if np.count_nonzero(reliable) >= 2:
        reliable_adjustment = np.clip(raw_adjustment[reliable], -max_local_warp, max_local_warp)
        smooth_adjustment = np.interp(x, x[reliable], reliable_adjustment)
    else:
        smooth_adjustment = np.zeros_like(x)
    for item, adjustment in zip(knots, smooth_adjustment):
        color_frame = item["color_depth"] * cmax
        item["ct_frame"] = float(slope * color_frame + intercept + adjustment)
    # Do not extrapolate into unobserved head/foot depths.  The viewer keeps
    # color visible there and treats density as outside registered coverage.
    low, high = knots[0]["color_depth"], knots[-1]["color_depth"]
    mapped_low, mapped_high = knots[0]["ct_frame"], knots[-1]["ct_frame"]
    unique = {}
    for item in sorted(knots, key=lambda value: value["color_depth"]):
        if low <= item["color_depth"] <= high:
            unique[round(item["color_depth"], 10)] = item
    result = list(unique.values())
    for index in range(1, len(result)):
        result[index]["ct_frame"] = max(result[index]["ct_frame"], result[index - 1]["ct_frame"] + 1e-3)
    return result, [low, high], [mapped_low, mapped_high]


def monotonic_depth_knots(color_depths, ct_depths, observations):
    """Compatibility adapter for explicit synthetic anchors."""
    ordered=sorted(observations); matches=[(min(range(len(color_depths)),key=lambda i:abs(color_depths[i]-color)), ordinal) for ordinal,color,_ in ordered]
    return _depth_knots(matches,color_depths,ct_depths,[confidence for _,_,confidence in ordered])


def register_modalities(color_slices,ct_slices,read_color:Callable,read_ct:Callable,sample_count=32,holdout_count=24):
    """Sequence-match corrected baked layers, then solve spatial profile/QC."""
    color_depths=[x.depth for x in color_slices]; ct_depths=[x.depth for x in ct_slices]
    ci=representative_indices(len(color_slices),sample_count); ti=representative_indices(len(ct_slices),sample_count)
    colors=[read_color(color_slices[i]) for i in ci]; cts=[read_ct(ct_slices[i]) for i in ti]
    orientation_scores={}; paths={}
    for name in ORIENTATIONS:
        path,cost=constrained_dtw([descriptor(x,"color") for x in colors],[descriptor(x,"ct",name) for x in cts])
        orientation_scores[name]=-cost; paths[name]=path
    orientation=max(orientation_scores,key=orientation_scores.get); raw_pairs=[(ci[i],ti[j]) for i,j in paths[orientation]]
    raw_samples=[]; confidences=[]
    for i,j in raw_pairs:
        solution=solve_affine(read_color(color_slices[i]),read_ct(ct_slices[j]),orientation); raw_samples.append((i,j,solution)); confidences.append(solution.confidence)
    depth_knots,coverage_color,coverage_ct=_depth_profile(raw_pairs,color_depths,ct_depths,confidences)
    depth_x=np.asarray([item["color_depth"] for item in depth_knots],float); depth_y=np.asarray([item["ct_frame"] for item in depth_knots],float)
    solution_cache={(i,j):solution for i,j,solution in raw_samples}
    pairs=[]; samples=[]
    fit_indices = [
        i for i in spatial_sample_indices(len(color_slices), sample_count)
        if coverage_color[0] <= color_depths[i] / (max(color_depths) or 1) <= coverage_color[1]
    ]
    for i in fit_indices:
        mapped=float(np.interp(color_depths[i]/(max(color_depths) or 1),depth_x,depth_y))
        j=min(range(len(ct_depths)),key=lambda ordinal:abs(ct_depths[ordinal]-mapped))
        pairs.append((i,j))
        solution=solution_cache.get((i,j)) or solve_affine(read_color(color_slices[i]),read_ct(ct_slices[j]),orientation)
        samples.append((i,j,solution))
    valid=[item for item in samples if item[2].confidence >= .25 and item[2].overlap >= .25 and SCALE_X_LIMITS[0] <= item[2].scale_x <= SCALE_X_LIMITS[1] and SCALE_Y_LIMITS[0] <= item[2].scale_y <= SCALE_Y_LIMITS[1]]
    if not valid:
        valid=samples
    valid_x=np.array([item[0] for item in valid],float)
    valid_raw=np.array([np.asarray(item[2].matrix,float).ravel() for item in valid])
    padded=np.pad(valid_raw,((1,1),(0,0)),mode="edge")
    smooth_valid=np.median(np.stack((padded[:-2],padded[1:-1],padded[2:])),axis=0)
    h,w=colors[0].shape[:2]; cmax=max(color_depths) or 1
    spatial=[]; profile_matrices=[]
    for i,j,s in samples:
        matrix=np.array([np.interp(i,valid_x,smooth_valid[:,column]) for column in range(6)]).reshape(2,3)
        projected,sx,sy,rotation=_project(matrix)
        support=float(np.interp(i,valid_x,[item[2].confidence for item in valid]))
        ss=AffineSolution(orientation,projected.tolist(),sx,sy,float(projected[0,2]),float(projected[1,2]),rotation,s.overlap,s.residual,support)
        profile_matrices.append((i,projected,support))
        spatial.append({"color_depth":color_depths[i]/cmax,"inverse_uv":inverse_uv(ss,w,h),"confidence":support})
    # Holdouts are disjoint evenly spaced color levels, predicted by depth knots.
    used={i for i,_ in pairs}
    candidate_pool=[i for i in representative_indices(len(color_slices),sample_count+holdout_count*2) if i not in used and coverage_color[0] <= color_depths[i]/cmax <= coverage_color[1]]
    candidate_positions=representative_indices(len(candidate_pool),holdout_count)
    candidates=[candidate_pool[position] for position in candidate_positions]
    matrix_x=np.array([item[0] for item in profile_matrices],float); held=[]
    for i in candidates:
        mapped=float(np.interp(color_depths[i]/cmax,depth_x,depth_y))
        j=min(range(len(ct_depths)),key=lambda ordinal:abs(ct_depths[ordinal]-mapped))
        matrix=np.array([np.interp(i,matrix_x,[item[1].ravel()[column] for item in profile_matrices]) for column in range(6)]).reshape(2,3)
        projected,sx,sy,rotation=_project(matrix)
        color,ct=read_color(color_slices[i]),read_ct(ct_slices[j]); fixed_mask=color_anatomy_mask(color); moving_mask=ct_anatomy_mask(orient(ct,orientation)); fixed_feature,_=feature(color,fixed_mask); moving_feature,_=feature(orient(ct,orientation),moving_mask)
        overlap,residual=_evaluate_warp(fixed_feature,moving_feature,fixed_mask,moving_mask,projected)
        support=float(np.interp(i,matrix_x,[item[2] for item in profile_matrices]))
        solution=AffineSolution(orientation,projected.tolist(),sx,sy,float(projected[0,2]),float(projected[1,2]),rotation,overlap,residual,support)
        before,after=_edge_distance(color,ct,solution)
        landmark_before,landmark_after=_internal_landmark_distance(color,ct,solution)
        held.append({"color_ordinal":i,"ct_ordinal":j,"silhouette_before":before,"silhouette_after":after,"coverage":overlap,"landmark_before":landmark_before,"landmark_after":landmark_after})
    regions=_regional_qc(held,color_depths,cmax)
    before=float(np.mean([x["silhouette_before"] for x in held])) if held else None; after=float(np.mean([x["silhouette_after"] for x in held])) if held else None
    landmark_before=float(np.mean([x["landmark_before"] for x in held])) if held else None; landmark_after=float(np.mean([x["landmark_after"] for x in held])) if held else None
    coverage=float(np.mean([x["coverage"] for x in held])) if held else 0.
    improvement=(before-after)/before if before and after is not None else 0.
    landmark_improvement=(landmark_before-landmark_after)/landmark_before if landmark_before and landmark_after is not None else 0.
    ranked_scores=sorted(orientation_scores.values(),reverse=True)
    orientation_margin=ranked_scores[0]-ranked_scores[1] if len(ranked_scores)>1 else 0.
    edge_limit=max(1,math.ceil(sample_count*.05)); depth_band_limited=bool(paths[orientation][0][0] >= edge_limit or paths[orientation][-1][0] <= len(ci)-1-edge_limit)
    regional_failures=[name for name,value in regions.items() if not value["accepted"]]
    accepted=bool(improvement>=.10 and coverage>=.55 and len(valid)>=max(4,len(samples)//3) and not regional_failures)
    qc={"accepted":accepted,"representative_count":len(samples),"accepted_spatial_count":len(valid),"holdout_count":len(held),"before_symmetric_edge_distance":before,"after_symmetric_edge_distance":after,"improvement":improvement,"coverage":coverage,"internal_landmark_before_distance":landmark_before,"internal_landmark_after_distance":landmark_after,"internal_landmark_improvement":landmark_improvement,"regional":regions,"regional_failures":regional_failures,"low_coverage":coverage<.60,"orientation_margin":orientation_margin,"orientation_ambiguous":orientation_margin<.01,"depth_band_limited":depth_band_limited}
    return {"orientation":orientation,"orientation_scores":orientation_scores,"depth_knots":depth_knots,"spatial_knots":sorted(spatial,key=lambda x:x["color_depth"]),"coverage_color":coverage_color,"coverage_ct":coverage_ct,"qc":qc,"representatives":[{"color_ordinal":i,"ct_ordinal":j,**asdict(s)} for i,j,s in samples]}
