"""
engine.py — memory-optimized for Render free (512 MB) + Qdrant free tier
Quality controlled by LOW_RAM flag:
  LOW_RAM=true  → light enhance + tiny sharpness (saves RAM)
  LOW_RAM=false → original full enhance + full sharpness
"""

import os
import gc
import threading
import time as _time
import uuid
import numpy as np
import cv2

# On a CPU-quota-limited container (e.g. Render's 0.1 vCPU), OpenCV's and
# ONNXRuntime's default thread pools are sized off the host's full core
# count. Every thread then fights the tiny quota for scheduling time, which
# is what turns sub-second inference into multi-second/minute stalls with
# wildly inconsistent latency. Force everything to a single thread so each
# request gets the whole quota instead of thrashing against itself.
cv2.setNumThreads(1)

import onnxruntime as ort

_ort_SessionOptions = ort.SessionOptions


class _SingleThreadSessionOptions(_ort_SessionOptions):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.intra_op_num_threads = 1
        self.inter_op_num_threads = 1
        self.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL


ort.SessionOptions = _SingleThreadSessionOptions

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    OptimizersConfigDiff,
)
from uniface.detection import SCRFD
from uniface.recognition import ArcFace
import config as C

try:
    from uniface.constants import SCRFDWeights
    _SCRFD_MODELS = {
        "scrfd_500m": SCRFDWeights.SCRFD_500M_KPS,
        "scrfd_10g": SCRFDWeights.SCRFD_10G_KPS,
    }
except Exception:
    SCRFDWeights = None
    _SCRFD_MODELS = {}


def _f(x):
    return float(x)

def _i(x):
    return int(x)


# Set DEBUG_TIMING=1 on Render to log a per-stage breakdown of every
# check-frame/quality/match call, so the real bottleneck on the actual
# throttled instance is visible instead of guessed at.
_DEBUG_TIMING = str(os.getenv("DEBUG_TIMING", "0")).strip().lower() in ("1", "true", "yes")


class _Lap:
    """Cheap stage-timer. report() is a no-op unless DEBUG_TIMING is set."""

    __slots__ = ("t0", "marks")

    def __init__(self):
        self.t0 = _time.perf_counter()
        self.marks = []

    def lap(self, name):
        self.marks.append((name, _time.perf_counter()))

    def report(self, label):
        if not _DEBUG_TIMING:
            return
        prev = self.t0
        parts = []
        for name, t in self.marks:
            parts.append(f"{name}={(t - prev) * 1000:.0f}ms")
            prev = t
        total = (prev - self.t0) * 1000
        print(f"[TIMING] {label} total={total:.0f}ms {' '.join(parts)}", flush=True)


class FaceEngine:
    def __init__(self):
        os.makedirs(C.DATA_DIR, exist_ok=True)
        self._lock = threading.Lock()

        # Under a 0.1 vCPU quota there is no real parallelism to gain from
        # running two inference calls "concurrently" — they just take turns
        # fighting each other for the same sliver of CPU, which costs more
        # in scheduling/cache thrash than running them back-to-back. This
        # lock forces every detect/embed call (across all requests) onto
        # one at a time, in arrival order.
        self._infer_lock = threading.Lock()

        # Models
        # SCRFD's own preprocessing letterboxes every frame up to `input_size`
        # before inference — our MAX_SIDE pre-resize doesn't reduce its cost.
        # SCRFD_10G at 640x640 (the library default) is ~20x more FLOPs than
        # SCRFD_500M, and this app's frames are close-up, cooperative faces
        # (webcam selfie), so the small model at a smaller input size is
        # plenty accurate and dramatically cheaper per frame.
        det_input_side = getattr(C, "DET_INPUT_SIZE", 320)
        det_model = _SCRFD_MODELS.get(getattr(C, "DET_MODEL", "scrfd_500m"))
        try:
            if det_model is None:
                raise TypeError("SCRFD model selection not available in this uniface version")
            self.detector = SCRFD(
                model_name=det_model,
                confidence_threshold=C.DET_CONF,
                input_size=(det_input_side, det_input_side),
                providers=C.PROVIDERS,
            )
        except TypeError:
            # Older/newer uniface version without model_name/input_size kwargs
            self.detector = SCRFD(confidence_threshold=C.DET_CONF, providers=C.PROVIDERS)
        self.recognizer = ArcFace(providers=C.PROVIDERS)

        # Qdrant (cloud free tier or local)
        if getattr(C, "QDRANT_URL", None) and getattr(C, "QDRANT_API_KEY", None):
            self.client = QdrantClient(
                url=C.QDRANT_URL,
                api_key=C.QDRANT_API_KEY,
                prefer_grpc=False,
                timeout=30,
            )
        else:
            qdrant_path = os.path.join(C.DATA_DIR, "qdrant_data")
            self.client = QdrantClient(path=qdrant_path)

        self.collection = "faces"
        self._ensure_collection()

        # Reuse one CLAHE instance
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # Max long-side size before detection
        self._max_side = getattr(C, "MAX_SIDE", 640)

        # Cache the flag once
        self._low_ram = bool(getattr(C, "LOW_RAM", True))

        # Pay any one-time ONNX Runtime lazy-init cost (buffer allocation,
        # kernel selection) now, at startup, instead of on the first real
        # request.
        self._warmup()

    def _warmup(self):
        try:
            dummy = np.zeros((self._max_side, self._max_side, 3), dtype=np.uint8)
            self.detector.detect(dummy)
            self._enhance(dummy.copy())
            self.recognizer.get_normalized_embedding(
                np.zeros((112, 112, 3), dtype=np.uint8), None
            )
        except Exception:
            pass

    def _ensure_collection(self):
        try:
            names = {c.name for c in self.client.get_collections().collections}
        except Exception:
            names = set()
        if self.collection in names:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=C.EMBED_DIM,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(
                max_optimization_threads=getattr(C, "MAX_OPTIMIZATION_THREADS", 1),
                indexing_threshold=getattr(C, "INDEXING_THRESHOLD", 20000),
            ),
        )

        for field in ("businessId", "studentId"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass

    # -------------------------------------------------- helpers
    @staticmethod
    def _l2n(v):
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n = np.maximum(n, 1e-10)
        return (v / n).astype(np.float32)

    @staticmethod
    def _decode(buf: bytes):
        arr = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        del arr
        return img

    def _resize_long_side(self, img, max_side=None):
        if max_side is None:
            max_side = self._max_side
        h, w = img.shape[:2]
        side = max(h, w)
        if side <= max_side:
            return img, 1.0
        scale = max_side / float(side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return out, scale

    @staticmethod
    def _crop_face(img, bbox, scale=None):
        if scale is None:
            scale = C.FACE_CROP_SCALE
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(float, bbox[:4])
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        size = max(x2 - x1, y2 - y1) * scale
        half = size * 0.5
        nx1 = max(0, int(cx - half))
        ny1 = max(0, int(cy - half))
        nx2 = min(w, int(cx + half))
        ny2 = min(h, int(cy + half))
        if nx2 <= nx1 or ny2 <= ny1:
            return img, (0, 0)
        return img[ny1:ny2, nx1:nx2].copy(), (nx1, ny1)

    def _enhance(self, img):
        """
        LOW_RAM=true  → light CLAHE + mild unsharp
        LOW_RAM=false → original strong pipeline (CLAHE + bilateral + sharpen)
        """
        if img is None or img.size == 0:
            return img

        # Common: CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        del lab, l, a, b

        if self._low_ram:
            # Light path (saves RAM)
            blur = cv2.GaussianBlur(img, (0, 0), 0.8)
            sharp = cv2.addWeighted(img, 1.25, blur, -0.25, 0)
            del blur
            return sharp
        else:
            # Original-quality path
            den = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
            blur = cv2.GaussianBlur(den, (0, 0), 1.0)
            sharp = cv2.addWeighted(den, 1.4, blur, -0.4, 0)
            del den, blur
            return sharp

    def _sharpness(self, img, bbox):
        """
        LOW_RAM=true  → tiny 48×48 patch
        LOW_RAM=false → full crop (original behaviour)
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        y1i, y2i = max(0, int(y1)), min(h, int(y2))
        x1i, x2i = max(0, int(x1)), min(w, int(x2))
        patch = img[y1i:y2i, x1i:x2i]

        if not patch.size:
            return 0.0

        if self._low_ram:
            small = cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA)
            sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
            del small
        else:
            # Full crop sharpness (same as original)
            sharp = float(cv2.Laplacian(patch, cv2.CV_64F).var())

        del patch
        return sharp

    def _largest_face(self, img):
        faces = self.detector.detect(img)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @staticmethod
    def _safe_score(face):
        val = getattr(face, "score", None)
        if val is None:
            for alt in ("confidence", "det_score", "prob"):
                val = getattr(face, alt, None)
                if val is not None:
                    break
        if val is None:
            return 1.0
        try:
            arr = np.asarray(val, dtype=np.float32).ravel()
            return float(arr[0]) if arr.size else 1.0
        except Exception:
            return 1.0

    def _embed_from_face(self, img, face):
        """Embedding from already-detected face. No second detection."""
        bbox = [float(x) for x in np.asarray(face.bbox, dtype=np.float32).ravel()[:4]]
        score = self._safe_score(face)
        lm = getattr(face, "landmarks", None)

        crop, (ox, oy) = self._crop_face(img, bbox)

        if lm is not None:
            lm = np.asarray(lm, dtype=np.float32).copy()
            if lm.ndim == 2 and lm.shape[1] == 2:
                lm[:, 0] -= ox
                lm[:, 1] -= oy

        try:
            raw = self.recognizer.get_normalized_embedding(crop, lm)
        except Exception:
            try:
                raw = self.recognizer.get_normalized_embedding(crop, None)
            except Exception:
                del crop
                return None

        emb = np.asarray(raw, dtype=np.float32).ravel()
        del crop, raw
        if emb.size != C.EMBED_DIM:
            return None
        return self._l2n(emb), bbox, score

    # -------------------------------------------------- quality (enrollment)
    def _quality(self, img):
        if img is None:
            return 0.0, None, {"reason": "bad_image"}

        lap = _Lap()
        with self._infer_lock:
            lap.lap("wait_lock")
            # 1. Downscale early
            img, _ = self._resize_long_side(img)
            lap.lap("resize")
            # 2. Enhance (light or full according to LOW_RAM)
            img = self._enhance(img)
            lap.lap("enhance")

            h, w = img.shape[:2]
            face = self._largest_face(img)
            lap.lap("detect")
            if face is None:
                del img
                lap.report("quality(no_face)")
                return 0.0, None, {"reason": "no_face"}

            x1, y1, x2, y2 = [float(v) for v in np.asarray(face.bbox, dtype=np.float32).ravel()[:4]]
            fw = max(x2 - x1, 1.0)
            fh = max(y2 - y1, 1.0)
            det = self._safe_score(face)

            area_ratio = (fw * fh) / float(w * h)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            off = max(abs(cx - w * 0.5) / (w * 0.5), abs(cy - h * 0.5) / (h * 0.5))

            # Sharpness (tiny or full according to LOW_RAM)
            sharp = self._sharpness(img, [x1, y1, x2, y2])
            lap.lap("sharp")

            s_det = min(max((det - 0.3) / 0.6, 0.0), 1.0)
            s_size = min(max((area_ratio - C.AREA_MIN) / (0.35 - C.AREA_MIN), 0.0), 1.0)
            if area_ratio > C.AREA_MAX:
                s_size *= 0.4
            s_center = 1.0 - min(off, 1.0)
            s_sharp = min(sharp / (C.SHARP_MIN * 3.0), 1.0)

            score = 0.35 * s_det + 0.25 * s_sharp + 0.25 * s_size + 0.15 * s_center

            emb = None
            if score > 0:
                res = self._embed_from_face(img, face)
                if res is not None:
                    emb = res[0]
            lap.lap("embed")

            del img

        lap.report("quality")
        detail = {
            "reason": "ok",
            "score": round(float(score), 3),
            "det": round(det, 3),
            "sharpness": round(sharp, 1),
            "area_ratio": round(area_ratio, 3),
            "off_center": round(float(off), 3),
        }
        return float(score), emb, detail

    # -------------------------------------------------- live check
    def check_frame(self, buf: bytes):
        lap = _Lap()
        with self._infer_lock:
            lap.lap("wait_lock")
            img = self._decode(buf)
            lap.lap("decode")
            if img is None:
                lap.report("check_frame(bad_image)")
                return {"ok": False, "reason": "bad_image"}

            img, _ = self._resize_long_side(img)
            lap.lap("resize")
            h, w = img.shape[:2]

            face = self._largest_face(img)
            lap.lap("detect")
            if face is None:
                del img
                lap.report("check_frame(no_face)")
                return {"ok": False, "reason": "no_face"}

            x1, y1, x2, y2 = [_f(v) for v in face.bbox]
            fw, fh = x2 - x1, y2 - y1
            score = self._safe_score(face)

            area_ratio = (fw * fh) / _f(w * h)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            centered = (
                abs(cx - w * 0.5) < w * C.CENTER_TOL
                and abs(cy - h * 0.5) < h * C.CENTER_TOL
            )

            # Sharpness (tiny or full according to LOW_RAM)
            sharp = self._sharpness(img, [x1, y1, x2, y2])
            lap.lap("sharp")
            del img

        lap.report("check_frame")
        ok, reason = True, "ok"
        if score < C.DET_SCORE_MIN:
            ok, reason = False, "low_confidence"
        elif area_ratio < C.AREA_MIN:
            ok, reason = False, "too_far"
        elif area_ratio > C.AREA_MAX:
            ok, reason = False, "too_close"
        elif not centered:
            ok, reason = False, "not_centered"
        elif sharp < C.SHARP_MIN:
            ok, reason = False, "blurry"

        return {
            "ok": bool(ok),
            "reason": reason,
            "score": round(score, 3),
            "area_ratio": round(area_ratio, 3),
            "sharpness": round(sharp, 1),
            "bbox": [x1, y1, x2, y2],
            "frame_w": _i(w),
            "frame_h": _i(h),
        }

    # -------------------------------------------------- enrollment
    def enroll(self, business_id: str, student_id: str, name: str, frame_bufs: list):
        scored = []
        t_start = _time.perf_counter()
        n_processed = 0

        for buf in frame_bufs:
            img = self._decode(buf)
            if img is None:
                continue
            n_processed += 1
            try:
                sc, emb, detail = self._quality(img)
            except Exception:
                emb = None
            del img
            if emb is None:
                continue
            scored.append((sc, emb, detail))

            # Early stop
            if len(scored) >= getattr(C, "BURST_KEEP", 5) * 2:
                break

            if len(scored) % 3 == 0:
                gc.collect()

        if _DEBUG_TIMING:
            print(
                f"[TIMING] enroll(scan) total={(_time.perf_counter() - t_start) * 1000:.0f}ms "
                f"frames_decoded={n_processed} frames_scored={len(scored)}",
                flush=True,
            )

        if not scored:
            return {
                "ok": False,
                "reason": "no_usable_faces",
                "vectors": 0,
                "received": len(frame_bufs),
            }

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score = _f(scored[0][0])

        accepted = [t for t in scored if t[0] >= C.ENROLL_ACCEPT]
        if not accepted:
            return {
                "ok": False,
                "reason": "below_threshold",
                "vectors": 0,
                "received": len(frame_bufs),
                "best_score": round(best_score, 3),
                "threshold": C.ENROLL_ACCEPT,
            }

        kept = []
        kept_details = []
        for sc, emb, detail in accepted:
            if len(kept) >= C.BURST_KEEP:
                break
            if kept:
                sims = np.dot(np.vstack(kept), emb.ravel())
                if float(sims.max()) > C.MIN_DIVERSITY:
                    continue
            kept.append(emb)
            kept_details.append(detail)

        if not kept:
            return {
                "ok": False,
                "reason": "no_diverse_faces",
                "vectors": 0,
                "received": len(frame_bufs),
            }

        vecs = np.vstack(kept)
        centroid = self._l2n(vecs.mean(axis=0))
        all_vecs = np.vstack([vecs, centroid]).astype(np.float32)
        del vecs, kept, scored
        gc.collect()

        t_qdrant = _time.perf_counter()
        with self._lock:
            self.client.delete(
                collection_name=self.collection,
                points_selector=rest.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(key="businessId", match=MatchValue(value=business_id)),
                            FieldCondition(key="studentId", match=MatchValue(value=student_id)),
                        ]
                    )
                ),
            )

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v.tolist(),
                    payload={
                        "businessId": business_id,
                        "studentId": student_id,
                        "name": name,
                        "is_centroid": i == len(all_vecs) - 1,
                    },
                )
                for i, v in enumerate(all_vecs)
            ]
            self.client.upsert(collection_name=self.collection, points=points)

        total = self.client.count(collection_name=self.collection).count
        if _DEBUG_TIMING:
            print(f"[TIMING] enroll(qdrant_write+count)={(_time.perf_counter() - t_qdrant) * 1000:.0f}ms", flush=True)
        del all_vecs, points
        gc.collect()

        return {
            "ok": True,
            "name": name,
            "studentId": student_id,
            "businessId": business_id,
            "vectors": _i(len(kept_details) + 1),
            "angles_used": _i(len(kept_details)),
            "received": _i(len(frame_bufs)),
            "accepted": _i(len(accepted)),
            "best_score": round(best_score, 3),
            "kept_scores": [d["score"] for d in kept_details],
            "threshold": C.ENROLL_ACCEPT,
            "index_total": _i(total),
        }

    # -------------------------------------------------- recognition
    def _match_single(self, img, business_id=None):
        lap = _Lap()
        with self._infer_lock:
            lap.lap("wait_lock")
            img, _ = self._resize_long_side(img)
            img = self._enhance(img)
            lap.lap("enhance")
            face = self._largest_face(img)
            lap.lap("detect")
            if face is None:
                del img
                lap.report("match_single(no_face)")
                return None

            res = self._embed_from_face(img, face)
            lap.lap("embed")
            del img
        lap.report("match_single(pre-qdrant)")
        if res is None:
            return None
        emb, bbox, _ = res

        query_filter = None
        if business_id:
            query_filter = Filter(
                must=[FieldCondition(key="businessId", match=MatchValue(value=business_id))]
            )

        t_q0 = _time.perf_counter()
        response = self.client.query_points(
            collection_name=self.collection,
            query=emb.ravel().tolist(),
            query_filter=query_filter,
            limit=C.TOPK,
            score_threshold=C.SIM_THRESHOLD,
        )
        if _DEBUG_TIMING:
            print(f"[TIMING] match_single(qdrant_query)={(_time.perf_counter() - t_q0) * 1000:.0f}ms", flush=True)
        hits = response.points
        del emb

        if not hits:
            return {"matched": False, "bbox": bbox}

        votes = {}
        for hit in hits:
            p = hit.payload
            sid = p["studentId"]
            if sid not in votes:
                votes[sid] = [0.0, 0, p["name"], p["businessId"]]
            votes[sid][0] += float(hit.score)
            votes[sid][1] += 1

        sid = max(votes, key=lambda k: votes[k][0])
        total, cnt, name, bid = votes[sid]
        return {
            "matched": True,
            "studentId": sid,
            "name": name,
            "businessId": bid,
            "score": total / cnt,
            "bbox": bbox,
        }

    def recognize(self, frame_bufs: list, business_id=None):
        per_frame = []
        for i, buf in enumerate(frame_bufs):
            img = self._decode(buf)
            if img is None:
                continue
            m = self._match_single(img, business_id)
            del img
            per_frame.append(m)

            hits_so_far = sum(1 for r in per_frame if r and r.get("matched"))
            if hits_so_far >= max(2, (len(frame_bufs) // 2) + 1):
                break

            if i % 2 == 1:
                gc.collect()

        hits = [r for r in per_frame if r and r.get("matched")]
        checked = len([r for r in per_frame if r is not None])

        if not hits:
            return {
                "exists": False,
                "studentId": None,
                "name": None,
                "businessId": None,
                "probability": 0.0,
                "avg_score": 0.0,
                "votes": 0,
                "checked": _i(checked or len(frame_bufs)),
            }

        tally = {}
        for r in hits:
            sid = r["studentId"]
            if sid not in tally:
                tally[sid] = [0, 0.0, r["name"], r["businessId"]]
            tally[sid][0] += 1
            tally[sid][1] += _f(r["score"])

        sid = max(tally, key=lambda k: (tally[k][0], tally[k][1]))
        count, sum_score, name, bid = tally[sid]
        denom = max(checked, 1)

        agreement = count / denom
        avg_score = sum_score / count
        probability = round(0.5 * agreement + 0.5 * avg_score, 4)
        exists = (count >= (denom // 2 + denom % 2)) if denom > 1 else True

        return {
            "exists": bool(exists),
            "studentId": sid,
            "name": name,
            "businessId": bid,
            "probability": _f(probability),
            "avg_score": round(_f(avg_score), 4),
            "agreement": round(_f(agreement), 3),
            "votes": _i(count),
            "checked": _i(denom),
        }

    # -------------------------------------------------- list students
    def list_students(self):
        seen = {}
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                key = (p.payload["businessId"], p.payload["studentId"])
                seen[key] = p.payload["name"]
            if next_offset is None:
                break
            offset = next_offset

        return [
            {"businessId": b, "studentId": s, "name": n}
            for (b, s), n in seen.items()
        ]