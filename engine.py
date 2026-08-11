"""
engine.py — face logic with Qdrant (embedded mode)
Uses: QdrantClient(path="./qdrant_data")
"""

import os
import json
import threading
import uuid
import numpy as np
import cv2

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from uniface.detection import SCRFD
from uniface.recognition import MobileFace
import config as C


# -------- helpers --------
def _f(x):
    return float(x)

def _i(x):
    return int(x)


class FaceEngine:
    def __init__(self):
        os.makedirs(C.DATA_DIR, exist_ok=True)
        self._lock = threading.Lock()

        # Models
        self.detector = SCRFD(confidence_threshold=C.DET_CONF, providers=C.PROVIDERS)
        self.recognizer = MobileFace(providers=C.PROVIDERS)

        # ---------- Embedded Qdrant ----------
        qdrant_path = os.path.join(C.DATA_DIR, "qdrant_data")
        self.client = QdrantClient(path=qdrant_path)
        self.collection = "faces"
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=C.EMBED_DIM,
                    distance=Distance.COSINE,
                ),
            )
            # Indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="businessId",
                field_schema="keyword",
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="studentId",
                field_schema="keyword",
            )

    # -------------------------------------------------- helpers
    @staticmethod
    def _l2n(v):
        v = np.asarray(v, dtype="float32")
        if v.ndim == 1:
            v = v[None, :]
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        return v / norm

    @staticmethod
    def _decode(buf: bytes):
        arr = np.frombuffer(buf, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _enhance(img):
        """Light denoise + mild sharpen for grainy phone frames"""
        if img is None or img.size == 0:
            return img
        den = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)
        blur = cv2.GaussianBlur(den, (0, 0), 1.2)
        sharp = cv2.addWeighted(den, 1.5, blur, -0.5, 0)
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
            arr = np.asarray(val, dtype="float32").ravel()
            return float(arr[0]) if arr.size else 1.0
        except Exception:
            return 1.0

    def _embed(self, img):
        img = self._enhance(img)
        face = self._largest_face(img)
        if face is None:
            return None
        score = self._safe_score(face)
        lm = getattr(face, "landmarks", None)
        try:
            raw = self.recognizer.get_normalized_embedding(img, lm)
        except Exception:
            try:
                raw = self.recognizer.get_normalized_embedding(img, None)
            except Exception:
                return None
        emb = np.asarray(raw, dtype="float32").ravel()
        if emb.size != C.EMBED_DIM:
            return None
        emb = self._l2n(emb)
        bbox = [_f(x) for x in np.asarray(face.bbox, dtype="float32").ravel()[:4]]
        return emb, bbox, score

    # -------------------------------------------------- quality
    def _quality(self, img):
        if img is None:
            return 0.0, None, {"reason": "bad_image"}
        img = self._enhance(img)
        h, w = img.shape[:2]
        face = self._largest_face(img)
        if face is None:
            return 0.0, None, {"reason": "no_face"}

        x1, y1, x2, y2 = [_f(v) for v in np.asarray(face.bbox, dtype="float32").ravel()[:4]]
        fw, fh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        det = self._safe_score(face)

        area_ratio = (fw * fh) / _f(w * h)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        off = max(abs(cx - w / 2) / (w / 2), abs(cy - h / 2) / (h / 2))
        crop = img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        sharp = _f(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0

        s_det = min(max((det - 0.3) / 0.6, 0.0), 1.0)
        s_size = min(max((area_ratio - C.AREA_MIN) / (0.35 - C.AREA_MIN), 0.0), 1.0)
        if area_ratio > C.AREA_MAX:
            s_size *= 0.4
        s_center = 1.0 - min(off, 1.0)
        s_sharp = min(sharp / (C.SHARP_MIN * 3.0), 1.0)

        score = 0.35 * s_det + 0.25 * s_sharp + 0.25 * s_size + 0.15 * s_center

        emb = None
        if score > 0:
            try:
                raw = self.recognizer.get_normalized_embedding(img, getattr(face, "landmarks", None))
                e = np.asarray(raw, dtype="float32").ravel()
                if e.size == C.EMBED_DIM:
                    emb = self._l2n(e)
            except Exception:
                emb = None

        detail = {
            "reason": "ok",
            "score": round(_f(score), 3),
            "det": round(det, 3),
            "sharpness": round(sharp, 1),
            "area_ratio": round(area_ratio, 3),
            "off_center": round(_f(off), 3),
        }
        return _f(score), emb, detail

    # -------------------------------------------------- check frame
    def check_frame(self, buf: bytes):
        img = self._decode(buf)
        if img is None:
            return {"ok": False, "reason": "bad_image"}
        img = self._enhance(img)
        h, w = img.shape[:2]

        face = self._largest_face(img)
        if face is None:
            return {"ok": False, "reason": "no_face"}

        x1, y1, x2, y2 = [_f(v) for v in face.bbox]
        fw, fh = x2 - x1, y2 - y1
        score = self._safe_score(face)

        area_ratio = (fw * fh) / _f(w * h)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        centered = abs(cx - w / 2) < w * C.CENTER_TOL and abs(cy - h / 2) < h * C.CENTER_TOL
        crop = img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        sharp = _f(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0

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
        for buf in frame_bufs:
            img = self._decode(buf)
            try:
                sc, emb, detail = self._quality(img)
            except Exception:
                continue
            if emb is None:
                continue
            scored.append((sc, emb, detail))

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
                sims = (np.vstack(kept) @ emb.T).ravel()
                if _f(sims.max()) > C.MIN_DIVERSITY:
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
        all_vecs = np.vstack([vecs, centroid]).astype("float32")

        with self._lock:
            # Delete old vectors of this student
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

            # Insert new vectors
            points = []
            for i, vec in enumerate(all_vecs):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec.tolist(),
                        payload={
                            "businessId": business_id,
                            "studentId": student_id,
                            "name": name,
                            "is_centroid": i == len(all_vecs) - 1,
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points)

        total = self.client.count(collection_name=self.collection).count

        return {
            "ok": True,
            "name": name,
            "studentId": student_id,
            "businessId": business_id,
            "vectors": _i(len(all_vecs)),
            "angles_used": _i(len(kept)),
            "received": _i(len(frame_bufs)),
            "accepted": _i(len(accepted)),
            "best_score": round(best_score, 3),
            "kept_scores": [d["score"] for d in kept_details],
            "threshold": C.ENROLL_ACCEPT,
            "index_total": _i(total),
        }

    # -------------------------------------------------- recognition
    def _match_single(self, img, business_id=None):
        res = self._embed(img)
        if res is None:
            return None
        emb, bbox, _ = res

        query_filter = None
        if business_id:
            query_filter = Filter(
                must=[FieldCondition(key="businessId", match=MatchValue(value=business_id))]
            )

        # New correct method for latest qdrant-client
        response = self.client.query_points(
            collection_name=self.collection,
            query=emb[0].tolist() if emb.ndim > 1 else emb.tolist(),
            query_filter=query_filter,
            limit=C.TOPK,
            score_threshold=C.SIM_THRESHOLD,
        )
        hits = response.points

        if not hits:
            return {"matched": False, "bbox": bbox}

        votes = {}
        for hit in hits:
            payload = hit.payload
            sid = payload["studentId"]
            if sid not in votes:
                votes[sid] = [0.0, 0, payload["name"], payload["businessId"]]
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
        for buf in frame_bufs:
            img = self._decode(buf)
            if img is None:
                continue
            per_frame.append(self._match_single(img, business_id))

        hits = [r for r in per_frame if r and r.get("matched")]
        checked = len([r for r in per_frame if r is not None])

        if not hits:
            return {
                "exists": False,
                "studentId": None,
                "name": None,
                "businessId": None,          # ← add this
                "probability": 0.0,
                "avg_score": 0.0,
                "votes": 0,
                "checked": _i(checked or len(frame_bufs)),
            }

        tally = {}
        for r in hits:
            sid = r["studentId"]
            if sid not in tally:
                # also keep businessId
                tally[sid] = [0, 0.0, r["name"], r["businessId"]]
            tally[sid][0] += 1
            tally[sid][1] += _f(r["score"])

        sid = max(tally, key=lambda k: (tally[k][0], tally[k][1]))
        count, sum_score, name, bid = tally[sid]   # ← now 4 values
        denom = max(checked, 1)

        agreement = count / denom
        avg_score = sum_score / count
        probability = round(0.5 * agreement + 0.5 * avg_score, 4)
        exists = (count >= (denom // 2 + denom % 2)) if denom > 1 else True

        return {
            "exists": bool(exists),
            "studentId": sid,
            "name": name,
            "businessId": bid,               # ← important
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
                limit=256,
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