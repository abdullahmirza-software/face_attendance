"""
config.py — every tunable read from environment / .env

Loaded once at import. Override any value in a .env file next to this module
(see .env.example), or via real environment variables (which win over .env).
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the working dir / project root if present
except Exception:
    pass  # dotenv optional; env vars still work without it


def _str(key, default):
    return os.getenv(key, default)

def _int(key, default):
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return int(default)

def _float(key, default):
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return float(default)

def _list(key, default):
    raw = os.getenv(key)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- ONNX / models ----
PROVIDERS = _list("PROVIDERS", ["CPUExecutionProvider"])
DET_CONF = _float("DET_CONF", 0.5)            # SCRFD detector confidence
EMBED_DIM = _int("EMBED_DIM", 512)

# ---- storage paths ----
DATA_DIR = _str("DATA_DIR", os.path.join(_HERE, "data"))
INDEX_PATH = _str("INDEX_PATH", os.path.join(DATA_DIR, "faiss_index.bin"))
META_PATH = _str("META_PATH", os.path.join(DATA_DIR, "faiss_meta.json"))
LOGIN_LOG = _str("LOGIN_LOG", os.path.join(DATA_DIR, "login_log.json"))

# ---- enrollment quality gates ----
DET_SCORE_MIN = _float("DET_SCORE_MIN", 0.55)     # reject weak detections
MIN_DIVERSITY = _float("MIN_DIVERSITY", 0.92)     # dedupe near-identical angles
SHARP_MIN = _float("SHARP_MIN", 40.0)             # min Laplacian variance
AREA_MIN = _float("AREA_MIN", 0.06)               # face too-far threshold
AREA_MAX = _float("AREA_MAX", 0.75)               # face too-close threshold
CENTER_TOL = _float("CENTER_TOL", 0.25)           # centering tolerance (frac of w/h)

# ---- enrollment burst / selection ----
BURST_TARGET = _int("BURST_TARGET", 20)          # frames to record per enrollment burst
BURST_KEEP = _int("BURST_KEEP", 8)               # best frames to actually store
ENROLL_ACCEPT = _float("ENROLL_ACCEPT", 0.80)    # min quality score (0..1) to accept a frame
BURST_MAX_MS = _int("BURST_MAX_MS", 6000)        # safety cap on burst duration (client side)

# ---- recognition ----
SIM_THRESHOLD = _float("SIM_THRESHOLD", 0.62)     # cosine sim to accept a match
TOPK = _int("TOPK", 5)                             # top-k vectors that vote

# ---- server ----
HOST = _str("HOST", "0.0.0.0")
PORT = _int("PORT", 8000)
