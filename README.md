# Sentry — Face Attendance (FastAPI + Web)

Two-mode face attendance system:

- **Register** — enroll a student from a short webcam video. The browser samples
  frames while the person slowly turns their head (phone-style guidance overlay);
  the backend keeps only diverse, sharp angles and stores each as a vector in FAISS.
- **Check in** — verify a live face against enrolled students. On a confident match
  it records the sign-in (name, studentId, probability, timestamp) to a JSON log.

`businessId` and `studentId` are **UUID strings** and `name` is a string. Because
UUIDs can't be packed into FAISS int64 ids, FAISS uses sequential int64 row ids and
`faiss_meta.json` maps each row id -> `{businessId, studentId, name}`. That file is
the source of truth for identity.

**All settings live in `.env`.** Copy `.env.example` to `.env` and edit — thresholds,
paths, ONNX providers, host/port. Real environment variables override `.env`.

## Install

```bash
cd face_attendance
python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
cp .env.example .env      # then edit as needed
```

> GPU: replace `uniface[cpu]` with `uniface[gpu]` in `requirements.txt`, and set
> `PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']` in `engine.py`.
> Keep `faiss-cpu` — the search is sub-millisecond at this scale; the GPU only
> helps the ONNX face models.

First run downloads the SCRFD + MobileFace ONNX weights automatically (needs
internet once).

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**.

> Webcam access needs a secure context. `localhost` counts as secure, so it works
> directly. To use another device on your network, put it behind HTTPS (a reverse
> proxy or `uvicorn --ssl-keyfile ... --ssl-certfile ...`), or browsers will block
> the camera.

## How it works (no-lag design)

- **One shared camera stream**, reused across both modes.
- **Downscaled JPEG frames** (≤480px) sent to the server — small uploads, fast decode.
- **Verify** captures 4 frames over ~0.7s and votes across them → a probability,
  not a single fragile guess.
- **Enroll** over-samples frames and lets the backend dedupe near-identical angles
  (`MIN_DIVERSITY`) and drop blurry/weak ones (`DET_SCORE_MIN`), so you get varied
  poses instead of 8 copies of the same frame. Each angle + a centroid go into FAISS.
- **Top-k voting** at match time (`TOPK=5`) lets multiple stored angles of one person
  reinforce each other, which is more robust than nearest-1.

## Tuning (.env)

| Variable | Meaning | Try |
|---|---|---|
| `SIM_THRESHOLD` | cosine sim to accept a match | 0.60 (loose) – 0.68 (strict) |
| `TOPK` | stored vectors that vote per query | 3–7 |
| `DET_SCORE_MIN` | reject weak detections on enroll | 0.5–0.65 |
| `MIN_DIVERSITY` | dedupe similar angles (higher = keep more) | 0.88–0.95 |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/check-frame` | per-frame quality feedback (face present / distance / sharpness) |
| POST | `/api/enroll` | `businessId, studentId, name, frames[]` → stores vectors |
| POST | `/api/verify` | `frames[]` (+ optional `businessId`) → match + probability, logs sign-in |
| GET | `/api/students` | enrolled students |
| GET | `/api/logins` | sign-in log |

## Data files (in `data/`)

- `faiss_index.bin` — the FAISS index (vectors + ids)
- `faiss_meta.json` — id → {businessId, studentId, name}
- `login_log.json` — sign-in records with probability + timestamp

## Note on the recognition backend

The scripts rely on `uniface`'s `SCRFD` (detection) and `MobileFace` (recognition).
If your installed uniface version exposes a different face-object attribute for
detection confidence than `.score`, adjust the `getattr(face, "score", 1.0)` calls
in `engine.py` — everything else is version-agnostic.