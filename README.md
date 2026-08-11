# Sentry — Face Attendance (FastAPI + Web)

Two-mode face attendance system:

- **Register** — enroll a student from a short webcam video. The browser samples
  frames while the person slowly turns their head (phone-style guidance overlay);
  the backend keeps only diverse, sharp angles and stores them in **Qdrant**.
- **Check in** — verify a live face against enrolled students. On a confident match
  it records the sign-in (name, studentId, businessId, probability, timestamp).

`businessId` and `studentId` are **UUID strings** and `name` is a string.  
All vectors are stored in an embedded Qdrant collection with payload filtering
by `businessId` / `studentId`.

**All settings live in `.env`.** Copy `.env.example` to `.env` and edit — thresholds,
paths, ONNX providers, host/port. Real environment variables override `.env`.

## Install

```bash
cd face_attendance
python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
cp .env.example .env      # then edit as needed