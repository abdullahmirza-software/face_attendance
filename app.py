import os
import json
import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from starlette.concurrency import run_in_threadpool

import config as C
from engine import FaceEngine

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(C.DATA_DIR, exist_ok=True)

# ---------- LOW_RAM switch ----------
LOW_RAM = bool(getattr(C, "LOW_RAM", True))

if LOW_RAM:
    MAX_FRAMES = min(getattr(C, "BURST_TARGET", 18) + 2, 22)
    MAX_FRAME_BYTES = getattr(C, "MAX_FRAME_BYTES", 300_000)   # ~300 KB
    MAX_LOGIN_LOGS = getattr(C, "MAX_LOGIN_LOGS", 300)
    LIMIT_CONCURRENCY = 6
    ENABLE_DOCS = False
else:
    MAX_FRAMES = getattr(C, "BURST_TARGET", 28) + 6
    MAX_FRAME_BYTES = getattr(C, "MAX_FRAME_BYTES", 600_000)
    MAX_LOGIN_LOGS = getattr(C, "MAX_LOGIN_LOGS", 2000)
    LIMIT_CONCURRENCY = 20
    ENABLE_DOCS = True

app = FastAPI(
    title="Face Attendance API",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
)

engine = FaceEngine()


def _read_logins():
    if not os.path.exists(C.LOGIN_LOG):
        return []
    try:
        with open(C.LOGIN_LOG, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _append_login(entry: dict):
    logs = _read_logins()
    logs.append(entry)
    if len(logs) > MAX_LOGIN_LOGS:
        logs = logs[-MAX_LOGIN_LOGS:]
    # compact JSON when LOW_RAM
    separators = (",", ":") if LOW_RAM else None
    with open(C.LOGIN_LOG, "w") as f:
        json.dump(logs, f, indent=None if LOW_RAM else 2, separators=separators)


@app.exception_handler(Exception)
async def _all_errors(request: Request, exc: Exception):
    print(f"[ERROR] {type(exc).__name__}: {exc}")
    return JSONResponse(
        {"ok": False, "error": type(exc).__name__, "detail": str(exc)},
        status_code=500,
    )


@app.post("/api/check-frame")
async def check_frame(frame: UploadFile = File(...)):
    buf = await frame.read()
    if len(buf) > MAX_FRAME_BYTES:
        raise HTTPException(413, "frame too large")
    result = await run_in_threadpool(engine.check_frame, buf)
    return JSONResponse(result)


@app.post("/api/enroll")
async def enroll(
    businessId: str = Form(...),
    studentId: str = Form(...),
    name: str = Form(...),
    frames: List[UploadFile] = File(...),
):
    if not frames:
        raise HTTPException(400, "no frames provided")
    if len(frames) > MAX_FRAMES:
        raise HTTPException(400, f"too many frames (max {MAX_FRAMES})")

    bufs = []
    for f in frames:
        data = await f.read()
        if len(data) > MAX_FRAME_BYTES:
            continue
        if data:
            bufs.append(data)
        if len(bufs) >= MAX_FRAMES:
            break

    if not bufs:
        raise HTTPException(400, "no usable frames")

    result = await run_in_threadpool(
        engine.enroll, businessId.strip(), studentId.strip(), name.strip(), bufs
    )
    del bufs

    if not result.get("ok"):
        return JSONResponse(result, status_code=422)
    return JSONResponse(result)


@app.post("/api/verify")
async def verify(
    frames: List[UploadFile] = File(...),
    businessId: Optional[str] = Form(None),
):
    if not frames:
        raise HTTPException(400, "no frames provided")
    if len(frames) > MAX_FRAMES:
        raise HTTPException(400, f"too many frames (max {MAX_FRAMES})")

    bufs = []
    for f in frames:
        data = await f.read()
        if len(data) > MAX_FRAME_BYTES:
            continue
        if data:
            bufs.append(data)
        if len(bufs) >= MAX_FRAMES:
            break

    if not bufs:
        raise HTTPException(400, "no usable frames")

    bid = businessId.strip() if businessId else None
    result = await run_in_threadpool(engine.recognize, bufs, business_id=bid)
    del bufs

    if result.get("exists"):
        entry = {
            "businessId": result.get("businessId") or bid,
            "studentId": result["studentId"],
            "name": result["name"],
            "probability": result["probability"],
            "avg_score": result["avg_score"],
            "loginTime": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        _append_login(entry)
        result["logged"] = entry

    return JSONResponse(result)


@app.get("/api/config")
async def client_config():
    return JSONResponse({
        "burstTarget": C.BURST_TARGET,
        "burstKeep": C.BURST_KEEP,
        "burstMaxMs": C.BURST_MAX_MS,
        "enrollAccept": C.ENROLL_ACCEPT,
        "captureMaxW": getattr(C, "CAPTURE_MAX_W", 640 if LOW_RAM else 720),
        "captureQuality": getattr(C, "CAPTURE_QUALITY", 0.72 if LOW_RAM else 0.82),
        "lowRam": LOW_RAM,
    })


@app.get("/api/students")
async def students():
    return JSONResponse(await run_in_threadpool(engine.list_students))


@app.get("/api/logins")
async def logins():
    return JSONResponse(_read_logins())


app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=C.HOST,
        port=C.PORT,
        reload=False,
        workers=1,
        limit_concurrency=LIMIT_CONCURRENCY,
        timeout_keep_alive=12 if LOW_RAM else 30,
    )