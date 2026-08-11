"""
app.py — FastAPI backend. businessId / studentId are strings (UUIDs); name is a string.
All settings come from config.py (.env).
"""

import os
import json
import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import traceback
import config as C
from engine import FaceEngine

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(C.DATA_DIR, exist_ok=True)

app = FastAPI(title="Face Attendance API")


from fastapi.requests import Request as _Req


@app.exception_handler(Exception)
async def _all_errors(request: _Req, exc: Exception):
    tb = traceback.format_exc()
    print(tb)  # full traceback in server console
    return JSONResponse(
        {"ok": False, "error": type(exc).__name__, "detail": str(exc),
         "where": tb.strip().splitlines()[-3:]},
        status_code=500,
    )
engine = FaceEngine()


def _read_logins():
    if os.path.exists(C.LOGIN_LOG):
        with open(C.LOGIN_LOG, "r") as f:
            return json.load(f)
    return []


def _append_login(entry: dict):
    logs = _read_logins()
    logs.append(entry)
    with open(C.LOGIN_LOG, "w") as f:
        json.dump(logs, f, indent=2)


@app.post("/api/check-frame")
async def check_frame(frame: UploadFile = File(...)):
    buf = await frame.read()
    return JSONResponse(engine.check_frame(buf))


@app.post("/api/enroll")
async def enroll(
    businessId: str = Form(...),
    studentId: str = Form(...),
    name: str = Form(...),
    frames: List[UploadFile] = File(...),
):
    if not frames:
        raise HTTPException(400, "no frames provided")
    bufs = [await f.read() for f in frames]
    result = engine.enroll(businessId.strip(), studentId.strip(), name.strip(), bufs)
    if not result["ok"]:
        return JSONResponse(result, status_code=422)
    return JSONResponse(result)


@app.post("/api/verify")
async def verify(
    frames: List[UploadFile] = File(...),
    businessId: Optional[str] = Form(None),
):
    if not frames:
        raise HTTPException(400, "no frames provided")
    bufs = [await f.read() for f in frames]
    bid = businessId.strip() if businessId else None
    result = engine.recognize(bufs, business_id=bid)

    if result.get("exists"):
        entry = {
            "businessId": result.get("businessId") or bid,   # ← prefer matched one
            "studentId": result["studentId"],
            "name": result["name"],
            "probability": result["probability"],
            "avg_score": result["avg_score"],
            "loginTime": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        _append_login(entry)
        result["logged"] = entry

    return JSONResponse(result)


@app.get("/api/config")
async def client_config():
    # expose only what the browser needs; server .env stays the source of truth
    return JSONResponse({
        "burstTarget": C.BURST_TARGET,
        "burstKeep": C.BURST_KEEP,
        "burstMaxMs": C.BURST_MAX_MS,
        "enrollAccept": C.ENROLL_ACCEPT,
        "captureMaxW": C.CAPTURE_MAX_W,
        "captureQuality": C.CAPTURE_QUALITY,
    })


@app.get("/api/students")
async def students():
    return JSONResponse(engine.list_students())


@app.get("/api/logins")
async def logins():
    return JSONResponse(_read_logins())


app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=C.HOST, port=C.PORT, reload=False)
