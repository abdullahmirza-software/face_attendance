# Performance session notes — 2026-08-11/12

Context: on Render (0.1 vCPU / 512 MB), `check-frame`/`verify`/`enroll` were taking
seconds to minutes (`enroll` up to ~2 min, `check-frame` up to ~30s, wildly
inconsistent run to run).

## Root causes found

1. **SCRFD always ran the heaviest model at full resolution.** `uniface`'s
   `SCRFD.detect()` letterboxes every frame up to its own `input_size`
   (default 640×640) before inference, regardless of any pre-resize we did —
   so `MAX_SIDE` had zero effect on detector cost. It also defaulted to the
   `SCRFD_10G` variant, the biggest one in the family.
2. **ONNXRuntime/OpenCV defaulted to multi-threaded execution**, sized off the
   host's full core count. Under a 0.1 vCPU cgroup quota that causes thread
   scheduling thrash — the likely cause of the wild run-to-run variance
   (7s vs 30s for the same call).
3. **FastAPI routes were `async def` calling fully synchronous, CPU-bound
   code inline**, blocking the single event loop for the whole request.
   Concurrent `check-frame` calls (fired every 5th frame during an enroll
   capture burst) queued up behind each other and behind the following
   `enroll` call instead of overlapping.
4. **No backpressure on the client** — a new `check-frame` call was fired
   every N frames regardless of whether the previous one had returned yet,
   so multiple could be in flight and pile up together.
5. **No serialization of CPU-bound work server-side** — under a CPU quota
   this small, two "concurrent" inference calls don't run in parallel, they
   just fight each other for the same sliver of CPU, which is *slower*
   overall than running them back-to-back.

## Fixes applied

- `engine.py`
  - Force single-threaded execution: `cv2.setNumThreads(1)` and a
    monkeypatched `ort.SessionOptions` (`intra_op_num_threads=1`,
    `inter_op_num_threads=1`, sequential execution mode).
  - Switch detector to `SCRFD_500M` at 320×320 input (`DET_MODEL` /
    `DET_INPUT_SIZE` env vars, defaults `scrfd_500m` / `320`), with a
    fallback to the old default if an incompatible `uniface` version is
    installed. Benchmarked ~95ms → ~2.6ms per detection on an unconstrained
    machine (36×) — the real-world gain on a throttled box should be larger.
  - Added `self._infer_lock` (a plain `threading.Lock`) around the CPU-bound
    portions of `check_frame`, `_quality`, and `_match_single` so inference
    work across all requests is serialized instead of thrashing. The Qdrant
    network call in `_match_single` stays outside the lock.
  - Added a startup warm-up pass (`_warmup`) that exercises the detector,
    `_enhance()`, and the recognizer once at boot, so the first real request
    doesn't pay ONNX/OpenCV lazy-init cost (measured ~147ms → ~12ms for the
    first `_enhance()` call).
  - Added `DEBUG_TIMING=1` (env-gated, off by default) — logs a per-stage ms
    breakdown (decode/resize/enhance/detect/embed/qdrant/lock-wait) for every
    check-frame/quality/match call, plus `enroll`'s scan and Qdrant-write
    totals. Use this to see exactly where time goes on the real Render
    instance, since local benchmarks don't reflect its actual throttling
    behavior.
- `config.py` / `.env.example`
  - New knobs: `DET_MODEL`, `DET_INPUT_SIZE`, `DEBUG_TIMING`.
- `app.py`
  - Wrapped the blocking `engine.check_frame` / `engine.enroll` /
    `engine.recognize` / `engine.list_students` calls in
    `starlette.concurrency.run_in_threadpool` so the async event loop is
    never blocked by CPU-bound work.
- `static/app.js`
  - Added a `guideInFlight` guard so at most one `check-frame` call is ever
    in flight during an enroll capture burst, instead of firing regardless
    of whether the previous call has returned.

## Verified locally

- All three files compile; the app was actually started with `uvicorn`
  (real deps installed in a scratch venv) and hit with real HTTP requests,
  not just read for correctness.
- Confirmed the inference lock queues concurrent `check-frame` calls in
  order (`wait_lock` growing 0 → 25 → 53 → 85ms) instead of letting them
  thrash concurrently.
- Confirmed `enroll` and `check-frame` still return correct results
  (`no_face` / `no_usable_faces` on synthetic no-face test images).
- Benchmarked SCRFD_10G@640 vs SCRFD_500M@320: 95ms → 2.6ms per call.

## Results so far (from the live Render instance, between rounds of fixes)

| Endpoint | Before | After round 1 |
|---|---|---|
| `enroll` | ~2 min | ~12.8s |
| `check-frame` | 7–30s (erratic) | ~5s (consistent) |
| `verify` | ~24s | ~1.9s |

Round 2 (inference lock + client-side guard + timing) had not yet been
measured on the live instance as of writing this file.

## Next steps

1. Redeploy with `DEBUG_TIMING=1` set on Render, trigger one enroll burst,
   and capture the `[TIMING]` lines from Render's log viewer. That's the
   fastest way to find out where any remaining time is going (detection?
   Qdrant network latency? lock contention?) instead of guessing further.
2. There is a hard ceiling here: on a 0.1 vCPU instance, no amount of
   software optimization gives real parallelism. If consistent sub-second
   responses under concurrent users are required, upgrading the Render plan
   is the highest-leverage remaining lever.
3. **Security note (unrelated to perf, still open):** `.env.example` is
   tracked in git and contains what looks like a live Qdrant Cloud URL +
   API key, in a repo tied to a public demo link. Recommend rotating that
   key and replacing it with a placeholder.
