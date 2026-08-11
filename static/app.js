/* ------------------------------------------------------------------
   Sentry face attendance — frontend logic
   - two modes: verify (check-in) and enroll (register from webcam video)
   - phone-style live guidance during enrollment via /api/check-frame
   - captures diverse frames, posts to FastAPI, renders result + probability
------------------------------------------------------------------ */

const API = "";  // same origin

// ---- tiny helpers ----
const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Grab a JPEG blob from a <video> at reduced size (fast uploads, less lag)
function grabBlob(video, maxW = 640, quality = 0.88) {
  const scale = Math.min(1, maxW / video.videoWidth);
  const c = document.createElement("canvas");
  c.width = Math.round(video.videoWidth * scale);
  c.height = Math.round(video.videoHeight * scale);
  const ctx = c.getContext("2d");
  ctx.drawImage(video, 0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob((b) => res(b), "image/jpeg", quality));
}

// ---- camera manager (one shared stream, attached per active view) ----
const cam = {
  stream: null,
  async start(videoEl) {
    if (!this.stream) {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 720 }, height: { ideal: 960 }, facingMode: "user" },
        audio: false,
      });
    }
    videoEl.srcObject = this.stream;
    await videoEl.play().catch(() => {});
  },
};

/* ================================================================
   MODE SWITCHING
================================================================ */
const views = { verify: $("#view-verify"), enroll: $("#view-enroll") };
let currentMode = "verify";

document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;
    views.verify.classList.toggle("d-none", currentMode !== "verify");
    views.enroll.classList.toggle("d-none", currentMode !== "enroll");
    if (currentMode === "verify") { abortCapture(); await cam.start($("#verifyVideo")); }
    else await cam.start($("#enrollVideo"));
  });
});

/* ================================================================
   VERIFY MODE
================================================================ */
const verifyVideo = $("#verifyVideo");
const verifyBtn = $("#verifyBtn");
const verifyHint = $("#verifyHint");

async function initVerify() {
  try {
    await cam.start(verifyVideo);
    verifyHint.textContent = "Ready — center your face";
    verifyBtn.disabled = false;
  } catch (e) {
    verifyHint.textContent = "Camera blocked. Allow access and reload.";
  }
}

verifyBtn.addEventListener("click", async () => {
  verifyBtn.disabled = true;
  const frame = $("#verifyFrame");
  frame.classList.add("scanning");
  verifyHint.textContent = "Scanning…";

  try {
    // capture a few frames over ~0.7s for a robust vote
    const blobs = [];
    for (let i = 0; i < 4; i++) {
      blobs.push(await grabBlob(verifyVideo));
      await sleep(160);
    }

    const fd = new FormData();
    blobs.forEach((b, i) => fd.append("frames", b, `f${i}.jpg`));

    // ---------- NEW: send businessId if filled ----------
    const businessId = $("#verifyBusiness")?.value.trim();
    if (businessId) {
      fd.append("businessId", businessId);
    }
    // ----------------------------------------------------

    const res = await fetch(`${API}/api/verify`, { method: "POST", body: fd });
    const data = await res.json();
    renderVerifyResult(data);
    if (data.exists) loadLogins();
  } catch (e) {
    renderVerifyError();
  } finally {
    frame.classList.remove("scanning");
    verifyHint.textContent = "Ready — center your face";
    verifyBtn.disabled = false;
  }
});

function renderVerifyResult(d) {
  const box = $("#verifyResult");
  if (d.exists) {
    const pct = Math.round(d.probability * 100);
    box.innerHTML = `
      <div class="res">
        <span class="res-badge ok"><i class="bi bi-check-circle-fill"></i> Match found</span>
        <h3 class="res-name">${escapeHtml(d.name)}</h3>
        <div class="meter-row"><span>Business</span><b>${d.businessId || "—"}</b></div>
        <div class="res-id">Student #${d.studentId}</div>
        <div class="lbl">Confidence</div>
        <div class="meter"><i style="width:${pct}%"></i></div>
        <div class="meter-row"><span>Probability</span><b>${pct}%</b></div>
        <div class="meter-row"><span>Frames agreed</span><b>${d.votes}/${d.checked}</b></div>
        <div class="meter-row"><span>Avg similarity</span><b>${d.avg_score}</b></div>
        <div class="meter-row"><span>Signed in</span><b>${d.logged ? fmtTime(d.logged.loginTime) : "—"}</b></div>
      </div>`;
  } else {
    box.innerHTML = `
      <div class="res">
        <span class="res-badge no"><i class="bi bi-x-circle-fill"></i> Not recognized</span>
        <h3 class="res-name">Unknown face</h3>
        <div class="res-id">No enrolled student matched (${d.checked} frame${d.checked === 1 ? "" : "s"} checked).</div>
        <p class="text-muted mt-3 mb-0" style="font-size:14px">If this is a new student, switch to <b>Register</b> to enroll them.</p>
      </div>`;
  }
}
function renderVerifyError() {
  $("#verifyResult").innerHTML = `
    <div class="res"><span class="res-badge no"><i class="bi bi-wifi-off"></i> Request failed</span>
    <p class="text-muted mt-2">Could not reach the server. Check it's running and try again.</p></div>`;
}

/* ================================================================
   ENROLL MODE
================================================================ */
const enrollVideo = $("#enrollVideo");
const enrollBtn = $("#enrollBtn");
const enrollHint = $("#enrollHint");
const enrollFrame = $("#enrollFrame");
const captureRing = $("#captureRing");
const ringFg = $("#ringFg");
const ringCount = $("#ringCount");
const quality = $("#enrollQuality");

// burst capture config (mirrors server .env defaults; server is source of truth)
let BURST_TARGET = 20;   // frames to record
let BURST_KEEP = 8;
let BURST_MAX_MS = 6000; // safety time cap
const RING_CIRCUM = 339;       // 2*pi*54

let liveTimer = null;
let formOK = false;

function validateForm() {
  formOK =
    $("#fBusiness").value.trim() !== "" &&
    $("#fStudent").value.trim() !== "" &&
    $("#fName").value.trim() !== "";
  enrollBtn.disabled = !formOK;
  if (formOK && enrollHint.textContent === "Fill in details to begin")
    enrollHint.textContent = "Ready — press Start capture";
}
["fBusiness", "fStudent", "fName"].forEach((id) =>
  $("#" + id).addEventListener("input", validateForm)
);

// UUID generator (browser-native, with fallback)
function uuid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}
$("#genStudent").addEventListener("click", () => {
  $("#fStudent").value = uuid();
  validateForm();
});

// Live quality feedback loop (throttled) — phone-style guidance
// Live guidance is driven from inside the capture burst (see runCapture).
// This is intentionally NOT on a background timer, so check-frame is never
// called while idle. Kept as a no-op to avoid touching other references.
async function liveGuide() { /* disabled: burst-driven guidance only */ }

function paintQuality(q) {
  quality.hidden = false;
  const face = $("#qFace"), dist = $("#qDist"), sharp = $("#qSharp");
  const msg = {
    no_face: ["No face", "bad"], low_confidence: ["Weak", "bad"],
    too_far: ["Move closer", "bad"], too_close: ["Move back", "bad"],
    not_centered: ["Center face", "bad"], blurry: ["Hold still", "bad"],
    ok: ["Good", "good"], bad_image: ["—", "bad"],
  };
  const [label, cls] = msg[q.reason] || ["—", "bad"];

  face.textContent = q.reason === "no_face" ? "Not found" : "Detected";
  face.className = q.reason === "no_face" ? "bad" : "good";
  dist.textContent = label; dist.className = cls;
  sharp.textContent = q.sharpness != null ? q.sharpness : "—";
  sharp.className = (q.sharpness || 0) > 40 ? "good" : "bad";

  enrollFrame.classList.toggle("good", !!q.ok);
  enrollFrame.classList.toggle("bad", !q.ok && q.reason !== "no_face");
  enrollHint.textContent = q.ok ? "Great — keep turning slowly" : label;
  enrollHint.className = "cam-hint " + (q.ok ? "good" : "bad");
}

async function initEnroll() {
  try {
    await cam.start(enrollVideo);
    validateForm();
    // NOTE: no live-guide polling here. check-frame is only called while a
    // capture is running, so the network stays idle until you press Start.
    stopLiveGuide();
    enrollHint.textContent = formOK ? "Ready — press Start capture" : "Fill in details to begin";
  } catch (e) {
    enrollHint.textContent = "Camera blocked. Allow access and reload.";
  }
}

function stopLiveGuide() {
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
}

enrollBtn.addEventListener("click", runCapture);

let capturing = false;

async function runCapture() {
  if (!formOK || capturing) return;
  capturing = true;
  enrollBtn.disabled = true;
  captureRing.hidden = false;
  enrollFrame.classList.add("scanning");
  $("#enrollResult").innerHTML = "";
  enrollHint.className = "cam-hint";
  enrollHint.textContent = "Recording… slowly turn your head";
  stopLiveGuide(); // ensure no duplicate polling

  const blobs = [];
  const started = performance.now();

  try {
    // ---- rapid burst; auto-stops at BURST_TARGET or the safety time cap ----
    for (let i = 0; i < BURST_TARGET; i++) {
      if (!capturing) break;                 // aborted (e.g. mode switch / error)
      const blob = await grabBlob(enrollVideo, window.CAPTURE_MAX_W || 640, window.CAPTURE_QUALITY || 0.88);
      blobs.push(blob);
      const done = i + 1;
      ringCount.textContent = done;
      ringFg.style.strokeDashoffset = RING_CIRCUM * (1 - done / BURST_TARGET);

      // live guidance during recording only: reuse this frame, ~every 4th, fire-and-forget
      if (i % 4 === 0) {
        const qfd = new FormData();
        qfd.append("frame", blob, "q.jpg");
        fetch(`${API}/api/check-frame`, { method: "POST", body: qfd })
          .then((r) => r.json())
          .then((q) => { if (capturing) paintQuality(q); })
          .catch(() => {});
      }
      await sleep(40);
      if (performance.now() - started > BURST_MAX_MS) break;
    }

    // ---- auto-stop recording, then process the whole burst at once ----
    enrollFrame.classList.remove("scanning");
    enrollHint.textContent = "Selecting best shots…";

    const fd = new FormData();
    fd.append("businessId", $("#fBusiness").value.trim());
    fd.append("studentId", $("#fStudent").value.trim());
    fd.append("name", $("#fName").value.trim());
    blobs.forEach((b, i) => fd.append("frames", b, `a${i}.jpg`));

    const res = await fetch(`${API}/api/enroll`, { method: "POST", body: fd });
    if (!res.ok && res.status >= 500) {
      // server error -> stop cleanly, surface it
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `server ${res.status}`);
    }
    const data = await res.json();
    renderEnrollResult(data);
    loadStudents();
  } catch (e) {
    // ANY failure: stop recording + detecting immediately, show a clear message
    abortCapture();
    $("#enrollResult").innerHTML = errBox(
      "Capture stopped — " + (e && e.message ? e.message : "something went wrong") + ". Press Start to try again."
    );
  } finally {
    captureRing.hidden = true;
    ringFg.style.strokeDashoffset = RING_CIRCUM;
    ringCount.textContent = "0";
    enrollFrame.classList.remove("scanning");
    enrollBtn.disabled = false;
    capturing = false;
    stopLiveGuide(); // idle again — no background check-frame calls
  }
}

// Hard stop: kills the burst loop and any polling. Used on error / mode change.
function abortCapture() {
  capturing = false;
  stopLiveGuide();
  enrollFrame.classList.remove("scanning", "good", "bad");
}

function renderEnrollResult(d) {
  const box = $("#enrollResult");
  if (d.ok) {
    const scores = (d.kept_scores || []).map((x) => Math.round(x * 100) + "%").join(", ");
    box.innerHTML = `
      <div class="res" style="margin-top:18px">
        <span class="res-badge ok"><i class="bi bi-check-circle-fill"></i> Enrolled</span>
        <h3 class="res-name" style="font-size:22px">${escapeHtml(d.name)}</h3>
        <div class="res-id">Student ${shortId(d.studentId)}</div>
        <div class="meter-row"><span>Frames received</span><b>${d.received}</b></div>
        <div class="meter-row"><span>Passed threshold (${Math.round(d.threshold*100)}%)</span><b>${d.accepted}</b></div>
        <div class="meter-row"><span>Angles stored</span><b>${d.angles_used}</b></div>
        <div class="meter-row"><span>Best frame score</span><b>${Math.round(d.best_score*100)}%</b></div>
        ${scores ? `<div class="meter-row"><span>Kept scores</span><b style="font-size:12px">${scores}</b></div>` : ""}
        <div class="meter-row"><span>Vectors in FAISS</span><b>${d.vectors}</b></div>
      </div>`;
  } else if (d.reason === "below_threshold") {
    box.innerHTML = errBox(
      `No frame reached the ${Math.round((d.threshold||0.8)*100)}% quality bar ` +
      `(best was ${Math.round((d.best_score||0)*100)}%). Improve lighting, hold steady, ` +
      `and fill the circle with your face, then retry.`
    );
  } else if (d.reason === "no_usable_faces") {
    box.innerHTML = errBox(
      `No clear face detected in ${d.received || "the"} frames. Move closer, ` +
      `ensure your face is well-lit and centered, then retry.`
    );
  } else {
    box.innerHTML = errBox("Could not enroll. Please retry.");
  }
}

/* ================================================================
   STATS + LOGS
================================================================ */
async function loadStudents() {
  try {
    const r = await fetch(`${API}/api/students`);
    const list = await r.json();
    $("#statStudents").textContent = list.length;
  } catch (_) {}
}

async function loadLogins() {
  try {
    const r = await fetch(`${API}/api/logins`);
    const logs = await r.json();
    const today = new Date().toDateString();
    const todays = logs.filter((l) => new Date(l.loginTime).toDateString() === today);
    $("#statLogins").textContent = todays.length;

    const body = $("#logBody");
    if (!logs.length) {
      body.innerHTML = `<tr><td colspan="4" class="text-muted text-center py-4">No sign-ins yet.</td></tr>`;
      return;
    }
    body.innerHTML = logs.slice().reverse().slice(0, 12).map((l) => `
      <tr>
        <td>${escapeHtml(l.name)}</td>
        <td class="text-muted">#${l.studentId}</td>
        <td><span class="pill">${Math.round(l.probability * 100)}%</span></td>
        <td class="text-muted">${fmtTime(l.loginTime)}</td>
      </tr>`).join("");
  } catch (_) {}
}
$("#refreshLogs").addEventListener("click", loadLogins);

/* ================================================================
   UTIL
================================================================ */
function shortId(id) {
  const s = String(id || "");
  return s.length > 12 ? `#${s.slice(0, 8)}…` : `#${s}`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
  catch { return iso; }
}
function errBox(msg) {
  return `<div class="res" style="margin-top:18px">
    <span class="res-badge no"><i class="bi bi-exclamation-triangle-fill"></i> Error</span>
    <p class="text-muted mt-2 mb-0">${escapeHtml(msg)}</p></div>`;
}

/* ================================================================
   BOOT
================================================================ */
async function loadConfig() {
  try {
    const r = await fetch(`${API}/api/config`);
    const c = await r.json();
    if (c.burstTarget) BURST_TARGET = c.burstTarget;
    if (c.burstKeep) BURST_KEEP = c.burstKeep;
    if (c.burstMaxMs) BURST_MAX_MS = c.burstMaxMs;
    // optional
    if (c.captureMaxW) window.CAPTURE_MAX_W = c.captureMaxW;
    if (c.captureQuality) window.CAPTURE_QUALITY = c.captureQuality;
  } catch (_) {}
}

(async function boot() {
  await loadConfig();
  await initVerify();
  await initEnroll();  // sets up live-guide loop; only fires while enroll view active
  loadStudents();
  loadLogins();
})();
