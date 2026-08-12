/* ------------------------------------------------------------------
   Sentry face attendance — frontend (optimized)
   Respects LOW_RAM from /api/config
------------------------------------------------------------------ */

const API = "";

// ---- helpers ----
const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Reuse one canvas (big win — no alloc every frame)
const _canvas = document.createElement("canvas");
const _ctx = _canvas.getContext("2d", { alpha: false });

function grabBlob(video, maxW = 640, quality = 0.75) {
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const scale = Math.min(1, maxW / vw);
  const w = Math.round(vw * scale);
  const h = Math.round(vh * scale);

  if (_canvas.width !== w || _canvas.height !== h) {
    _canvas.width = w;
    _canvas.height = h;
  }
  _ctx.drawImage(video, 0, 0, w, h);
  return new Promise((res) =>
    _canvas.toBlob((b) => res(b), "image/jpeg", quality)
  );
}

// ---- camera ----
const cam = {
  stream: null,
  async start(videoEl) {
    if (!this.stream) {
      // Lower ideal resolution on low-RAM mode
      const idealW = window.LOW_RAM ? 560 : 720;
      const idealH = window.LOW_RAM ? 720 : 960;
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: idealW },
          height: { ideal: idealH },
          facingMode: "user",
        },
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
    if (currentMode === "verify") {
      abortCapture();
      await cam.start($("#verifyVideo"));
    } else {
      await cam.start($("#enrollVideo"));
    }
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
  const businessId = $("#verifyBusiness")?.value.trim() || "";

  // Allow empty (search all) or a valid UUID v4
  if (businessId && !isValidUuid(businessId)) {
    renderVerifyError("Invalid Business ID — must be a valid UUID v4 or left empty.");
    return;
  }

  verifyBtn.disabled = true;
  const frame = $("#verifyFrame");
  frame.classList.add("scanning");
  verifyHint.textContent = "Scanning…";

  try {
    // Fewer frames + lower quality when LOW_RAM
    const nFrames = window.LOW_RAM ? 3 : 4;
    const maxW = window.CAPTURE_MAX_W || (window.LOW_RAM ? 512 : 640);
    const quality = window.CAPTURE_QUALITY || (window.LOW_RAM ? 0.70 : 0.80);

    const blobs = [];
    for (let i = 0; i < nFrames; i++) {
      blobs.push(await grabBlob(verifyVideo, maxW, quality));
      await sleep(window.LOW_RAM ? 120 : 150);
    }

    const fd = new FormData();
    blobs.forEach((b, i) => fd.append("frames", b, `f${i}.jpg`));

    if (businessId) fd.append("businessId", businessId);

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
const qualityEl = $("#enrollQuality");

let BURST_TARGET = 18;
let BURST_KEEP = 4;
let BURST_MAX_MS = 7000;
const RING_CIRCUM = 339;

let liveTimer = null;
let formOK = false;
let capturing = false;
let guideInFlight = false;

function isValidUuid(v) {
  return /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[4][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$/.test(v.trim());
}

function validateForm() {
  const business = $("#fBusiness").value.trim();
  const student  = $("#fStudent").value.trim();
  const name     = $("#fName").value.trim();

  formOK =
    isValidUuid(business) &&
    isValidUuid(student) &&
    name.length > 0 &&
    /\S/.test(name);

  enrollBtn.disabled = !formOK;

  if (formOK && enrollHint.textContent === "Fill in details to begin") {
    enrollHint.textContent = "Ready — press Start capture";
  }
}

["fBusiness", "fStudent", "fName"].forEach((id) =>
  $("#" + id).addEventListener("input", validateForm)
);

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

function paintQuality(q) {
  qualityEl.hidden = false;
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

async function runCapture() {
  if (!formOK || capturing) return;
  capturing = true;
  enrollBtn.disabled = true;
  captureRing.hidden = false;
  enrollFrame.classList.add("scanning");
  $("#enrollResult").innerHTML = "";
  enrollHint.className = "cam-hint";
  enrollHint.textContent = "Recording… slowly turn your head";
  stopLiveGuide();

  const maxW = window.CAPTURE_MAX_W || (window.LOW_RAM ? 512 : 640);
  const quality = window.CAPTURE_QUALITY || (window.LOW_RAM ? 0.70 : 0.80);
  const guideEvery = window.LOW_RAM ? 5 : 4;   // fewer check-frame calls

  const blobs = [];
  const started = performance.now();

  try {
    for (let i = 0; i < BURST_TARGET; i++) {
      if (!capturing) break;
      const blob = await grabBlob(enrollVideo, maxW, quality);
      blobs.push(blob);

      const done = i + 1;
      ringCount.textContent = done;
      ringFg.style.strokeDashoffset = RING_CIRCUM * (1 - done / BURST_TARGET);

      // Live guidance — only every N frames, and only if the previous
      // check-frame call has already returned. The backend has ~0.1 CPU;
      // firing another one before the last finishes just makes both queue
      // up behind each other instead of either finishing sooner.
      if (i % guideEvery === 0 && !guideInFlight) {
        guideInFlight = true;
        const qfd = new FormData();
        qfd.append("frame", blob, "q.jpg");
        fetch(`${API}/api/check-frame`, { method: "POST", body: qfd })
          .then((r) => r.json())
          .then((q) => { if (capturing) paintQuality(q); })
          .catch(() => {})
          .finally(() => { guideInFlight = false; });
      }

      await sleep(window.LOW_RAM ? 50 : 40);
      if (performance.now() - started > BURST_MAX_MS) break;
    }

    enrollFrame.classList.remove("scanning");
    enrollHint.textContent = "Selecting best shots…";

    const fd = new FormData();
    const bizVal = $("#fBusiness").value.trim();
    if (bizVal) fd.append("businessId", bizVal);
    fd.append("studentId", $("#fStudent").value.trim());
    fd.append("name", $("#fName").value.trim());
    blobs.forEach((b, i) => fd.append("frames", b, `a${i}.jpg`));

    const res = await fetch(`${API}/api/enroll`, { method: "POST", body: fd });
    if (!res.ok && res.status >= 500) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `server ${res.status}`);
    }
    const data = await res.json();
    renderEnrollResult(data);
    loadStudents();
  } catch (e) {
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
    stopLiveGuide();
  }
}

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
        <div class="meter-row"><span>Vectors stored</span><b>${d.vectors}</b></div>
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
    if (c.captureMaxW) window.CAPTURE_MAX_W = c.captureMaxW;
    if (c.captureQuality) window.CAPTURE_QUALITY = c.captureQuality;
    if (typeof c.lowRam === "boolean") window.LOW_RAM = c.lowRam;
  } catch (_) {
    window.LOW_RAM = true; // safe default
  }
}

(async function boot() {
  await loadConfig();
  await initVerify();
  await initEnroll();
  loadStudents();
  loadLogins();
})();