/* Interactive mask editor.
   Tools: Remove (click a mask), Draw outline (freehand contour -> fills), Pan, Fit.
   Wheel zooms; Undo and Save. In batch mode: Previous / Next / Save, with an
   unsaved-changes prompt (Save / Discard / Cancel) when navigating between images.
   Self-contained; exposes window.Editor.open(token, onSaved) and
   window.Editor.openBatch(list, index, onSavedIdx). */
(function () {
  const el = (id) => document.getElementById(id);

  let CUR_TOKEN = null, SINGLE_CB = null, BATCH = null, dirty = false;
  let W = 0, H = 0;
  let baseGray = null, labels = null, maxLabel = 0;
  let tool = "draw", history = [];
  let off = null, offCtx = null, offImg = null;
  let scale = 1, tx = 0, ty = 0, FIT = 1;
  let drawing = false, panning = false, panStart = null, pathPts = null;
  const colorCache = new Map();

  function colorOf(L) {
    let c = colorCache.get(L);
    if (!c) { c = [60 + (L * 67) % 196, 60 + (L * 97) % 196, 60 + (L * 137) % 196]; colorCache.set(L, c); }
    return c;
  }

  function decodePng(dataurl) {
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => {
        const c = document.createElement("canvas");
        c.width = im.width; c.height = im.height;
        const g = c.getContext("2d"); g.drawImage(im, 0, 0);
        resolve({ data: g.getImageData(0, 0, im.width, im.height).data, w: im.width, h: im.height });
      };
      im.src = dataurl;
    });
  }

  // ---- load one image into the editor ----
  async function loadImage(token) {
    const fd = new FormData(); fd.append("token", token);
    const r = await (await fetch("/api/single/editor", { method: "POST", body: fd })).json();
    if (r.error) { alert(r.error); return false; }
    CUR_TOKEN = token;
    W = r.w; H = r.h;
    const base = await decodePng(r.base), idm = await decodePng(r.idmap);
    baseGray = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++) baseGray[i] = base.data[i * 4];
    labels = new Int32Array(W * H); maxLabel = 0;
    for (let i = 0; i < W * H; i++) {
      const L = idm.data[i * 4] + (idm.data[i * 4 + 1] << 8) + (idm.data[i * 4 + 2] << 16);
      labels[i] = L; if (L > maxLabel) maxLabel = L;
    }
    off = document.createElement("canvas"); off.width = W; off.height = H;
    offCtx = off.getContext("2d"); offImg = offCtx.createImageData(W, H);
    history = []; colorCache.clear(); dirty = false;
    sizeCanvas(); setTool("draw"); renderComposite(); fitView();
    return true;
  }

  async function open(token, onSaved) {
    SINGLE_CB = onSaved; BATCH = null;
    el("editor").classList.add("show");
    updateNavUI();
    await loadImage(token);
  }

  async function openBatch(list, index, onSavedIdx) {
    BATCH = { list, index, onSavedIdx }; SINGLE_CB = null;
    el("editor").classList.add("show");
    if (await loadImage(list[index].token)) updateNavUI();
  }

  function close() { el("editor").classList.remove("show"); labels = null; baseGray = null; BATCH = null; }

  function markDirty() { dirty = true; }

  function sizeCanvas() {
    const cv = el("ed-canvas"), st = el("ed-stage");
    cv.width = st.clientWidth; cv.height = st.clientHeight;
  }
  function fitView() {
    const cv = el("ed-canvas");
    scale = Math.min(cv.width / W, cv.height / H) * 0.98; FIT = scale;
    tx = (cv.width - W * scale) / 2; ty = (cv.height - H * scale) / 2;
    redraw();
  }
  function clampView() {
    const cv = el("ed-canvas"), iw = W * scale, ih = H * scale, m = 60;
    tx = Math.max(m - iw, Math.min(cv.width - m, tx));
    ty = Math.max(m - ih, Math.min(cv.height - m, ty));
  }

  function renderComposite() {
    const d = offImg.data;
    for (let y = 0, i = 0; y < H; y++) {
      for (let x = 0; x < W; x++, i++) {
        const L = labels[i], g = baseGray[i];
        let r, gg, b;
        if (L > 0) { const c = colorOf(L); r = (g * 0.5 + c[0] * 0.5) | 0; gg = (g * 0.5 + c[1] * 0.5) | 0; b = (g * 0.5 + c[2] * 0.5) | 0; }
        else { r = gg = b = g; }
        const right = x < W - 1 ? labels[i + 1] : L, down = y < H - 1 ? labels[i + W] : L;
        if ((right !== L || down !== L) && (L > 0 || right > 0 || down > 0)) { r = 255; gg = 255; b = 0; }
        const p = i * 4; d[p] = r; d[p + 1] = gg; d[p + 2] = b; d[p + 3] = 255;
      }
    }
    offCtx.putImageData(offImg, 0, 0);
    updateCount();
  }
  function redraw() {
    const cv = el("ed-canvas"), ctx = cv.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(scale, 0, 0, scale, tx, ty); ctx.drawImage(off, 0, 0);
    if (drawing && pathPts && pathPts.length > 1) {
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.beginPath();
      const p0 = imgToCanvas(pathPts[0]); ctx.moveTo(p0.x, p0.y);
      for (let k = 1; k < pathPts.length; k++) { const p = imgToCanvas(pathPts[k]); ctx.lineTo(p.x, p.y); }
      ctx.strokeStyle = "#00e5ff"; ctx.lineWidth = 2; ctx.stroke();
    }
  }
  function imgToCanvas(pt) { return { x: tx + pt[0] * scale, y: ty + pt[1] * scale }; }
  function evToImg(e) {
    const cv = el("ed-canvas"), rect = cv.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (cv.width / rect.width);
    const cy = (e.clientY - rect.top) * (cv.height / rect.height);
    return [(cx - tx) / scale, (cy - ty) / scale];
  }
  function updateCount() {
    const s = new Set();
    for (let i = 0; i < labels.length; i++) if (labels[i] > 0) s.add(labels[i]);
    el("ed-count").textContent = s.size + " objects" + (dirty ? " (unsaved)" : "");
  }
  function snapshot() { history.push(labels.slice()); if (history.length > 15) history.shift(); }

  function removeAt(ix, iy) {
    const x = Math.floor(ix), y = Math.floor(iy);
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const L = labels[y * W + x]; if (L <= 0) return;
    snapshot(); for (let i = 0; i < labels.length; i++) if (labels[i] === L) labels[i] = 0;
    markDirty(); renderComposite(); redraw();
  }
  function fillOutline(pts) {
    if (!pts || pts.length < 3) return;
    const t = document.createElement("canvas"); t.width = W; t.height = H;
    const g = t.getContext("2d");
    g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
    for (let k = 1; k < pts.length; k++) g.lineTo(pts[k][0], pts[k][1]);
    g.closePath(); g.fillStyle = "#fff"; g.fill();
    const dd = g.getImageData(0, 0, W, H).data;
    snapshot(); const L = ++maxLabel;
    for (let i = 0; i < W * H; i++) if (dd[i * 4 + 3] > 0) labels[i] = L;
    markDirty(); renderComposite(); redraw();
  }
  function setTool(t) {
    tool = t;
    ["remove", "draw", "pan"].forEach(k => { const b = el("ed-" + k); if (b) b.classList.toggle("active", k === t); });
    el("ed-canvas").style.cursor = t === "pan" ? "grab" : "crosshair";
  }

  // ---- save / navigation ----
  async function commit() {
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const g = c.getContext("2d"); const out = g.createImageData(W, H);
    for (let i = 0; i < labels.length; i++) {
      const L = labels[i], p = i * 4;
      out.data[p] = L & 255; out.data[p + 1] = (L >> 8) & 255; out.data[p + 2] = (L >> 16) & 255; out.data[p + 3] = 255;
    }
    g.putImageData(out, 0, 0);
    const fd = new FormData(); fd.append("token", CUR_TOKEN); fd.append("idmap", c.toDataURL("image/png"));
    const r = await (await fetch("/api/single/commit", { method: "POST", body: fd })).json();
    if (r.error) throw new Error(r.error);
    dirty = false;
    return r;
  }

  async function doSave() {
    el("ed-save").disabled = true;
    try {
      const r = await commit();
      if (BATCH) { BATCH.onSavedIdx(BATCH.index, r.overlay, r.count); updateCount(); }
      else { if (SINGLE_CB) SINGLE_CB(r.overlay, r.count); close(); }
    } catch (e) { alert(e); } finally { el("ed-save").disabled = false; }
  }

  function askSave() {
    return new Promise((res) => {
      const ov = document.createElement("div");
      ov.className = "surface";
      ov.style.cssText = "position:fixed;inset:0;background:rgba(8,8,7,.72);z-index:40;display:flex;align-items:center;justify-content:center";
      ov.innerHTML = `<div class="modal-card">
        <p style="margin:0 0 16px">This image has unsaved changes.</p>
        <div style="display:flex;gap:8px;justify-content:flex-end">
          <button data-a="cancel" class="sm">Cancel</button>
          <button data-a="discard" class="sm">Discard</button>
          <button data-a="save" class="sm primary">Save</button>
        </div></div>`;
      ov.addEventListener("click", (e) => {
        const a = e.target.dataset && e.target.dataset.a;
        if (a) { document.body.removeChild(ov); res(a); }
      });
      document.body.appendChild(ov);
    });
  }

  async function navTo(idx) {
    if (!BATCH) return;
    idx = Math.max(0, Math.min(BATCH.list.length - 1, idx));
    if (idx === BATCH.index) return;
    if (dirty) {
      const a = await askSave();
      if (a === "cancel") return;
      if (a === "save") {
        try { const r = await commit(); BATCH.onSavedIdx(BATCH.index, r.overlay, r.count); }
        catch (e) { alert(e); return; }
      }
    }
    BATCH.index = idx; await loadImage(BATCH.list[idx].token); updateNavUI();
  }

  function updateNavUI() {
    const on = !!BATCH;
    ["ed-prev", "ed-next", "ed-pos"].forEach(id => { const e = el(id); if (e) e.style.display = on ? "" : "none"; });
    document.querySelectorAll(".ed-bar .nav-sep").forEach(e => { e.style.display = on ? "" : "none"; });
    if (on) {
      const f = BATCH.list[BATCH.index];
      el("ed-pos").textContent = `${BATCH.index + 1} / ${BATCH.list.length}` + (f.condition ? ` (${f.condition})` : "");
      el("ed-prev").disabled = BATCH.index <= 0;
      el("ed-next").disabled = BATCH.index >= BATCH.list.length - 1;
    }
  }

  function wire() {
    el("ed-remove").onclick = () => setTool("remove");
    el("ed-draw").onclick = () => setTool("draw");
    el("ed-pan").onclick = () => setTool("pan");
    el("ed-fit").onclick = fitView;
    el("ed-undo").onclick = () => { if (history.length) { labels = history.pop(); markDirty(); renderComposite(); redraw(); } };
    el("ed-save").onclick = doSave;
    el("ed-cancel").onclick = close;
    el("ed-prev").onclick = () => navTo(BATCH ? BATCH.index - 1 : 0);
    el("ed-next").onclick = () => navTo(BATCH ? BATCH.index + 1 : 0);

    const cv = el("ed-canvas");
    cv.addEventListener("mousedown", (e) => {
      if (!labels) return;
      if (tool === "pan" || e.button === 1) { panning = true; panStart = [e.clientX - tx, e.clientY - ty]; return; }
      const [ix, iy] = evToImg(e);
      if (tool === "remove") { removeAt(ix, iy); return; }
      if (tool === "draw") { drawing = true; pathPts = [[ix, iy]]; }
    });
    cv.addEventListener("mousemove", (e) => {
      if (!labels) return;
      if (panning) { tx = e.clientX - panStart[0]; ty = e.clientY - panStart[1]; clampView(); redraw(); return; }
      if (drawing) { pathPts.push(evToImg(e)); redraw(); }
    });
    window.addEventListener("mouseup", () => {
      if (panning) panning = false;
      if (drawing) { drawing = false; const pts = pathPts; pathPts = null; fillOutline(pts); }
    });
    cv.addEventListener("wheel", (e) => {
      if (!labels) return;
      e.preventDefault();
      const [ix, iy] = evToImg(e);
      let f = Math.exp(-e.deltaY * 0.0015); f = Math.max(0.9, Math.min(1.1, f));
      scale = Math.max(FIT * 0.9, Math.min(FIT * 40, scale * f));
      const rect = cv.getBoundingClientRect();
      const cx = (e.clientX - rect.left) * (cv.width / rect.width);
      const cy = (e.clientY - rect.top) * (cv.height / rect.height);
      tx = cx - ix * scale; ty = cy - iy * scale; clampView(); redraw();
    }, { passive: false });
    window.addEventListener("resize", () => { if (el("editor").classList.contains("show")) { sizeCanvas(); redraw(); } });
  }

  document.addEventListener("DOMContentLoaded", wire);
  if (document.readyState !== "loading") wire();

  window.Editor = { open, openBatch };
})();
