// 2D interference playground: draggable complex reflectors rendered by
// separable Dirichlet splats and by coherent (complex) Gaussian splats.
"use strict";

(() => {
  const root = document.getElementById("demo-interference");
  if (!root) return;
  const cvD = root.querySelector('[data-panel="dirichlet"]');
  const cvG = root.querySelector('[data-panel="gaussian"]');
  const overlays = [...root.querySelectorAll(".panel-overlay")];
  const phaseIn = root.querySelector('[name="phase"]');
  const phaseOut = root.querySelector('[data-out="phase"]');
  const selOut = root.querySelector('[data-out="selected"]');
  const resetBtn = root.querySelector('[data-action="reset"]');

  const N = 16;               // DFT length per axis (bins)
  const PX_PER_BIN = 16;      // internal render resolution
  const RES = N * PX_PER_BIN; // 256 x 256 image per panel
  const MAX_REFLECTORS = 6;
  const view = { scale: "lin", sampling: "cont" };

  // Three reflectors with unequal amplitudes, close to the paper's
  // forward-fidelity scene (Fig. 3).
  const initial = () => [
    { x: 6.2, y: 9.1, amp: 1.0, phase: 0 },
    { x: 8.6, y: 10.4, amp: 0.75, phase: 2.1 },
    { x: 9.9, y: 8.3, amp: 0.6, phase: 4.0 },
  ];
  let refl = initial();
  let selected = 0;

  for (const cv of [cvD, cvG]) { cv.width = RES; cv.height = RES; }
  const imgD = cvD.getContext("2d").createImageData(RES, RES);
  const imgG = cvG.getContext("2d").createImageData(RES, RES);
  const re = new Float64Array(RES * RES), im = new Float64Array(RES * RES);
  const tmp = [0, 0];

  // Evaluate each reflector's kernel once per axis, then form the
  // separable product per pixel: O(K * RES^2) multiply-adds.
  function render(kind, img) {
    re.fill(0); im.fill(0);
    const ax = new Float64Array(RES * 2), ay = new Float64Array(RES * 2);
    for (const r of refl) {
      for (let i = 0; i < RES; i++) {
        // Screen coordinates are shifted half a bin so that bin k fills the
        // cell [k, k+1); "bins" mode holds each cell at its centre sample,
        // which is exactly what the sensor records.
        let f = (i + 0.5) / PX_PER_BIN;
        if (view.sampling === "bins") f = Math.floor(f) + 0.5;
        if (kind === "dirichlet") {
          DS.dirichlet(f - r.x, N, tmp); ax[i * 2] = tmp[0]; ax[i * 2 + 1] = tmp[1];
          DS.dirichlet(f - r.y, N, tmp); ay[i * 2] = tmp[0]; ay[i * 2 + 1] = tmp[1];
        } else {
          ax[i * 2] = DS.gaussian(DS.wrap(f - r.x, N)); ax[i * 2 + 1] = 0;
          ay[i * 2] = DS.gaussian(DS.wrap(f - r.y, N)); ay[i * 2 + 1] = 0;
        }
      }
      const br = r.amp * Math.cos(r.phase), bi = r.amp * Math.sin(r.phase);
      for (let j = 0; j < RES; j++) {
        // Row j is the y axis; flip so y grows upward like a plot.
        const yr = ay[(RES - 1 - j) * 2], yi = ay[(RES - 1 - j) * 2 + 1];
        const cr = br * yr - bi * yi, ci = br * yi + bi * yr;
        const row = j * RES;
        for (let i = 0; i < RES; i++) {
          const xr = ax[i * 2], xi = ax[i * 2 + 1];
          re[row + i] += cr * xr - ci * xi;
          im[row + i] += cr * xi + ci * xr;
        }
      }
    }
    let peak = 1e-12;
    const mag = new Float64Array(RES * RES);
    for (let p = 0; p < mag.length; p++) {
      mag[p] = Math.hypot(re[p], im[p]);
      if (mag[p] > peak) peak = mag[p];
    }
    const d = img.data;
    for (let p = 0; p < mag.length; p++) {
      let t = mag[p] / peak;
      // dB view spans 40 dB, matching the dynamic range readers can see.
      if (view.scale === "db") t = Math.max(0, 1 + (20 * Math.log10(t + 1e-12)) / 40);
      const li = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
      d[p * 4] = DS.LUT[li]; d[p * 4 + 1] = DS.LUT[li + 1]; d[p * 4 + 2] = DS.LUT[li + 2]; d[p * 4 + 3] = 255;
    }
  }

  function drawOverlay() {
    for (const ov of overlays) {
      const { ctx, w, h } = DS.fitCanvas(ov);
      ctx.clearRect(0, 0, w, h);
      refl.forEach((r, idx) => {
        const x = (r.x / N) * w, y = h - (r.y / N) * h;
        const rad = 5 + 3 * r.amp;
        ctx.lineWidth = idx === selected ? 2.2 : 1.4;
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.beginPath(); ctx.arc(x, y, rad, 0, 2 * Math.PI); ctx.stroke();
        // Phase hand: shows each reflector's complex phase like a clock.
        ctx.beginPath(); ctx.moveTo(x, y);
        ctx.lineTo(x + rad * Math.cos(-r.phase), y + rad * Math.sin(-r.phase)); ctx.stroke();
        if (idx === selected) {
          ctx.strokeStyle = "rgba(23,32,44,0.9)"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(x, y, rad + 3, 0, 2 * Math.PI); ctx.stroke();
        }
      });
    }
  }

  let pending = false;
  function update() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      render("dirichlet", imgD); cvD.getContext("2d").putImageData(imgD, 0, 0);
      render("gaussian", imgG); cvG.getContext("2d").putImageData(imgG, 0, 0);
      drawOverlay();
      const r = refl[selected];
      selOut.textContent = r ? `Reflector ${selected + 1} of ${refl.length}` : "None selected";
      phaseIn.disabled = !r;
      if (r) {
        phaseIn.value = r.phase.toFixed(2);
        phaseOut.textContent = `${Math.round((r.phase * 180) / Math.PI)}°`;
      }
    });
  }

  // Shared gestures (DS.pointEditor): drag to move, click empty space to
  // add, right-click or long-press to delete. Either panel edits the scene.
  for (const ov of overlays) {
    const toBins = (x, y) => {
      const w = ov.getBoundingClientRect().width;
      return [Math.max(0, Math.min(N, (x / w) * N)), Math.max(0, Math.min(N, (1 - y / w) * N)), w / N];
    };
    DS.pointEditor(ov, {
      hit: (x, y) => {
        const [bx, by, perBin] = toBins(x, y);
        let best = null, bd = 14 / perBin;
        refl.forEach((r, i) => { const d = Math.hypot(r.x - bx, r.y - by); if (d < bd) { bd = d; best = i; } });
        return best;
      },
      add: (x, y) => {
        if (refl.length >= MAX_REFLECTORS) return null;
        const [bx, by] = toBins(x, y);
        refl.push({ x: bx, y: by, amp: 0.8, phase: 0 });
        selected = refl.length - 1;
        update();
        return selected;
      },
      move: (i, x, y) => {
        const [bx, by] = toBins(x, y);
        selected = i; refl[i].x = bx; refl[i].y = by;
        update();
      },
      remove: (i) => {
        if (refl.length <= 1) return;
        refl.splice(i, 1);
        selected = Math.min(selected, refl.length - 1);
        update();
      },
    });
  }

  phaseIn.addEventListener("input", () => {
    if (!refl[selected]) return;
    refl[selected].phase = +phaseIn.value;
    update();
  });
  resetBtn.addEventListener("click", () => { refl = initial(); selected = 0; update(); });
  for (const group of root.querySelectorAll("[data-group]")) {
    group.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      for (const b of group.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
      view[group.dataset.group] = btn.value;
      update();
    });
  }
  DS.onResize(root, update);
})();
