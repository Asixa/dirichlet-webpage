// Kernel explorer: one reflector at a sub-bin offset, rendered by the exact
// Dirichlet kernel and by a main-lobe-matched Gaussian.
"use strict";

(() => {
  const root = document.getElementById("demo-kernel");
  if (!root) return;
  const canvas = root.querySelector("canvas");
  const nIn = root.querySelector('[name="N"]');
  const dIn = root.querySelector('[name="delta"]');
  const nOut = root.querySelector('[data-out="N"]');
  const dOut = root.querySelector('[data-out="delta"]');
  const readSide = root.querySelector('[data-out="sidelobe"]');
  const readPeak = root.querySelector('[data-out="peak"]');
  const readGauss = root.querySelector('[data-out="gauss"]');

  const state = { N: 16, delta: 0.3, view: "db" };
  const FLOOR_DB = -60;
  const tmp = [0, 0];

  const kernel = (delta, N, out) => DS.dirichlet(delta, N, out);
  // Main-lobe FWHM of the rectangular window is 1.207 bins; the Gaussian
  // gets the same FWHM so the two curves agree at the peak.
  const SIGMA = 1.2067 / 2.3548;
  // Half-width of the main lobe to the first null, in bins.
  const LOBE = 1;

  function draw() {
    const { ctx, w, h } = DS.fitCanvas(canvas);
    const N = state.N, mu = N / 2 + state.delta;
    const L = 44, R = 12, T = 14, B = 38;
    const pw = w - L - R, ph = h - T - B;
    const xOf = (f) => L + (f / N) * pw;
    const muted = DS.token("--muted"), rule = DS.token("--rule");
    const dir = DS.token("--dir"), gau = DS.token("--gau");
    ctx.clearRect(0, 0, w, h);

    let yOf, ticks;
    if (state.view === "db") {
      yOf = (v) => {
        const db = Math.max(FLOOR_DB, 20 * Math.log10(Math.abs(v) + 1e-15));
        return T + (db / FLOOR_DB) * ph;
      };
      ticks = [[1, "0 dB"], [10 ** (-20 / 20), "−20"], [10 ** (-40 / 20), "−40"], [10 ** (-60 / 20), "−60"]];
    } else if (state.view === "lin") {
      yOf = (v) => T + (1 - v) * ph;
      ticks = [[1, "1"], [0.5, "0.5"], [0, "0"]];
    } else {
      yOf = (v) => T + ((1 - v) / 2) * ph;
      ticks = [[1, "1"], [0, "0"], [-1, "−1"]];
    }

    // Axes and grid.
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    ctx.lineWidth = 1;
    for (const [v, label] of ticks) {
      const y = Math.round(yOf(v)) + 0.5;
      ctx.strokeStyle = rule;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      ctx.fillStyle = muted; ctx.fillText(label, L - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const step = N <= 16 ? 2 : N <= 32 ? 4 : 8;
    for (let k = 0; k <= N; k += step) ctx.fillText(String(k), xOf(k), T + ph + 6);
    ctx.fillText("frequency bin", L + pw / 2, T + ph + 22);

    // Reflector position.
    ctx.strokeStyle = muted;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(xOf(mu), T); ctx.lineTo(xOf(mu), T + ph); ctx.stroke();
    ctx.setLineDash([]);

    const curve = (fn, color, width, dash) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
      ctx.beginPath();
      for (let px = 0; px <= pw; px++) {
        const f = (px / pw) * N;
        const y = yOf(fn(f));
        if (px === 0) ctx.moveTo(L + px, y); else ctx.lineTo(L + px, y);
      }
      ctx.stroke(); ctx.setLineDash([]);
    };
    const gaussAt = (f) => DS.gaussian(DS.wrap(f - mu, N), SIGMA);

    if (state.view === "reim") {
      curve((f) => kernel(f - mu, N, tmp)[0], dir, 1.6);
      curve((f) => kernel(f - mu, N, tmp)[1], "rgba(43,102,166,0.45)", 1.4);
      curve(gaussAt, gau, 1.6, [5, 4]);
    } else {
      curve((f) => Math.hypot(...kernel(f - mu, N, tmp)), dir, 1.6);
      curve(gaussAt, gau, 1.6, [5, 4]);
    }

    // Integer-bin samples, coloured by phase.
    let eAll = 0, eOut = 0, gAll = 0, gOut = 0, peak = 0;
    for (let k = 0; k < N; k++) {
      kernel(k - mu, N, tmp);
      const mag = Math.hypot(tmp[0], tmp[1]);
      const g = DS.gaussian(DS.wrap(k - mu, N), SIGMA);
      const outside = Math.abs(DS.wrap(k - mu, N)) >= LOBE;
      eAll += mag * mag; gAll += g * g;
      if (outside) { eOut += mag * mag; gOut += g * g; }
      peak = Math.max(peak, mag);
      const v = state.view === "reim" ? tmp[0] : mag;
      const x = xOf(k), y = yOf(v), y0 = state.view === "reim" ? yOf(0) : T + ph;
      ctx.strokeStyle = "rgba(23,32,44,0.22)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillStyle = DS.phaseColor(tmp[0], tmp[1]);
      ctx.beginPath(); ctx.arc(x, y, N > 40 ? 2.4 : 3.4, 0, 2 * Math.PI); ctx.fill();
    }

    readSide.textContent = `${((100 * eOut) / eAll).toFixed(1)}%`;
    readGauss.textContent = `${((100 * gOut) / gAll).toFixed(1)}%`;
    readPeak.textContent = `${(20 * Math.log10(peak)).toFixed(2)} dB`;
  }

  function sync() {
    state.N = +nIn.value;
    state.delta = +dIn.value;
    nOut.textContent = state.N;
    dOut.textContent = state.delta.toFixed(2);
    draw();
  }

  nIn.addEventListener("input", sync);
  dIn.addEventListener("input", sync);
  for (const group of root.querySelectorAll("[data-group]")) {
    group.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      for (const b of group.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
      state[group.dataset.group] = btn.value;
      draw();
    });
  }
  DS.onResize(canvas, sync);
})();
