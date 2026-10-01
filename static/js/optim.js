// Optimization demos.
//   1. Landscape: the variable-projected loss of a single Dirichlet splat
//      against a single reflector. Gradient descent stalls in sidelobe
//      minima; the residual certificate finds the true peak in one scan.
//   2. Race: AdamW vs. a compact DSFW (VarPro + certificate replacement +
//      Levenberg-Marquardt sliding) on a cold-started 1D sparse scene.
"use strict";

// ---------------------------------------------------------------------------
// Shared 1D forward model on integer bins k = 0..N-1.
// ---------------------------------------------------------------------------
const Model1D = (() => {
  const tmp = [0, 0], der = [0, 0];

  // Columns a_j[k] = d_N(k - c_j), interleaved complex, shape N x S.
  function columns(C, N) {
    const S = C.length, A = new Float64Array(N * S * 2);
    for (let j = 0; j < S; j++) {
      for (let k = 0; k < N; k++) {
        DS.dirichlet(k - C[j], N, tmp);
        A[(k * S + j) * 2] = tmp[0];
        A[(k * S + j) * 2 + 1] = tmp[1];
      }
    }
    return A;
  }

  function synth(C, b, N) {
    const S = C.length, A = columns(C, N), y = new Float64Array(N * 2);
    for (let k = 0; k < N; k++) {
      for (let j = 0; j < S; j++) {
        const ar = A[(k * S + j) * 2], ai = A[(k * S + j) * 2 + 1];
        y[k * 2] += ar * b[j * 2] - ai * b[j * 2 + 1];
        y[k * 2 + 1] += ar * b[j * 2 + 1] + ai * b[j * 2];
      }
    }
    return y;
  }

  // Closed-form ridge solve b* = (A^H A / N + lambda I)^-1 A^H y / N (Eq. 7).
  function varpro(C, y, N, lambda = 1e-7) {
    const S = C.length, A = columns(C, N);
    const G = new Float64Array(S * S * 2), q = new Float64Array(S * 2);
    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        let sr = 0, si = 0;
        for (let k = 0; k < N; k++) {
          const ar = A[(k * S + i) * 2], ai = -A[(k * S + i) * 2 + 1]; // conj
          const br = A[(k * S + j) * 2], bi = A[(k * S + j) * 2 + 1];
          sr += ar * br - ai * bi; si += ar * bi + ai * br;
        }
        G[(i * S + j) * 2] = sr / N + (i === j ? lambda : 0);
        G[(i * S + j) * 2 + 1] = si / N;
      }
      let sr = 0, si = 0;
      for (let k = 0; k < N; k++) {
        const ar = A[(k * S + i) * 2], ai = -A[(k * S + i) * 2 + 1];
        sr += ar * y[k * 2] - ai * y[k * 2 + 1];
        si += ar * y[k * 2 + 1] + ai * y[k * 2];
      }
      q[i * 2] = sr / N; q[i * 2 + 1] = si / N;
    }
    return { b: DS.csolve(G, q, S), A };
  }

  // Loss normalized by ||y||^2 so 1 means "explains nothing".
  function loss(C, b, y, N) {
    const m = synth(C, b, N);
    let e = 0, n = 0;
    for (let k = 0; k < N * 2; k++) { const d = m[k] - y[k]; e += d * d; n += y[k] * y[k]; }
    return e / n;
  }

  // Certificate s(x) = |<a(x), r>| / ||a(x)|| on a candidate grid (Eq. 9).
  function certificate(r, N, grid) {
    const s = new Float64Array(grid.length);
    for (let g = 0; g < grid.length; g++) {
      let sr = 0, si = 0, nn = 0;
      for (let k = 0; k < N; k++) {
        DS.dirichlet(k - grid[g], N, tmp);
        sr += tmp[0] * r[k * 2] + tmp[1] * r[k * 2 + 1];
        si += tmp[0] * r[k * 2 + 1] - tmp[1] * r[k * 2];
        nn += tmp[0] * tmp[0] + tmp[1] * tmp[1];
      }
      s[g] = Math.hypot(sr, si) / Math.sqrt(nn);
    }
    return s;
  }

  // Coherent-cancellation efficiency eff_j = ||P_perp a_j|| / ||a_j|| (Eq. 10).
  function efficiency(C, N) {
    const S = C.length, A = columns(C, N), eff = new Float64Array(S);
    for (let j = 0; j < S; j++) {
      const others = C.filter((_, i) => i !== j);
      const aj = new Float64Array(N * 2);
      for (let k = 0; k < N; k++) { aj[k * 2] = A[(k * S + j) * 2]; aj[k * 2 + 1] = A[(k * S + j) * 2 + 1]; }
      let nrm = 0;
      for (let k = 0; k < N * 2; k++) nrm += aj[k] * aj[k];
      if (others.length === 0) { eff[j] = 1; continue; }
      const { b } = varpro(others, aj, N, 1e-9);
      const proj = synth(others, b, N);
      let res = 0;
      for (let k = 0; k < N * 2; k++) { const d = aj[k] - proj[k]; res += d * d; }
      eff[j] = Math.sqrt(res / nrm);
    }
    return eff;
  }

  // Levenberg-Marquardt on centres with amplitudes re-solved by VarPro
  // after every accepted step. Returns refined centres and their loss.
  function slide(C0, y, N, iters = 8) {
    let C = C0.slice();
    let { b } = varpro(C, y, N);
    let L = loss(C, b, y, N);
    let damp = 1e-2;
    const S = C.length;
    for (let it = 0; it < iters; it++) {
      const m = synth(C, b, N);
      // J[k][j] = d model_k / d c_j = -b_j d'(k - c_j)
      const J = new Float64Array(N * S * 2);
      for (let j = 0; j < S; j++) {
        for (let k = 0; k < N; k++) {
          DS.dirichletDeriv(k - C[j], N, der);
          const br = b[j * 2], bi = b[j * 2 + 1];
          J[(k * S + j) * 2] = -(br * der[0] - bi * der[1]);
          J[(k * S + j) * 2 + 1] = -(br * der[1] + bi * der[0]);
        }
      }
      const H = new Float64Array(S * S), g = new Float64Array(S);
      for (let i = 0; i < S; i++) {
        for (let k = 0; k < N; k++) {
          const jr = J[(k * S + i) * 2], ji = J[(k * S + i) * 2 + 1];
          const er = m[k * 2] - y[k * 2], ei = m[k * 2 + 1] - y[k * 2 + 1];
          g[i] += jr * er + ji * ei;
          for (let j = 0; j < S; j++) {
            H[i * S + j] += jr * J[(k * S + j) * 2] + ji * J[(k * S + j) * 2 + 1];
          }
        }
      }
      let accepted = false;
      for (let tries = 0; tries < 6 && !accepted; tries++) {
        const M = Float64Array.from(H);
        for (let i = 0; i < S; i++) M[i * S + i] += damp * (H[i * S + i] + 1e-6);
        const step = DS.rsolve(M, g.map((v) => -v), S);
        // Cap a single slide at half a bin: larger moves belong to the
        // certificate, and capping keeps LM from hopping across sidelobes.
        const Cn = C.map((c, i) => Math.min(N - 0.5, Math.max(0.5, c + Math.max(-0.5, Math.min(0.5, step[i])))));
        const vp = varpro(Cn, y, N);
        const Ln = loss(Cn, vp.b, y, N);
        if (Ln < L) { C = Cn; b = vp.b; L = Ln; damp = Math.max(1e-6, damp / 3); accepted = true; }
        else damp *= 4;
      }
      if (!accepted) break;
    }
    return { C, b, L };
  }

  return { columns, synth, varpro, loss, certificate, efficiency, slide };
})();

// ---------------------------------------------------------------------------
// Demo 1: single-splat landscape.
// ---------------------------------------------------------------------------
(() => {
  const root = document.getElementById("demo-landscape");
  if (!root) return;
  const canvas = root.querySelector("canvas");
  const status = root.querySelector('[data-out="status"]');
  const certBtn = root.querySelector('[data-action="certificate"]');

  const N = 32, MU = 21.3;
  const y = Model1D.synth([MU], new Float64Array([1, 0]), N);
  const GRID = Array.from({ length: 641 }, (_, i) => (i / 640) * N);
  // With one splat, VarPro gives F(c) = 1 - |<a(c), y>|^2 / (||a||^2 ||y||^2),
  // which is 1 - (certificate / ||y||)^2: same scan, read two ways.
  const cert = Model1D.certificate(y, N, GRID);
  let ynorm = 0;
  for (let k = 0; k < 2 * N; k++) ynorm += y[k] * y[k];
  ynorm = Math.sqrt(ynorm);
  const F = cert.map((s) => 1 - (s / ynorm) ** 2);
  const Fat = (c) => {
    const s = Model1D.certificate(y, N, [c])[0];
    return 1 - (s / ynorm) ** 2;
  };

  let ball = null;      // { c, trail: [] }
  let scan = null;      // certificate sweep progress in [0, 1]
  let anim = null;

  function draw() {
    const { ctx, w, h } = DS.fitCanvas(canvas);
    const L = 44, R = 12, T = 16, B = 30, gap = 40;
    const pw = w - L - R;
    const h1 = (h - T - B - gap) * 0.62, h2 = h - T - B - gap - h1;
    const xOf = (c) => L + (c / N) * pw;
    // Far sidelobe valleys are only 0.1-2% deep in F, invisible on a linear
    // axis. -10 log10(1 - F) is monotone in F, so every minimum stays where
    // it is; F = 0 sits at the panel bottom and F -> 1 at the top (40 dB).
    const tr = (v) => Math.min(40, -10 * Math.log10(Math.max(1e-4, 1 - v)));
    const yFv = (v) => T + h1 - (tr(v) / 40) * h1;
    const top2 = T + h1 + gap;
    const yS = (v) => top2 + (1 - v / ynorm) * h2;
    const muted = DS.token("--muted"), rule = DS.token("--rule"), ink = DS.token("--ink");
    const dir = DS.token("--dir"), gau = DS.token("--gau");
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";

    // Panel labels.
    ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    const narrow = pw < 480;
    ctx.fillText(narrow ? "Loss F(c), lower is better" : "Loss F(c) of one splat at centre c (lower is better)", L, T - 3);
    ctx.fillText(narrow ? "Certificate s(x)" : "Certificate s(x): where a new splat would reduce the residual most", L, top2 - 3);

    // Frames.
    ctx.strokeStyle = rule; ctx.lineWidth = 1;
    ctx.strokeRect(L + 0.5, T + 0.5, pw, h1);
    ctx.strokeRect(L + 0.5, top2 + 0.5, pw, h2);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of [0, 0.9, 0.99, 0.999]) ctx.fillText(String(v), L - 6, yFv(v));

    ctx.beginPath();
    GRID.forEach((c, i) => { const yy = yFv(F[i]); if (i === 0) ctx.moveTo(xOf(c), yy); else ctx.lineTo(xOf(c), yy); });
    ctx.strokeStyle = ink; ctx.lineWidth = 1.6; ctx.stroke();

    // True reflector.
    ctx.strokeStyle = muted; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(xOf(MU), T); ctx.lineTo(xOf(MU), top2 + h2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("true reflector", xOf(MU), top2 + h2 + 4);

    // Certificate curve, revealed by the sweep.
    const upto = scan === null ? 0 : scan;
    if (upto > 0) {
      ctx.beginPath();
      const last = Math.floor(upto * (GRID.length - 1));
      for (let i = 0; i <= last; i++) {
        const yy = yS(cert[i]);
        if (i === 0) ctx.moveTo(xOf(GRID[i]), yy); else ctx.lineTo(xOf(GRID[i]), yy);
      }
      ctx.strokeStyle = dir; ctx.lineWidth = 1.6; ctx.stroke();
      if (upto < 1) {
        ctx.strokeStyle = dir; ctx.globalAlpha = 0.35;
        ctx.beginPath(); ctx.moveTo(xOf(GRID[last]), top2); ctx.lineTo(xOf(GRID[last]), top2 + h2); ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Ball and its gradient-descent trail.
    if (ball) {
      ctx.strokeStyle = gau; ctx.lineWidth = 1.2;
      ctx.beginPath();
      ball.trail.forEach((c, i) => { const yy = yFv(Fat(c)); if (i === 0) ctx.moveTo(xOf(c), yy); else ctx.lineTo(xOf(c), yy); });
      ctx.stroke();
      const yy = yFv(Fat(ball.c));
      ctx.fillStyle = ball.kind === "cert" ? dir : gau;
      ctx.beginPath(); ctx.arc(xOf(ball.c), yy - 6, 6, 0, 2 * Math.PI); ctx.fill();
    }

    // x ticks.
    ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let k = 0; k <= N; k += 4) ctx.fillText(String(k), xOf(k), T + h1 + 3);
  }

  function runDescent(c0) {
    cancelAnimationFrame(anim);
    ball = { c: c0, trail: [c0], kind: "gd" };
    // lr 0.15 converges inside the main lobe without oscillating (checked
    // offline); larger steps bounce across the peak and blur the point.
    const STEPS = 300, lr = 0.15, h = 1e-3;
    // Reduced motion: run every step in one frame instead of 3 per frame.
    const perFrame = DS.prefersReducedMotion() ? STEPS : 3;
    let it = 0;
    const tick = () => {
      for (let s = 0; s < perFrame && it < STEPS; s++, it++) {
        const g = (Fat(ball.c + h) - Fat(ball.c - h)) / (2 * h);
        ball.c = Math.min(N - 0.01, Math.max(0.01, ball.c - lr * g));
        ball.trail.push(ball.c);
      }
      draw();
      const done = it >= STEPS;
      status.textContent = !done
        ? `Gradient descent, step ${it}`
        : Math.abs(ball.c - MU) < 0.1
          ? `Gradient descent reached the reflector at c = ${ball.c.toFixed(2)}, because it started inside the main lobe.`
          : `Gradient descent stopped at c = ${ball.c.toFixed(2)}, a sidelobe minimum. The reflector is at ${MU.toFixed(2)}.`;
      if (!done) anim = requestAnimationFrame(tick);
    };
    tick();
  }

  function runCertificate() {
    cancelAnimationFrame(anim);
    scan = 0;
    const tick = () => {
      scan = Math.min(1, scan + (DS.prefersReducedMotion() ? 1 : 0.025));
      draw();
      if (scan < 1) { anim = requestAnimationFrame(tick); return; }
      let best = 0;
      for (let i = 1; i < cert.length; i++) if (cert[i] > cert[best]) best = i;
      ball = { c: GRID[best], trail: [GRID[best]], kind: "cert" };
      draw();
      status.textContent = `The certificate peaks at x = ${GRID[best].toFixed(2)}. One global scan places the splat on the reflector, with no descent.`;
    };
    tick();
  }

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const c = ((e.clientX - rect.left - 44) / (rect.width - 56)) * N;
    if (c < 0 || c > N) return;
    runDescent(c);
  });
  canvas.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    runDescent(4 + Math.random() * 10);
  });
  certBtn.addEventListener("click", runCertificate);
  DS.onResize(canvas, draw);
})();

// ---------------------------------------------------------------------------
// Demo 2: AdamW vs DSFW race.
// ---------------------------------------------------------------------------
(() => {
  const root = document.getElementById("demo-race");
  if (!root) return;
  const cvAdam = root.querySelector('[data-panel="adam"]');
  const cvDsfw = root.querySelector('[data-panel="dsfw"]');
  const cvLoss = root.querySelector('[data-panel="loss"]');
  const runBtn = root.querySelector('[data-action="run"]');
  const newBtn = root.querySelector('[data-action="new"]');
  const outAdam = root.querySelector('[data-out="adam"]');
  const outDsfw = root.querySelector('[data-out="dsfw"]');
  const logOut = root.querySelector('[data-out="log"]');

  const N = 48, K = 4, S = 6;
  const ADAM_STEPS = 900, DSFW_STEPS = 24;
  const FRAMES = 300;             // both methods finish in the same wall time
  const GRID = Array.from({ length: 481 }, (_, i) => 0.5 + (i / 480) * (N - 1));
  let seed = 7;
  let truth, y, adam, dsfw, frame, running = false, raf = null;

  function newScene() {
    const rand = DS.rng(seed);
    const mus = [];
    while (mus.length < K) {
      const m = 6 + rand() * (N - 12);
      if (mus.every((v) => Math.abs(v - m) > 3)) mus.push(m);
    }
    mus.sort((a, b) => a - b);
    const b = new Float64Array(K * 2);
    for (let i = 0; i < K; i++) {
      const a = 0.55 + 0.45 * rand(), p = rand() * 2 * Math.PI;
      b[i * 2] = a * Math.cos(p); b[i * 2 + 1] = a * Math.sin(p);
    }
    truth = { C: mus, b };
    y = Model1D.synth(mus, b, N);
    reset();
  }

  function reset() {
    const rand = DS.rng(seed * 31 + 1);
    // Cold start shared by both methods: uniform centres, equal amplitude.
    const C0 = Array.from({ length: S }, () => 2 + rand() * (N - 4));
    const b0 = new Float64Array(S * 2).map((_, i) => (i % 2 === 0 ? 0.3 : 0));
    adam = {
      C: C0.slice(), b: Float64Array.from(b0),
      m: new Float64Array(S * 3), v: new Float64Array(S * 3), t: 0, hist: [],
    };
    adam.hist.push([0, Model1D.loss(adam.C, adam.b, y, N)]);
    dsfw = { C: C0.slice(), b: Float64Array.from(b0), t: 0, hist: [], cert: null, last: null };
    dsfw.hist.push([0, Model1D.loss(dsfw.C, dsfw.b, y, N)]);
    frame = 0;
    logOut.textContent = "Press Run. Both methods start from the same random centres.";
    drawAll();
  }

  // One AdamW step on (c_j, Re b_j, Im b_j) with the normalized loss.
  const der = [0, 0], tmp = [0, 0];
  function adamStep() {
    const { C, b } = adam;
    const m = Model1D.synth(C, b, N);
    let yn = 0;
    for (let k = 0; k < 2 * N; k++) yn += y[k] * y[k];
    const grad = new Float64Array(S * 3);
    for (let j = 0; j < S; j++) {
      for (let k = 0; k < N; k++) {
        const er = m[k * 2] - y[k * 2], ei = m[k * 2 + 1] - y[k * 2 + 1];
        DS.dirichlet(k - C[j], N, tmp);
        DS.dirichletDeriv(k - C[j], N, der);
        const br = b[j * 2], bi = b[j * 2 + 1];
        // d model / d c_j = -b_j d'(k - c_j)
        const jr = -(br * der[0] - bi * der[1]), ji = -(br * der[1] + bi * der[0]);
        grad[j * 3] += 2 * (er * jr + ei * ji);
        grad[j * 3 + 1] += 2 * (er * tmp[0] + ei * tmp[1]);
        grad[j * 3 + 2] += 2 * (-er * tmp[1] + ei * tmp[0]);
      }
    }
    adam.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-4;
    const lr = [0.05, 0.03, 0.03];
    for (let p = 0; p < S * 3; p++) {
      const g = grad[p] / yn;
      adam.m[p] = b1 * adam.m[p] + (1 - b1) * g;
      adam.v[p] = b2 * adam.v[p] + (1 - b2) * g * g;
      const mh = adam.m[p] / (1 - b1 ** adam.t), vh = adam.v[p] / (1 - b2 ** adam.t);
      const j = Math.floor(p / 3), q = p % 3, rate = lr[q];
      const upd = rate * (mh / (Math.sqrt(vh) + eps));
      if (q === 0) adam.C[j] = Math.min(N - 0.5, Math.max(0.5, adam.C[j] - upd));
      else {
        const idx = j * 2 + (q - 1);
        adam.b[idx] -= upd + rate * wd * adam.b[idx];
      }
    }
  }

  // One DSFW outer step (Algorithm 1): VarPro, certificate, utility,
  // hard replacement, then a short LM slide. A replacement is kept only
  // if it lowers the loss, so the loss never increases.
  function dsfwStep() {
    const { b } = Model1D.varpro(dsfw.C, y, N);
    const m = Model1D.synth(dsfw.C, b, N);
    const r = new Float64Array(N * 2);
    for (let k = 0; k < N * 2; k++) r[k] = y[k] - m[k];
    const cert = Model1D.certificate(r, N, GRID);
    let best = 0;
    for (let i = 1; i < cert.length; i++) if (cert[i] > cert[best]) best = i;
    const xStar = GRID[best];

    const eff = Model1D.efficiency(dsfw.C, N);
    const mags = dsfw.C.map((_, j) => Math.hypot(b[j * 2], b[j * 2 + 1]));
    const mmax = Math.max(...mags) || 1;
    const util = mags.map((a, j) => 0.5 * (a / mmax) + 0.5 * eff[j]);
    let worst = 0;
    util.forEach((u, j) => { if (u < util[worst]) worst = j; });

    const keep = Model1D.slide(dsfw.C, y, N);
    const Cr = dsfw.C.slice(); Cr[worst] = xStar;
    const swap = Model1D.slide(Cr, y, N);
    const takeSwap = swap.L < keep.L;
    const res = takeSwap ? swap : keep;
    dsfw.last = takeSwap ? { from: dsfw.C[worst], to: xStar, j: worst } : null;
    dsfw.C = res.C; dsfw.b = res.b; dsfw.cert = cert; dsfw.t++;
    return { takeSwap, worst, xStar, L: res.L };
  }

  // Success = every true centre has a splat within 0.1 bin (paper's criterion).
  function matched(C) {
    return truth.C.filter((mu) => C.some((c) => Math.abs(c - mu) < 0.1)).length;
  }

  function drawPanel(canvas, st, color, title, showCert) {
    const { ctx, w, h } = DS.fitCanvas(canvas);
    const L = 12, R = 12, T = 20, B = 18;
    const pw = w - L - R, ph = h - T - B;
    const xOf = (f) => L + (f / N) * pw;
    const muted = DS.token("--muted"), rule = DS.token("--rule"), ink = DS.token("--ink");
    ctx.clearRect(0, 0, w, h);
    ctx.font = "12px 'Hanken Grotesk', system-ui, sans-serif";
    ctx.fillStyle = ink; ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(title, L, 2);

    const yMax = 1.25;
    const yOf = (v) => T + (1 - Math.min(v, yMax) / yMax) * ph;
    ctx.strokeStyle = rule; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, yOf(0) + 0.5); ctx.lineTo(L + pw, yOf(0) + 0.5); ctx.stroke();

    const evalAt = (C, b, f) => {
      let r = 0, i = 0;
      for (let j = 0; j < C.length; j++) {
        DS.dirichlet(f - C[j], N, tmp);
        r += b[j * 2] * tmp[0] - b[j * 2 + 1] * tmp[1];
        i += b[j * 2] * tmp[1] + b[j * 2 + 1] * tmp[0];
      }
      return Math.hypot(r, i);
    };

    if (showCert && st.cert) {
      let cmax = 0;
      for (const v of st.cert) cmax = Math.max(cmax, v);
      if (cmax > 1e-6) {
        ctx.beginPath();
        GRID.forEach((x, i) => { const yy = yOf((st.cert[i] / cmax) * 0.35); if (i === 0) ctx.moveTo(xOf(x), yy); else ctx.lineTo(xOf(x), yy); });
        ctx.lineTo(xOf(GRID[GRID.length - 1]), yOf(0)); ctx.lineTo(xOf(GRID[0]), yOf(0)); ctx.closePath();
        ctx.fillStyle = "rgba(43,102,166,0.12)"; ctx.fill();
      }
    }

    // Target (filled grey) and current rendering (coloured line).
    ctx.beginPath();
    for (let px = 0; px <= pw; px += 1) {
      const f = (px / pw) * N, yy = yOf(evalAt(truth.C, truth.b, f));
      if (px === 0) ctx.moveTo(L + px, yy); else ctx.lineTo(L + px, yy);
    }
    ctx.lineTo(L + pw, yOf(0)); ctx.lineTo(L, yOf(0)); ctx.closePath();
    ctx.fillStyle = "rgba(23,32,44,0.10)"; ctx.fill();
    ctx.beginPath();
    for (let px = 0; px <= pw; px += 1) {
      const f = (px / pw) * N, yy = yOf(evalAt(st.C, st.b, f));
      if (px === 0) ctx.moveTo(L + px, yy); else ctx.lineTo(L + px, yy);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();

    // True centres (grey ticks below axis) and splat centres (coloured dots).
    ctx.fillStyle = muted;
    for (const mu of truth.C) {
      ctx.beginPath();
      ctx.moveTo(xOf(mu), yOf(0) + 2); ctx.lineTo(xOf(mu) - 4, yOf(0) + 10); ctx.lineTo(xOf(mu) + 4, yOf(0) + 10);
      ctx.closePath(); ctx.fill();
    }
    if (st.last) {
      ctx.strokeStyle = color; ctx.globalAlpha = 0.5; ctx.setLineDash([3, 3]);
      const yA = yOf(0) - 4;
      ctx.beginPath();
      ctx.moveTo(xOf(st.last.from), yA);
      ctx.quadraticCurveTo((xOf(st.last.from) + xOf(st.last.to)) / 2, yA - 40, xOf(st.last.to), yA);
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    st.C.forEach((c) => {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(xOf(c), yOf(0), 4.2, 0, 2 * Math.PI); ctx.fill();
    });
  }

  function drawLoss() {
    const { ctx, w, h } = DS.fitCanvas(cvLoss);
    const L = 44, R = 12, T = 10, B = 26;
    const pw = w - L - R, ph = h - T - B;
    // log10 range; cold starts can exceed 1 because the initial splats add
    // energy the target does not have.
    const lo = -10, hi = 1;
    const yOf = (v) => T + ((hi - Math.max(lo, Math.log10(v + 1e-300))) / (hi - lo)) * ph;
    const xOf = (p) => L + p * pw;
    const muted = DS.token("--muted"), rule = DS.token("--rule");
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let e = lo; e <= 0; e += 2) {
      const yy = Math.round(yOf(10 ** e)) + 0.5;
      ctx.strokeStyle = rule; ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(L + pw, yy); ctx.stroke();
      ctx.fillStyle = muted; ctx.fillText(e === 0 ? "1" : `1e${e}`, L - 6, yy);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("fraction of each method's iteration budget", L + pw / 2, T + ph + 8);
    const line = (hist, total, color) => {
      ctx.beginPath();
      hist.forEach(([t, v], i) => { const xx = xOf(t / total), yy = yOf(v); if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); });
      ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.stroke();
    };
    line(adam.hist, ADAM_STEPS, DS.token("--gau"));
    line(dsfw.hist, DSFW_STEPS, DS.token("--dir"));
  }

  function drawAll() {
    drawPanel(cvAdam, adam, DS.token("--gau"), `AdamW, step ${adam.t} of ${ADAM_STEPS}`, false);
    drawPanel(cvDsfw, dsfw, DS.token("--dir"), `DSFW, outer step ${dsfw.t} of ${DSFW_STEPS}`, true);
    drawLoss();
    const la = adam.hist[adam.hist.length - 1][1], ld = dsfw.hist[dsfw.hist.length - 1][1];
    outAdam.textContent = `loss ${la.toExponential(1)}, ${matched(adam.C)} of ${K} found`;
    outDsfw.textContent = `loss ${ld.toExponential(1)}, ${matched(dsfw.C)} of ${K} found`;
  }

  function tick() {
    if (!running) return;
    frame++;
    const adamTarget = Math.round((frame / FRAMES) * ADAM_STEPS);
    while (adam.t < adamTarget && adam.t < ADAM_STEPS) {
      adamStep();
      if (adam.t % 5 === 0) adam.hist.push([adam.t, Model1D.loss(adam.C, adam.b, y, N)]);
    }
    const dsfwTarget = Math.round((frame / FRAMES) * DSFW_STEPS);
    while (dsfw.t < dsfwTarget && dsfw.t < DSFW_STEPS) {
      const info = dsfwStep();
      dsfw.hist.push([dsfw.t, info.L]);
      logOut.textContent = info.takeSwap
        ? `DSFW step ${dsfw.t}: the certificate peaks at bin ${info.xStar.toFixed(2)}, so the lowest-utility splat jumps there.`
        : `DSFW step ${dsfw.t}: no replacement lowers the loss, so the splats only slide.`;
    }
    drawAll();
    if (frame >= FRAMES) {
      running = false;
      runBtn.textContent = "Run again";
      const ok = matched(dsfw.C) === K, okA = matched(adam.C) === K;
      logOut.textContent = `Done. DSFW found ${matched(dsfw.C)} of ${K} reflectors${ok ? "" : " (try another scene)"}; AdamW found ${matched(adam.C)}${okA ? "" : ", stuck next to sidelobes"}.`;
      return;
    }
    raf = requestAnimationFrame(tick);
  }

  runBtn.addEventListener("click", () => {
    if (running) { running = false; runBtn.textContent = "Resume"; return; }
    if (frame >= FRAMES) reset();
    running = true;
    runBtn.textContent = "Pause";
    raf = requestAnimationFrame(tick);
  });
  newBtn.addEventListener("click", () => {
    cancelAnimationFrame(raf);
    running = false;
    runBtn.textContent = "Run";
    seed = (seed * 1103515245 + 12345) % 2147483647;
    newScene();
  });
  DS.whenVisible(root, (visible) => { if (!visible && running) { running = false; runBtn.textContent = "Resume"; } });

  newScene();
  DS.onResize(root, drawAll);
})();
