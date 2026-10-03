// 2D optimizer race: AdamW vs. DSFW on a 2D Fourier-space target built
// from a template shape (square, star, letter A) of complex reflectors.
// The model, VarPro, LM sliding, and certificate reuse Fidelity's
// Dirichlet fitter; this file adds the DSFW outer loop, AdamW, templates,
// and the interactive view.
"use strict";

const Race2D = (() => {
  const tmp = [0, 0];

  // Reflector centres for a template, in bins, centred in an N x N grid.
  // Neighbouring points sit about 2 bins apart: resolvable (the main lobe
  // is 1 bin to its first null) yet close enough to interfere strongly.
  function template(name, N) {
    const c = N / 2, pts = [];
    const along = (x0, y0, x1, y1, spacing, includeEnd) => {
      const len = Math.hypot(x1 - x0, y1 - y0);
      const n = Math.max(1, Math.round(len / spacing));
      for (let i = 0; i < n + (includeEnd ? 1 : 0); i++) {
        const t = i / n;
        pts.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]);
      }
    };
    if (name === "square") {
      const h = N * 0.22;
      const v = [[c - h, c - h], [c + h, c - h], [c + h, c + h], [c - h, c + h]];
      for (let i = 0; i < 4; i++) along(...v[i], ...v[(i + 1) % 4], 2.2, false);
    } else if (name === "star") {
      const ro = N * 0.3, ri = ro * 0.42, v = [];
      for (let i = 0; i < 10; i++) {
        const r = i % 2 ? ri : ro, a = Math.PI / 2 + (i * Math.PI) / 5;
        v.push([c + r * Math.cos(a), c + r * Math.sin(a)]);
      }
      for (let i = 0; i < 10; i++) along(...v[i], ...v[(i + 1) % 10], 2.3, false);
    } else {
      // Letter A: two legs and a crossbar.
      const top = [c, c + N * 0.3], bl = [c - N * 0.2, c - N * 0.28], br = [c + N * 0.2, c - N * 0.28];
      along(...bl, ...top, 2.2, true);
      along(...top, ...br, 2.2, true);
      const lerp = (p, q, t) => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
      const l = lerp(bl, top, 0.42), r = lerp(br, top, 0.42);
      along(l[0] + 2.2, l[1], r[0] - 0.4, r[1], 2.2, false);
    }
    // Drop near-duplicates (stroke joints).
    return pts.filter((p, i) => pts.findIndex((q) => Math.hypot(p[0] - q[0], p[1] - q[1]) < 1) === i);
  }

  function makeModel(N) {
    const S = N * N;
    const ax = new Float64Array(N * 2), ay = new Float64Array(N * 2);
    const dx = new Float64Array(N * 2), dy = new Float64Array(N * 2);

    // Separable atom d(k - cx) d(l - cy) and its derivatives.
    function atom(p, col, dcols) {
      DS.dirichletRow(p[0], N, ax, dx);
      DS.dirichletRow(p[1], N, ay, dy);
      // dirichletRow gives d/dDelta with Delta = k - c; d/dc is its negative.
      for (let k = 0; k < N * 2; k++) { dx[k] = -dx[k]; dy[k] = -dy[k]; }
      for (let l = 0; l < N; l++) {
        const yr = ay[l * 2], yi = ay[l * 2 + 1], dyr = dy[l * 2], dyi = dy[l * 2 + 1];
        for (let k = 0; k < N; k++) {
          const s = (l * N + k) * 2;
          const xr = ax[k * 2], xi = ax[k * 2 + 1], dxr = dx[k * 2], dxi = dx[k * 2 + 1];
          col[s] = xr * yr - xi * yi; col[s + 1] = xr * yi + xi * yr;
          dcols[0][s] = dxr * yr - dxi * yi; dcols[0][s + 1] = dxr * yi + dxi * yr;
          dcols[1][s] = xr * dyr - xi * dyi; dcols[1][s + 1] = xr * dyi + xi * dyr;
        }
      }
    }

    // Certificate s(x) = |<a(x), r>| on a grid of G x G candidates, by the
    // separable trick (O(G N^2 + G^2 N) instead of O(G^2 N^2)). Returns the
    // argmax and, if wantMap, the whole map for display.
    const G = 4 * N, step = N / G;
    const tab = new Float64Array(G * N * 2);
    for (let g = 0; g < G; g++) for (let k = 0; k < N; k++) {
      DS.dirichlet(k - g * step, N, tmp); tab[(g * N + k) * 2] = tmp[0]; tab[(g * N + k) * 2 + 1] = tmp[1];
    }
    function certificateMap(e, wantMap) {
      const map = wantMap ? new Float64Array(G * G) : null;
      const v = new Float64Array(N * 2);
      let best = -1, bp = [0, 0];
      for (let gx = 0; gx < G; gx++) {
        for (let l = 0; l < N; l++) {
          let sr = 0, si = 0;
          for (let k = 0; k < N; k++) {
            const ar = tab[(gx * N + k) * 2], ai = -tab[(gx * N + k) * 2 + 1];
            const er = e[(l * N + k) * 2], ei = e[(l * N + k) * 2 + 1];
            sr += ar * er - ai * ei; si += ar * ei + ai * er;
          }
          v[l * 2] = sr; v[l * 2 + 1] = si;
        }
        for (let gy = 0; gy < G; gy++) {
          let sr = 0, si = 0;
          for (let l = 0; l < N; l++) {
            const ar = tab[(gy * N + l) * 2], ai = -tab[(gy * N + l) * 2 + 1];
            sr += ar * v[l * 2] - ai * v[l * 2 + 1]; si += ar * v[l * 2 + 1] + ai * v[l * 2];
          }
          const m = sr * sr + si * si;
          if (map) map[gy * G + gx] = Math.sqrt(m);
          if (m > best) { best = m; bp = [gx * step, gy * step]; }
        }
      }
      return { best: bp, map };
    }

    const fitter = Fidelity.makeDirichletFitter({
      S, P: 2, atom,
      certificate: (e) => certificateMap(e, false).best,
      clamp: (p) => p.map((v) => Math.min(N - 0.5, Math.max(0.5, v))),
    });

    function synth(C, b) {
      const y = new Float64Array(S * 2);
      const col = new Float64Array(S * 2), dc = [new Float64Array(S * 2), new Float64Array(S * 2)];
      C.forEach((p, j) => {
        atom(p, col, dc);
        const br = b[j * 2], bi = b[j * 2 + 1];
        for (let s = 0; s < S; s++) {
          y[s * 2] += br * col[s * 2] - bi * col[s * 2 + 1];
          y[s * 2 + 1] += br * col[s * 2 + 1] + bi * col[s * 2];
        }
      });
      return y;
    }

    // Coherent-cancellation efficiency of each splat (paper Eq. 10): the
    // fraction of its footprint the other splats cannot explain. With the
    // Gram matrix G = A^H A, ||P_perp a_j||^2 = 1 / (G^-1)_jj (Schur
    // complement), so one K x K inverse gives every splat's value. The
    // per-splat least-squares version cost ~100 ms per step at K = 26.
    function efficiencies(C) {
      const K = C.length;
      if (K < 2) return [1];
      const col = new Float64Array(S * 2), dc = [new Float64Array(S * 2), new Float64Array(S * 2)];
      const A = new Float64Array(S * K * 2);
      C.forEach((p, j) => {
        atom(p, col, dc);
        for (let s = 0; s < S; s++) { A[(s * K + j) * 2] = col[s * 2]; A[(s * K + j) * 2 + 1] = col[s * 2 + 1]; }
      });
      const Gm = new Float64Array(K * K * 2);
      for (let s = 0; s < S; s++) {
        const row = s * K;
        for (let i = 0; i < K; i++) {
          const ar = A[(row + i) * 2], ai = -A[(row + i) * 2 + 1];
          for (let j = i; j < K; j++) {
            const br = A[(row + j) * 2], bi = A[(row + j) * 2 + 1];
            Gm[(i * K + j) * 2] += ar * br - ai * bi; Gm[(i * K + j) * 2 + 1] += ar * bi + ai * br;
          }
        }
      }
      for (let i = 0; i < K; i++) for (let j = 0; j < i; j++) {
        Gm[(i * K + j) * 2] = Gm[(j * K + i) * 2]; Gm[(i * K + j) * 2 + 1] = -Gm[(j * K + i) * 2 + 1];
      }
      // Ridge keeps coincident splats from making G singular; they then
      // get an efficiency near zero, which is the right answer.
      for (let i = 0; i < K; i++) Gm[(i * K + i) * 2] += 1e-9 * Gm[(i * K + i) * 2] + 1e-12;
      return C.map((_, j) => {
        const e = new Float64Array(K * 2); e[j * 2] = 1;
        const x = DS.csolve(Gm, e, K);           // column j of G^-1
        const inv = x[j * 2];                     // (G^-1)_jj is real
        return Math.min(1, Math.sqrt(1 / Math.max(inv * Gm[(j * K + j) * 2], 1)));
      });
    }

    let yNorm = 1;
    function lossOf(L) { return L / yNorm; }

    // One DSFW outer step (Algorithm 1). Returns the new state.
    function dsfwStep(st, y) {
      const cur = fitter.evaluate(st.C, y);
      const { best, map } = certificateMap(cur.e, true);
      const eff = efficiencies(st.C);
      const mags = st.C.map((_, j) => Math.hypot(cur.b[j * 2], cur.b[j * 2 + 1]));
      const mmax = Math.max(...mags) || 1;
      let worst = 0, wu = Infinity;
      mags.forEach((m, j) => { const u = 0.5 * m / mmax + 0.5 * eff[j]; if (u < wu) { wu = u; worst = j; } });
      const keep = fitter.slide(st.C, y, 6);
      const Cr = st.C.map((p) => p.slice()); Cr[worst] = best.slice();
      const swap = fitter.slide(Cr, y, 6);
      const take = swap.L < keep.L;
      const res = take ? swap : keep;
      return {
        C: res.C.map((p) => p.slice()), b: res.b, L: lossOf(res.L), t: st.t + 1,
        cert: map, moved: take ? { from: st.C[worst], to: best } : null,
      };
    }

    // AdamW on (cx, cy, Re b, Im b) per splat, normalized squared error.
    function adamStep(st, y) {
      const K = st.C.length;
      const col = new Float64Array(S * 2), dc = [new Float64Array(S * 2), new Float64Array(S * 2)];
      const m = synth(st.C, st.b);
      const e = new Float64Array(S * 2);
      for (let s = 0; s < S * 2; s++) e[s] = m[s] - y[s];
      const grad = new Float64Array(K * 4);
      st.C.forEach((p, j) => {
        atom(p, col, dc);
        const br = st.b[j * 2], bi = st.b[j * 2 + 1];
        let g0 = 0, g1 = 0, g2 = 0, g3 = 0;
        for (let s = 0; s < S; s++) {
          const er = e[s * 2], ei = e[s * 2 + 1];
          // d model / d c = b * d col / d c
          const d0r = br * dc[0][s * 2] - bi * dc[0][s * 2 + 1], d0i = br * dc[0][s * 2 + 1] + bi * dc[0][s * 2];
          const d1r = br * dc[1][s * 2] - bi * dc[1][s * 2 + 1], d1i = br * dc[1][s * 2 + 1] + bi * dc[1][s * 2];
          g0 += er * d0r + ei * d0i;
          g1 += er * d1r + ei * d1i;
          g2 += er * col[s * 2] + ei * col[s * 2 + 1];
          g3 += -er * col[s * 2 + 1] + ei * col[s * 2];
        }
        grad[j * 4] = 2 * g0 / yNorm; grad[j * 4 + 1] = 2 * g1 / yNorm;
        grad[j * 4 + 2] = 2 * g2 / yNorm; grad[j * 4 + 3] = 2 * g3 / yNorm;
      });
      const t = st.t + 1, b1 = 0.9, b2 = 0.999, lr = [0.05, 0.05, 0.03, 0.03];
      const mo = Float64Array.from(st.m), vo = Float64Array.from(st.v);
      const C = st.C.map((p) => p.slice()), b = Float64Array.from(st.b);
      for (let q = 0; q < K * 4; q++) {
        mo[q] = b1 * mo[q] + (1 - b1) * grad[q];
        vo[q] = b2 * vo[q] + (1 - b2) * grad[q] * grad[q];
        const upd = lr[q % 4] * (mo[q] / (1 - b1 ** t)) / (Math.sqrt(vo[q] / (1 - b2 ** t)) + 1e-8);
        const j = Math.floor(q / 4), k = q % 4;
        if (k < 2) C[j][k] = Math.min(N - 0.5, Math.max(0.5, C[j][k] - upd));
        else b[j * 2 + (k - 2)] -= upd + lr[k] * 1e-4 * b[j * 2 + (k - 2)];
      }
      let L = 0;
      const mn = synth(C, b);
      for (let s = 0; s < S * 2; s++) { const d = mn[s] - y[s]; L += d * d; }
      return { C, b, m: mo, v: vo, t, L: L / yNorm };
    }

    function setTarget(y) {
      yNorm = 0;
      for (let s = 0; s < S * 2; s++) yNorm += y[s] * y[s];
    }
    function lossFor(C, b, y) {
      const m = synth(C, b);
      let L = 0;
      for (let s = 0; s < S * 2; s++) { const d = m[s] - y[s]; L += d * d; }
      return L / yNorm;
    }

    return { N, S, G, atom, synth, dsfwStep, adamStep, setTarget, lossFor, certificateMap, fitter };
  }

  // Message-driven engine shared by the Web Worker and the main-thread
  // fallback. Messages:
  //   {type: "reset", N, target: {C, b}, S, seed}  cold start both methods
  //   {type: "target", target: {C, b}}             new target, keep splats
  //   {type: "step", dsfw, adam}                   run that many steps
  //   {type: "move", which, j, p}                  user moved splat j
  //   {type: "add", which, p} / {type: "remove", which, j}  user edits
  // Every message returns a snapshot of both optimizer states.
  function createEngine() {
    let model = null, y = null, adam = null, dsfw = null;

    function setTarget(t) {
      y = model.synth(t.C, Float64Array.from(t.b));
      model.setTarget(y);
    }
    function refreshDsfw(C) {
      const r = model.fitter.evaluate(C, y);
      dsfw = { ...dsfw, C, b: r.b, L: model.lossFor(C, r.b, y) };
    }
    function snapshot() {
      return {
        adam: { C: adam.C, b: adam.b, L: adam.L, t: adam.t },
        dsfw: { C: dsfw.C, b: dsfw.b, L: dsfw.L, t: dsfw.t, cert: dsfw.cert, moved: dsfw.moved },
      };
    }

    return function handle(msg) {
      if (msg.type === "reset") {
        if (!model || model.N !== msg.N) model = makeModel(msg.N);
        setTarget(msg.target);
        // Cold start shared by both methods: uniform centres, equal amplitude.
        const rand = DS.rng(msg.seed), N = msg.N;
        const C0 = Array.from({ length: msg.S }, () => [2 + rand() * (N - 4), 2 + rand() * (N - 4)]);
        const b0 = new Float64Array(msg.S * 2).map((_, i) => (i % 2 ? 0 : 0.3));
        adam = { C: C0.map((p) => p.slice()), b: Float64Array.from(b0), m: new Float64Array(msg.S * 4), v: new Float64Array(msg.S * 4), t: 0 };
        adam.L = model.lossFor(adam.C, adam.b, y);
        dsfw = { C: C0.map((p) => p.slice()), b: Float64Array.from(b0), t: 0, cert: null, moved: null };
        dsfw.L = model.lossFor(dsfw.C, dsfw.b, y);
      } else if (msg.type === "target") {
        setTarget(msg.target);
        adam.L = model.lossFor(adam.C, adam.b, y);
        refreshDsfw(dsfw.C);
        dsfw.cert = model.certificateMap(model.fitter.evaluate(dsfw.C, y).e, true).map;
      } else if (msg.type === "step") {
        for (let i = 0; i < msg.adam; i++) adam = model.adamStep(adam, y);
        for (let i = 0; i < msg.dsfw; i++) dsfw = model.dsfwStep(dsfw, y);
      } else if (msg.type === "add" || msg.type === "remove") {
        const edit = (C) => (msg.type === "add" ? [...C.map((p) => p.slice()), msg.p.slice()] : C.filter((_, j) => j !== msg.j));
        if (msg.which === "adam") {
          // New AdamW splats start like the cold start: amplitude 0.3, fresh moments.
          const grow = (arr, k, fill) => {
            const a = Array.from(arr);
            if (msg.type === "add") a.push(...fill); else a.splice(msg.j * k, k);
            return Float64Array.from(a);
          };
          adam.C = edit(adam.C);
          adam.b = grow(adam.b, 2, [0.3, 0]);
          adam.m = grow(adam.m, 4, [0, 0, 0, 0]);
          adam.v = grow(adam.v, 4, [0, 0, 0, 0]);
          adam.L = model.lossFor(adam.C, adam.b, y);
        } else {
          refreshDsfw(edit(dsfw.C));
          dsfw.moved = null;
        }
      } else if (msg.type === "move") {
        if (msg.which === "adam") {
          adam.C = adam.C.map((p, j) => (j === msg.j ? msg.p.slice() : p));
          adam.L = model.lossFor(adam.C, adam.b, y);
        } else {
          refreshDsfw(dsfw.C.map((p, j) => (j === msg.j ? msg.p.slice() : p)));
          dsfw.moved = null;
        }
      }
      return snapshot();
    };
  }

  return { template, makeModel, createEngine };
})();

// =========================================================================
// 2D view. Panels: target (editable reflectors), AdamW fit, DSFW fit or
// certificate. Splats in the two fit panels can be dragged mid-run.
// =========================================================================
(() => {
  if (typeof document === "undefined") return;
  // Captured now: currentScript is only set while this file first runs.
  const thisScript = document.currentScript;
  const root = document.getElementById("demo-race2d");
  if (!root) return;
  const bar = root.closest(".demo").querySelector(".demo-bar");
  const heat = Object.fromEntries([...root.querySelectorAll("canvas.heat")].map((c) => [c.dataset.panel, c]));
  const over = Object.fromEntries([...root.querySelectorAll("canvas.panel-overlay")].map((c) => [c.dataset.panel, c]));
  const cvLoss = root.querySelector('[data-panel="loss"]');
  const tools = bar.querySelector('.bar-actions[data-only="2d"]');
  const runBtn = tools.querySelector('[data-action="run"]');
  const resetBtn = tools.querySelector('[data-action="reset"]');
  const out = Object.fromEntries([...root.querySelectorAll("[data-out]")].map((e) => [e.dataset.out, e]));

  const N = 32, RES = 128;                 // 32 x 32 bins, drawn at 4 px per bin
  const EXTRA = 2;                         // splat budget = reflectors + 2
  const ADAM_PER_ROUND = 20, DSFW_PER_ROUND = 1;
  let budget = { dsfw: 40, adam: 800 };
  let shape = "square", dsfwView = "fit", seed = 1;
  let target = null, state = null, hist = { adam: [], dsfw: [] };
  let running = false, busy = false, drag = null;
  const tmp = [0, 0];

  // ---- Engine: Web Worker if possible, else the same code on this thread.
  // Created on the first send, i.e. when the demo first nears the viewport,
  // so readers who never scroll here never start the worker.
  let post = null, lastMsg = null;
  function startEngine() {
    let handle = null;
    const fallback = (msg) => {
      handle = handle || Race2D.createEngine();
      setTimeout(() => onState(handle(msg)), 0);
    };
    try {
      // Worker URL (stamped with its own content hash) plus the versions of
      // the scripts it imports, so a deploy never mixes old and new code.
      const url = new URL(thisScript.dataset.worker, document.baseURI);
      for (const name of ["dsp", "fidelity", "race2d"]) {
        const tag = document.querySelector(`script[src*="static/js/${name}.js"]`);
        const v = tag && new URL(tag.src).searchParams.get("v");
        if (v) url.searchParams.set(name, v);
      }
      const src = url.href;
      const w = new Worker(src);
      w.onmessage = (e) => onState(e.data);
      // file:// pages and strict CSPs refuse workers; fall back quietly.
      w.onerror = (e) => { e.preventDefault(); post = fallback; if (lastMsg) post(lastMsg); };
      post = (msg) => w.postMessage(msg);
    } catch (err) { post = fallback; }
  }
  function send(msg) {
    if (!post) startEngine();
    lastMsg = msg; busy = true; post(msg);
  }

  function templateTarget(name) {
    const C = Race2D.template(name, N);
    return { C, b: C.flatMap(() => [1, 0]) };
  }

  function coldStart() {
    budget = { dsfw: 40, adam: 800 };
    hist = { adam: [], dsfw: [] };
    send({ type: "reset", N, target, S: target.C.length + EXTRA, seed: seed++ });
    DS.setRunButton(runBtn, "run");
  }

  function onState(s) {
    busy = false;
    state = s;
    const ha = hist.adam, hd = hist.dsfw;
    if (!ha.length || ha[ha.length - 1][0] !== s.adam.t) ha.push([s.adam.t, s.adam.L]);
    if (!hd.length || hd[hd.length - 1][0] !== s.dsfw.t) hd.push([s.dsfw.t, s.dsfw.L]);
    drawAll();
    if (!running || drag) return;
    if (s.dsfw.t >= budget.dsfw && s.adam.t >= budget.adam) {
      running = false;
      DS.setRunButton(runBtn, "again");
      return;
    }
    step();
  }
  function step() {
    send({
      type: "step",
      dsfw: state.dsfw.t < budget.dsfw ? DSFW_PER_ROUND : 0,
      adam: state.adam.t < budget.adam ? ADAM_PER_ROUND : 0,
    });
  }

  // A reflector counts as found when a splat lies within 0.1 bin of it.
  function found(C) {
    return target.C.filter((m) => C.some((c) => Math.hypot(c[0] - m[0], c[1] - m[1]) < 0.1)).length;
  }

  // ---- Rendering --------------------------------------------------------
  // |sum_j b_j d(x - cx_j) d(y - cy_j)| on a RES x RES grid, by separable rows.
  // Pixel i covers bin coordinate (i + 0.5) N / RES - 0.5, so bin k is centred
  // in its 4-pixel cell.
  function field(C, b) {
    const re = new Float64Array(RES * RES), im = new Float64Array(RES * RES);
    const rx = new Float64Array(RES * 2), ry = new Float64Array(RES * 2);
    C.forEach((p, j) => {
      for (let i = 0; i < RES; i++) {
        const f = ((i + 0.5) * N) / RES - 0.5;
        DS.dirichlet(f - p[0], N, tmp); rx[i * 2] = tmp[0]; rx[i * 2 + 1] = tmp[1];
        DS.dirichlet(f - p[1], N, tmp); ry[i * 2] = tmp[0]; ry[i * 2 + 1] = tmp[1];
      }
      const br = b[j * 2], bi = b[j * 2 + 1];
      for (let r = 0; r < RES; r++) {
        const yr = ry[(RES - 1 - r) * 2], yi = ry[(RES - 1 - r) * 2 + 1];
        const cr = br * yr - bi * yi, ci = br * yi + bi * yr;
        for (let i = 0; i < RES; i++) {
          const xr = rx[i * 2], xi = rx[i * 2 + 1], o = r * RES + i;
          re[o] += cr * xr - ci * xi; im[o] += cr * xi + ci * xr;
        }
      }
    });
    const mag = new Float64Array(RES * RES);
    for (let o = 0; o < mag.length; o++) mag[o] = Math.hypot(re[o], im[o]);
    return mag;
  }
  function paint(cv, mag, peak) {
    cv.width = RES; cv.height = RES;
    const ctx = cv.getContext("2d"), img = ctx.createImageData(RES, RES);
    for (let o = 0; o < mag.length; o++) {
      const li = Math.max(0, Math.min(255, Math.round((mag[o] / peak) * 255))) * 3;
      img.data[o * 4] = DS.LUT[li]; img.data[o * 4 + 1] = DS.LUT[li + 1]; img.data[o * 4 + 2] = DS.LUT[li + 2]; img.data[o * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  // The certificate arrives on the model's G x G candidate grid (gx, gy
  // from 0 in steps of N / G); resample it onto the display grid.
  function paintCert(cv, map) {
    const G = Math.round(Math.sqrt(map.length));
    let mx = 1e-30;
    for (const v of map) mx = Math.max(mx, v);
    const mag = new Float64Array(RES * RES);
    for (let r = 0; r < RES; r++) for (let i = 0; i < RES; i++) {
      const fx = ((i + 0.5) * N) / RES - 0.5, fy = ((RES - 1 - r + 0.5) * N) / RES - 0.5;
      const gx = Math.min(G - 1, Math.max(0, Math.round((fx * G) / N)));
      const gy = Math.min(G - 1, Math.max(0, Math.round((fy * G) / N)));
      mag[r * RES + i] = map[gy * G + gx];
    }
    paint(cv, mag, mx);
  }

  const toPx = (p, w) => [((p[0] + 0.5) / N) * w, w - ((p[1] + 0.5) / N) * w];
  const toBins = (x, y, w) => [
    Math.min(N - 0.5, Math.max(0.5, (x / w) * N - 0.5)),
    Math.min(N - 0.5, Math.max(0.5, ((w - y) / w) * N - 0.5)),
  ];

  function drawMarkers(name) {
    const { ctx, w } = DS.fitCanvas(over[name]);
    ctx.clearRect(0, 0, w, w);
    if (name === "target") {
      target.C.forEach((p, i) => {
        const [x, y] = toPx(p, w);
        ctx.strokeStyle = "rgba(255,255,255,0.95)"; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.stroke();
      });
      return;
    }
    // True reflectors as small white dots, splats as hollow rings: the
    // heat map stays visible under both (solid splat dots hid the peaks).
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    for (const p of target.C) { const [x, y] = toPx(p, w); ctx.beginPath(); ctx.arc(x, y, 1.4, 0, 2 * Math.PI); ctx.fill(); }
    const st = state[name];
    if (name === "dsfw" && st.moved) {
      const [x0, y0] = toPx(st.moved.from, w), [x1, y1] = toPx(st.moved.to, w);
      ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.3;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.setLineDash([]);
    }
    const color = name === "adam" ? DS.token("--gau") : DS.token("--dir");
    st.C.forEach((p) => {
      const [x, y] = toPx(p, w);
      ctx.lineWidth = 1.6; ctx.strokeStyle = color;
      ctx.beginPath(); ctx.arc(x, y, 5.5, 0, 2 * Math.PI); ctx.stroke();
      ctx.lineWidth = 0.8; ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.beginPath(); ctx.arc(x, y, 6.8, 0, 2 * Math.PI); ctx.stroke();
    });
  }

  function drawLoss() {
    const { ctx, w, h } = DS.fitCanvas(cvLoss);
    const L = 44, R = 12, T = 10, B = 26, pw = w - L - R, ph = h - T - B;
    const lo = -14, hi = 1;
    const yOf = (v) => T + ((hi - Math.max(lo, Math.log10(v + 1e-300))) / (hi - lo)) * ph;
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let e = lo; e <= 0; e += 2) {
      const yy = Math.round(yOf(10 ** e)) + 0.5;
      ctx.strokeStyle = DS.token("--rule"); ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(L + pw, yy); ctx.stroke();
      ctx.fillStyle = DS.token("--muted"); ctx.fillText(e === 0 ? "1" : `1e${e}`, L - 6, yy);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("fraction of each method's iteration budget", L + pw / 2, T + ph + 8);
    const line = (pts, total, color) => {
      ctx.beginPath();
      pts.forEach(([t, v], i) => {
        const x = L + Math.min(1, t / total) * pw, y = yOf(v);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.stroke();
    };
    line(hist.adam, budget.adam, DS.token("--gau"));
    line(hist.dsfw, budget.dsfw, DS.token("--dir"));
  }

  let targetPeak = 1;
  function drawAll() {
    if (!state || !target) return;
    const tm = field(target.C, target.b);
    targetPeak = 0;
    for (const v of tm) targetPeak = Math.max(targetPeak, v);
    paint(heat.target, tm, targetPeak || 1);
    // Fits share the target's colour scale so they read as comparable.
    paint(heat.adam, field(state.adam.C, state.adam.b), targetPeak || 1);
    if (dsfwView === "cert" && state.dsfw.cert) paintCert(heat.dsfw, state.dsfw.cert);
    else paint(heat.dsfw, field(state.dsfw.C, state.dsfw.b), targetPeak || 1);
    drawMarkers("target"); drawMarkers("adam"); drawMarkers("dsfw");
    drawLoss();
    const K = target.C.length;
    out.adam.textContent = `loss ${state.adam.L.toExponential(1)}, ${found(state.adam.C)} of ${K} found`;
    out.dsfw.textContent = `loss ${state.dsfw.L.toExponential(1)}, ${found(state.dsfw.C)} of ${K} found`;
    out.dsfwTitle.textContent = dsfwView === "cert" ? "DSFW (Ours), certificate" : "DSFW (Ours)";
  }

  // ---- Interaction -------------------------------------------------------
  // Shared gestures (DS.pointEditor) on all three panels: drag to move,
  // click empty space to add, right-click or long-press to delete. In the
  // target panel the points are reflectors; in the fit panels, splats.
  function hit(points, x, y, w) {
    let best = null, bd = 10;
    points.forEach((p, i) => {
      const [px, py] = toPx(p, w), d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }
  // After an edit on a finished run, extend the budget so Run continues.
  function editedAfterRun() {
    if (running || !state) return;
    budget = { dsfw: state.dsfw.t + 20, adam: state.adam.t + 400 };
    DS.setRunButton(runBtn, "continue");
  }
  const MAX_POINTS = 40, MIN_REFLECTORS = 2, MIN_SPLATS = 1;
  const width = (cv) => cv.getBoundingClientRect().width;

  for (const [name, cv] of Object.entries(over)) {
    const isTarget = name === "target";
    const points = () => (isTarget ? target.C : state[name].C);
    DS.pointEditor(cv, {
      hit: (x, y) => (state ? hit(points(), x, y, width(cv)) : null),
      add: (x, y) => {
        if (!state || points().length >= MAX_POINTS) return null;
        const p = toBins(x, y, width(cv));
        drag = true;
        if (isTarget) {
          target.C.push(p); target.b.push(1, 0);
        } else {
          // Shown at once; the engine adds it when the drag ends.
          state[name].C = [...state[name].C, p];
          state[name].b = Float64Array.from([...state[name].b, 0, 0]);
          cv.dataset.pendingAdd = "1";
        }
        drawAll();
        return points().length - 1;
      },
      move: (i, x, y) => {
        drag = true;
        const p = toBins(x, y, width(cv));
        if (isTarget) target.C[i] = p;
        else state[name].C = state[name].C.map((q, j) => (j === i ? p : q));
        drawAll();
      },
      end: (i) => {
        drag = null;
        editedAfterRun();
        // The engine replies with a fresh state; onState resumes a running loop.
        if (isTarget) send({ type: "target", target });
        else if (cv.dataset.pendingAdd) {
          delete cv.dataset.pendingAdd;
          send({ type: "add", which: name, p: state[name].C[i] });
        } else send({ type: "move", which: name, j: i, p: state[name].C[i] });
      },
      remove: (i) => {
        if (isTarget) {
          if (target.C.length <= MIN_REFLECTORS) return;
          target.C.splice(i, 1); target.b.splice(i * 2, 2);
          editedAfterRun();
          send({ type: "target", target });
        } else {
          if (state[name].C.length <= MIN_SPLATS) return;
          editedAfterRun();
          send({ type: "remove", which: name, j: i });
        }
      },
    });
  }

  runBtn.addEventListener("click", () => {
    if (running) { running = false; DS.setRunButton(runBtn, "resume"); return; }
    // Finished and not edited since: start over from a new random start.
    const finished = state && state.dsfw.t >= budget.dsfw && state.adam.t >= budget.adam;
    running = true;
    DS.setRunButton(runBtn, "pause");
    if (finished) coldStart();
    else if (!busy) step();
  });
  resetBtn.addEventListener("click", () => { running = false; coldStart(); });

  const shapeGroup = bar.querySelector('[data-group="shape"]');
  shapeGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    for (const b of shapeGroup.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
    shape = btn.value; target = templateTarget(shape); running = false;
    coldStart();
  });
  const viewGroup = bar.querySelector('[data-group="dsfwview"]');
  viewGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    for (const b of viewGroup.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
    dsfwView = btn.value;
    drawAll();
  });

  // Pause when scrolled away or when the 1D view is selected.
  DS.whenVisible(root, (v) => { if (!v && running) { running = false; DS.setRunButton(runBtn, "resume"); } });
  DS.onResize(root, () => {
    if (!target) { target = templateTarget(shape); coldStart(); } else drawAll();
  });
})();

// =========================================================================
// 1D / 2D switch for the optimizer demo (2D by default).
// =========================================================================
(() => {
  if (typeof document === "undefined") return;
  const root = document.getElementById("demo-opt");
  if (!root) return;
  const group = root.querySelector('[data-group="dim"]');
  const apply = (mode) => {
    for (const v of root.querySelectorAll("[data-view]")) v.hidden = v.dataset.view !== mode;
    for (const t of root.querySelectorAll("[data-only]")) t.hidden = t.dataset.only !== mode;
    for (const b of group.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === mode));
  };
  group.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) apply(b.value); });
  apply("2d");
})();
