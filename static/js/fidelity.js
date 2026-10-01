// Forward-fidelity demos (paper Fig. 3, made interactive).
// A random scene of K complex reflectors is rendered with the exact
// Dirichlet kernel. Two representations are then fitted to it:
//   - K Dirichlet splats: complex amplitudes by VarPro, centres placed by
//     the residual certificate and refined by Levenberg-Marquardt;
//   - M Gaussian splats (free weight, centre, width), fitted to the dB
//     magnitude profile by Levenberg-Marquardt, the setting in which
//     Gaussians do best.
// Both are scored by magnitude NMSE on a dense continuous grid, in dB.
"use strict";

const Fidelity = (() => {
  const tmp = [0, 0], der = [0, 0];

  // ---------------------------------------------------------------------
  // Dirichlet fit: complex VarPro + LM over atom parameters.
  // atom(p, col, dcols) fills one complex column (length 2*S) and its
  // derivative columns w.r.t. each of the atom's P parameters.
  // ---------------------------------------------------------------------
  function makeDirichletFitter({ S, P, atom, certificate, clamp }) {
    const lambda = 1e-9;

    function build(C) {
      const K = C.length;
      const A = new Float64Array(S * K * 2);
      const D = new Float64Array(S * K * P * 2);
      const col = new Float64Array(S * 2);
      const dcols = Array.from({ length: P }, () => new Float64Array(S * 2));
      for (let j = 0; j < K; j++) {
        atom(C[j], col, dcols);
        for (let s = 0; s < S; s++) {
          A[(s * K + j) * 2] = col[s * 2]; A[(s * K + j) * 2 + 1] = col[s * 2 + 1];
          for (let p = 0; p < P; p++) {
            D[((s * K + j) * P + p) * 2] = dcols[p][s * 2];
            D[((s * K + j) * P + p) * 2 + 1] = dcols[p][s * 2 + 1];
          }
        }
      }
      return { A, D };
    }

    function varpro(A, K, y) {
      const G = new Float64Array(K * K * 2), q = new Float64Array(K * 2);
      for (let i = 0; i < K; i++) {
        for (let j = i; j < K; j++) {
          let sr = 0, si = 0;
          for (let s = 0; s < S; s++) {
            const ar = A[(s * K + i) * 2], ai = -A[(s * K + i) * 2 + 1];
            const br = A[(s * K + j) * 2], bi = A[(s * K + j) * 2 + 1];
            sr += ar * br - ai * bi; si += ar * bi + ai * br;
          }
          G[(i * K + j) * 2] = sr + (i === j ? lambda : 0); G[(i * K + j) * 2 + 1] = si;
          G[(j * K + i) * 2] = sr + (i === j ? lambda : 0); G[(j * K + i) * 2 + 1] = -si;
        }
        let sr = 0, si = 0;
        for (let s = 0; s < S; s++) {
          const ar = A[(s * K + i) * 2], ai = -A[(s * K + i) * 2 + 1];
          sr += ar * y[s * 2] - ai * y[s * 2 + 1];
          si += ar * y[s * 2 + 1] + ai * y[s * 2];
        }
        q[i * 2] = sr; q[i * 2 + 1] = si;
      }
      const b = DS.csolve(G, q, K);
      // residual e = A b - y
      const e = new Float64Array(S * 2);
      let L = 0;
      for (let s = 0; s < S; s++) {
        let r = -y[s * 2], im = -y[s * 2 + 1];
        for (let j = 0; j < K; j++) {
          const ar = A[(s * K + j) * 2], ai = A[(s * K + j) * 2 + 1];
          r += ar * b[j * 2] - ai * b[j * 2 + 1];
          im += ar * b[j * 2 + 1] + ai * b[j * 2];
        }
        e[s * 2] = r; e[s * 2 + 1] = im; L += r * r + im * im;
      }
      return { b, e, L };
    }

    function evaluate(C, y) {
      if (C.length === 0) {
        let L = 0;
        const e = new Float64Array(S * 2);
        for (let s = 0; s < S * 2; s++) { e[s] = -y[s]; L += y[s] * y[s]; }
        return { C, b: new Float64Array(0), e, L };
      }
      const { A } = build(C);
      return { C, ...varpro(A, C.length, y) };
    }

    // Levenberg-Marquardt on all atom parameters, amplitudes re-solved
    // in closed form after each accepted step.
    function slide(C0, y, iters) {
      let cur = evaluate(C0, y);
      let damp = 1e-3;
      const K = C0.length, NP = K * P;
      for (let it = 0; it < iters; it++) {
        const { D } = build(cur.C);
        // J[s][j*P+p] = b_j * dA[s][j][p]
        const H = new Float64Array(NP * NP), g = new Float64Array(NP);
        const J = new Float64Array(NP * 2);
        for (let s = 0; s < S; s++) {
          for (let j = 0; j < K; j++) {
            const br = cur.b[j * 2], bi = cur.b[j * 2 + 1];
            for (let p = 0; p < P; p++) {
              const dr = D[((s * K + j) * P + p) * 2], di = D[((s * K + j) * P + p) * 2 + 1];
              J[(j * P + p) * 2] = br * dr - bi * di;
              J[(j * P + p) * 2 + 1] = br * di + bi * dr;
            }
          }
          const er = cur.e[s * 2], ei = cur.e[s * 2 + 1];
          for (let a = 0; a < NP; a++) {
            const ar = J[a * 2], ai = J[a * 2 + 1];
            g[a] += ar * er + ai * ei;
            for (let c = a; c < NP; c++) H[a * NP + c] += ar * J[c * 2] + ai * J[c * 2 + 1];
          }
        }
        for (let a = 0; a < NP; a++) for (let c = 0; c < a; c++) H[a * NP + c] = H[c * NP + a];
        let accepted = false;
        for (let t = 0; t < 8 && !accepted; t++) {
          const M = Float64Array.from(H);
          for (let a = 0; a < NP; a++) M[a * NP + a] += damp * H[a * NP + a] + 1e-12;
          const step = DS.rsolve(M, g.map((v) => -v), NP);
          const Cn = cur.C.map((p, j) => clamp(p.map((v, k) => v + Math.max(-0.5, Math.min(0.5, step[j * P + k])))));
          const nxt = evaluate(Cn, y);
          if (nxt.L < cur.L) { cur = nxt; damp = Math.max(1e-9, damp / 5); accepted = true; }
          else damp *= 6;
        }
        if (!accepted) break;
      }
      return cur;
    }

    // Greedy: add one atom at the certificate peak, slide, repeat.
    function fit(y, K) {
      let cur = evaluate([], y);
      for (let k = 0; k < K; k++) {
        const p = certificate(cur.e);
        cur = slide([...cur.C, p], y, 12);
      }
      return slide(cur.C, y, 80);
    }

    return { fit, evaluate };
  }

  // ---------------------------------------------------------------------
  // Gaussian fit on the magnitude profile, minimizing the reported metric
  // (linear magnitude NMSE). Each splat has nonlinear shape parameters
  // (centre, log width per axis); its real, signed weight is solved in
  // closed form (VarPro), so splats may also cancel each other.
  // basis(p, x, grads, wantGrad) returns the unit-weight splat at x and
  // fills d/dp for its P shape parameters.
  // Signed weights, not only positive ones: an earlier non-negative,
  // log-domain fit stalled near -17 dB with 96 splats because it could not
  // carve the deep nulls between interfering reflectors.
  // ---------------------------------------------------------------------
  function makeGaussianFitter({ X, target, P, basis, clamp }) {
    const S = X.length;
    let tn = 0;
    for (const t of target) tn += t * t;

    // Returns weights, model, and normalized loss for shape parameters G.
    function solve(G) {
      const K = G.length;
      const Phi = new Float64Array(S * K);
      const gr = new Float64Array(P);
      for (let j = 0; j < K; j++) for (let s = 0; s < S; s++) Phi[s * K + j] = basis(G[j], X[s], gr, false);
      const A = new Float64Array(K * K), q = new Float64Array(K);
      let tr = 0;
      for (let s = 0; s < S; s++) {
        const row = s * K, t = target[s];
        for (let i = 0; i < K; i++) {
          const v = Phi[row + i];
          if (v === 0) continue;
          q[i] += v * t;
          for (let j = i; j < K; j++) A[i * K + j] += v * Phi[row + j];
        }
      }
      for (let i = 0; i < K; i++) { tr += A[i * K + i]; for (let j = 0; j < i; j++) A[i * K + j] = A[j * K + i]; }
      // Tiny ridge keeps near-duplicate splats from exploding the weights.
      const ridge = 1e-9 * (tr / K + 1e-12);
      for (let i = 0; i < K; i++) A[i * K + i] += ridge;
      const w = DS.rsolve(A, q, K);
      const m = new Float64Array(S);
      let e = 0;
      for (let s = 0; s < S; s++) {
        let v = 0;
        for (let j = 0; j < K; j++) v += Phi[s * K + j] * w[j];
        m[s] = v;
        const d = Math.abs(v) - target[s]; e += d * d;
      }
      return { G, w, m, L: e / tn };
    }

    // One LM iteration on the shape parameters (weights held at their
    // VarPro optimum when forming the Jacobian).
    function step(state) {
      const { G, w, m } = state;
      const K = G.length, NP = K * P;
      const H = new Float64Array(NP * NP), g = new Float64Array(NP);
      const row = new Float64Array(NP), gr = new Float64Array(P);
      const nz = new Int32Array(NP);
      for (let s = 0; s < S; s++) {
        // Residual of the magnitude; d|m|/dm = sign(m).
        const sg = m[s] >= 0 ? 1 : -1;
        const r = Math.abs(m[s]) - target[s];
        let n = 0;
        for (let j = 0; j < K; j++) {
          const v = basis(G[j], X[s], gr, true);
          // Skip splats that are zero here: the Gauss-Newton build then
          // costs O(S * active^2) instead of O(S * NP^2).
          if (v === 0) continue;
          for (let p = 0; p < P; p++) { row[j * P + p] = sg * w[j] * gr[p]; nz[n++] = j * P + p; }
        }
        for (let a = 0; a < n; a++) {
          const ia = nz[a], va = row[ia];
          g[ia] += va * r;
          for (let c = a; c < n; c++) H[ia * NP + nz[c]] += va * row[nz[c]];
        }
      }
      for (let a = 0; a < NP; a++) for (let c = a + 1; c < NP; c++) {
        const v = H[a * NP + c] + H[c * NP + a]; H[a * NP + c] = v; H[c * NP + a] = v;
      }
      let damp = state.damp;
      for (let t = 0; t < 8; t++) {
        const M = Float64Array.from(H);
        for (let a = 0; a < NP; a++) M[a * NP + a] += damp * (H[a * NP + a] + 1e-9);
        const d = DS.rsolve(M, g.map((v) => -v), NP);
        const Gn = G.map((p, j) => clamp(p.map((v, k) => v + Math.max(-0.5, Math.min(0.5, d[j * P + k])))));
        const nxt = solve(Gn);
        if (nxt.L < state.L) {
          // Stop after 5 accepted steps in a row that each gain under 0.01%.
          const tiny = state.L - nxt.L < 1e-4 * state.L ? (state.tiny || 0) + 1 : 0;
          return { ...nxt, damp: Math.max(1e-9, damp / 4), tiny, stalled: tiny >= 5 };
        }
        damp *= 5;
      }
      return { ...state, damp, stalled: true };
    }

    function init(G) { return { ...solve(G), damp: 1e-3, stalled: false }; }

    return { init, step };
  }

  // Values under -150 dB are float64 round-off, so they print as a bound.
  function fmtDb(db) { return db < -150 ? "below −150 dB" : `${db.toFixed(1)} dB`; }

  function nmseDb(model, target) {
    let e = 0, n = 0;
    for (let i = 0; i < target.length; i++) {
      const d = model[i] - target[i]; e += d * d; n += target[i] * target[i];
    }
    return 10 * Math.log10(Math.max(e / n, 1e-16));
  }

  return { makeDirichletFitter, makeGaussianFitter, nmseDb, fmtDb };
})();

// =========================================================================
// 1D view: a 48-bin range profile in dB, Dirichlet and Gaussian fits in
// separate panels. `out` holds the shared readout elements.
// =========================================================================
function makeFit1D(stage, out) {
  const cvD = stage.querySelector('[data-panel="dirichlet"]');
  const cvG = stage.querySelector('[data-panel="gaussian"]');

  const N = 48;            // DFT length = number of measured bins
  const OS = 8;            // continuous samples per bin for fitting/scoring
  const FLOOR_DB = -60;
  const X = Array.from({ length: N * OS }, (_, i) => (i + 0.5) / OS);
  let seed = 3, M = 16;
  let scene, dfit, gfit, gstate, iter, raf = null;

  const SIG_MIN = 0.12, SIG_MAX = N / 4;
  const wrap = (d) => d - N * Math.round(d / N);

  function newScene() {
    const rand = DS.rng(seed);
    const K = 2 + Math.floor(rand() * 3);   // 2-4 reflectors
    const C = [];
    while (C.length < K) {
      const c = 16 + rand() * (N - 32);
      if (C.every((v) => Math.abs(v - c) > 1.6)) C.push(c);
    }
    const b = new Float64Array(K * 2);
    for (let i = 0; i < K; i++) {
      const a = 0.45 + 0.55 * rand(), ph = rand() * 2 * Math.PI;
      b[i * 2] = a * Math.cos(ph); b[i * 2 + 1] = a * Math.sin(ph);
    }
    // Continuous profile, normalized to unit peak.
    const prof = X.map((f) => {
      let r = 0, im = 0;
      C.forEach((c, j) => {
        DS.dirichlet(f - c, N, tmp1);
        r += b[j * 2] * tmp1[0] - b[j * 2 + 1] * tmp1[1];
        im += b[j * 2] * tmp1[1] + b[j * 2 + 1] * tmp1[0];
      });
      return Math.hypot(r, im);
    });
    const peak = Math.max(...prof);
    for (let i = 0; i < b.length; i++) b[i] /= peak;
    // Sensor data: the integer bins.
    const y = new Float64Array(N * 2);
    for (let k = 0; k < N; k++) {
      C.forEach((c, j) => {
        DS.dirichlet(k - c, N, tmp1);
        y[k * 2] += b[j * 2] * tmp1[0] - b[j * 2 + 1] * tmp1[1];
        y[k * 2 + 1] += b[j * 2] * tmp1[1] + b[j * 2 + 1] * tmp1[0];
      });
    }
    scene = { K, C, b, y, target: prof.map((v) => v / peak) };
  }
  const tmp1 = [0, 0], tmp2 = [0, 0];

  const dirFitter = Fidelity.makeDirichletFitter({
    S: N, P: 1,
    atom: (p, col, dcols) => {
      for (let k = 0; k < N; k++) {
        DS.dirichlet(k - p[0], N, tmp1); col[k * 2] = tmp1[0]; col[k * 2 + 1] = tmp1[1];
        DS.dirichletDeriv(k - p[0], N, tmp2); dcols[0][k * 2] = -tmp2[0]; dcols[0][k * 2 + 1] = -tmp2[1];
      }
    },
    // Certificate on a 0.02-bin grid: |<a(x), e>| (all atoms have equal norm).
    certificate: (e) => {
      let best = 0, bx = 0;
      for (let x = 0; x < N; x += 0.02) {
        let sr = 0, si = 0;
        for (let k = 0; k < N; k++) {
          DS.dirichlet(k - x, N, tmp1);
          sr += tmp1[0] * e[k * 2] + tmp1[1] * e[k * 2 + 1];
          si += tmp1[0] * e[k * 2 + 1] - tmp1[1] * e[k * 2];
        }
        const v = sr * sr + si * si;
        if (v > best) { best = v; bx = x; }
      }
      return [bx];
    },
    clamp: (p) => [((p[0] % N) + N) % N],
  });

  function dirichletProfile(C, b) {
    return X.map((f) => {
      let r = 0, im = 0;
      C.forEach((p, j) => {
        DS.dirichlet(f - p[0], N, tmp1);
        r += b[j * 2] * tmp1[0] - b[j * 2 + 1] * tmp1[1];
        im += b[j * 2] * tmp1[1] + b[j * 2 + 1] * tmp1[0];
      });
      return Math.hypot(r, im);
    });
  }

  // Unit-weight Gaussian; shape params [centre, log sigma]. The weight is
  // solved in closed form, so each splat still has 3 degrees of freedom.
  const basis = (p, x, gr, wantGrad) => {
    const sg = Math.exp(p[1]);
    const d = wrap(x - p[0]);
    const z = d / sg;
    if (z * z > 60) return 0;
    const v = Math.exp(-0.5 * z * z);
    if (wantGrad) { gr[0] = (v * d) / (sg * sg); gr[1] = v * z * z; }
    return v;
  };

  // Init: the M highest local maxima of the target (main lobes first,
  // then sidelobes), then evenly spaced fill if M exceeds the maxima.
  function gaussInit() {
    const t = scene.target, n = t.length;
    const peaks = [];
    for (let i = 0; i < n; i++) {
      if (t[i] >= t[(i - 1 + n) % n] && t[i] > t[(i + 1) % n]) peaks.push(i);
    }
    peaks.sort((a, b) => t[b] - t[a]);
    const s0 = Math.log(0.45);
    const G = peaks.slice(0, M).map((i) => [X[i], s0]);
    const rest = M - G.length;
    for (let k = 0; k < rest; k++) G.push([((k + 0.5) * N) / rest, s0]);
    return G;
  }

  const gFitter = () => Fidelity.makeGaussianFitter({
    X, target: scene.target, P: 2, basis,
    clamp: (p) => [((p[0] % N) + N) % N, Math.log(Math.min(SIG_MAX, Math.max(SIG_MIN, Math.exp(p[1]))))],
  });

  function drawPanel(cv, profile, centers, color, title, nmse, dashed) {
    const { ctx, w, h } = DS.fitCanvas(cv);
    const L = 40, R = 10, T = 26, B = 22;
    const pw = w - L - R, ph = h - T - B;
    const xOf = (f) => L + (f / N) * pw;
    const yOf = (v) => T + (Math.max(FLOOR_DB, 20 * Math.log10(v + 1e-12)) / FLOOR_DB) * ph;
    const muted = DS.token("--muted"), rule = DS.token("--rule"), ink = DS.token("--ink");
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
    for (const db of [0, -20, -40, -60]) {
      const y = Math.round(T + (db / FLOOR_DB) * ph) + 0.5;
      ctx.strokeStyle = rule; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      ctx.fillStyle = muted; ctx.fillText(db === 0 ? "0 dB" : `${db}`, L - 5, y);
    }
    // Ground truth.
    ctx.beginPath();
    X.forEach((f, i) => { const yy = yOf(scene.target[i]); if (i === 0) ctx.moveTo(xOf(f), yy); else ctx.lineTo(xOf(f), yy); });
    ctx.strokeStyle = "rgba(23,32,44,0.45)"; ctx.lineWidth = 1; ctx.stroke();
    ctx.lineTo(xOf(N), T + ph); ctx.lineTo(xOf(0), T + ph); ctx.closePath();
    ctx.fillStyle = "rgba(23,32,44,0.08)"; ctx.fill();
    // Fit.
    if (profile) {
      ctx.beginPath();
      X.forEach((f, i) => { const yy = yOf(profile[i]); if (i === 0) ctx.moveTo(xOf(f), yy); else ctx.lineTo(xOf(f), yy); });
      ctx.strokeStyle = color; ctx.lineWidth = 1.7; ctx.setLineDash(dashed ? [5, 3] : []); ctx.stroke(); ctx.setLineDash([]);
    }
    // Splat centres along the bottom edge.
    ctx.fillStyle = color; ctx.globalAlpha = 0.75;
    for (const c of centers) { ctx.beginPath(); ctx.arc(xOf(c), T + ph + 7, 3, 0, 2 * Math.PI); ctx.fill(); }
    ctx.globalAlpha = 1;
    // Labels.
    ctx.textBaseline = "top"; ctx.font = "600 12px 'Hanken Grotesk', system-ui, sans-serif";
    ctx.textAlign = "left"; ctx.fillStyle = ink; ctx.fillText(title, L, 4);
    if (nmse !== null) {
      ctx.textAlign = "right"; ctx.fillStyle = color;
      ctx.fillText(`NMSE ${Fidelity.fmtDb(nmse)}`, L + pw, 4);
    }
  }

  function draw() {
    const dProf = dfit ? dirichletProfile(dfit.C, dfit.b) : null;
    const dN = dProf ? Fidelity.nmseDb(dProf, scene.target) : null;
    drawPanel(cvD, dProf, dfit ? dfit.C.map((p) => p[0]) : [], DS.token("--dir"), `Dirichlet × ${scene.K}`, dN, false);
    const gm = gstate ? Array.from(gstate.m, Math.abs) : null;
    const gN = gm ? Fidelity.nmseDb(gm, scene.target) : null;
    drawPanel(cvG, gm, gstate ? gstate.G.map((p) => p[0]) : [], DS.token("--gau"), `Gaussian × ${M}`, gN, false);
    out.dparams.textContent = `${scene.K * 3}`;
    out.gparams.textContent = `${M * 3}`;
    out.dnmse.textContent = dN === null ? "–" : Fidelity.fmtDb(dN);
    out.gnmse.textContent = gN === null ? "–" : Fidelity.fmtDb(gN);
  }

  function startFit() {
    cancelAnimationFrame(raf);
    dfit = dirFitter.fit(scene.y, scene.K);
    gfit = gFitter();
    gstate = gfit.init(gaussInit());
    iter = 0;
    out.status.textContent = "Fitting the Gaussians…";
    draw();
    const MAX_IT = 150;
    const tick = () => {
      const t0 = performance.now();
      while (iter < MAX_IT && !gstate.stalled && performance.now() - t0 < 14) {
        gstate = gfit.step(gstate); iter++;
      }
      draw();
      if (iter < MAX_IT && !gstate.stalled) {
        out.status.textContent = `Fitting the Gaussians, iteration ${iter}`;
        raf = requestAnimationFrame(tick);
      } else {
        out.status.textContent = `Done after ${iter} Levenberg–Marquardt iterations.`;
      }
    };
    raf = requestAnimationFrame(tick);
  }

  newScene();
  return {
    start(m) { M = m; startFit(); },
    stop() { cancelAnimationFrame(raf); },
    reseed() { seed = (seed * 48271) % 2147483647; newScene(); },
    redraw() { if (scene) draw(); },
  };
}

// =========================================================================
// 2D view: a 16 x 16-bin spectrum as three heatmaps (ground truth,
// Dirichlet fit, Gaussian fit).
// =========================================================================
function makeFit2D(stage, out) {
  const canv = {
    gt: stage.querySelector('[data-panel="gt"]'),
    dir: stage.querySelector('[data-panel="dirichlet"]'),
    gau: stage.querySelector('[data-panel="gaussian"]'),
  };
  const titles = {
    dir: stage.querySelector('[data-out="dtitle"]'),
    gau: stage.querySelector('[data-out="gtitle"]'),
  };

  const N = 16;            // bins per axis
  const OS = 3;            // fitting samples per bin per axis (48 x 48)
  const RES = 128;         // display resolution per axis (8 px per bin)
  let seed = 11, M = 16, scale = "lin";
  let scene, dfit, gfit, gstate, iter, raf = null;
  const tmp1 = [0, 0], tmp2 = [0, 0];
  const wrap = (d) => d - N * Math.round(d / N);

  const SX = N * OS;
  const X = [];
  for (let j = 0; j < SX; j++) for (let i = 0; i < SX; i++) X.push([(i + 0.5) / OS, (j + 0.5) / OS]);
  const XD = [];
  for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) XD.push([((i + 0.5) * N) / RES, ((j + 0.5) * N) / RES]);

  function complexAt(C, b, f) {
    let r = 0, im = 0;
    C.forEach((p, j) => {
      DS.dirichlet(f[0] - p[0], N, tmp1); DS.dirichlet(f[1] - p[1], N, tmp2);
      const kr = tmp1[0] * tmp2[0] - tmp1[1] * tmp2[1], ki = tmp1[0] * tmp2[1] + tmp1[1] * tmp2[0];
      r += b[j * 2] * kr - b[j * 2 + 1] * ki;
      im += b[j * 2] * ki + b[j * 2 + 1] * kr;
    });
    return Math.hypot(r, im);
  }

  function newScene() {
    const rand = DS.rng(seed);
    const K = 2 + Math.floor(rand() * 3);
    const C = [];
    while (C.length < K) {
      const p = [5 + rand() * 6, 5 + rand() * 6];
      if (C.every((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) > 1.5)) C.push(p);
    }
    const b = new Float64Array(K * 2);
    for (let i = 0; i < K; i++) {
      const a = 0.45 + 0.55 * rand(), ph = rand() * 2 * Math.PI;
      b[i * 2] = a * Math.cos(ph); b[i * 2 + 1] = a * Math.sin(ph);
    }
    let peak = 0;
    for (const f of X) peak = Math.max(peak, complexAt(C, b, f));
    for (let i = 0; i < b.length; i++) b[i] /= peak;
    const y = new Float64Array(N * N * 2);
    for (let l = 0; l < N; l++) for (let k = 0; k < N; k++) {
      let r = 0, im = 0;
      C.forEach((p, j) => {
        DS.dirichlet(k - p[0], N, tmp1); DS.dirichlet(l - p[1], N, tmp2);
        const kr = tmp1[0] * tmp2[0] - tmp1[1] * tmp2[1], ki = tmp1[0] * tmp2[1] + tmp1[1] * tmp2[0];
        r += b[j * 2] * kr - b[j * 2 + 1] * ki; im += b[j * 2] * ki + b[j * 2 + 1] * kr;
      });
      y[(l * N + k) * 2] = r; y[(l * N + k) * 2 + 1] = im;
    }
    scene = {
      K, C, b, y,
      target: X.map((f) => complexAt(C, b, f)),
      display: XD.map((f) => complexAt(C, b, f)),
    };
  }

  const dirFitter = Fidelity.makeDirichletFitter({
    S: N * N, P: 2,
    atom: (p, col, dcols) => {
      const ax = new Float64Array(N * 2), ay = new Float64Array(N * 2);
      const dx = new Float64Array(N * 2), dy = new Float64Array(N * 2);
      for (let k = 0; k < N; k++) {
        DS.dirichlet(k - p[0], N, tmp1); ax[k * 2] = tmp1[0]; ax[k * 2 + 1] = tmp1[1];
        DS.dirichlet(k - p[1], N, tmp1); ay[k * 2] = tmp1[0]; ay[k * 2 + 1] = tmp1[1];
        DS.dirichletDeriv(k - p[0], N, tmp1); dx[k * 2] = -tmp1[0]; dx[k * 2 + 1] = -tmp1[1];
        DS.dirichletDeriv(k - p[1], N, tmp1); dy[k * 2] = -tmp1[0]; dy[k * 2 + 1] = -tmp1[1];
      }
      const mul = (ar, ai, br, bi, out, idx) => { out[idx] = ar * br - ai * bi; out[idx + 1] = ar * bi + ai * br; };
      for (let l = 0; l < N; l++) for (let k = 0; k < N; k++) {
        const s = (l * N + k) * 2;
        mul(ax[k * 2], ax[k * 2 + 1], ay[l * 2], ay[l * 2 + 1], col, s);
        mul(dx[k * 2], dx[k * 2 + 1], ay[l * 2], ay[l * 2 + 1], dcols[0], s);
        mul(ax[k * 2], ax[k * 2 + 1], dy[l * 2], dy[l * 2 + 1], dcols[1], s);
      }
    },
    // Separable certificate on a 0.1-bin grid:
    // <a(x,y), e> = sum_l conj d(l-y) * (sum_k conj d(k-x) e[l][k]).
    certificate: (e) => {
      const G = 10 * N, step = N / G;
      const tab = new Float64Array(G * N * 2);
      for (let g = 0; g < G; g++) for (let k = 0; k < N; k++) {
        DS.dirichlet(k - g * step, N, tmp1); tab[(g * N + k) * 2] = tmp1[0]; tab[(g * N + k) * 2 + 1] = tmp1[1];
      }
      let best = -1, bp = [0, 0];
      const v = new Float64Array(N * 2);
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
          if (m > best) { best = m; bp = [gx * step, gy * step]; }
        }
      }
      return bp;
    },
    clamp: (p) => p.map((v) => ((v % N) + N) % N),
  });

  // Unit-weight axis-aligned 2D Gaussian: [cx, cy, log sx, log sy]; the
  // weight is solved in closed form (5 degrees of freedom per splat).
  const basis = (p, x, gr, wantGrad) => {
    const sx = Math.exp(p[2]), sy = Math.exp(p[3]);
    const dx = wrap(x[0] - p[0]), dy = wrap(x[1] - p[1]);
    const zx = dx / sx, zy = dy / sy, q = zx * zx + zy * zy;
    if (q > 60) return 0;
    const v = Math.exp(-0.5 * q);
    if (wantGrad) {
      gr[0] = (v * dx) / (sx * sx); gr[1] = (v * dy) / (sy * sy);
      gr[2] = v * zx * zx; gr[3] = v * zy * zy;
    }
    return v;
  };

  function gaussInit() {
    const t = scene.target;
    const idx = (i, j) => ((j + SX) % SX) * SX + ((i + SX) % SX);
    const peaks = [];
    for (let j = 0; j < SX; j++) for (let i = 0; i < SX; i++) {
      const v = t[idx(i, j)];
      if (v > t[idx(i - 1, j)] && v >= t[idx(i + 1, j)] && v > t[idx(i, j - 1)] && v >= t[idx(i, j + 1)]) peaks.push(idx(i, j));
    }
    peaks.sort((a, b) => t[b] - t[a]);
    const s0 = Math.log(0.45);
    const G = peaks.slice(0, M).map((k) => [X[k][0], X[k][1], s0, s0]);
    const rand = DS.rng(seed + 7);
    while (G.length < M) G.push([rand() * N, rand() * N, s0, s0]);
    return G;
  }
  const clampS = (s) => Math.log(Math.min(N / 4, Math.max(0.12, Math.exp(s))));
  const gFitter = () => Fidelity.makeGaussianFitter({
    X, target: scene.target, P: 4, basis,
    clamp: (p) => [((p[0] % N) + N) % N, ((p[1] % N) + N) % N, clampS(p[2]), clampS(p[3])],
  });

  function paint(cv, values) {
    cv.width = RES; cv.height = RES;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(RES, RES);
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
      let t = values[(RES - 1 - j) * RES + i];
      if (scale === "db") t = Math.max(0, 1 + (20 * Math.log10(t + 1e-12)) / 40);
      const li = Math.max(0, Math.min(255, Math.round(Math.min(1, t) * 255))) * 3, o = (j * RES + i) * 4;
      img.data[o] = DS.LUT[li]; img.data[o + 1] = DS.LUT[li + 1]; img.data[o + 2] = DS.LUT[li + 2]; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function draw() {
    paint(canv.gt, scene.display);
    if (dfit) {
      const prof = XD.map((f) => complexAt(dfit.C, dfit.b, f));
      paint(canv.dir, prof);
      const nm = Fidelity.nmseDb(X.map((f) => complexAt(dfit.C, dfit.b, f)), scene.target);
      out.dnmse.textContent = Fidelity.fmtDb(nm);
    }
    if (gstate) {
      const gr = new Float64Array(4);
      const prof = XD.map((f) => Math.abs(gstate.G.reduce((s, p, j) => s + gstate.w[j] * basis(p, f, gr, false), 0)));
      paint(canv.gau, prof);
      out.gnmse.textContent = Fidelity.fmtDb(Fidelity.nmseDb(Array.from(gstate.m, Math.abs), scene.target));
    }
    titles.dir.textContent = `Dirichlet × ${scene.K}`;
    titles.gau.textContent = `Gaussian × ${M}`;
    out.dparams.textContent = `${scene.K * 4}`;
    out.gparams.textContent = `${M * 5}`;
  }

  function startFit() {
    cancelAnimationFrame(raf);
    dfit = dirFitter.fit(scene.y, scene.K);
    gfit = gFitter();
    gstate = gfit.init(gaussInit());
    iter = 0;
    draw();
    const MAX_IT = 80;
    const tick = () => {
      const t0 = performance.now();
      while (iter < MAX_IT && !gstate.stalled && performance.now() - t0 < 14) { gstate = gfit.step(gstate); iter++; }
      draw();
      if (iter < MAX_IT && !gstate.stalled) {
        out.status.textContent = `Fitting the Gaussians, iteration ${iter}`;
        raf = requestAnimationFrame(tick);
      } else out.status.textContent = `Done after ${iter} Levenberg–Marquardt iterations.`;
    };
    raf = requestAnimationFrame(tick);
  }
  newScene();
  return {
    start(m) { M = m; startFit(); },
    stop() { cancelAnimationFrame(raf); },
    reseed() { seed = (seed * 48271) % 2147483647; newScene(); },
    redraw() { if (scene) draw(); },
    setScale(v) { scale = v; if (scene) draw(); },
  };
}

// =========================================================================
// Controller: one demo with a 1D / 2D switch (2D by default). Only the
// active view runs; fitting starts the first time the demo scrolls into view.
// =========================================================================
(() => {
  const root = document.getElementById("demo-fit");
  if (!root) return;
  const out = Object.fromEntries([...root.querySelectorAll(".demo-side [data-out]")].map((e) => [e.dataset.out, e]));
  const stages = { "1d": root.querySelector('[data-view="1d"]'), "2d": root.querySelector('[data-view="2d"]') };
  const views = { "1d": makeFit1D(stages["1d"], out), "2d": makeFit2D(stages["2d"], out) };
  // Per-view limits: 2D Gaussian fits above 64 splats take several seconds.
  const MAX_COUNT = { "1d": 96, "2d": 64 };
  const NOTES = {
    "1d": "48 bins, shown in dB. A Dirichlet splat has a centre and a complex amplitude; a 1D Gaussian has a weight, centre, and width. NMSE compares magnitudes on a dense grid, as in the paper.",
    "2d": "16 × 16 bins. A Dirichlet splat has a 2D centre and a complex amplitude; an axis-aligned Gaussian has a weight, 2D centre, and two widths. NMSE compares magnitudes on a dense grid.",
  };
  const modeGroup = root.querySelector('[data-group="dim"]');
  const countGroup = root.querySelector('[data-group="count"]');
  const scaleGroup = root.querySelector('[data-group="scale"]');
  const scaleTool = scaleGroup.closest(".tool");
  const note = root.querySelector('[data-out="note"]');
  let mode = "2d", M = 16, visible = false, dirty = true;

  const press = (group, value) => {
    for (const b of group.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.value === String(value)));
  };

  function applyMode() {
    for (const k of Object.keys(stages)) stages[k].hidden = k !== mode;
    for (const b of countGroup.querySelectorAll("button")) b.hidden = +b.value > MAX_COUNT[mode];
    if (M > MAX_COUNT[mode]) M = MAX_COUNT[mode];
    press(countGroup, M);
    scaleTool.hidden = mode !== "2d";
    note.textContent = NOTES[mode];
  }
  function run() {
    if (!visible) { dirty = true; return; }
    dirty = false;
    for (const k of Object.keys(views)) if (k !== mode) views[k].stop();
    views[mode].start(M);
  }

  modeGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn || btn.value === mode) return;
    mode = btn.value; press(modeGroup, mode); applyMode(); run();
  });
  countGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    M = +btn.value; press(countGroup, M); run();
  });
  scaleGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    press(scaleGroup, btn.value); views["2d"].setScale(btn.value);
  });
  root.querySelector('[data-action="new"]').addEventListener("click", () => { views[mode].reseed(); run(); });
  DS.whenVisible(root, (v) => { visible = v; if (v && dirty) run(); });
  DS.onResize(root, () => views[mode].redraw());
  applyMode();
})();
