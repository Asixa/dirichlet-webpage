// Shared signal-processing and drawing helpers for the interactive figures.
// Complex numbers are plain [re, im] pairs; arrays of complex values are
// stored interleaved (re0, im0, re1, im1, ...) in Float64Array for speed.
"use strict";

const DS = (() => {
  const PI = Math.PI;

  // Unit-peak complex Dirichlet kernel d_N(delta) = kappa_N(delta) / N
  // (paper Eq. 1). delta is in bin units. Period is N; removable
  // singularities at delta in N*Z are resolved with the exact limit
  // (-1)^{m(N-1)} instead of a Taylor series, which is enough at float64.
  function dirichlet(delta, N, out) {
    const s = Math.sin(PI * delta / N);
    let mag;
    if (Math.abs(s) < 1e-10) {
      const m = Math.round(delta / N);
      mag = Math.abs(m * (N - 1)) % 2 === 0 ? 1 : -1;
    } else {
      mag = Math.sin(PI * delta) / (N * s);
    }
    const ph = -PI * (N - 1) * delta / N;
    out[0] = mag * Math.cos(ph);
    out[1] = mag * Math.sin(ph);
    return out;
  }

  // d/d(delta) of the kernel by central difference. h = 1e-4 bins keeps the
  // truncation error near 1e-8, far below anything the demos display.
  const _a = [0, 0], _b = [0, 0];
  function dirichletDeriv(delta, N, out) {
    const h = 1e-4;
    dirichlet(delta + h, N, _a);
    dirichlet(delta - h, N, _b);
    out[0] = (_a[0] - _b[0]) / (2 * h);
    out[1] = (_a[1] - _b[1]) / (2 * h);
    return out;
  }

  // Real Gaussian whose main lobe has the same FWHM as |d_N| (about 1.207
  // bins for large N). This is the "diffraction-matched" Gaussian used as
  // the fairest single-splat Gaussian stand-in.
  const GAUSS_SIGMA = 1.2067 / 2.3548;
  function gaussian(delta, sigma = GAUSS_SIGMA) {
    return Math.exp(-0.5 * (delta * delta) / (sigma * sigma));
  }

  // Wrap delta into [-N/2, N/2) so a Gaussian sees the same periodic
  // domain as the DFT; otherwise reflectors near the edge would look wrong.
  function wrap(delta, N) {
    return delta - N * Math.round(delta / N);
  }

  // matplotlib "Spectral" reversed: purple (low) -> yellow -> dark red (high),
  // the colormap used in every heatmap of the paper.
  const SPECTRAL = [
    [0x5e, 0x4f, 0xa2], [0x32, 0x88, 0xbd], [0x66, 0xc2, 0xa5],
    [0xab, 0xdd, 0xa4], [0xe6, 0xf5, 0x98], [0xff, 0xff, 0xbf],
    [0xfe, 0xe0, 0x8b], [0xfd, 0xae, 0x61], [0xf4, 0x6d, 0x43],
    [0xd5, 0x3e, 0x4f], [0x9e, 0x01, 0x42],
  ];
  // Precomputed 256-entry LUT so the 2D demos can colormap per pixel cheaply.
  const LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (SPECTRAL.length - 1);
    const k = Math.min(Math.floor(t), SPECTRAL.length - 2);
    const f = t - k;
    for (let c = 0; c < 3; c++) {
      LUT[i * 3 + c] = SPECTRAL[k][c] * (1 - f) + SPECTRAL[k + 1][c] * f;
    }
  }
  function spectral(t) {
    const i = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
    return `rgb(${LUT[i]},${LUT[i + 1]},${LUT[i + 2]})`;
  }

  // Phase -> hue. Used for the stems in the kernel explorer so the linear
  // phase law of the Dirichlet kernel is visible as a colour ramp.
  function phaseColor(re, im, light = 46) {
    const deg = ((Math.atan2(im, re) * 180) / PI + 360) % 360;
    return `hsl(${deg.toFixed(0)}, 62%, ${light}%)`;
  }

  // Resize a canvas to its CSS box at device pixel ratio and return a
  // context already scaled to CSS pixels. Call on every layout change.
  function fitCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  // Read design tokens from CSS so canvases follow the stylesheet.
  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ---- Small complex linear algebra (S <= 8, so O(S^3) is free) ----

  // Solve M x = v for complex M (S x S, interleaved rows) by Gaussian
  // elimination with partial pivoting. M and v are copied, not modified.
  function csolve(Min, vin, S) {
    const M = Float64Array.from(Min), v = Float64Array.from(vin);
    for (let col = 0; col < S; col++) {
      let piv = col, best = -1;
      for (let r = col; r < S; r++) {
        const re = M[(r * S + col) * 2], im = M[(r * S + col) * 2 + 1];
        const m = re * re + im * im;
        if (m > best) { best = m; piv = r; }
      }
      if (piv !== col) {
        for (let c = 0; c < S; c++) {
          for (let q = 0; q < 2; q++) {
            const a = (col * S + c) * 2 + q, b = (piv * S + c) * 2 + q;
            const t = M[a]; M[a] = M[b]; M[b] = t;
          }
        }
        for (let q = 0; q < 2; q++) {
          const t = v[col * 2 + q]; v[col * 2 + q] = v[piv * 2 + q]; v[piv * 2 + q] = t;
        }
      }
      const pr = M[(col * S + col) * 2], pi = M[(col * S + col) * 2 + 1];
      const pd = pr * pr + pi * pi || 1e-300;
      for (let r = col + 1; r < S; r++) {
        const ar = M[(r * S + col) * 2], ai = M[(r * S + col) * 2 + 1];
        // f = a / p
        const fr = (ar * pr + ai * pi) / pd, fi = (ai * pr - ar * pi) / pd;
        for (let c = col; c < S; c++) {
          const br = M[(col * S + c) * 2], bi = M[(col * S + c) * 2 + 1];
          M[(r * S + c) * 2] -= fr * br - fi * bi;
          M[(r * S + c) * 2 + 1] -= fr * bi + fi * br;
        }
        const br = v[col * 2], bi = v[col * 2 + 1];
        v[r * 2] -= fr * br - fi * bi;
        v[r * 2 + 1] -= fr * bi + fi * br;
      }
    }
    const x = new Float64Array(S * 2);
    for (let r = S - 1; r >= 0; r--) {
      let sr = v[r * 2], si = v[r * 2 + 1];
      for (let c = r + 1; c < S; c++) {
        const mr = M[(r * S + c) * 2], mi = M[(r * S + c) * 2 + 1];
        sr -= mr * x[c * 2] - mi * x[c * 2 + 1];
        si -= mr * x[c * 2 + 1] + mi * x[c * 2];
      }
      const pr = M[(r * S + r) * 2], pi = M[(r * S + r) * 2 + 1];
      const pd = pr * pr + pi * pi || 1e-300;
      x[r * 2] = (sr * pr + si * pi) / pd;
      x[r * 2 + 1] = (si * pr - sr * pi) / pd;
    }
    return x;
  }

  // Solve a real S x S system (row-major) by Gaussian elimination.
  function rsolve(Min, vin, S) {
    const M = Float64Array.from(Min), v = Float64Array.from(vin);
    for (let col = 0; col < S; col++) {
      let piv = col;
      for (let r = col + 1; r < S; r++) {
        if (Math.abs(M[r * S + col]) > Math.abs(M[piv * S + col])) piv = r;
      }
      if (piv !== col) {
        for (let c = 0; c < S; c++) {
          const t = M[col * S + c]; M[col * S + c] = M[piv * S + c]; M[piv * S + c] = t;
        }
        const t = v[col]; v[col] = v[piv]; v[piv] = t;
      }
      const p = M[col * S + col] || 1e-300;
      for (let r = col + 1; r < S; r++) {
        const f = M[r * S + col] / p;
        for (let c = col; c < S; c++) M[r * S + c] -= f * M[col * S + c];
        v[r] -= f * v[col];
      }
    }
    const x = new Float64Array(S);
    for (let r = S - 1; r >= 0; r--) {
      let s = v[r];
      for (let c = r + 1; c < S; c++) s -= M[r * S + c] * x[c];
      x[r] = s / (M[r * S + r] || 1e-300);
    }
    return x;
  }

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  // Seeded PRNG (mulberry32) so "New scene" buttons are reproducible per seed.
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Run fn the first time el comes within 400px of the viewport, then again
  // whenever el changes size. Deferring the first run keeps off-screen demos
  // from rendering (and forcing layout) during page load.
  function onResize(el, fn) {
    let started = false;
    const start = () => {
      if (started) return;
      started = true;
      fn();
      if ("ResizeObserver" in window) {
        // ResizeObserver reports once on observe(); fn already ran, skip it.
        let initial = true;
        new ResizeObserver(() => { if (initial) { initial = false; return; } fn(); }).observe(el);
      } else window.addEventListener("resize", fn);
    };
    if (!("IntersectionObserver" in window)) { start(); return; }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io.disconnect(); start(); }
    }, { rootMargin: "400px 0px" });
    io.observe(el);
  }

  // Only animate while the element is on screen; saves CPU on long pages.
  function whenVisible(el, cb) {
    if (!("IntersectionObserver" in window)) { cb(true); return; }
    new IntersectionObserver((entries) => {
      for (const e of entries) cb(e.isIntersecting);
    }, { rootMargin: "100px" }).observe(el);
  }

  return {
    dirichlet, dirichletDeriv, gaussian, GAUSS_SIGMA, wrap,
    LUT, spectral, phaseColor, fitCanvas, token,
    csolve, rsolve, prefersReducedMotion, rng, onResize, whenVisible,
  };
})();
