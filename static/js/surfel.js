// Surfel appearance demo: a bistatic Tx/Rx pair looks at three planar
// surfels. Each surfel's complex amplitude is sqrt(A) * V * F * (R/R0)^-gamma
// times the round-trip phase, and its splat is a Dirichlet kernel centred at
// its round-trip range (range profile) and, with a receive array, at its
// angle too (range-angle map) (paper Eq. 2-5).
"use strict";

(() => {
  const root = document.getElementById("demo-surfel");
  if (!root) return;
  const scene = root.querySelector('[data-panel="scene"]');
  const profile = root.querySelector('[data-panel="profile"]');
  const raCanvas = root.querySelector('[data-panel="ra"]');
  const areaIn = root.querySelector('[name="area"]');
  const areaOut = root.querySelector('[data-out="area"]');
  const isoIn = root.querySelector('[name="isotropic"]');
  const bars = Object.fromEntries(
    [...root.querySelectorAll("[data-bar]")].map((el) => [el.dataset.bar, el])
  );
  const materialGroup = root.querySelector('[data-group="material"]');

  // Sensor constants, close to the paper's 120 GHz / 15 GHz bunny setup.
  const LAMBDA = 2.5;        // carrier wavelength, mm (120 GHz)
  const RANGE_BIN = 10;      // c / 2B for B = 15 GHz, mm per range bin
  const N = 64;              // range FFT length (bins)
  // Receive array for the range-angle map: NA elements at lambda/2 along x,
  // centred on Rx. 16 elements give about 7 degrees of resolution at
  // broadside, typical of a compact radar; the 19 mm aperture puts the
  // scene in the near field, which the exact per-element paths reproduce.
  const NA = 16;
  const EL_PITCH = LAMBDA / 2;
  const R0 = 200;            // reference range for normalising path loss, mm
  const GAMMA = 2;           // path-loss exponent (two-way spreading)
  const BASE = 26;           // half Tx-Rx separation, mm (~20 deg at 150 mm)
  const VIEW = { xMin: -180, xMax: 180, yMin: -32, yMax: 335 }; // mm; yMin leaves room for the Tx/Rx labels
  const A_MAX = 900;         // mm^2, top of the area slider

  // Relative wave impedance eta / eta0. Non-magnetic media: eta/eta0 = 1/n.
  const MATERIALS = {
    metal: { eta: 0, label: "Metal", color: "#4f5d73" },      // steel grey
    glass: { eta: 1 / 1.5, label: "Glass", color: "#2b9fc4" },  // cyan
    silicon: { eta: 1 / 3.4, label: "Silicon", color: "#7b4fa0" }, // violet
  };
  // Surfels are coloured by material (MATERIALS[mat].color), everywhere.
  const colorOf = (s) => MATERIALS[s.mat].color;

  // Surfel 1 sits alone in range; surfels 2 and 3 are about one range
  // cell apart, so their splats overlap and interfere.
  const initial = () => [
    { x: -70, y: 140, theta: -Math.PI / 2 + 0.5, area: 400, mat: "metal" },
    { x: 10, y: 215, theta: -Math.PI / 2, area: 400, mat: "metal" },
    { x: 75, y: 218, theta: -Math.PI / 2 - 0.35, area: 400, mat: "glass" },
  ];
  let surfels = initial();
  let selected = 1;
  const tx = { x: -BASE, y: 0 }, rx = { x: BASE, y: 0 };
  const tmp = [0, 0];

  // TE Fresnel reflection coefficient (paper Eq. 4) with eta0 = 1.
  function fresnel(cosI, eta) {
    if (eta === 0) return -1;
    const sinI = Math.sqrt(Math.max(0, 1 - cosI * cosI));
    const sinT = sinI * eta;
    const cosT = Math.sqrt(Math.max(0, 1 - sinT * sinT));
    return (eta * cosI - cosT) / (eta * cosI + cosT);
  }

  function splat(s) {
    const nx = Math.cos(s.theta), ny = Math.sin(s.theta);
    const dtx = { x: tx.x - s.x, y: tx.y - s.y }, drx = { x: rx.x - s.x, y: rx.y - s.y };
    const Rtx = Math.hypot(dtx.x, dtx.y), Rrx = Math.hypot(drx.x, drx.y);
    const cosTx = (dtx.x * nx + dtx.y * ny) / Rtx;
    const cosRx = (drx.x * nx + drx.y * ny) / Rrx;
    const iso = isoIn.checked;
    const V = iso ? 1 : Math.max(cosTx, 0) * Math.max(cosRx, 0);
    const F = iso ? 1 : fresnel(Math.max(cosTx, 0), MATERIALS[s.mat].eta);
    const R = 0.5 * (Rtx + Rrx);
    const sqrtA = Math.sqrt(s.area / A_MAX);
    const loss = Math.pow(R / R0, -GAMMA);
    const amp = sqrtA * V * F * loss;
    const phase = (-2 * Math.PI * (Rtx + Rrx)) / LAMBDA;
    return {
      sqrtA, V, F, loss, amp, R,
      re: amp * Math.cos(phase), im: amp * Math.sin(phase),
      mu: R / RANGE_BIN,
    };
  }

  // Uniform scale: fit a mm view box into the canvas, centred.
  function fitView(w, h, view) {
    const s = Math.min(w / (view.xMax - view.xMin), h / (view.yMax - view.yMin));
    const ox = w / 2 - ((view.xMin + view.xMax) / 2) * s;
    const oy = h / 2 + ((view.yMin + view.yMax) / 2) * s;
    return {
      s,
      X: (x) => ox + x * s, Y: (y) => oy - y * s,
      inv: (px, py) => ({ x: (px - ox) / s, y: (oy - py) / s }),
    };
  }
  const sceneMap = (w, h) => fitView(w, h, VIEW);
  const halfLen = (s) => Math.sqrt(s.area / Math.PI) * 1.6; // drawn 1.6x for legibility
  const handle = (s) => ({ x: s.x + Math.cos(s.theta) * 36, y: s.y + Math.sin(s.theta) * 36 });

  function drawScene(sp) {
    const { ctx, w, h } = DS.fitCanvas(scene);
    const m = sceneMap(w, h);
    const rule = DS.token("--rule"), ink = DS.token("--ink");
    ctx.clearRect(0, 0, w, h);

    // Range rings every 50 mm, unlabelled: labels sat on top of the surfels,
    // and the range-angle map beside this view carries the range scale.
    ctx.strokeStyle = rule; ctx.lineWidth = 1;
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";
    for (let r = 50; r <= 300; r += 50) {
      ctx.beginPath(); ctx.arc(m.X(0), m.Y(0), r * m.s, Math.PI, 2 * Math.PI); ctx.stroke();
    }

    // Rays Tx -> surfel -> Rx, opacity follows the visibility factor V.
    surfels.forEach((s, i) => {
      const a = isoIn.checked ? 0.35 : 0.12 + 0.6 * sp[i].V;
      ctx.strokeStyle = `rgba(23,32,44,${a.toFixed(3)})`;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(m.X(tx.x), m.Y(tx.y)); ctx.lineTo(m.X(s.x), m.Y(s.y)); ctx.lineTo(m.X(rx.x), m.Y(rx.y));
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Antennas.
    for (const [p, label] of [[tx, "Tx"], [rx, "Rx"]]) {
      ctx.fillStyle = ink;
      ctx.fillRect(m.X(p.x) - 5, m.Y(p.y) - 5, 10, 10);
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(label, m.X(p.x), m.Y(p.y) + 8);
    }

    // Surfels as oriented disks seen edge-on, with a normal handle.
    surfels.forEach((s, i) => {
      const L = halfLen(s);
      const tx_ = -Math.sin(s.theta), ty_ = Math.cos(s.theta);
      ctx.strokeStyle = colorOf(s);
      ctx.lineCap = "round";
      ctx.lineWidth = i === selected ? 6 : 4.5;
      ctx.beginPath();
      ctx.moveTo(m.X(s.x - tx_ * L), m.Y(s.y - ty_ * L));
      ctx.lineTo(m.X(s.x + tx_ * L), m.Y(s.y + ty_ * L));
      ctx.stroke();
      const hd = handle(s);
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(m.X(s.x), m.Y(s.y)); ctx.lineTo(m.X(hd.x), m.Y(hd.y)); ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.beginPath(); ctx.arc(m.X(hd.x), m.Y(hd.y), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.lineCap = "butt";
    });
  }

  function drawProfile(sp) {
    const { ctx, w, h } = DS.fitCanvas(profile);
    const L = 40, R = 12, T = 12, B = 36;
    const pw = w - L - R, ph = h - T - B;
    // Show 100-300 mm: wide enough for the draggable area, narrow enough
    // that a 10 mm range cell and its sidelobes are visible.
    const binMin = 10, binMax = 30;
    const xOf = (k) => L + ((k - binMin) / (binMax - binMin)) * pw;
    const muted = DS.token("--muted"), rule = DS.token("--rule"), ink = DS.token("--ink");
    ctx.clearRect(0, 0, w, h);

    // Fixed y scale: a metal surfel at R0 facing the sensor head-on has
    // |amp| = sqrt(400/900) ~ 0.67 for a head-on metal surfel at R0; 1.5
    // leaves room for two splats adding constructively. Values above clip.
    const yMax = 1.5;
    const yOf = (v) => T + (1 - Math.min(v, yMax) / yMax) * ph;
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";
    ctx.strokeStyle = rule; ctx.lineWidth = 1;
    ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of [0, 0.5, 1, 1.5]) {
      const y = Math.round(yOf(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      ctx.fillText(v.toFixed(1), L - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let mm = 100; mm <= 300; mm += 25) ctx.fillText(`${mm}`, xOf(mm / RANGE_BIN), T + ph + 6);
    ctx.fillText("range (mm)", L + pw / 2, T + ph + 21);

    const sample = (k, only) => {
      let r = 0, i = 0;
      sp.forEach((p, idx) => {
        if (only !== undefined && only !== idx) return;
        DS.dirichlet(k - p.mu, N, tmp);
        r += p.re * tmp[0] - p.im * tmp[1];
        i += p.re * tmp[1] + p.im * tmp[0];
      });
      return Math.hypot(r, i);
    };

    // Per-surfel splats (dashed) and the coherent sum (filled).
    sp.forEach((_, idx) => {
      ctx.strokeStyle = colorOf(surfels[idx]); ctx.lineWidth = 1.3; ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (let px = 0; px <= pw; px++) {
        const k = binMin + (px / pw) * (binMax - binMin);
        const y = yOf(sample(k, idx));
        if (px === 0) ctx.moveTo(L + px, y); else ctx.lineTo(L + px, y);
      }
      ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let px = 0; px <= pw; px++) {
      const k = binMin + (px / pw) * (binMax - binMin);
      const y = yOf(sample(k));
      if (px === 0) ctx.moveTo(L + px, y); else ctx.lineTo(L + px, y);
    }
    ctx.strokeStyle = ink; ctx.lineWidth = 1.8; ctx.stroke();
    ctx.lineTo(L + pw, yOf(0)); ctx.lineTo(L, yOf(0)); ctx.closePath();
    ctx.fillStyle = "rgba(23,32,44,0.07)"; ctx.fill();
  }

  // Range-angle map, computed the way a radar does it rather than from the
  // separable far-field model: (1) each array element's range profile from
  // its exact bistatic path length Tx -> surfel -> element (so near-field
  // curvature and per-element phase are included); (2) an angle FFT across
  // elements (evaluated as a DTFT on a fine sin(theta) grid, i.e. a
  // zero-padded FFT); (3) the polar (range, angle) result drawn in Cartesian
  // space, the usual fan-shaped radar view, aligned with the scene.
  const POLAR = { rMin: 80, rMax: 300, dr: 1, thMax: 60, dth: 0.5 };  // mm, degrees
  const NR = Math.round((POLAR.rMax - POLAR.rMin) / POLAR.dr) + 1;
  // The fan's own view box: just wide enough for +-thMax at rMax, plus room
  // for the Tx/Rx labels below the array. Its aspect matches .plot-fan (5:3).
  const FAN_HALF = POLAR.rMax * Math.sin((POLAR.thMax * Math.PI) / 180) + 20;
  const FAN_VIEW = { xMin: rx.x - FAN_HALF, xMax: rx.x + FAN_HALF, yMin: -26, yMax: POLAR.rMax + 10 };
  const NT = Math.round((2 * POLAR.thMax) / POLAR.dth) + 1;
  const elemX = Array.from({ length: NA }, (_, m) => rx.x + (m - (NA - 1) / 2) * EL_PITCH);
  // Steering phases e^{-j 2 pi (x_m - x_c) sin(theta) / lambda}, one row per angle.
  const steer = new Float64Array(NT * NA * 2);
  for (let a = 0; a < NT; a++) {
    const u = Math.sin(((-POLAR.thMax + a * POLAR.dth) * Math.PI) / 180);
    for (let m = 0; m < NA; m++) {
      const ph = (-2 * Math.PI * (elemX[m] - rx.x) * u) / LAMBDA;
      steer[(a * NA + m) * 2] = Math.cos(ph); steer[(a * NA + m) * 2 + 1] = Math.sin(ph);
    }
  }
  // Colour reference: the array gain NA times a head-on metal surfel's
  // |amp| (~0.67) is ~11; 12 leaves a little headroom.
  const RA_REF = NA * 0.75;
  let raScale = "lin";
  let raImg = null;

  function rangeAngle(sp) {
    // (1) Per-element range profiles Y[m][r].
    const Y = new Float64Array(NA * NR * 2);
    surfels.forEach((s, i) => {
      const amp = sp[i].amp;
      if (amp === 0) return;
      const dTx = Math.hypot(s.x - tx.x, s.y - tx.y);
      for (let m = 0; m < NA; m++) {
        const L = dTx + Math.hypot(s.x - elemX[m], s.y);         // round trip, mm
        const ph = (-2 * Math.PI * L) / LAMBDA;
        const ar = amp * Math.cos(ph), ai = amp * Math.sin(ph);
        const mu = L / 2 / RANGE_BIN;                              // bistatic range bin
        for (let r = 0; r < NR; r++) {
          DS.dirichlet((POLAR.rMin + r * POLAR.dr) / RANGE_BIN - mu, N, tmp);
          const o = (m * NR + r) * 2;
          Y[o] += ar * tmp[0] - ai * tmp[1]; Y[o + 1] += ar * tmp[1] + ai * tmp[0];
        }
      }
    });
    // (2) Angle FFT per range sample.
    const P = new Float32Array(NR * NT);
    for (let r = 0; r < NR; r++) {
      for (let a = 0; a < NT; a++) {
        let sr = 0, si = 0;
        for (let m = 0; m < NA; m++) {
          const yr = Y[(m * NR + r) * 2], yi = Y[(m * NR + r) * 2 + 1];
          const cr = steer[(a * NA + m) * 2], ci = steer[(a * NA + m) * 2 + 1];
          sr += yr * cr - yi * ci; si += yr * ci + yi * cr;
        }
        P[r * NT + a] = Math.hypot(sr, si) / RA_REF;
      }
    }
    return P;
  }

  function drawRangeAngle(sp) {
    const { ctx, w, h } = DS.fitCanvas(raCanvas);
    const m = fitView(w, h, FAN_VIEW);
    const P = rangeAngle(sp);
    // (3) Fan view: look up every pixel's (range, angle) about the array centre.
    const W = Math.round(w), H = Math.round(h);
    if (!raImg || raImg.width !== W || raImg.height !== H) raImg = new ImageData(W, H);
    const d = raImg.data;
    d.fill(0);
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const q = m.inv(px + 0.5, py + 0.5);
        const dx = q.x - rx.x, dy = q.y - rx.y;
        const r = Math.hypot(dx, dy);
        if (r < POLAR.rMin || r > POLAR.rMax || dy <= 0) continue;
        const th = (Math.atan2(dx, dy) * 180) / Math.PI;
        if (Math.abs(th) > POLAR.thMax) continue;
        const ri = Math.round((r - POLAR.rMin) / POLAR.dr), ai = Math.round((th + POLAR.thMax) / POLAR.dth);
        let t = P[ri * NT + ai];
        if (raScale === "db") t = 1 + (20 * Math.log10(t + 1e-12)) / 40;
        const li = Math.max(0, Math.min(255, Math.round(t * 255))) * 3, o = (py * W + px) * 4;
        d[o] = DS.LUT[li]; d[o + 1] = DS.LUT[li + 1]; d[o + 2] = DS.LUT[li + 2]; d[o + 3] = 255;
      }
    }
    ctx.clearRect(0, 0, w, h);
    // putImageData ignores the context transform, so draw via a bitmap at CSS scale.
    const off = drawRangeAngle.off || (drawRangeAngle.off = document.createElement("canvas"));
    off.width = W; off.height = H;
    off.getContext("2d").putImageData(raImg, 0, 0);
    ctx.drawImage(off, 0, 0, w, h);

    // Range arcs and angle spokes about the array, labelled like a radar display.
    const cx = m.X(rx.x), cy = m.Y(rx.y);
    ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 1;
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";
    ctx.fillStyle = DS.token("--muted");
    const a0 = -Math.PI / 2 - (POLAR.thMax * Math.PI) / 180, a1 = -Math.PI / 2 + (POLAR.thMax * Math.PI) / 180;
    for (let r = 100; r <= 300; r += 100) {
      ctx.beginPath(); ctx.arc(cx, cy, r * m.s, a0, a1); ctx.stroke();
    }
    ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillStyle = "rgba(255,255,255,0.85)";
    for (const r of [100, 200]) ctx.fillText(`${r} mm`, cx + 3, cy - r * m.s - 2);
    for (const deg of [-60, -30, 0, 30, 60]) {
      const t = (deg * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(cx + POLAR.rMin * m.s * Math.sin(t), cy - POLAR.rMin * m.s * Math.cos(t));
      ctx.lineTo(cx + POLAR.rMax * m.s * Math.sin(t), cy - POLAR.rMax * m.s * Math.cos(t));
      ctx.stroke();
      // Label just inside the rim, rotated 8% toward broadside so the edge
      // labels clear the fan border.
      const tl = t * 0.92, rl = POLAR.rMax - 16;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fillText(`${deg}°`, cx + rl * m.s * Math.sin(tl), cy - rl * m.s * Math.cos(tl));
    }
    // Array, Tx, and the true surfel positions for reference.
    ctx.fillStyle = DS.token("--ink");
    ctx.textBaseline = "top";
    ctx.fillRect(m.X(elemX[0]), cy - 2, (elemX[NA - 1] - elemX[0]) * m.s, 4);
    ctx.fillRect(m.X(tx.x) - 4, m.Y(tx.y) - 4, 8, 8);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("Tx", m.X(tx.x), m.Y(tx.y) + 7);
    ctx.fillText("Rx array", cx, cy + 7);
    surfels.forEach((s, i) => {
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.beginPath(); ctx.arc(m.X(s.x), m.Y(s.y), 7, 0, 2 * Math.PI); ctx.stroke();
      ctx.lineWidth = 1.6; ctx.strokeStyle = colorOf(s);
      ctx.beginPath(); ctx.arc(m.X(s.x), m.Y(s.y), 7, 0, 2 * Math.PI); ctx.stroke();
    });
  }

  function setBar(key, value, max, text) {
    const el = bars[key];
    if (!el) return;
    el.querySelector(".bar-fill").style.width = `${Math.min(100, (100 * Math.abs(value)) / max).toFixed(1)}%`;
    el.querySelector(".bar-value").textContent = text;
  }

  function update() {
    const sp = surfels.map(splat);
    drawScene(sp);
    drawRangeAngle(sp);
    drawProfile(sp);
    const s = surfels[selected], p = sp[selected];
    // No "Surfel n" label: the bars take the selected surfel's colour instead.
    root.style.setProperty("--sel", colorOf(surfels[selected]));
    areaIn.value = s.area;
    areaOut.textContent = `${s.area} mm²`;
    for (const b of materialGroup.querySelectorAll("button")) {
      b.setAttribute("aria-pressed", String(b.value === s.mat));
      b.disabled = isoIn.checked;
    }
    setBar("area", p.sqrtA, 1, p.sqrtA.toFixed(2));
    setBar("V", p.V, 1, p.V.toFixed(2));
    setBar("F", p.F, 1, p.F.toFixed(2));
    setBar("loss", p.loss, 2, p.loss.toFixed(2));
    setBar("amp", p.amp, 1, Math.abs(p.amp).toFixed(3));
  }

  // Shared gestures (DS.pointEditor): drag a surfel to move it or its white
  // handle to turn it, click empty space to add one, right-click or
  // long-press to delete. 1 to MAX_SURFELS surfels.
  const MAX_SURFELS = 6;
  const toScene = (x, y) => {
    const r = scene.getBoundingClientRect();
    const m = sceneMap(r.width, r.height);
    return { q: m.inv(x, y), tol: 12 / m.s };
  };
  // Keep the round-trip range inside the 100-300 mm profile window.
  function place(s, q) {
    s.x = Math.max(VIEW.xMin + 10, Math.min(VIEW.xMax - 10, q.x));
    s.y = Math.max(100, Math.min(290, q.y));
    const r = Math.hypot(s.x, s.y);
    if (r > 290) { s.x *= 290 / r; s.y *= 290 / r; }
  }
  DS.pointEditor(scene, {
    hit: (x, y) => {
      const { q, tol } = toScene(x, y);
      let tok = null;
      surfels.forEach((s, i) => {
        const hd = handle(s);
        if (Math.hypot(hd.x - q.x, hd.y - q.y) < tol) tok = { i, mode: "rotate" };
      });
      if (!tok) {
        surfels.forEach((s, i) => {
          if (!tok && Math.hypot(s.x - q.x, s.y - q.y) < Math.max(tol, halfLen(s))) tok = { i, mode: "move" };
        });
      }
      return tok;
    },
    add: (x, y) => {
      if (surfels.length >= MAX_SURFELS) return null;
      const { q } = toScene(x, y);
      const s = { x: 0, y: 0, theta: 0, area: 400, mat: "metal" };
      place(s, q);
      s.theta = Math.atan2(-s.y, -s.x);   // face the array
      surfels.push(s);
      selected = surfels.length - 1;
      update();
      return { i: selected, mode: "move" };
    },
    move: (tok, x, y) => {
      const { q } = toScene(x, y);
      const s = surfels[tok.i];
      selected = tok.i;
      if (tok.mode === "rotate") s.theta = Math.atan2(q.y - s.y, q.x - s.x);
      else place(s, q);
      update();
    },
    remove: (tok) => {
      if (surfels.length <= 1) return;
      surfels.splice(tok.i, 1);
      selected = Math.min(selected, surfels.length - 1);
      update();
    },
  });

  // Keyboard: arrows rotate the selected surfel, Tab-like cycling with S.
  scene.addEventListener("keydown", (e) => {
    const s = surfels[selected];
    if (e.key === "ArrowLeft") s.theta += 0.05;
    else if (e.key === "ArrowRight") s.theta -= 0.05;
    else if (e.key === "s" || e.key === "S") selected = (selected + 1) % surfels.length;
    else return;
    e.preventDefault();
    update();
  });

  areaIn.addEventListener("input", () => { surfels[selected].area = +areaIn.value; update(); });
  isoIn.addEventListener("change", update);
  const raGroup = root.querySelector('[data-group="rascale"]');
  raGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    for (const b of raGroup.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
    raScale = btn.value;
    update();
  });
  materialGroup.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    surfels[selected].mat = btn.value;
    update();
  });
  root.querySelector('[data-action="reset"]').addEventListener("click", () => {
    surfels = initial(); selected = 1; isoIn.checked = false; update();
  });
  DS.onResize(root, update);
})();
