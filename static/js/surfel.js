// Surfel appearance demo: a bistatic Tx/Rx pair looks at three planar
// surfels. Each surfel's complex amplitude is sqrt(A) * V * F * (R/R0)^-gamma
// times the round-trip phase, and its range-profile splat is a Dirichlet
// kernel centred at its round-trip range (paper Eq. 2-4).
"use strict";

(() => {
  const root = document.getElementById("demo-surfel");
  if (!root) return;
  const scene = root.querySelector('[data-panel="scene"]');
  const profile = root.querySelector('[data-panel="profile"]');
  const areaIn = root.querySelector('[name="area"]');
  const areaOut = root.querySelector('[data-out="area"]');
  const isoIn = root.querySelector('[name="isotropic"]');
  const selOut = root.querySelector('[data-out="selected"]');
  const bars = Object.fromEntries(
    [...root.querySelectorAll("[data-bar]")].map((el) => [el.dataset.bar, el])
  );
  const materialGroup = root.querySelector('[data-group="material"]');

  // Sensor constants, close to the paper's 120 GHz / 15 GHz bunny setup.
  const LAMBDA = 2.5;        // carrier wavelength, mm (120 GHz)
  const RANGE_BIN = 10;      // c / 2B for B = 15 GHz, mm per range bin
  const N = 64;              // range FFT length (bins)
  const R0 = 200;            // reference range for normalising path loss, mm
  const GAMMA = 2;           // path-loss exponent (two-way spreading)
  const BASE = 26;           // half Tx-Rx separation, mm (~20 deg at 150 mm)
  const VIEW = { xMin: -180, xMax: 180, yMin: -20, yMax: 340 }; // mm
  const A_MAX = 900;         // mm^2, top of the area slider

  // Relative wave impedance eta / eta0. Non-magnetic media: eta/eta0 = 1/n.
  const MATERIALS = {
    metal: { eta: 0, label: "Metal" },
    glass: { eta: 1 / 1.5, label: "Glass" },
    silicon: { eta: 1 / 3.4, label: "Silicon" },
  };
  const COLORS = ["#2b66a6", "#c2512f", "#4b8a52"];

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

  function sceneMap(w, h) {
    // Uniform scale: fit the mm view box into the canvas.
    const s = Math.min(w / (VIEW.xMax - VIEW.xMin), h / (VIEW.yMax - VIEW.yMin));
    const ox = w / 2 - ((VIEW.xMin + VIEW.xMax) / 2) * s;
    const oy = h / 2 + ((VIEW.yMin + VIEW.yMax) / 2) * s;
    return {
      s,
      X: (x) => ox + x * s, Y: (y) => oy - y * s,
      inv: (px, py) => ({ x: (px - ox) / s, y: (oy - py) / s }),
    };
  }
  const halfLen = (s) => Math.sqrt(s.area / Math.PI) * 1.6; // drawn 1.6x for legibility
  const handle = (s) => ({ x: s.x + Math.cos(s.theta) * 36, y: s.y + Math.sin(s.theta) * 36 });

  function drawScene(sp) {
    const { ctx, w, h } = DS.fitCanvas(scene);
    const m = sceneMap(w, h);
    const muted = DS.token("--muted"), rule = DS.token("--rule"), ink = DS.token("--ink");
    ctx.clearRect(0, 0, w, h);

    // Range rings every 50 mm from the array centre.
    ctx.strokeStyle = rule; ctx.lineWidth = 1;
    ctx.font = "11px 'Hanken Grotesk', system-ui, sans-serif";
    // Labels hug the centre line at 97 deg: always inside the view box
    // (rings reach 300 mm but the box is only +-180 mm wide).
    ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    const la = (97 * Math.PI) / 180;
    for (let r = 50; r <= 300; r += 50) {
      ctx.beginPath(); ctx.arc(m.X(0), m.Y(0), r * m.s, Math.PI, 2 * Math.PI); ctx.stroke();
      ctx.fillText(`${r} mm`, m.X(r * Math.cos(la)) - 2, m.Y(r * Math.sin(la)) - 1);
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
      ctx.strokeStyle = COLORS[i];
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
      ctx.strokeStyle = COLORS[idx]; ctx.lineWidth = 1.3; ctx.setLineDash([4, 3]);
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

  function setBar(key, value, max, text) {
    const el = bars[key];
    if (!el) return;
    el.querySelector(".bar-fill").style.width = `${Math.min(100, (100 * Math.abs(value)) / max).toFixed(1)}%`;
    el.querySelector(".bar-value").textContent = text;
  }

  function update() {
    const sp = surfels.map(splat);
    drawScene(sp);
    drawProfile(sp);
    const s = surfels[selected], p = sp[selected];
    selOut.textContent = `Surfel ${selected + 1}`;
    selOut.style.color = COLORS[selected];
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

  // Dragging: grab the normal handle to rotate, the disk to move.
  let drag = null;
  scene.addEventListener("pointerdown", (e) => {
    const rect = scene.getBoundingClientRect();
    const m = sceneMap(rect.width, rect.height);
    const q = m.inv(e.clientX - rect.left, e.clientY - rect.top);
    const tol = 12 / m.s;
    let hit = null;
    surfels.forEach((s, i) => {
      const hd = handle(s);
      if (Math.hypot(hd.x - q.x, hd.y - q.y) < tol) hit = { i, mode: "rotate" };
    });
    if (!hit) {
      surfels.forEach((s, i) => {
        if (Math.hypot(s.x - q.x, s.y - q.y) < Math.max(tol, halfLen(s))) hit = hit || { i, mode: "move" };
      });
    }
    if (!hit) return;
    selected = hit.i;
    drag = { ...hit, id: e.pointerId };
    scene.setPointerCapture(e.pointerId);
    update();
  });
  scene.addEventListener("pointermove", (e) => {
    if (!drag || drag.id !== e.pointerId) return;
    const rect = scene.getBoundingClientRect();
    const m = sceneMap(rect.width, rect.height);
    const q = m.inv(e.clientX - rect.left, e.clientY - rect.top);
    const s = surfels[drag.i];
    if (drag.mode === "rotate") s.theta = Math.atan2(q.y - s.y, q.x - s.x);
    else {
      s.x = Math.max(VIEW.xMin + 10, Math.min(VIEW.xMax - 10, q.x));
      // Keep the round-trip range inside the 100-300 mm profile window.
      s.y = Math.max(100, Math.min(290, q.y));
      const r = Math.hypot(s.x, s.y);
      if (r > 290) { s.x *= 290 / r; s.y *= 290 / r; }
    }
    update();
  });
  const end = () => { drag = null; };
  scene.addEventListener("pointerup", end);
  scene.addEventListener("pointercancel", end);

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
