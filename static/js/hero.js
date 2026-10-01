// PSF strip under the teaser: |d_N| of one reflector placed at the pointer,
// drawn in dB with the integer-bin samples a sensor would record. It draws in
// once when first scrolled into view; afterwards it only changes when the
// pointer moves over the teaser.
"use strict";

(() => {
  const canvas = document.getElementById("hero-trace");
  if (!canvas) return;

  const FLOOR_DB = -42;          // bottom of the trace; below this is clipped
  const BIN_PX = 26;             // target on-screen spacing of one DFT bin
  let mu = null;                 // reflector position in bins; null = centre
  let target = null;
  let reveal = DS.prefersReducedMotion() ? 1 : 0;
  const tmp = [0, 0];

  function draw() {
    const { ctx, w, h } = DS.fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const N = Math.max(8, Math.round(w / BIN_PX));
    const scale = w / N;
    const m = mu === null ? N / 2 + 0.35 : mu;
    const top = 6, base = h - 6;
    const yOf = (v) => {
      const db = Math.max(FLOOR_DB, 20 * Math.log10(Math.abs(v) + 1e-12));
      return base - ((db - FLOOR_DB) / -FLOOR_DB) * (base - top);
    };
    const xMax = w * reveal;

    // Integer-bin samples: what the sensor actually records.
    ctx.lineWidth = 1;
    for (let k = 0; k < N; k++) {
      const x = (k + 0.5) * scale;
      if (x > xMax) break;
      DS.dirichlet(k + 0.5 - m, N, tmp);
      const mag = Math.hypot(tmp[0], tmp[1]);
      const y = yOf(mag);
      ctx.strokeStyle = "rgba(43,102,166,0.28)";
      ctx.beginPath(); ctx.moveTo(x, base); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillStyle = DS.phaseColor(tmp[0], tmp[1], 50);
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 2 * Math.PI); ctx.fill();
    }

    // Continuous kernel.
    ctx.strokeStyle = DS.token("--dir") || "#2b66a6";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let px = 0; px <= xMax; px += 1) {
      DS.dirichlet(px / scale - m, N, tmp);
      const y = yOf(Math.hypot(tmp[0], tmp[1]));
      if (px === 0) ctx.moveTo(px, y); else ctx.lineTo(px, y);
    }
    ctx.stroke();
  }

  let scheduled = false;
  function schedule() {
    if (!scheduled) { scheduled = true; requestAnimationFrame(frame); }
  }

  function frame() {
    scheduled = false;
    let again = false;
    if (reveal < 1) { reveal = Math.min(1, reveal + 0.022); again = true; }
    if (target !== null) {
      if (mu === null) mu = target;
      const d = target - mu;
      if (Math.abs(d) > 0.002) { mu += d * 0.18; again = true; } else mu = target;
    }
    draw();
    if (again) schedule();
  }

  const area = canvas.closest(".teaser") || canvas;
  area.addEventListener("pointermove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const N = Math.max(8, Math.round(rect.width / BIN_PX));
    target = ((e.clientX - rect.left) / rect.width) * N;
    schedule();
  });

  // First call (when the strip nears the viewport) starts the draw-in.
  let started = false;
  DS.onResize(canvas, () => { if (started) draw(); else { started = true; schedule(); } });
})();
