// Runs the 2D optimizer race off the main thread: one DSFW outer step
// takes 50-90 ms, which would stall drawing and dragging.
"use strict";
importScripts("dsp.js", "fidelity.js", "race2d.js");

const handle = Race2D.createEngine();
self.onmessage = (e) => {
  const out = handle(e.data);
  out.id = e.data.id;
  self.postMessage(out);
};
