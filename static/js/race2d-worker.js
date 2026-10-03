// Runs the 2D optimizer race off the main thread: one DSFW outer step
// takes 50-90 ms, which would stall drawing and dragging.
"use strict";
// The page passes each script's content version in the query string
// (see race2d.js) so the worker never imports a stale cached copy.
const q = new URLSearchParams(self.location.search);
const versioned = (name) => (q.get(name) ? `${name}.js?v=${q.get(name)}` : `${name}.js`);
importScripts(versioned("dsp"), versioned("fidelity"), versioned("race2d"));

const handle = Race2D.createEngine();
self.onmessage = (e) => {
  const out = handle(e.data);
  out.id = e.data.id;
  self.postMessage(out);
};
