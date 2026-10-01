// Page glue: math rendering, static charts, the bunny comparison slider,
// BibTeX copy, and placeholder links.
"use strict";

(() => {
  // KaTeX loads with `defer` before this file, so it is ready here.
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }

  // Horizontal bar charts. data-v is the value; log charts map
  // log10(v) from [data-min, data-max] onto the bar width.
  for (const list of document.querySelectorAll(".hbars")) {
    const isLog = list.classList.contains("hbars-log");
    const lo = +list.dataset.min || 0, hi = +list.dataset.max;
    for (const li of list.children) {
      const v = +li.dataset.v;
      const t = isLog ? (Math.log10(v) - lo) / (hi - lo) : v / hi;
      li.style.setProperty("--w", `${Math.max(0, Math.min(1, t)) * 100}%`);
      // Long bars carry their label inside, or it would overflow on phones.
      if (t > 0.7) li.classList.add("label-inside");
      const label = document.createElement("b");
      label.textContent = isLog ? formatSci(v) : String(v);
      li.appendChild(label);
    }
  }
  function formatSci(v) {
    const e = Math.floor(Math.log10(v));
    const m = v / 10 ** e;
    return `${m.toFixed(2)}×10${superscript(e)}`;
  }
  function superscript(n) {
    const map = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    return String(n).split("").map((c) => map[c]).join("");
  }

  // Bunny before/after slider with view tabs.
  const cmp = document.getElementById("bunny-compare");
  if (cmp) {
    const box = cmp.querySelector(".compare-box");
    const range = cmp.querySelector(".compare-range");
    const ours = cmp.querySelector(".compare-ours");
    const ref = cmp.querySelector(".compare-ref");
    range.addEventListener("input", () => box.style.setProperty("--pos", `${range.value}%`));
    const views = cmp.querySelector(".seg-views");
    views.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      for (const b of views.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
      ours.src = `static/images/bunny3d_view_ours_${btn.value}.png`;
      ref.src = `static/images/bunny3d_view_ref_${btn.value}.png`;
    });
  }

  // BibTeX copy.
  const copyBtn = document.querySelector(".bib-copy");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const text = document.getElementById("bibtex-text").textContent;
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = "Copied";
      } catch {
        // Clipboard API needs a secure context; fall back to selecting it.
        const range = document.createRange();
        range.selectNodeContents(document.getElementById("bibtex-text"));
        const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
        copyBtn.textContent = "Selected, press Ctrl+C";
      }
      setTimeout(() => { copyBtn.textContent = "Copy"; }, 1800);
    });
  }

  // Links marked as pending (arXiv, code) do nothing until filled in.
  for (const a of document.querySelectorAll('a[aria-disabled="true"]')) {
    a.addEventListener("click", (e) => e.preventDefault());
  }
})();
