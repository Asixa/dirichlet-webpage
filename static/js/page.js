// Page glue: math rendering, the bunny comparison slider,
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
