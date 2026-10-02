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

  // Segmented controls get a sliding pill under the pressed button. Every
  // demo already toggles aria-pressed, so a MutationObserver keeps the pill
  // in sync without touching demo code; a ResizeObserver re-places it when
  // a hidden control appears or fonts change button widths.
  for (const seg of document.querySelectorAll(".seg")) {
    const pill = document.createElement("span");
    pill.className = "seg-pill";
    pill.setAttribute("aria-hidden", "true");
    seg.prepend(pill);
    seg.classList.add("has-pill");
    const place = () => {
      const btn = seg.querySelector('button[aria-pressed="true"]:not([hidden])');
      if (!btn || !btn.offsetWidth) { pill.style.opacity = "0"; return; }
      pill.style.opacity = "1";
      pill.style.width = `${btn.offsetWidth}px`;
      pill.style.height = `${btn.offsetHeight}px`;
      pill.style.transform = `translate(${btn.offsetLeft}px, ${btn.offsetTop}px)`;
    };
    place();
    // Enable the slide only after the first placement, so pages load still.
    requestAnimationFrame(() => seg.classList.add("pill-ready"));
    new MutationObserver(place).observe(seg, { subtree: true, attributes: true, attributeFilter: ["aria-pressed", "hidden"] });
    if ("ResizeObserver" in window) new ResizeObserver(place).observe(seg);
  }

  // Views and tool groups that are switched in (1D / 2D) fade in.
  for (const el of document.querySelectorAll("[data-view], [data-only]")) {
    new MutationObserver(() => {
      if (el.hidden) return;
      el.classList.remove("view-enter");
      void el.offsetWidth;   // restart the animation
      el.classList.add("view-enter");
    }).observe(el, { attributes: true, attributeFilter: ["hidden"] });
    el.addEventListener("animationend", (e) => { if (e.target === el) el.classList.remove("view-enter"); });
  }

  // Buttons that pulse for attention stop once they have been pressed.
  for (const btn of document.querySelectorAll(".icon-btn.is-attention")) {
    btn.addEventListener("click", () => btn.classList.remove("is-attention"), { once: true });
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

  // Analytics events (GA4). PDF downloads and outbound links are recorded by
  // GA4 enhanced measurement; these add what it cannot see. No-op when the
  // tag is blocked or absent.
  const track = (name, params) => { if (typeof window.gtag === "function") window.gtag("event", name, params); };
  // First use of each interactive figure, once per page view.
  for (const demo of document.querySelectorAll(".demo[id]")) {
    const first = () => {
      track("demo_interact", { demo: demo.id.replace(/^demo-/, "") });
      demo.removeEventListener("pointerdown", first, true);
      demo.removeEventListener("input", first, true);
    };
    demo.addEventListener("pointerdown", first, true);
    demo.addEventListener("input", first, true);
  }

  // BibTeX copy.
  const copyBtn = document.querySelector(".bib-copy");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      track("copy_bibtex");
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
