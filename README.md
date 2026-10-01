# Dirichlet Splatting project page

Static project page for *Dirichlet Splatting: Differentiable Rendering for
Wave-Based Inverse Problems* (SIGGRAPH Asia 2026, ACM TOG 45(6)).
No build step: open `index.html` through any static server.

```sh
python -m http.server 8000   # then visit http://localhost:8000
```

To deploy on GitHub Pages, push this folder as the repository root and enable
Pages on the default branch.

## Before publishing

- `index.html`: fill in the arXiv and code links (search for `TODO`), then
  remove `class="is-pending"` and `aria-disabled="true"` from those two pills.
- Author names link to ORCID; swap in homepages if you prefer.
- `static/pdf/Dirichlet_Splatting.pdf` is the camera-ready PDF (12 MB). Replace
  it, or point the Paper button at the ACM DL / arXiv PDF instead.

## Layout

| Path | Contents |
| --- | --- |
| `index.html` | Page content |
| `static/css/style.css` | Styles; colour tokens at the top |
| `static/js/dsp.js` | Dirichlet kernel, colormap, small complex solvers |
| `static/js/hero.js` | Header PSF trace |
| `static/js/kernel.js` | Kernel explorer (Dirichlet vs. Gaussian, rect/Hann) |
| `static/js/interference.js` | 2D coherent interference playground |
| `static/js/surfel.js` | Surfel appearance model (V, Fresnel, path loss) |
| `static/js/optim.js` | Loss landscape + AdamW vs. DSFW race |
| `static/js/fidelity.js` | 1D/2D Dirichlet vs. Gaussian fits to a random scene |
| `static/js/page.js` | KaTeX, bar charts, bunny slider, BibTeX copy |
| `static/images/` | Figures exported from `Dirichlet_Arxiv/figures` |

The interactive figures are simplified 1D/2D browser re-implementations for
illustration; numbers quoted on the page come from the paper.
