# Dirichlet Splatting project page

Static project page for *Dirichlet Splatting: Differentiable Rendering for
Wave-Based Inverse Problems* (SIGGRAPH Asia 2026, ACM TOG 45(6)).
No build step: open `index.html` through any static server.

```sh
python -m http.server 8000   # then visit http://localhost:8000
```

The site is meant to live at <https://dirichlet.xingyuchen.me/>. On GitHub
Pages, push this folder as the repository root, enable Pages on the default
branch, and add a DNS `CNAME` record `dirichlet` -> `<user>.github.io`; the
`CNAME` file here tells Pages which domain to serve.

## Search engines and link previews

- `index.html` `<head>` carries the description, canonical URL, Open Graph /
  Twitter tags, Google Scholar `citation_*` tags, and a schema.org
  `ScholarlyArticle` JSON-LD block.
- `robots.txt` and `sitemap.xml` sit at the site root.
- `static/images/social-card.jpg` (1200 x 630) is the link-preview image.
- If the domain changes, replace `https://dirichlet.xingyuchen.me/` in
  `index.html`, `robots.txt`, `sitemap.xml`, and `CNAME`.

## Before publishing

- `index.html`: fill in the arXiv and code links (search for `TODO`), then
  remove `class="is-pending"` and `aria-disabled="true"` from those two pills.
- `static/pdf/Dirichlet_Splatting.pdf` is the camera-ready PDF (12 MB). Replace
  it, or point the Paper button at the ACM DL / arXiv PDF instead.

## Layout

| Path | Contents |
| --- | --- |
| `index.html` | Page content |
| `static/css/style.css` | Styles; colour tokens at the top |
| `static/js/dsp.js` | Dirichlet kernel, colormap, small complex solvers |
| `static/js/kernel.js` | Kernel explorer (Dirichlet vs. Gaussian) |
| `static/js/interference.js` | 2D coherent interference playground |
| `static/js/surfel.js` | Surfel appearance model (V, Fresnel, path loss) and range-angle map |
| `static/js/optim.js` | 1D AdamW vs. DSFW race |
| `static/js/race2d.js` | 2D race: shape templates, DSFW outer loop, AdamW, editable view, 1D/2D switch |
| `static/js/race2d-worker.js` | Runs the 2D race off the main thread (falls back to the main thread on `file://`) |
| `static/js/fidelity.js` | 1D/2D Dirichlet vs. Gaussian fits to a random scene |
| `static/js/page.js` | KaTeX, bunny slider, BibTeX copy |
| `static/images/` | Figures exported from `Dirichlet_Arxiv/figures`; teaser also as 1200/2000 px WebP |
| `static/vendor/` | Self-hosted web fonts (latin subsets) and KaTeX 0.16.11, so first paint never waits on a CDN |

Icons are [Lucide](https://lucide.dev) 1.49.0 (ISC license), inlined as an SVG sprite at the top of `index.html`.

Every interactive figure draws for the first time when it nears the viewport
and stops any running computation when scrolled away, so off-screen figures
cost nothing. The interactive figures are simplified 1D/2D browser re-implementations for
illustration; numbers quoted on the page come from the paper.
