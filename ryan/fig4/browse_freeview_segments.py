"""Web browser for picking a free-viewing eye-trace segment for fig4 panel A.

Enumerates every valid 5-s window in the panel-B session's `backimage.dset`
(stride 30 samples = 0.25 s; loose filter: all `dpi_valid`), and serves a
Flask page that scrubs through candidates. Each candidate overlays the
selected trace on **its own trial's** natural image. Copy the displayed
pin string back into the chat to bake the chosen segment into panel A.

Usage:
    uv run ryan/fig4/browse_freeview_segments.py [--port 8787] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import io
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, request, send_file

from _fig4_helpers import PANEL_B_SESSION


FS = 120.0
WINDOW_S = 5.0
WINDOW_N = int(round(WINDOW_S * FS))
STRIDE = 30   # 0.25 s


# ---------------------------------------------------------------------------
# Session-wide state, loaded once at startup.
# ---------------------------------------------------------------------------
class State:
    pass


S = State()


def _load_session():
    from DataYatesV1.utils.io import get_session
    from DataYatesV1.exp.general import get_trial_protocols
    from models.data.datasets import DictDataset

    print(f"Loading session {PANEL_B_SESSION}...")
    subject, date = PANEL_B_SESSION.split("_")
    sess = get_session(subject, date)
    if sess is None:
        raise RuntimeError(f"Could not load session {PANEL_B_SESSION}")
    exp = sess.exp
    protocols = get_trial_protocols(exp)

    sr = exp["S"]["screenRect"].astype(int)
    screen_shape = (int(sr[3] - sr[1]), int(sr[2] - sr[0]))
    ppd = float(exp["S"]["pixPerDeg"])

    dset_path = sess.sess_dir / "datasets" / "backimage.dset"
    print(f"Loading {dset_path}...")
    dset = DictDataset.load(str(dset_path))
    eyepos = np.asarray(dset["eyepos"])
    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel().astype(int)
    if "dpi_valid" in dset.covariates:
        valid = np.asarray(dset.covariates["dpi_valid"]).ravel().astype(bool)
    else:
        valid = np.ones(len(eyepos), dtype=bool)

    # Enumerate dense candidate windows.
    candidates = []   # list of (trial_idx, start_global, start_local)
    for t in np.unique(trial_inds):
        idxs = np.where(trial_inds == int(t))[0]
        if len(idxs) < WINDOW_N:
            continue
        # idxs must be contiguous (they are, by construction in the dset).
        for local_start in range(0, len(idxs) - WINDOW_N + 1, STRIDE):
            sl = slice(local_start, local_start + WINDOW_N)
            if not valid[idxs[sl]].all():
                continue
            candidates.append((int(t), int(idxs[local_start]), int(local_start)))
    print(f"Enumerated {len(candidates)} valid candidate windows "
          f"across {len(np.unique([c[0] for c in candidates]))} trials.")

    S.exp = exp
    S.protocols = protocols
    S.screen_shape = screen_shape
    S.ppd = ppd
    S.eyepos = eyepos
    S.trial_inds = trial_inds
    S.candidates = candidates


@lru_cache(maxsize=64)
def _render_trial_image(trial_idx: int):
    """Render the full-screen natural image for one BackImage trial."""
    from DataYatesV1.exp.backimage import BackImageTrial
    from PIL import Image as PILImage

    trial = BackImageTrial(S.exp["D"][trial_idx], S.exp["S"])
    img = trial.get_image()
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.uint8)

    H, W = S.screen_shape
    canvas = np.full((H, W), int(trial.bkgnd), dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in trial.dest_rect]
    h, w = y1 - y0, x1 - x0
    if img.shape[:2] != (h, w):
        img = np.array(PILImage.fromarray(img).resize((w, h), resample=2))
    y0c, y1c = max(0, y0), min(H, y1)
    x0c, x1c = max(0, x0), min(W, x1)
    sy0, sy1 = y0c - y0, h - (y1 - y1c)
    sx0, sx1 = x0c - x0, w - (x1 - x1c)
    canvas[y0c:y1c, x0c:x1c] = img[sy0:sy1, sx0:sx1]
    return canvas


def _render_candidate_png(cand_idx: int) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trial_idx, start_global, _ = S.candidates[cand_idx]
    image = _render_trial_image(trial_idx)
    H, W = S.screen_shape
    cx, cy = W / 2.0, H / 2.0

    ep = S.eyepos[start_global:start_global + WINDOW_N]   # (N, 2) deg
    trace_px = np.column_stack([
        ep[:, 0] * S.ppd + cx,
        -ep[:, 1] * S.ppd + cy,
    ])

    fig_w = 8.0
    fig_h = fig_w * H / W
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax.plot(trace_px[:, 0], trace_px[:, 1], color="#ffd84d",
            linewidth=1.4, alpha=0.95)
    ax.plot(trace_px[0, 0], trace_px[0, 1], "o", color="#1ec8c8",
            markersize=6, markeredgecolor="black", markeredgewidth=0.6)
    ax.plot(trace_px[-1, 0], trace_px[-1, 1], "s", color="#e64b4b",
            markersize=6, markeredgecolor="black", markeredgewidth=0.6)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def _candidate_summary(cand_idx: int):
    trial_idx, start_global, start_local = S.candidates[cand_idx]
    ep = S.eyepos[start_global:start_global + WINDOW_N]
    dxy = np.diff(ep, axis=0, prepend=ep[:1])
    speed = np.linalg.norm(dxy, axis=1) * FS
    fix_frac = float(np.mean(speed < 20.0))
    spread = float(ep.std(axis=0).mean())
    xy_range = float(np.max(np.abs(ep)))
    return {
        "cand_idx": cand_idx,
        "n_total": len(S.candidates),
        "trial_idx": trial_idx,
        "start_global": start_global,
        "start_local": start_local,
        "fix_frac": round(fix_frac, 3),
        "spread_deg": round(spread, 3),
        "max_abs_deg": round(xy_range, 2),
        "pin": f"pin: trial={trial_idx} start_global={start_global} "
               f"start_local={start_local}",
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>fig4 freeview picker</title>
<style>
  body { font-family: ui-monospace, monospace; background:#111; color:#ddd;
         margin:0; padding:12px; }
  #img { display:block; max-width: 100%; height:auto; background:#000;
         border:1px solid #333; }
  .row { display:flex; align-items:center; gap:12px; margin:8px 0;
         flex-wrap:wrap; }
  button { background:#222; color:#ddd; border:1px solid #555;
           padding:6px 12px; font: inherit; cursor:pointer; }
  button:hover { background:#333; }
  input[type=range] { flex:1; min-width:300px; }
  input[type=number] { width:80px; background:#222; color:#ddd;
                       border:1px solid #555; font:inherit; padding:4px; }
  #pin { background:#1a1a1a; padding:8px; border:1px solid #444;
         user-select:all; cursor:copy; }
  .meta span { margin-right: 14px; }
  .meta b { color:#ffd84d; }
</style></head><body>
<h2 style="margin:4px 0 8px">Free-viewing segment picker — fig4 panel A</h2>
<div class="row">
  <button id="prev">◀ prev</button>
  <input type="range" id="slider" min="0" max="0" value="0">
  <button id="next">next ▶</button>
  <span>idx <input type="number" id="goto" min="0" value="0"></span>
  <span id="counter">– / –</span>
</div>
<div class="meta row" id="meta"></div>
<div class="row"><div id="pin" title="click to select, copy with Cmd/Ctrl-C"></div>
  <button id="copy">copy pin</button></div>
<img id="img" src=""/>
<script>
const slider = document.getElementById('slider');
const img = document.getElementById('img');
const counter = document.getElementById('counter');
const meta = document.getElementById('meta');
const pin = document.getElementById('pin');
const goto_ = document.getElementById('goto');
let total = 0;

async function load(i) {
  const r = await fetch('/info/' + i);
  const d = await r.json();
  total = d.n_total;
  slider.max = total - 1;
  slider.value = d.cand_idx;
  goto_.max = total - 1;
  goto_.value = d.cand_idx;
  counter.textContent = (d.cand_idx + 1) + ' / ' + total;
  meta.innerHTML = `
    <span>trial <b>${d.trial_idx}</b></span>
    <span>start_global <b>${d.start_global}</b></span>
    <span>start_local <b>${d.start_local}</b> (${(d.start_local/120).toFixed(2)}s)</span>
    <span>fix_frac <b>${d.fix_frac}</b></span>
    <span>spread <b>${d.spread_deg}°</b></span>
    <span>max|xy| <b>${d.max_abs_deg}°</b></span>
  `;
  pin.textContent = d.pin;
  img.src = '/img/' + d.cand_idx + '?v=' + Date.now();
}

slider.addEventListener('input', e => load(parseInt(e.target.value)));
document.getElementById('prev').onclick = () =>
  load(Math.max(0, parseInt(slider.value) - 1));
document.getElementById('next').onclick = () =>
  load(Math.min(total - 1, parseInt(slider.value) + 1));
goto_.addEventListener('change', e => load(parseInt(e.target.value)));
document.getElementById('copy').onclick = () => {
  navigator.clipboard.writeText(pin.textContent);
};
window.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft')  load(Math.max(0, parseInt(slider.value)-1));
  if (e.key === 'ArrowRight') load(Math.min(total-1, parseInt(slider.value)+1));
  if (e.key === 'PageDown')   load(Math.min(total-1, parseInt(slider.value)+10));
  if (e.key === 'PageUp')     load(Math.max(0, parseInt(slider.value)-10));
});
load(0);
</script></body></html>
"""


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/info/<int:i>")
def info(i: int):
    i = max(0, min(len(S.candidates) - 1, i))
    return jsonify(_candidate_summary(i))


@app.get("/img/<int:i>")
def img(i: int):
    i = max(0, min(len(S.candidates) - 1, i))
    png = _render_candidate_png(i)
    return send_file(io.BytesIO(png), mimetype="image/png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args()
    _load_session()
    print(f"Serving on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
