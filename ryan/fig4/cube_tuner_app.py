"""Interactive Flask app to tune the lag-cube pose in panel A.

Sliders adjust the cube's world rotation (yaw / pitch / roll about world
Y / X / Z, composed as R_y @ R_x @ R_z) and its size (W, H, D). The page
re-renders the stimulus half of panel A on every change. Current slider
values are echoed verbatim at the bottom of the page so they can be copied
back into _fig4a_stimulus.py module constants.

Usage:
    uv run python ryan/fig4/cube_tuner_app.py
    # then open http://127.0.0.1:5050/
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, send_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _fig4_data import configure_matplotlib                       # noqa: E402
from _fig4a_data import load_panel_a_assets                       # noqa: E402
import _fig4a_stimulus as stim                                    # noqa: E402


configure_matplotlib()
ASSETS = load_panel_a_assets()


SLIDERS = [
    # (key,           min,    max,   step,  default-from-module)
    ("yaw_deg",       -90.0,   90.0,  0.5,  stim.CUBE_YAW_DEG),
    ("pitch_deg",     -60.0,   60.0,  0.5,  stim.CUBE_PITCH_DEG),
    ("roll_deg",      -60.0,   60.0,  0.5,  stim.CUBE_ROLL_DEG),
    ("cube_w",         0.50,   5.00,  0.05, stim.CUBE_W),
    ("cube_h",         0.50,   5.00,  0.05, stim.CUBE_H),
    ("cube_d",         0.50,   5.00,  0.05, stim.CUBE_D),
]


HTML = """<!doctype html>
<html><head><title>Lag-cube tuner</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         margin: 16px; max-width: 1100px; }
  h2 { margin: 0 0 8px 0; }
  .row { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
  .row label { width: 110px; font-family: ui-monospace, monospace; font-size: 13px; }
  .row input[type=range] { flex: 1; max-width: 420px; }
  .row .val { width: 70px; text-align: right; font-family: ui-monospace, monospace;
              font-size: 13px; }
  .row .reset { font-size: 11px; padding: 2px 6px; cursor: pointer; }
  img { max-width: 100%; border: 1px solid #ccc; margin-top: 10px; display: block; }
  pre.params { background: #f4f4f4; padding: 10px; font-family: ui-monospace, monospace;
               font-size: 13px; white-space: pre; user-select: all; }
  .hdr { display: flex; align-items: baseline; gap: 12px; }
  .hdr small { color: #666; }
</style></head><body>
<div class="hdr">
  <h2>Lag-cube tuner</h2>
  <small>R<sub>y</sub>(yaw) · R<sub>x</sub>(pitch) · R<sub>z</sub>(roll) applied to local cube coords.
         Rendering reuses <code>plot_panel_a_stimulus</code> via module-constant patching.</small>
</div>

<div id="controls"></div>

<img id="preview" src="" alt="rendered stimulus half"/>

<h3>Copy these into <code>_fig4a_stimulus.py</code>:</h3>
<pre class="params" id="params"></pre>

<script>
const SLIDERS = __SLIDER_SPECS__;
const state = Object.fromEntries(SLIDERS.map(s => [s[0], s[4]]));

const controls = document.getElementById("controls");
SLIDERS.forEach(([k, lo, hi, step, def]) => {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `
    <label>${k}</label>
    <input type="range" min="${lo}" max="${hi}" step="${step}" value="${def}" id="s_${k}"/>
    <div class="val" id="v_${k}">${def}</div>
    <button class="reset" data-key="${k}" data-def="${def}">reset</button>
  `;
  controls.appendChild(row);
  const slider = row.querySelector(`#s_${k}`);
  slider.addEventListener("input", (e) => {
    state[k] = parseFloat(e.target.value);
    document.getElementById(`v_${k}`).textContent = state[k].toFixed(2);
    refreshDebounced();
  });
});
document.querySelectorAll(".reset").forEach(btn => {
  btn.addEventListener("click", () => {
    const k = btn.dataset.key;
    const v = parseFloat(btn.dataset.def);
    state[k] = v;
    const s = document.getElementById(`s_${k}`);
    s.value = v;
    document.getElementById(`v_${k}`).textContent = v.toFixed(2);
    refreshDebounced();
  });
});

function refresh() {
  const q = new URLSearchParams(state).toString();
  document.getElementById("preview").src = "/render?" + q + "&t=" + Date.now();
  document.getElementById("params").textContent =
    `CUBE_YAW_DEG   = ${state.yaw_deg.toFixed(2)}\n` +
    `CUBE_PITCH_DEG = ${state.pitch_deg.toFixed(2)}\n` +
    `CUBE_ROLL_DEG  = ${state.roll_deg.toFixed(2)}\n` +
    `CUBE_W = ${state.cube_w.toFixed(2)}\n` +
    `CUBE_H = ${state.cube_h.toFixed(2)}\n` +
    `CUBE_D = ${state.cube_d.toFixed(2)}`;
}
let timer = null;
function refreshDebounced() {
  clearTimeout(timer);
  timer = setTimeout(refresh, 120);
}
refresh();
</script>
</body></html>
"""

import json

app = Flask(__name__)


@app.route("/")
def index():
    page = HTML.replace("__SLIDER_SPECS__", json.dumps(SLIDERS))
    return page


@app.route("/render")
def render():
    overrides = {
        "CUBE_YAW_DEG":   float(request.args.get("yaw_deg",   stim.CUBE_YAW_DEG)),
        "CUBE_PITCH_DEG": float(request.args.get("pitch_deg", stim.CUBE_PITCH_DEG)),
        "CUBE_ROLL_DEG":  float(request.args.get("roll_deg",  stim.CUBE_ROLL_DEG)),
        "CUBE_W":         float(request.args.get("cube_w",    stim.CUBE_W)),
        "CUBE_H":         float(request.args.get("cube_h",    stim.CUBE_H)),
        "CUBE_D":         float(request.args.get("cube_d",    stim.CUBE_D)),
    }
    saved = {k: getattr(stim, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(stim, k, v)
        fig, ax = plt.subplots(figsize=(10, 4))
        stim.plot_panel_a_stimulus(ax, ASSETS)
        fig.tight_layout(pad=0.05)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    finally:
        for k, v in saved.items():
            setattr(stim, k, v)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    import argparse, socket
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0",
                   help="bind address (default 0.0.0.0 → reachable over LAN/ZeroTier)")
    p.add_argument("--port", type=int, default=5050)
    args = p.parse_args()
    print(f"Cube tuner binding {args.host}:{args.port}")
    try:
        print(f"  hostname: {socket.gethostname()}")
    except Exception:
        pass
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
