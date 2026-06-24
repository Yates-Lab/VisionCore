"""Interactive tuner for Figure 2's middle row.

Sliders drive the four middle-row layout knobs and re-render the whole figure so
you can judge them in context:

    row1_height      height ratio of the middle row (rows 0/2 are fixed at 1.05)
    d_width_frac     fraction of the row width taken by panel D; the remainder
                     is split evenly between E and F
    mid_wspace       horizontal padding between D | E | F
    d_title_fontsize font size of the decomposition matrix titles/labels

The fig2 data bundle is loaded once at startup and reused, so each re-render is
just the matplotlib draw. When you're happy, copy the read-out values back and
I'll bake them into ``compose``'s defaults.

Run (adds Flask on top of the workspace env, no project changes):

    uv run --with flask python paper/fig2/tune_midrow_app.py

then open http://127.0.0.1:5057
"""
import sys
import threading
from pathlib import Path

from flask import Flask, request, Response

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from generate_figure2 import compose, load_prepared_data  # noqa: E402

app = Flask(__name__)

# Load the (cached) data bundle once; matplotlib draws aren't thread-safe so we
# serialize re-renders behind a lock.
print("Loading fig2 data bundle ...", flush=True)
DATA = load_prepared_data(refresh=False, split_subjects=False)
print("Ready.", flush=True)
_RENDER_LOCK = threading.Lock()

DEFAULTS = dict(
    row1_height=1.10,
    d_width_frac=0.60,
    mid_wspace=0.30,
    d_title_fontsize=8.0,
)

PARAMS = [
    # key, label, min, max, step
    ("row1_height", "Middle-row height ratio", 0.90, 1.90, 0.02),
    ("d_width_frac", "Panel D width fraction", 0.45, 0.72, 0.01),
    ("mid_wspace", "Inter-panel padding (wspace)", 0.15, 1.40, 0.05),
    ("d_title_fontsize", "Panel D title font size", 4.5, 9.0, 0.5),
]

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Fig2 middle-row tuner</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }}
  #controls {{ width: 340px; padding: 18px; box-sizing: border-box;
               border-right: 1px solid #ddd; overflow-y: auto; }}
  #preview {{ flex: 1; display: flex; align-items: flex-start;
              justify-content: center; padding: 16px; overflow: auto; }}
  #preview img {{ max-width: 100%; box-shadow: 0 1px 6px rgba(0,0,0,.2); }}
  .row {{ margin-bottom: 16px; }}
  label {{ font-size: 13px; font-weight: 600; display: block; }}
  input[type=range] {{ width: 100%; }}
  .val {{ font-variant-numeric: tabular-nums; color: #1a6; font-weight: 700; }}
  pre {{ background:#f4f4f4; padding:10px; border-radius:6px; font-size:12px;
         white-space: pre-wrap; }}
  button {{ font-size: 13px; padding: 6px 10px; margin-right: 6px; cursor: pointer; }}
  #status {{ font-size: 12px; color:#888; height: 14px; }}
</style></head>
<body>
<div id="controls">
  <h3 style="margin-top:0">Fig 2 middle row</h3>
  {sliders}
  <div class="row">
    <button onclick="resetAll()">Reset to current defaults</button>
  </div>
  <div id="status"></div>
  <h4 style="margin-bottom:4px">Values to paste back</h4>
  <pre id="readout"></pre>
</div>
<div id="preview"><img id="fig" src=""></div>

<script>
const PARAMS = {params_json};
const DEFAULTS = {defaults_json};

function vals() {{
  const o = {{}};
  for (const p of PARAMS) o[p.key] = parseFloat(document.getElementById(p.key).value);
  return o;
}}
function fmt(o) {{
  return `row1_height={{}}, d_width_frac={{}}, mid_wspace={{}}, d_title_fontsize={{}}`
    .replace('{{}}', o.row1_height.toFixed(2))
    .replace('{{}}', o.d_width_frac.toFixed(2))
    .replace('{{}}', o.mid_wspace.toFixed(2))
    .replace('{{}}', o.d_title_fontsize.toFixed(1));
}}
let timer = null;
function render() {{
  const o = vals();
  for (const p of PARAMS) {{
    const v = o[p.key];
    document.getElementById(p.key + "_v").textContent =
      (p.key === "d_title_fontsize") ? v.toFixed(1) : v.toFixed(2);
  }}
  document.getElementById("readout").textContent = fmt(o);
  document.getElementById("status").textContent = "rendering ...";
  const qs = new URLSearchParams(o).toString();
  clearTimeout(timer);
  timer = setTimeout(() => {{
    const img = document.getElementById("fig");
    const next = new Image();
    next.onload = () => {{ img.src = next.src;
        document.getElementById("status").textContent = ""; }};
    next.src = "/preview?" + qs + "&_t=" + Date.now();
  }}, 180);
}}
function resetAll() {{
  for (const p of PARAMS) document.getElementById(p.key).value = DEFAULTS[p.key];
  render();
}}
for (const p of PARAMS)
  document.getElementById(p.key).addEventListener("input", render);
render();
</script>
</body></html>
"""


def _slider_html():
    import json
    rows = []
    for key, label, lo, hi, step in PARAMS:
        d = DEFAULTS[key]
        rows.append(
            f'<div class="row"><label>{label}: '
            f'<span class="val" id="{key}_v">{d}</span></label>'
            f'<input type="range" id="{key}" min="{lo}" max="{hi}" '
            f'step="{step}" value="{d}"></div>'
        )
    params_json = json.dumps([
        {"key": k, "label": lbl, "min": lo, "max": hi, "step": st}
        for k, lbl, lo, hi, st in PARAMS
    ])
    return "\n".join(rows), params_json


@app.route("/")
def index():
    import json
    sliders, params_json = _slider_html()
    return PAGE.format(
        sliders=sliders,
        params_json=params_json,
        defaults_json=json.dumps(DEFAULTS),
    )


@app.route("/preview")
def preview():
    kw = {}
    for key, _, lo, hi, _step in PARAMS:
        try:
            v = float(request.args.get(key, DEFAULTS[key]))
        except (TypeError, ValueError):
            v = DEFAULTS[key]
        kw[key] = max(lo, min(hi, v))
    with _RENDER_LOCK:
        png = compose(prepared_data=DATA, return_png_bytes=True,
                      preview_dpi=110, **kw)
    return Response(png, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5057, debug=False, threaded=False)
