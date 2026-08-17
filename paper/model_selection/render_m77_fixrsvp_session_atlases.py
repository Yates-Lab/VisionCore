#!/usr/bin/env python3
"""Render per-session FixRSVP prediction atlases for the Twin and M77.

Two separate atlas sets are written:

* ``120hz`` uses the exact Figure-3 comparison frame for data, Twin, and M77.
* ``240hz`` uses native 240-Hz data and M77 predictions.  The Twin is a
  120-Hz model, so its rate is repeated across each pair of display bins and
  is explicitly labeled as a 120-Hz reference; it is never presented as a
  native-240 prediction.

Each session PDF contains every common unit in the frozen Figure-3 neuron
order.  The script refuses mismatched sessions, neuron order, observations, or
time axes before rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_M77 = (
    ROOT
    / "outputs/dekel240_paper/m77_epoch279/figure3_predictions/"
    "full_true240_trace.pkl"
)
DEFAULT_TWIN = ROOT / "outputs/cache/fig3_digitaltwin.pkl"
DEFAULT_OUT = (
    ROOT
    / "outputs/dekel240_paper/m77_epoch279/fixrsvp_session_prediction_atlases"
)

DATA_COLOR = "#202020"
TWIN_COLOR = "#E67E22"
M77_COLOR = "#2C7FB8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m77-trace-cache", type=Path, default=DEFAULT_M77)
    parser.add_argument("--twin-cache", type=Path, default=DEFAULT_TWIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--cells-per-page", type=int, default=12)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument(
        "--rate",
        action="append",
        choices=("120", "240"),
        default=None,
        help="Atlas rate to render; repeat to select both. Default: both.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_list(path: Path) -> list[dict]:
    with path.open("rb") as stream:
        value = dill.load(stream)
    if not isinstance(value, list) or not value:
        raise TypeError(f"Expected a nonempty list in {path}")
    return value


def finite_maximum(*values: np.ndarray) -> float:
    finite = np.concatenate(
        [np.asarray(value, dtype=float)[np.isfinite(value)] for value in values]
    )
    return max(float(np.max(finite)) if finite.size else 1.0, 1e-6)


def validate_pair(m77: dict, twin: dict) -> None:
    session = str(m77["session"])
    if session != str(twin["session"]):
        raise ValueError(f"Session mismatch: {session} versus {twin['session']}")
    m77_mask = np.asarray(m77["neuron_mask"], dtype=np.int64)
    twin_mask = np.asarray(twin["neuron_mask"], dtype=np.int64)
    if not np.array_equal(m77_mask, twin_mask):
        raise ValueError(f"{session}: Twin and M77 neuron order differs")

    for key in ("robs_mean", "rhat_mean", "ccnorm", "ve_model"):
        if key not in m77 or key not in twin:
            raise KeyError(f"{session}: missing required 120-Hz field {key}")
    if np.asarray(m77["robs_mean"]).shape != np.asarray(twin["robs_mean"]).shape:
        raise ValueError(f"{session}: Twin and M77 120-Hz PSTH shapes differ")
    np.testing.assert_allclose(
        np.asarray(m77["robs_mean"], dtype=float),
        np.asarray(twin["robs_mean"], dtype=float),
        # The legacy Twin cache is float64 while the regenerated M77 cache
        # passed through a float32 serialization boundary.  Require numerical
        # identity at substantially tighter precision than the plotted/scored
        # rates, without treating round-off at that boundary as different data.
        rtol=2e-7,
        atol=1e-8,
        equal_nan=True,
        err_msg=f"{session}: cached 120-Hz observations differ",
    )

    native = m77.get("native240")
    if not isinstance(native, dict):
        raise KeyError(f"{session}: M77 cache lacks native240 predictions")
    native_data = np.asarray(native["robs_mean"], dtype=float)
    native_prediction = np.asarray(native["rhat_mean"], dtype=float)
    if native_data.shape != native_prediction.shape:
        raise ValueError(f"{session}: native M77 observation/prediction shapes differ")
    if native_data.shape[0] != 2 * np.asarray(m77["robs_mean"]).shape[0]:
        raise ValueError(f"{session}: native time axis is not exactly twice 120 Hz")
    if native_data.shape[1] != len(m77_mask):
        raise ValueError(f"{session}: native unit axis differs from neuron mask")


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def metric_text(m77: dict, twin: dict, unit: int) -> str:
    return (
        f"CCnorm Twin {float(twin['ccnorm'][unit]):.2f} · M77 "
        f"{float(m77['ccnorm'][unit]):.2f}\n"
        f"single-trial $R^2$ Twin {float(twin['ve_model'][unit]):.3f} · M77 "
        f"{float(m77['ve_model'][unit]):.3f}"
    )


def new_page(
    *,
    session: str,
    page: int,
    n_pages: int,
    rate: int,
    rows: int,
    columns: int,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.0, 8.5),
        squeeze=False,
        constrained_layout=False,
    )
    if rate == 120:
        title = f"{session} · FixRSVP PSTHs at 120 Hz · data, Twin, M77"
        footer = (
            "Canonical Figure-3 frame; Twin and M77 use the same data-defined "
            "support and positive affine adjustment."
        )
    else:
        title = f"{session} · native 240-Hz FixRSVP PSTHs · data, M77, Twin reference"
        footer = (
            "Data and M77 are native 240 Hz. Twin is a 120-Hz prediction repeated "
            "across paired display bins; it is not a native-240 estimate."
        )
    fig.suptitle(
        f"{title}   |   page {page + 1}/{n_pages}",
        x=0.055,
        y=0.982,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=7, color="#555555")
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.075, top=0.925, wspace=0.24, hspace=0.48)
    return fig, axes


def render_session(
    m77: dict,
    twin: dict,
    *,
    out_path: Path,
    rate: int,
    cells_per_page: int,
    columns: int,
) -> int:
    validate_pair(m77, twin)
    session = str(m77["session"])
    unit_ids = np.asarray(m77["neuron_mask"], dtype=np.int64)
    n_units = len(unit_ids)
    n_pages = int(math.ceil(n_units / cells_per_page))
    rows = int(math.ceil(cells_per_page / columns))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rate == 120:
        data = np.asarray(m77["robs_mean"], dtype=float) * 120.0
        m77_prediction = np.asarray(m77["rhat_mean"], dtype=float) * 120.0
        twin_prediction = np.asarray(twin["rhat_mean"], dtype=float) * 120.0
        time_ms = (np.arange(data.shape[0]) + 0.5) / 120.0 * 1000.0
    elif rate == 240:
        native = m77["native240"]
        data = np.asarray(native["robs_mean"], dtype=float) * 240.0
        m77_prediction = np.asarray(native["rhat_mean"], dtype=float) * 240.0
        twin_rate = np.asarray(twin["rhat_mean"], dtype=float) * 120.0
        twin_prediction = np.repeat(twin_rate, 2, axis=0)
        time_ms = (np.arange(data.shape[0]) + 0.5) / 240.0 * 1000.0
    else:
        raise ValueError(rate)

    with PdfPages(out_path) as pdf:
        for page in range(n_pages):
            fig, axes = new_page(
                session=session,
                page=page,
                n_pages=n_pages,
                rate=rate,
                rows=rows,
                columns=columns,
            )
            start = page * cells_per_page
            stop = min(start + cells_per_page, n_units)
            for slot, unit in enumerate(range(start, stop)):
                ax = axes.flat[slot]
                ax.plot(time_ms, data[:, unit], color=DATA_COLOR, lw=1.0, label="data")
                if rate == 240:
                    ax.step(
                        time_ms,
                        twin_prediction[:, unit],
                        where="mid",
                        color=TWIN_COLOR,
                        lw=1.0,
                        alpha=0.9,
                        label="Twin (120-Hz reference)",
                    )
                else:
                    ax.plot(
                        time_ms,
                        twin_prediction[:, unit],
                        color=TWIN_COLOR,
                        lw=1.1,
                        label="Twin",
                    )
                ax.plot(
                    time_ms,
                    m77_prediction[:, unit],
                    color=M77_COLOR,
                    lw=1.1,
                    label="M77",
                )
                ymax = 1.08 * finite_maximum(
                    data[:, unit], twin_prediction[:, unit], m77_prediction[:, unit]
                )
                ax.set_ylim(0, ymax)
                ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))
                ax.grid(alpha=0.16, linewidth=0.5)
                ax.set_title(f"unit {int(unit_ids[unit])} · {metric_text(m77, twin, unit)}", loc="left")
                if slot % columns == 0:
                    ax.set_ylabel("spikes/s")
                if slot // columns == rows - 1 or unit + columns >= stop:
                    ax.set_xlabel("time (ms)")
            for slot in range(stop - start, rows * columns):
                axes.flat[slot].set_axis_off()
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.985),
                frameon=False,
                ncol=len(labels),
                fontsize=7.2,
            )
            pdf.savefig(fig)
            plt.close(fig)
    return n_pages


def main() -> int:
    args = parse_args()
    if args.cells_per_page < 1 or args.columns < 1:
        raise ValueError("cells-per-page and columns must be positive")
    configure()
    m77_results = load_list(args.m77_trace_cache)
    twin_by_session = {
        str(result["session"]): result for result in load_list(args.twin_cache)
    }
    m77_by_session = {str(result["session"]): result for result in m77_results}
    sessions = list(m77_by_session)
    if args.session:
        requested = list(dict.fromkeys(args.session))
        missing = [name for name in requested if name not in m77_by_session]
        if missing:
            raise KeyError(f"M77 trace cache lacks requested sessions: {missing}")
        sessions = requested
    if args.max_sessions is not None:
        sessions = sessions[: int(args.max_sessions)]
    rates = [int(value) for value in (args.rate or ["120", "240"])]

    records = []
    for ordinal, session in enumerate(sessions, 1):
        if session not in twin_by_session:
            raise KeyError(f"Twin cache lacks {session}")
        m77 = m77_by_session[session]
        twin = twin_by_session[session]
        print(f"[{ordinal}/{len(sessions)}] {session}", flush=True)
        for rate in rates:
            suffix = "120hz" if rate == 120 else "native240"
            output = args.out_dir / f"{rate}hz" / f"{session}_twin_vs_m77_{suffix}.pdf"
            pages = render_session(
                m77,
                twin,
                out_path=output,
                rate=rate,
                cells_per_page=int(args.cells_per_page),
                columns=int(args.columns),
            )
            records.append(
                {
                    "session": session,
                    "rate_hz": rate,
                    "n_units": int(len(m77["neuron_mask"])),
                    "n_pages": pages,
                    "path": str(output.resolve()),
                }
            )

    manifest = {
        "analysis": "per-session Twin versus M77 FixRSVP prediction atlases",
        "m77_trace_cache": str(args.m77_trace_cache.resolve()),
        "m77_trace_cache_sha256": sha256(args.m77_trace_cache),
        "twin_cache": str(args.twin_cache.resolve()),
        "twin_cache_sha256": sha256(args.twin_cache),
        "label_contract": {
            "reference_model": "Twin",
            "selected_model": "M77",
            "native_240_caveat": (
                "Twin is a 120-Hz prediction repeated for display; only data and "
                "M77 are native 240 Hz"
            ),
        },
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "atlas_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} session atlases to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
