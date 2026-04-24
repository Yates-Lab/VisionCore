"""
Generate a PDF showing extreme FEM cells: bottom 10 (least FEM-selective)
followed by top 20 (most FEM-selective), each with main_unit_panel tuning
info and the two-stage model diagnostics PNG side by side.
"""
import sys
sys.path.insert(0, "/home/tejas/DataYatesV1")
sys.path.insert(0, "/home/tejas/VisionCore/tejas/model")

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image

from tejas.metrics.gratings import get_gratings_for_dataset
from tejas.metrics.fixrsvp import get_fixrsvp_for_dataset, plot_fixrsvp_psth_and_spike_raster
from tejas.metrics.grating_utils import plot_phase_tuning_with_fit, plot_ori_tuning
from tejas.metrics.gaborium import plot_unit_sta_ste, get_rf_contour_metrics, get_rf_gaussian_fit_metrics, get_unit_sta_ste_dict
from tejas.metrics.mcfarland import get_mcfarland_analysis_for_dataset, plot_mcfarland_analysis_for_unit

RUNS_ROOT = Path("/home/tejas/VisionCore/tejas/model/final_runs_4levels")
CONFIGS_DIR = Path("/home/tejas/VisionCore/experiments/dataset_configs/sessions")
TXT_PATH = RUNS_ROOT / "allen_extreme_fem_cells.txt"
OUTPUT_PDF = RUNS_ROOT / "fem_extreme_cells_overview.pdf"


def parse_full_list(txt_path):
    lines = open(txt_path).readlines()
    rows = []
    in_full = False
    for line in lines:
        if "FULL SORTED LIST" in line:
            in_full = True
            continue
        if in_full and line.strip() and not line.startswith("-") and not line.startswith("Session"):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append({
                        "session": parts[0],
                        "unit_idx": int(parts[1]),
                        "one_minus_alpha": float(parts[2]),
                    })
                except ValueError:
                    pass
    return rows


def load_cids(session):
    with open(CONFIGS_DIR / f"{session}.yaml") as f:
        return yaml.safe_load(f)["cids"]


def unit_idx_to_cell_id(session, unit_idx):
    cids = load_cids(session)
    if unit_idx in cids:
        return cids.index(unit_idx)
    return None


def get_diagnostics_png_path(session, cell_id):
    return RUNS_ROOT / session / "pngs" / f"cell_{int(cell_id):03d}_best_diagnostics.png"


def add_separator_page(pdf, title):
    fig = plt.figure(figsize=(16, 10))
    fig.text(0.5, 0.5, title, ha="center", va="center", fontsize=28, fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def load_session_panel_data(date, subject="Allen"):
    gratings_info = get_gratings_for_dataset(date, subject, cache=True)
    try:
        fixrsvp_info = get_fixrsvp_for_dataset(date, subject, cache=True)
    except Exception:
        fixrsvp_info = None
    try:
        mcfarland_info = get_mcfarland_analysis_for_dataset(date, subject, cache=True)
    except Exception:
        mcfarland_info = None
    rf_contour_metrics = get_rf_contour_metrics(date, subject)
    rf_gaussian_fit_metrics = get_rf_gaussian_fit_metrics(date, subject, cache=True)
    sta_ste_dict = get_unit_sta_ste_dict(date, subject)
    return {
        "gratings_info": gratings_info,
        "fixrsvp_info": fixrsvp_info,
        "mcfarland_info": mcfarland_info,
        "rf_contour_metrics": rf_contour_metrics,
        "rf_gaussian_fit_metrics": rf_gaussian_fit_metrics,
        "unit_sta_ste_dict": sta_ste_dict,
    }


def build_tuning_panel(unit_idx, date, panel_data):
    fig = plt.figure(figsize=(10, 8), dpi=50)
    gs = GridSpec(3, 2, figure=fig, hspace=0, wspace=0.3, top=0.95, bottom=0,
                  height_ratios=[1.2, 1, 1.1])

    ax_sta = fig.add_subplot(gs[0, :])
    plot_unit_sta_ste("Allen", date, unit_idx, panel_data["unit_sta_ste_dict"],
                      contour_metrics=panel_data["rf_contour_metrics"],
                      gaussian_fit_metrics=panel_data["rf_gaussian_fit_metrics"],
                      sampling_rate=240, ax=ax_sta)

    ax_phase = fig.add_subplot(gs[1, 0])
    plot_phase_tuning_with_fit(panel_data["gratings_info"], unit_idx, ax=ax_phase)

    ax_ori = fig.add_subplot(gs[1, 1])
    plot_ori_tuning(panel_data["gratings_info"], unit_idx, ax=ax_ori)

    ax_fixrsvp = fig.add_subplot(gs[2, 0])
    if panel_data["fixrsvp_info"] is not None:
        plot_fixrsvp_psth_and_spike_raster(panel_data["fixrsvp_info"], unit_idx, ax=ax_fixrsvp)
    else:
        ax_fixrsvp.text(0.5, 0.5, "No FixRSVP", ha="center", va="center")
        ax_fixrsvp.axis("off")

    ax_mcf = fig.add_subplot(gs[2, 1])
    if panel_data["mcfarland_info"] is not None:
        plot_mcfarland_analysis_for_unit(unit_idx, panel_data["mcfarland_info"],
                                         contour_metrics=panel_data["rf_contour_metrics"],
                                         ax=ax_mcf, bins=37, show_fit=True)
    else:
        ax_mcf.text(0.5, 0.5, "No McFarland", ha="center", va="center")
        ax_mcf.axis("off")

    pos_phase = ax_phase.get_position()
    pos_ori = ax_ori.get_position()
    pos_fixrsvp = ax_fixrsvp.get_position()
    pos_mcf = ax_mcf.get_position()
    shift_up = 0.03
    shift_down = 0.05
    ax_phase.set_position([pos_phase.x0, pos_phase.y0 + shift_up, pos_phase.width, pos_phase.height])
    ax_ori.set_position([pos_ori.x0, pos_ori.y0 + shift_up, pos_ori.width, pos_ori.height])
    ax_fixrsvp.set_position([pos_fixrsvp.x0, pos_fixrsvp.y0 - shift_down, pos_fixrsvp.width, pos_fixrsvp.height])
    ax_mcf.set_position([pos_mcf.x0, pos_mcf.y0 - shift_down, pos_mcf.width, pos_mcf.height])

    return fig


def add_cell_page(pdf, row, cell_id, panel_dict_cache, is_fem_selective):
    session = row["session"]
    unit_idx = row["unit_idx"]
    one_minus_alpha = row["one_minus_alpha"]
    date = session.replace("Allen_", "")

    if session not in panel_dict_cache:
        print(f"  Loading panel data for {session}...")
        panel_dict_cache[session] = load_session_panel_data(date)

    panel_fig = build_tuning_panel(unit_idx, date, panel_dict_cache[session])
    panel_fig.savefig("/tmp/_panel_tmp.png", dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(panel_fig)
    panel_img = np.array(Image.open("/tmp/_panel_tmp.png"))

    fig = plt.figure(figsize=(20, 10), dpi=72)
    gs_outer = GridSpec(1, 2, figure=fig, wspace=0.02, left=0.01, right=0.99, top=0.92, bottom=0.01)

    ax_left = fig.add_subplot(gs_outer[0, 0])
    ax_left.imshow(panel_img)
    ax_left.axis("off")
    ax_left.set_title(f"Tuning Panel (unit {unit_idx})", fontsize=11, pad=4)

    ax_right = fig.add_subplot(gs_outer[0, 1])
    png_path = get_diagnostics_png_path(session, cell_id)
    if png_path.exists():
        diag_img = np.array(Image.open(png_path))
        ax_right.imshow(diag_img)
        ax_right.set_title(f"Two-Stage Model (cell {cell_id:03d})", fontsize=11, pad=4)
    else:
        ax_right.text(0.5, 0.5, "PNG not found", ha="center", va="center", fontsize=14)
    ax_right.axis("off")

    alpha_color = "green" if is_fem_selective else "red"
    title_prefix = f"{session}  |  unit_idx={unit_idx}  |  cell_id={cell_id:03d}  |  "
    alpha_str = f"1-α = {one_minus_alpha:.4f}"
    fig.text(0.5, 0.97, title_prefix, ha="center", va="center", fontsize=14, fontweight="bold",
             transform=fig.transFigure)
    prefix_width = len(title_prefix) / (len(title_prefix) + len(alpha_str))
    fig.text(0.5 + prefix_width * 0.28, 0.97, alpha_str, ha="center", va="center",
             fontsize=14, fontweight="bold", color=alpha_color, transform=fig.transFigure)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    N_BOTTOM = 40
    N_TOP = 40

    full_list = parse_full_list(TXT_PATH)
    bottom = full_list[:N_BOTTOM]
    top_descending = list(reversed(full_list[-N_TOP:]))
    top_ascending = list(reversed(top_descending))

    panel_dict_cache = {}

    with PdfPages(str(OUTPUT_PDF)) as pdf:
        add_separator_page(pdf, f"LEAST FEM-SELECTIVE (Bottom {N_BOTTOM})\n1-α ascending (most stimulus-driven first)")

        for i, row in enumerate(bottom):
            cell_id = unit_idx_to_cell_id(row["session"], row["unit_idx"])
            if cell_id is None:
                print(f"  Skipping {row['session']} unit {row['unit_idx']} — not in model config")
                continue
            print(f"  [{i+1}/{N_BOTTOM}] Bottom: {row['session']} unit={row['unit_idx']} cell={cell_id} 1-α={row['one_minus_alpha']:.4f}")
            add_cell_page(pdf, row, cell_id, panel_dict_cache, is_fem_selective=False)

        add_separator_page(pdf, f"MOST FEM-SELECTIVE (Top {N_TOP})\n1-α ascending (within top {N_TOP})")

        for i, row in enumerate(top_ascending):
            cell_id = unit_idx_to_cell_id(row["session"], row["unit_idx"])
            if cell_id is None:
                print(f"  Skipping {row['session']} unit {row['unit_idx']} — not in model config")
                continue
            print(f"  [{i+1}/{N_TOP}] Top: {row['session']} unit={row['unit_idx']} cell={cell_id} 1-α={row['one_minus_alpha']:.4f}")
            add_cell_page(pdf, row, cell_id, panel_dict_cache, is_fem_selective=True)

    print(f"\nSaved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
