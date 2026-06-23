"""
Exploratory: per-session significance rates for panel G (subspace alignment).

For each session, compute one-sided empirical p-values vs that session's own
eye-shuffle cloud for:
    X = PSTH variance captured by the FEM subspace (var_p_given_f)
    Y = FEM variance captured by the PSTH subspace (var_f_given_p)
and report, per animal, the fraction of sessions significant at α = 0.05 and
0.01 for X, Y, and (X and Y) jointly.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig2"))
from compute_fig2_data import load_fig2_data  # noqa: E402


def emp_p_greater(null_vals, observed):
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if null_vals.size == 0 or not np.isfinite(observed):
        return np.nan
    return (np.sum(null_vals >= observed) + 1) / (null_vals.size + 1)


def main():
    data = load_fig2_data()
    sub_names = np.asarray(data["sub_names"])
    sub_subjects = np.asarray(data["sub_subjects"])
    x_real = np.asarray(data["var_p_given_f"], dtype=float)
    y_real = np.asarray(data["var_f_given_p"], dtype=float)

    null_idx = np.asarray(data["null_session_idx"], dtype=int)
    null_x = np.asarray(data["null_var_p_given_f"], dtype=float)
    null_y = np.asarray(data["null_var_f_given_p"], dtype=float)

    n_sessions = len(sub_names)
    rows = []
    for i in range(n_sessions):
        mask = null_idx == i
        nx = null_x[mask]
        ny = null_y[mask]
        px = emp_p_greater(nx, x_real[i])
        py = emp_p_greater(ny, y_real[i])
        rows.append((sub_names[i], sub_subjects[i], x_real[i], y_real[i],
                     int(mask.sum()), px, py))

    print(f"\n{'session':<35} {'subj':<8} {'X':>6} {'Y':>6} {'nshuf':>6} "
          f"{'pX':>8} {'pY':>8}")
    print("-" * 88)
    for name, subj, x, y, n, px, py in rows:
        print(f"{name:<35} {subj:<8} {x:6.3f} {y:6.3f} {n:6d} "
              f"{px:8.4f} {py:8.4f}")

    print("\n=== Per-animal significance rates ===")
    alphas = [0.05, 0.01]
    header = (f"\n{'subject':<10} {'n_sess':>7} "
              + " ".join([f"{'pX<' + str(a):>10}" for a in alphas])
              + " "
              + " ".join([f"{'pY<' + str(a):>10}" for a in alphas])
              + " "
              + " ".join([f"{'both<' + str(a):>10}" for a in alphas]))
    print(header)
    print("-" * len(header))

    subjects = sorted(set(sub_subjects.tolist()))
    rows_arr = np.array([(r[1], r[5], r[6]) for r in rows],
                        dtype=[("subj", object), ("px", float), ("py", float)])
    for subj in subjects:
        m = rows_arr["subj"] == subj
        n = int(m.sum())
        if n == 0:
            continue
        px = rows_arr["px"][m]
        py = rows_arr["py"][m]
        cells = [f"{n:>7d}"]
        for a in alphas:
            f = np.mean(px < a)
            k = int(np.sum(px < a))
            cells.append(f"{k}/{n} ({f:.2f})".rjust(10))
        for a in alphas:
            f = np.mean(py < a)
            k = int(np.sum(py < a))
            cells.append(f"{k}/{n} ({f:.2f})".rjust(10))
        for a in alphas:
            both = (px < a) & (py < a)
            f = np.mean(both)
            k = int(np.sum(both))
            cells.append(f"{k}/{n} ({f:.2f})".rjust(10))
        print(f"{subj:<10} " + " ".join(cells))

    # Pooled across animals for reference
    print()
    px_all = rows_arr["px"]
    py_all = rows_arr["py"]
    n = px_all.size
    for a in alphas:
        kx = int(np.sum(px_all < a))
        ky = int(np.sum(py_all < a))
        kb = int(np.sum((px_all < a) & (py_all < a)))
        print(f"  pooled (α={a}): X {kx}/{n} ({kx/n:.2f}), "
              f"Y {ky}/{n} ({ky/n:.2f}), both {kb}/{n} ({kb/n:.2f})")


if __name__ == "__main__":
    main()
