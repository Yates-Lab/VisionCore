#!/usr/bin/env python3
"""Utility script to compare stored contour areas with recomputed mask areas."""
import argparse
import numpy as np

from tejas.metrics.gaborium import get_rf_contour_metrics
from DataYatesV1.exp.general import get_session
from DataYatesV1.utils.data.loading import get_gaborium_sta_ste
from DataYatesV1.utils.rf import get_mask_from_contour


def normalize_for_contour(ste_slice: np.ndarray) -> np.ndarray:
    """Normalize STE slice to [0, 1] as in get_contour_metrics."""
    wspace = ste_slice.squeeze()
    ptp = np.ptp(wspace)
    if ptp != 0:
        return (wspace - np.min(wspace)) / ptp
    return wspace


def recompute_mask_area(contour: np.ndarray, ste_slice: np.ndarray) -> float:
    """Recompute contour mask area using the stored contour coordinates."""
    normalized = normalize_for_contour(ste_slice)
    mask = get_mask_from_contour(normalized, contour)
    return float(mask.sum())


def main():
    parser = argparse.ArgumentParser(description="Verify contour areas for selected units")
    parser.add_argument('--subject', required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--units', nargs='+', type=int, required=True,
                        help='Unit IDs to inspect')
    parser.add_argument('--n_lags', type=int, default=20)
    args = parser.parse_args()

    metrics = get_rf_contour_metrics(args.date, args.subject, cache=True)
    sess = get_session(args.subject, args.date)
    _, stes = get_gaborium_sta_ste(sess, args.n_lags)

    for unit_id in args.units:
        entry = metrics.get(unit_id)
        if entry is None:
            print(f"Unit {unit_id}: no contour metrics available")
            continue
        contour = entry.get('contour')
        if contour is None or len(contour) == 0 or not entry.get('valid', False):
            print(f"Unit {unit_id}: contour invalid or missing")
            continue
        peak_lag = entry['ste_peak_lag']
        ste_slice = stes[unit_id, peak_lag]

        stored_area = entry['area_contour']
        recomputed_area = recompute_mask_area(contour, ste_slice)

        print(f"Unit {unit_id}: stored_area={stored_area:.2f}, recomputed_area={recomputed_area:.2f}, "
              f"sqrt_area_deg={entry['sqrt_area_contour_deg']:.3f}")


if __name__ == '__main__':
    main()
