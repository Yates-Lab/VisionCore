"""Frozen snapshot of the production fig2 pipeline (2026-06-02).

Files snapshotted (verbatim except for intra-snapshot import rewrites and a
local CACHE_DIR override in compute_fig2_data.py):

- covariance.py            <- VisionCore/VisionCore/covariance.py
- stats.py                 <- VisionCore/VisionCore/stats.py
- subspace.py              <- VisionCore/VisionCore/subspace.py
- compute_fig2_data.py     <- VisionCore/ryan/fig2/compute_fig2_data.py

This snapshot is the "legacy" comparator for the methods-folder parallel
pipeline. Do not edit it -- if the production code changes and you want to
re-pin the comparator, copy the new files in and bump SNAPSHOT_DATE.
"""

SNAPSHOT_DATE = "2026-06-02"
