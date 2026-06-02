# legacy/ — frozen production-pipeline snapshot

Pinned 2026-06-02 from `VisionCore/VisionCore/{covariance,stats,subspace}.py`
and `VisionCore/ryan/fig2/compute_fig2_data.py`.

Do not edit. The only modifications relative to the live files are:

- intra-snapshot imports rewritten to relative (`from .covariance import ...`),
  so the snapshot binds to its own frozen siblings rather than to whatever the
  live `VisionCore.covariance` module becomes after future refactors;
- `CACHE_DIR` overridden inside `compute_fig2_data.py` to point at
  `../cache/legacy_native/`, so the snapshot's outputs do not collide with
  the production `~/.cache/v1-fovea/` files.

Bump `SNAPSHOT_DATE` in `__init__.py` if you re-pin.
