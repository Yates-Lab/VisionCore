#!/bin/bash
set -e
cd /home/tejas/VisionCore

SESSIONS=(
  Allen_2022-02-16
  Allen_2022-02-18
  Allen_2022-02-24
  Allen_2022-03-02
  Allen_2022-03-04
  Allen_2022-03-30
  Allen_2022-04-01
  Allen_2022-04-06
  Allen_2022-04-08
  Allen_2022-04-13
  Allen_2022-04-15
  Allen_2022-06-01
  Allen_2022-06-10
  Allen_2022-08-05
)

for sess in "${SESSIONS[@]}"; do
  echo "========================================"
  echo "$(date): Processing $sess"
  echo "========================================"

  echo "  -> Baseline + scaled PDFs..."
  SESSION_NAME="$sess" uv run python tejas/model/two_stage_fixrsvp_psth_pdf.py 2>&1 || {
    echo "  !! FAILED baseline PDF for $sess, continuing..."
  }

  echo "  -> Eta-calibrated PDF..."
  SESSION_NAME="$sess" uv run python tejas/model/two_stage_fixrsvp_eta_calibrated_pdf.py 2>&1 || {
    echo "  !! FAILED eta-calibrated PDF for $sess, continuing..."
  }

  echo "  -> Done with $sess"
  echo ""
done

echo "========================================"
echo "$(date): ALL SESSIONS COMPLETE"
echo "========================================"
