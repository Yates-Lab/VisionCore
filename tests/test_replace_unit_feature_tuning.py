import json

import pandas as pd

from paper.fig4.upstream.replace_unit_feature_tuning import replace_tuning


def test_replace_tuning_removes_stale_schema_and_preserves_base_columns(tmp_path):
    unit_table = tmp_path / "unit_feature_table.csv"
    previous_tuning = tmp_path / "previous_tuning.csv"
    new_tuning = tmp_path / "new_tuning.csv"
    output = tmp_path / "updated.csv"

    pd.DataFrame(
        {
            "unit_index": [0, 1],
            "unit_label": ["u000", "u001"],
            "canonical_channel": [10, 11],
            "sf_group": ["old-low", "old-high"],
            "old_only": [1.0, 2.0],
        }
    ).to_csv(unit_table, index=False)
    pd.DataFrame(
        {
            "unit_index": [0, 1],
            "unit_label": ["u000", "u001"],
            "sf_group": ["old-low", "old-high"],
            "old_only": [1.0, 2.0],
        }
    ).to_csv(previous_tuning, index=False)
    pd.DataFrame(
        {
            "unit_index": [0, 1],
            "unit_label": ["new-label-ignored", "new-label-ignored"],
            "sf_group": ["m70-high", "m70-low"],
            "new_only": [3.0, 4.0],
        }
    ).to_csv(new_tuning, index=False)

    report = replace_tuning(
        unit_feature_table=unit_table,
        new_unit_tuning_csv=new_tuning,
        previous_unit_tuning_csv=previous_tuning,
        out_path=output,
    )
    updated = pd.read_csv(output)

    assert updated["unit_label"].tolist() == ["u000", "u001"]
    assert updated["canonical_channel"].tolist() == [10, 11]
    assert updated["sf_group"].tolist() == ["m70-high", "m70-low"]
    assert updated["new_only"].tolist() == [3.0, 4.0]
    assert "old_only" not in updated
    assert report["n_units"] == 2
    assert set(report["removed_stale_columns"]) == {"old_only", "sf_group"}

    provenance = json.loads(
        (tmp_path / "updated_tuning_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["output_unit_feature_table_sha256"] == report[
        "output_unit_feature_table_sha256"
    ]
