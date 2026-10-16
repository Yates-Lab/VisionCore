"""The Rowley canary training import path must not require DataYatesV1."""

import subprocess
import sys
import textwrap


def test_rowley_training_modules_and_transforms_do_not_import_datayates():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import torch

        class RejectDataYates(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "DataYatesV1" or fullname.startswith("DataYatesV1."):
                    raise ModuleNotFoundError(
                        "DataYatesV1 import attempted on the Rowley-only path"
                    )
                return None

        sys.meta_path.insert(0, RejectDataYates())

        from models.data.loading import remove_pixel_norm
        from models.data.transforms import make_pipeline
        from training.pl_modules.multidataset_dm import MultiDatasetDM

        config, removed = remove_pixel_norm({
            "transforms": {
                "stim": {
                    "source": "stim",
                    "ops": [{"pixelnorm": {}}, {"unsqueeze": 1}],
                }
            }
        })
        assert removed is True
        assert config["transforms"]["stim"]["ops"] == [{"unsqueeze": 1}]

        transform = make_pipeline([
            {"symlog": {}},
            {"temporal_basis": {
                "num_delta_funcs": 0,
                "num_cosine_funcs": 2,
                "history_bins": 4,
                "causal": True,
            }},
            {"splitrelu": {"split_dim": 1, "trainable_gain": False}},
        ])
        output = transform(torch.tensor([
            [0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, -2.0]
        ]))
        assert output.shape == (4, 8)
        assert MultiDatasetDM is not None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
