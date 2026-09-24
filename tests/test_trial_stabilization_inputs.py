import hashlib
import json
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TrialStabilizationInputTest(unittest.TestCase):
    def test_load_model_forwards_dataset_config_override(self):
        import eval.eval_stack_multidataset as stack

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            dataset_config = root / "datasets.yaml"
            dataset_config.write_text("sessions: []\n")
            calls = []

            class FakeLightningModel:
                names = []
                model = SimpleNamespace(activation=object())

                @classmethod
                def load_from_checkpoint(cls, path, **kwargs):
                    calls.append((path, kwargs))
                    return cls()

                def to(self, _device):
                    return self

                def eval(self):
                    return self

            with patch.object(stack, "MultiDatasetModel", FakeLightningModel), patch.object(
                stack.torch, "load", return_value={"state_dict": {"weight": torch.zeros(1)}}
            ):
                try:
                    stack.load_model(
                        checkpoint_path=checkpoint,
                        dataset_config_path=dataset_config,
                        device="cpu",
                        verbose=False,
                    )
                except TypeError as error:
                    self.fail(f"dataset config override is unavailable: {error}")

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["cfg_dir"], str(dataset_config.resolve()))

    def test_replay_inputs_use_selected_source_and_local_verified_overrides(self):
        from paper.fig3 import run_history_stabilization

        self.assertTrue(
            hasattr(run_history_stabilization, "resolve_replay_inputs"),
            "mixed-root replay input resolver is unavailable",
        )
        resolve_replay_inputs = run_history_stabilization.resolve_replay_inputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "external"
            local_root = root / "local"
            bundle = source_root / "outputs" / "selected"
            (bundle / "figure3").mkdir(parents=True)
            final_manifest = bundle / "FINAL_MANIFEST.json"
            checkpoint = source_root / "outputs" / "training" / "model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            final_manifest.write_text(
                json.dumps({"status": "complete", "checkpoint_sha256": checkpoint_sha256})
            )
            global_cache = bundle / "figure3" / "cache.pkl"
            global_cache.write_bytes(b"global")
            local_config = local_root / "datasets.yaml"
            local_config.parent.mkdir(parents=True)
            local_config.write_text("sessions: []\n")
            empirical = local_root / "cache"
            empirical.mkdir()
            for name in (
                "covdecomp_empirical.pkl",
                "covdecomp_derived.pkl",
                "covdecomp_aligned_sessions.pkl",
            ):
                (empirical / name).write_bytes(name.encode())

            run_manifest = {
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "dataset_config_sha256": hashlib.sha256(local_config.read_bytes()).hexdigest(),
                "environment": {
                    "FIG3_MODEL_CHECKPOINT": "/original/outputs/training/model.ckpt",
                    "FIG3_DATASET_CONFIGS": "/original/paper/configs/inaccessible.yaml",
                    "FIG3_ABLATION_CACHE_PATH": "/original/outputs/selected/figure3/cache.pkl",
                    "FIG3_COVDECOMP_CACHE_PATH": "/original/outputs/inaccessible/empirical.pkl",
                    "FIG3_COVDECOMP_DERIVED_CACHE_PATH": "/original/outputs/inaccessible/derived.pkl",
                    "COVDECOMP_ALIGNED_CACHE_PATH": "/original/outputs/inaccessible/aligned.pkl",
                },
            }
            (bundle / "figure3" / "run_manifest.json").write_text(json.dumps(run_manifest))
            selection = {
                "bundle": "outputs/selected",
                "checkpoint_sha256": run_manifest["checkpoint_sha256"],
                "final_manifest_sha256": hashlib.sha256(final_manifest.read_bytes()).hexdigest(),
            }

            manifest, environment, provenance = resolve_replay_inputs(
                selection,
                source_root=source_root,
                empirical_cache_dir=empirical,
                dataset_config_path=local_config,
            )

            self.assertEqual(manifest, run_manifest)
            self.assertEqual(environment["FIG3_MODEL_CHECKPOINT"], str(checkpoint))
            self.assertEqual(environment["FIG3_ABLATION_CACHE_PATH"], str(global_cache))
            self.assertEqual(environment["FIG3_DATASET_CONFIGS"], str(local_config.resolve()))
            self.assertEqual(
                environment["FIG3_COVDECOMP_CACHE_PATH"],
                str(empirical / "covdecomp_empirical.pkl"),
            )
            self.assertEqual(
                environment["FIG3_COVDECOMP_DERIVED_CACHE_PATH"],
                str(empirical / "covdecomp_derived.pkl"),
            )
            self.assertEqual(
                environment["COVDECOMP_ALIGNED_CACHE_PATH"],
                str(empirical / "covdecomp_aligned_sessions.pkl"),
            )
            self.assertEqual(provenance["source_root"], str(source_root.resolve()))
            self.assertEqual(
                provenance["dataset_config_sha256"],
                run_manifest["dataset_config_sha256"],
            )
            self.assertEqual(provenance["empirical_cache_dir"], str(empirical.resolve()))
            self.assertEqual(
                provenance["input_sha256"]["FIG3_ABLATION_CACHE_PATH"],
                hashlib.sha256(global_cache.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["input_sha256"]["FIG3_COVDECOMP_CACHE_PATH"],
                hashlib.sha256((empirical / "covdecomp_empirical.pkl").read_bytes()).hexdigest(),
            )

    def test_crossing_history_audit_uses_every_native_prediction_source_trial(self):
        from paper.fig3 import _fig3_data
        import numpy as np

        self.assertTrue(
            hasattr(_fig3_data, "history_crosses_trial_boundary"),
            "replay cannot audit native predictions whose histories cross trials",
        )
        trial_ids = np.array([0, 0, 0, 1, 1, 1, 2])
        # Two native predictions contribute to each 120-Hz endpoint.
        model_indices = np.array([1, 2, 3, 4])
        crossing = _fig3_data.history_crosses_trial_boundary(
            model_indices,
            np.array([0, 1]),
            trial_ids,
            n_endpoints=2,
        )

        np.testing.assert_array_equal(crossing, [False, True])

    def test_renderer_assembles_history_and_trial_controls_in_global_session_order(self):
        manuscript = Path(__file__).resolve().parents[1] / "manuscript"
        sys.path.insert(0, str(manuscript))
        import render_stabilization_control as render

        self.assertTrue(
            hasattr(render, "assemble_control_rows"),
            "renderer cannot assemble a separate trial cache",
        )
        identities = {
            "neuron_mask": [1, 2],
            "ccmax": [0.5, 0.6],
            "ccnorm_unstable": [False, False],
            "matched_var_y": [1.0, 2.0],
            "matched_n_windows": [10, 11],
            "n_base_windows": 12,
        }

        def row(session, reference):
            return {"session": session, "stabilization_reference": reference, **identities}

        global_payload = {
            "schema_version": 7,
            "checkpoint_path": "checkpoint",
            "complete": True,
            "results": [row("A", "session_global"), row("B", "session_global")],
        }
        history_payload = {
            "schema_version": 7,
            "checkpoint_path": "checkpoint",
            "complete": True,
            "results": [row("B", "history_endpoint"), row("A", "history_endpoint")],
        }
        trial_payload = {
            "schema_version": 7,
            "checkpoint_path": "checkpoint",
            "complete": True,
            "results": [row("B", "trial_centroid"), row("A", "trial_centroid")],
        }

        history, trial, checks = render.assemble_control_rows(
            global_payload, history_payload, trial_payload
        )

        self.assertEqual([record["session"] for record in history], ["A", "B"])
        self.assertEqual([record["session"] for record in trial], ["A", "B"])
        self.assertEqual(set(checks), {"history_endpoint", "trial_centroid"})
        self.assertTrue(all(all(values.values()) for control in checks.values() for values in control.values()))

    def test_renderer_rejects_trial_replay_drift_by_default(self):
        manuscript = ROOT / "manuscript"
        sys.path.insert(0, str(manuscript))
        import render_stabilization_control as render

        self.assertTrue(hasattr(render, "validate_trial_replay_acceptance"))
        self.assertTrue(hasattr(render, "replay_check_record"))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "trial.pkl"
            cache.write_bytes(b"trial")
            accepted = render.validate_trial_replay_acceptance(cache, None)
            self.assertIsNone(accepted)
            with self.assertRaisesRegex(AssertionError, "trial.*explainable_fraction/zeroed"):
                render.replay_check_record(
                    "trial", "explainable_fraction", "zeroed", .0067, .005, accepted
                )

    def test_renderer_accepts_only_the_matching_trial_cache_hash(self):
        manuscript = ROOT / "manuscript"
        sys.path.insert(0, str(manuscript))
        import render_stabilization_control as render

        self.assertTrue(hasattr(render, "validate_trial_replay_acceptance"))
        self.assertTrue(hasattr(render, "replay_check_record"))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "trial.pkl"
            cache.write_bytes(b"trial")
            expected = hashlib.sha256(cache.read_bytes()).hexdigest()
            accepted = render.validate_trial_replay_acceptance(cache, expected)
            record = render.replay_check_record(
                "trial", "explainable_fraction", "zeroed", .0067, .005, accepted
            )
            self.assertEqual(record["status"], "accepted")
            self.assertEqual(record["accepted_trial_replay_sha256"], expected)
            with self.assertRaisesRegex(AssertionError, "history.*explainable_fraction/zeroed"):
                render.replay_check_record(
                    "history", "explainable_fraction", "zeroed", .0067, .005, accepted
                )

    def test_renderer_refuses_a_wrong_trial_cache_hash(self):
        manuscript = ROOT / "manuscript"
        sys.path.insert(0, str(manuscript))
        import render_stabilization_control as render

        self.assertTrue(hasattr(render, "validate_trial_replay_acceptance"))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "trial.pkl"
            cache.write_bytes(b"trial")
            with self.assertRaisesRegex(ValueError, "does not match"):
                render.validate_trial_replay_acceptance(cache, "0" * 64)

    def test_renderer_computes_direct_adjacent_scope_contrasts(self):
        manuscript = Path(__file__).resolve().parents[1] / "manuscript"
        sys.path.insert(0, str(manuscript))
        import numpy as np
        import render_stabilization_control as render

        self.assertTrue(
            hasattr(render, "adjacent_scope_contrasts"),
            "renderer lacks directly paired adjacent-scope contrasts",
        )
        values = {
            "full": np.array([4.0, 8.0]),
            "history": np.array([3.0, 6.0]),
            "trial": np.array([2.0, 4.0]),
            "global": np.array([1.0, 2.0]),
        }
        calls = []

        def paired(first, second, sessions, mask):
            calls.append((first.copy(), second.copy(), sessions.copy(), mask.copy()))
            return {"median": float(np.median(second[mask] - first[mask]))}

        contrasts = render.adjacent_scope_contrasts(
            values,
            np.array(["A", "B"]),
            np.array([True, True]),
            paired,
        )

        self.assertEqual(
            list(contrasts),
            ["history_minus_full", "trial_minus_history", "global_minus_trial"],
        )
        self.assertEqual([record["median"] for record in contrasts.values()], [-1.5, -1.5, -1.5])
        np.testing.assert_array_equal(calls[1][0], values["history"])
        np.testing.assert_array_equal(calls[1][1], values["trial"])


if __name__ == "__main__":
    unittest.main()
