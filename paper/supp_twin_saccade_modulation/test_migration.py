"""Relocatable input and cache contracts for the migrated analysis."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from paper.supp_twin_saccade_modulation.analysis import input_paths, validate_session_cache, ensure_empty_session_cache


class MigrationTests(unittest.TestCase):
    def test_input_paths_resolve_selected_bundle_without_temp_or_machine_specific_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / 'outputs/no_phase_readout_comparison_20260910/rank1/figure3'
            bundle.mkdir(parents=True)
            (bundle / 'run_manifest.json').write_text(json.dumps({'environment': {
                'FIG3_MODEL_CHECKPOINT': '/old/machine/outputs/no_phase_readout_comparison_20260910/training/model.ckpt',
                'FIG3_CACHE_PATH': '/old/machine/outputs/no_phase_readout_comparison_20260910/rank1/figure3/cache/fig3_model.pkl',
            }}))
            paths = input_paths(root, Path('/example/empirical'), Path('/example/data'))
            self.assertEqual(paths['checkpoint'], root / 'outputs/no_phase_readout_comparison_20260910/training/model.ckpt')
            self.assertEqual(paths['scores'], root / 'outputs/no_phase_readout_comparison_20260910/rank1/figure3/cache/fig3_model.pkl')
            self.assertEqual(paths['aligned'], Path('/example/empirical/covdecomp_aligned_sessions.pkl'))
            self.assertEqual(paths['data'], Path('/example/data'))

    def test_registry_guard_accepts_same_fixrsvp_file(self):
        from paper.supp_twin_saccade_modulation.compute import assert_registry_fixrsvp_path
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'processed/S/datasets/fixrsvp.dset'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'dataset')
            assert_registry_fixrsvp_path(SimpleNamespace(sess_dir=source.parent.parent),
                                         {'data': root / 'processed'}, 'S')

    def test_registry_guard_accepts_symlink_alias(self):
        from paper.supp_twin_saccade_modulation.compute import assert_registry_fixrsvp_path
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'processed/S/datasets/fixrsvp.dset'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'dataset')
            (root / 'alias').symlink_to(root / 'processed', target_is_directory=True)
            assert_registry_fixrsvp_path(SimpleNamespace(sess_dir=source.parent.parent),
                                         {'data': root / 'alias'}, 'S')

    def test_registry_guard_rejects_different_dataset_even_with_same_bytes(self):
        from paper.supp_twin_saccade_modulation.compute import assert_registry_fixrsvp_path
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registered = root / 'registered/S/datasets/fixrsvp.dset'
            requested = root / 'requested/S/datasets/fixrsvp.dset'
            for source in (registered, requested):
                source.parent.mkdir(parents=True)
                source.write_bytes(b'otherwise matching dataset')
            with self.assertRaisesRegex(ValueError, '--data-root.*registry'):
                assert_registry_fixrsvp_path(SimpleNamespace(sess_dir=registered.parent.parent),
                                             {'data': root / 'requested'}, 'S')

    def test_partial_cache_without_manifest_is_not_overwritten(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            ensure_empty_session_cache(folder)
            (folder / 'components.npz').write_bytes(b'partial')
            with self.assertRaisesRegex(AssertionError, 'partial'):
                ensure_empty_session_cache(folder)

    def test_relocated_cache_preserves_manifest_provenance_and_rejects_bad_inputs(self):
        from paper.supp_twin_saccade_modulation.analysis import sha256
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'components.npz').write_bytes(b'original components')
            (folder / 'results.json').write_text('{}')
            manifest = dict(inputs={'/old/checkpoint': 'abc', '/old/config': 'cfg'},
                            source_hashes={'fixrsvp': 'def'},
                            components_sha256=sha256(folder / 'components.npz'),
                            results_sha256=sha256(folder / 'results.json'))
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            valid = {'/new/checkpoint': 'abc', '/new/config': 'cfg'}
            validate_session_cache(folder, valid, {'fixrsvp': 'def'})
            self.assertEqual(json.loads((folder / 'manifest.json').read_text()), manifest)
            with self.assertRaisesRegex(AssertionError, 'mismatched'):
                validate_session_cache(folder, {'/new/checkpoint': 'wrong', '/new/config': 'cfg'}, {'fixrsvp': 'def'})
            with self.assertRaisesRegex(AssertionError, 'mismatched'):
                validate_session_cache(folder, {'/new/checkpoint': 'cfg', '/new/config': 'abc'}, {'fixrsvp': 'def'})
            (folder / 'components.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(AssertionError, 'mismatched'):
                validate_session_cache(folder, valid, {'fixrsvp': 'def'})
            (folder / 'components.npz').unlink()
            with self.assertRaisesRegex(AssertionError, 'mismatched'):
                validate_session_cache(folder, valid, {'fixrsvp': 'def'})
