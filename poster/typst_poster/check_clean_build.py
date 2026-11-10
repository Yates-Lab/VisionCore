"""Build twice in a clean Git clone of only the poster sources, then validate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from fonts import HERE, checked_font_paths


def run(args, cwd, **kwargs):
    return subprocess.run([str(x) for x in args], cwd=cwd, check=True, **kwargs)


def require(condition, message):
    if not condition:
        raise SystemExit(f'FAIL: {message}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use only cached locked packages')
    args = parser.parse_args()
    checked_font_paths()
    repo = Path(run(['git', 'rev-parse', '--show-toplevel'], HERE,
                    capture_output=True, text=True).stdout.strip())
    relative = HERE.relative_to(repo)
    paths = run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
                 '--', 'poster/.gitignore', 'poster/.gitattributes', relative], repo, capture_output=True).stdout.split(b'\0')
    files = sorted({os.fsdecode(p) for p in paths if p} | {'.gitignore'})
    require(all('.fonts' not in Path(p).parts and '.venv' not in Path(p).parts
                and 'output' not in Path(p).parts for p in files), 'Generated/private files selected')
    env = dict(os.environ, PYTHONPATH='', PYTHONNOUSERSITE='1')
    with tempfile.TemporaryDirectory(prefix='visioncore-poster-clean-') as temp:
        temp = Path(temp)
        fixture = temp / 'fixture'
        fixture.mkdir()
        for name in files:
            source, target = repo / name, fixture / name
            require(source.is_file() and not source.is_symlink(), f'Non-portable input: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        run(['git', '-c', 'init.defaultBranch=poster-test', 'init', '-q'], fixture)
        run(['git', 'add', '.'], fixture)
        run(['git', '-c', 'user.name=Poster build test', '-c', 'user.email=poster-build-test@localhost',
             '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Poster isolation test fixture'], fixture)
        clone = temp / 'clean checkout'
        run(['git', 'clone', '-q', '--no-hardlinks', fixture, clone], temp)
        project = clone / relative
        require(not (project / 'assets/panel5_pilot.pdf').exists(), 'Generated panel leaked into clone')
        require(not (project / '.fonts').exists(), 'Proprietary fonts leaked into clone')
        missing = subprocess.run([sys.executable, str(project / 'build.py')], cwd=temp,
                                 env=env, capture_output=True, text=True)
        require(missing.returncode != 0 and 'Font dependency check failed' in missing.stderr,
                'Missing-font guard did not fail helpfully')
        bad_fonts = temp / 'bad fonts'
        bad_fonts.mkdir()
        (bad_fonts / 'arial.ttf').write_bytes(b'not the approved font')
        bad = subprocess.run([sys.executable, str(project / 'setup_fonts.py'), '--from-dir', str(bad_fonts)],
                             cwd=temp, env=env, capture_output=True, text=True)
        require(bad.returncode != 0 and 'missing or different font bytes' in bad.stderr,
                'Mismatched-font guard failed')
        sync = ['uv', 'sync', '--project', project, '--frozen']
        if args.offline:
            sync.append('--offline')
        run(sync, temp, env=env)
        python = project / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        run([python, project / 'setup_fonts.py', '--from-dir', HERE / '.fonts'], temp, env=env)
        run([python, project / 'build.py', '--compare'], temp, env=env)
        run([python, project / 'validate.py'], temp, env=env)
        run([python, project / 'make_print_proof.py'], temp, env=env)
        require(all((project / 'output' / name).is_file() for name in
                    ('raster_print_proof.pdf', 'panel1_print_300dpi.png', 'panel2_print_300dpi.png')),
                'Print inspection outputs missing from clean build')
        names = ['poster_typst_full_60x36.pdf', 'poster_typst_full.png',
                 'poster_typst_overview.png', 'poster_panels_actual_size.pdf']
        first = {name: hashlib.sha256((project / 'output' / name).read_bytes()).hexdigest() for name in names}
        run([python, project / 'build.py'], temp, env=env)
        second = {name: hashlib.sha256((project / 'output' / name).read_bytes()).hexdigest() for name in names}
        require(first == second, 'Repeated PDF/PNG exports are not byte-identical')
        require(not run(['git', 'status', '--porcelain'], clone, capture_output=True, text=True).stdout,
                'Build modified tracked files or created unignored output')
        report = dict(source_files=len(files), source_bytes=sum((repo / p).stat().st_size for p in files),
                      clean_clone=True, print_proof_generated=True,
                      fresh_locked_environment=True, missing_fonts_rejected=True,
                      mismatched_fonts_rejected=True, repeat_build_bytes_identical=True,
                      sha256=second, validation=json.loads((project / 'output/validation.json').read_text()))
    (HERE / 'output').mkdir(exist_ok=True)
    (HERE / 'output/clean_build_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: clean-clone build/validation and byte-identical repeated PDF/PNG exports')


if __name__ == '__main__':
    main()
