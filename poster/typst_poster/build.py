"""Build the poster from local Typst sources, scientific assets and saved data."""
import argparse
import os
import shutil
import subprocess
import sys

from fonts import HERE, checked_font_paths

# Fixed dates remove wall-clock metadata from Matplotlib and Typst exports.
SOURCE_DATE_EPOCH = 1794009600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', action='store_true',
                        help='Also render the bundled approved reference and a difference image')
    args = parser.parse_args()
    font_paths = sorted({str(p.parent) for p in checked_font_paths()})
    import fitz
    import typst
    from PIL import Image, ImageChops
    import json

    component = HERE / 'components/panel5'
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(SOURCE_DATE_EPOCH),
               PYTHONHASHSEED='0', MPLBACKEND='Agg')
    subprocess.run([sys.executable, str(component / 'render_plots.py')],
                   cwd=HERE, env=env, check=True, stdout=subprocess.DEVNULL)
    options = dict(font_paths=font_paths, ignore_system_fonts=True,
                   timestamp=SOURCE_DATE_EPOCH)
    typst.compile(str(component / 'panel5.typ'), output=str(component / 'panel5_pilot.pdf'),
                  root=str(component), **options)
    shutil.copyfile(component / 'panel5_pilot.pdf', HERE / 'assets/panel5_pilot.pdf')
    # Intro and panel 1 (gaze shifts within an RF): Matplotlib plots from frozen
    # data, intro text composed natively in Typst.
    fem = HERE / 'components/panel_fem'
    subprocess.run([sys.executable, str(fem / 'render_plots.py')],
                   cwd=HERE, env=env, check=True, stdout=subprocess.DEVNULL)
    typst.compile(str(fem / 'intro.typ'), output=str(HERE / 'assets/intro.pdf'),
                  root=str(HERE), **options)
    typst.compile(str(fem / 'panel_fem.typ'), output=str(HERE / 'assets/panel_fem.pdf'),
                  root=str(HERE), **options)
    out = HERE / 'output'
    out.mkdir(exist_ok=True)
    pdf = out / 'poster_typst_full_60x36.pdf'
    typst.compile(str(HERE / 'poster.typ'), output=str(pdf), root=str(HERE), **options)
    with fitz.open(pdf) as doc:
        if len(doc) != 1 or list(doc[0].rect) != [0, 0, 4320, 2592]:
            raise RuntimeError('Unexpected poster page dimensions')
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        pix.save(out / 'poster_typst_full.png')
    with Image.open(out / 'poster_typst_full.png') as image:
        image.resize((1728, 1037), Image.Resampling.LANCZOS).save(out / 'poster_typst_overview.png')
        content = json.loads((HERE / 'poster_content.json').read_text())
        for r in content['regions']:
            box = (r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height'])
            image.crop(tuple(round(v) for v in box)).save(out / f"{r['name']}.png")
        if args.compare:
            with fitz.open(HERE / 'reference/approved.pdf') as reference:
                pix = reference[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                ref = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            ref.save(out / 'approved_reference.png')
            ImageChops.difference(ref, image).save(out / 'difference.png')
    subprocess.run([sys.executable, str(HERE / 'audit_layout.py')], cwd=HERE, check=True)
    print(f'Built full poster with Typst {typst.__version__}: {pdf}')


if __name__ == '__main__':
    main()
