"""Vector-only typography fixes for imported publication artwork."""
from pathlib import Path
import math
import re


def restore_svg_labels(source: Path, destination: Path) -> None:
    """Restore labeled Inkscape text outlines without changing the drawing.

    These seven paths retain their original text and font size in aria-label
    and style. Recover the baseline from the first DejaVu Sans glyph vertex.
    This makes the labels measurable in the final PDF font audit.
    """
    from lxml import etree
    from matplotlib.textpath import TextPath
    tree = etree.parse(str(source))
    for old in tree.xpath('//*[@aria-label]'):
        if etree.QName(old).localname != 'path':
            raise ValueError('Unexpected labeled SVG element')
        label = old.get('aria-label')
        style = old.get('style', '')
        size = float(re.search(r'font-size:([\d.]+)px', style).group(1))
        first = re.match(r'\s*[mM]\s*([-\d.eE]+)[ ,]+([-\d.eE]+)', old.get('d'))
        if first is None:
            raise ValueError('Cannot recover the text baseline')
        vx, vy = TextPath((0, 0), label, size=size).vertices[0]
        new = etree.Element('{http://www.w3.org/2000/svg}text')
        for key in ('id', 'transform', 'style'):
            if old.get(key):
                new.set(key, old.get(key))
        new.set('style', style + ';font-family:DejaVu Sans;stroke:none')
        new.set('x', str(float(first[1]) - vx))
        new.set('y', str(float(first[2]) + vy))
        new.text = label
        old.getparent().replace(old, new)
    tree.write(str(destination), encoding='UTF-8', xml_declaration=True)


def enlarge_pdf_text(source: Path, destination: Path, minimum: float) -> dict:
    """Re-typeset existing vector text at the same baselines, with a font floor.

    Used for imported supplementary panels whose generators were not exported.
    Raster images, vector plots, and text values remain unchanged. The output
    requires visual inspection for collisions after any increase in type size.
    """
    import pymupdf as fitz
    from matplotlib import font_manager
    document = fitz.open(source)
    n_changed = 0
    for page in document:
        lines = [line for block in page.get_text('dict')['blocks'] if block['type'] == 0
                 for line in block['lines']]
        spans = [(line, span) for line in lines for span in line['spans']]
        # Remove only text; leave underlying images and vector geometry intact.
        for _, span in spans:
            page.add_redact_annot(span['bbox'], fill=False)
        page.apply_redactions(images=0, graphics=0)
        for line, span in spans:
            bold = bool(span['flags'] & 16)
            italic = bool(span['flags'] & 2)
            prop = font_manager.FontProperties(family='DejaVu Sans',
                                               weight='bold' if bold else 'normal',
                                               style='oblique' if italic else 'normal')
            font = font_manager.findfont(prop)
            name = 'publication' + ('B' if bold else '') + ('I' if italic else '')
            page.insert_font(fontname=name, fontfile=font)
            angle = int(round(math.degrees(math.atan2(-line['dir'][1], line['dir'][0])))) % 360
            if angle not in (0, 90, 180, 270):
                raise ValueError(f'Unsupported text rotation: {angle}')
            rgb = fitz.sRGB_to_pdf(span['color'])
            size = max(minimum, span['size'])
            n_changed += span['size'] < minimum
            page.insert_text(span['origin'], span['text'], fontname=name,
                             fontsize=size, color=rgb, rotate=angle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination, garbage=4, deflate=True)
    return {'source': str(source), 'output': str(destination), 'minimum_source_pt': minimum,
            'spans_enlarged': n_changed}
