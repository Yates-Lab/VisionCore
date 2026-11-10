"""Export 300 dpi raster checks and actual-size, vector-preserving print crops.

Run after build.py. Print the proof on Letter paper in landscape orientation at
100% / Actual size; its one-inch calibration line checks printer scaling.
"""
import json
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent


def main():
    out = HERE / "output"
    regions = {r["name"]: r for r in json.loads(
        (HERE / "poster_content.json").read_text())["regions"]}
    crops = [
        ("panel_fem", "Panel 1: trials 1-5", (468, 169, 1180, 456)),
        ("panel_fem", "Panel 1: trials 6-10 and population response", (468, 456, 1180, 890)),
        ("panel2", "Panel 2: original trial order", (450, 85, 1165, 350)),
        ("panel2", "Panel 2: sorted trial order", (450, 365, 1165, 660)),
    ]
    with fitz.open(out / "poster_typst_full_60x36.pdf") as source, fitz.open() as proof:
        for key, filename in (("panel_fem", "panel1"), ("panel2", "panel2")):
            r = regions[key]
            clip = fitz.Rect(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])
            source[0].get_pixmap(clip=clip, dpi=300, alpha=False).save(
                out / f"{filename}_print_300dpi.png")
        for key, title, local in crops:
            r = regions[key]
            clip = fitz.Rect(local) + (r["x"], r["y"], r["x"], r["y"])
            page = proof.new_page(width=792, height=612)
            destination = fitz.Rect(40, 70, 40 + clip.width, 70 + clip.height)
            assert destination.width == clip.width and destination.height == clip.height
            assert page.rect.contains(destination)
            page.insert_text((40, 30), title, fontsize=15)
            page.insert_text((40, 49), "Actual poster size. Print at 100% / Actual size, without Fit to page.", fontsize=10)
            page.show_pdf_page(destination, source, 0, clip=clip)
            page.draw_line((40, 555), (112, 555), width=1)
            for x in (40, 112):
                page.draw_line((x, 551), (x, 559), width=1)
            page.insert_text((122, 558), "1 inch / 25.4 mm - check this after printing", fontsize=10)
        proof.save(out / "raster_print_proof.pdf", garbage=4, deflate=True, no_new_id=True)
    print("Created panel1_print_300dpi.png, panel2_print_300dpi.png and raster_print_proof.pdf")


if __name__ == "__main__":
    main()
