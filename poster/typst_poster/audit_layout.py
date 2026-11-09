"""Geometric text audit of the final rendered PDF (heuristic, not design proof)."""
from pathlib import Path
import json
import fitz
HERE=Path(__file__).resolve().parent
content=json.loads((HERE/'poster_content.json').read_text())
doc=fitz.open(HERE/'output/poster_typst_full_60x36.pdf')
runs=[];overflow=[];overlaps=[]
regions=[(r['name'],fitz.Rect(r['x'],r['y'],r['x']+r['width'],r['y']+r['height'])) for r in content['regions']]
for t in doc[0].get_texttrace():
 chars=[c for c in t['chars'] if not chr(c[0]).isspace()]
 if not chars:continue
 text=''.join(chr(c[0]) for c in t['chars'])
 box=fitz.Rect(chars[0][3])
 for c in chars:
  box |= fitz.Rect(c[3])
  for name,region in regions:
   if fitz.Point(c[2]) in region and not region.contains(fitz.Rect(c[3])):
    overflow.append({'region':name,'character':chr(c[0]),'box':list(c[3])})
 runs.append((box,text))
for i,(a,ta) in enumerate(runs):
 for b,tb in runs[i+1:]:
  intersect=a&b
  if intersect.is_empty:continue
  fraction=intersect.get_area()/min(a.get_area(),b.get_area())
  if fraction>.15:overlaps.append({'first':ta,'second':tb,'fraction':fraction,'box':list(intersect)})
report={'text_runs_checked':len(runs),'regions_checked':len(regions),'characters_outside_own_region':overflow,'text_run_overlap_above_15_percent':overlaps,'scope':'PDF text bounding boxes. Does not detect text over data marks, establish reading-distance legibility, or replace visual inspection.'}
(HERE/'output/layout_audit.json').write_text(json.dumps(report,indent=2))
if overflow or overlaps:
 raise SystemExit(json.dumps(report, indent=2))
print(f'PASS: {len(runs)} text runs; no panel-boundary overflow or text overlap above the 15% screening threshold')
