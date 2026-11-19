from pathlib import Path
from html.parser import HTMLParser
import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET

import pymupdf
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/architecture'


class Diagram(HTMLParser):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.equations = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'data-node' in attrs:
            self.nodes.append(attrs['data-node'])
        if 'data-equation' in attrs:
            self.equations[attrs['data-equation']] = attrs['data-source']


text = (ROOT / 'bvp-pinn-architecture.html').read_text()
diagram = Diagram()
diagram.feed(text)
expected = ['interior','normalize','network','fields','endpoints','endpoint-eval','parameters','derivatives','residuals','boundaries','boundary-loss','equation-loss','total-loss','gradients','optimizer','failed-update','monitor','lock','final-test','report','reference']
if diagram.nodes != expected:
    raise ValueError('The expected computational graph nodes are missing or duplicated.')
spec = importlib.util.spec_from_file_location('bvp_architecture_math', ROOT / 'tools/math_markup.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
canonical = module.equation_expressions((ROOT / 'docs/equations.md').read_text())
if diagram.equations != canonical:
    raise ValueError('A canonical residual or boundary expression differs.')
for formula in re.findall(r'<math\b.*?</math>', text, flags=re.S):
    ET.fromstring(formula)
if re.search(r'BVPFORMULA\d+TOKEN|PHYSICAL RUN BLOCKED|Pr = 6\.8', text):
    raise ValueError('The architecture contains stale or unrendered content.')
reader = PdfReader(CACHE / 'render.pdf')
if len(reader.pages) != 1:
    raise ValueError('The architecture should remain one content-sized vector sheet.')
writer = PdfWriter()
writer.append(reader)
writer.add_metadata({
    '/Title':'BVP PINN architecture and training methodology',
    '/Author':'Boundary BVP research project',
    '/Subject':'Lorenz-style PINN computational graph adapted to the confirmed five-field BVP',
    '/Creator':'Boundary BVP architecture builder',
})
output = ROOT / 'bvp-pinn-architecture.pdf'
with output.open('wb') as stream:
    writer.write(stream)
doc = pymupdf.open(output)
page = doc[0]
spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans']]
if any('\ufffd' in span['text'] or '\x00' in span['text'] for span in spans):
    raise ValueError('Missing glyph markers in PDF text.')
if any(not page.rect.contains(pymupdf.Rect(span['bbox'])) for span in spans):
    raise ValueError('Text extends beyond the page.')
svg = page.get_svg_image(text_as_path=True)
ET.fromstring(svg)
if '<image' in svg:
    raise ValueError('Unexpected raster image in vector SVG.')
(ROOT / 'bvp-pinn-architecture.svg').write_text(svg)
page.get_pixmap(matrix=pymupdf.Matrix(1500/page.rect.width,1500/page.rect.width),alpha=False).save(ROOT / 'bvp-pinn-architecture-preview.png')
# Crop fixed rendered sections without reflowing the final page.
sections = [('01-computational-graph',0,2550),('02-confirmed-math',2550,3600),('03-training-contract',3600,4450),('04-evaluation-contract',4450,page.rect.height)]
images = []
for name,top,bottom in sections:
    top = min(top,page.rect.height)
    bottom = min(bottom,page.rect.height)
    clip = pymupdf.Rect(0,top,page.rect.width,bottom)
    path = CACHE / f'{name}.png'
    page.get_pixmap(matrix=pymupdf.Matrix(2400/page.rect.width,2400/page.rect.width),clip=clip,alpha=False).save(path)
    images.append(str(path))
report = {'page_count':1,'size_points':list(page.rect),'minimum_text_points':min(s['size'] for s in spans if s['text'].strip()),'nodes':diagram.nodes,'canonical_expressions':len(diagram.equations),'svg_raster_images':0,'review_images':images,'pdf_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
(CACHE / 'check.json').write_text(json.dumps(report,indent=2))
print('ARCHITECTURE CHECK PASSED: 21 distinct graph nodes, 17 canonical expressions, valid MathML, no missing glyphs or out-of-page text, vector SVG.')
print(json.dumps(report,indent=2))
