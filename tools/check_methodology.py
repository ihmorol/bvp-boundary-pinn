from pathlib import Path
from html.parser import HTMLParser
import argparse
import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET

import pymupdf
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/methodology"


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tasks = []
        self.equations = {}
        self.parameters = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-task" in attrs:
            self.tasks.append(attrs["data-task"])
        if "data-equation" in attrs:
            self.equations[attrs["data-equation"]] = attrs["data-source"]
        if attrs.get("class") == "parameter":
            self.parameters += 1


def check(html_path):
    text = html_path.read_text()
    inventory = Inventory()
    inventory.feed(text)
    expected = [f"{phase}.{task}" for phase in range(1, 11) for task in range(1, 6)]
    if inventory.tasks != expected:
        raise ValueError("The diagram does not contain fifty ordered small tasks.")
    if inventory.parameters != 22:
        raise ValueError("The diagram needs all 22 locked parameter values.")
    spec = importlib.util.spec_from_file_location("bvp_math_check", ROOT / "tools/math_markup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    canonical = module.equation_expressions((ROOT / "docs/equations.md").read_text())
    if inventory.equations != canonical:
        raise ValueError("A displayed equation source differs from the canonical record.")
    for fragment in re.findall(r"<math\b.*?</math>", text, flags=re.S):
        ET.fromstring(fragment)
    if re.search(r"BVPFORMULA\d+TOKEN|PHYSICAL RUN BLOCKED|Pr = 6\.8|7 missing values", text):
        raise ValueError("The diagram contains an unfilled formula or stale parameter status.")
    method = (ROOT / "docs/methodology.md").read_text()
    if "\u2014" in method:
        raise ValueError("The method contains an em dash.")
    print("CONTENT CHECK PASSED: 50 ordered tasks, 22 locked values, 17 canonical residuals and valid MathML.")


def prepare_pdf(source, output):
    reader = PdfReader(source)
    if len(reader.pages) != 1:
        raise ValueError("The methodology must remain one content-sized sheet.")
    writer = PdfWriter()
    writer.append(reader)
    writer.add_metadata({
        "/Title": "Five-field boundary-value problem: mathematical methodology",
        "/Author": "Boundary BVP research project",
        "/Subject": "Locked supervisor parameters and fifty small tasks for reference, PINN training, validation, testing and reporting",
        "/Creator": "Boundary BVP methodology builder",
    })
    with output.open("wb") as stream:
        writer.write(stream)
    print(f"Prepared {output.name} with complete metadata and vector page content.")


def render(pdf_path):
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    inventory = json.loads((CACHE / "content-inventory.json").read_text())
    spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"]]
    positions = []
    for number in range(1, 11):
        title = inventory["stages"][str(number)]
        headings = [span for span in spans if span["text"] == title and abs(span["size"] - 28.5) < 0.1]
        if len(headings) != 1:
            raise ValueError(f"Cannot uniquely locate phase {number} in the PDF.")
        positions.append(max(0, headings[0]["bbox"][1] - 15))
    if positions != sorted(positions):
        raise ValueError("PDF phases are not in reading order.")
    scale = 2400 / page.rect.width
    boundaries = [0, *positions, page.rect.height]
    images = []
    for index, (top, bottom) in enumerate(zip(boundaries, boundaries[1:])):
        clip = pymupdf.Rect(0, max(0, top - 5), page.rect.width, min(page.rect.height, bottom + 5))
        name = "00-equations-parameters.png" if index == 0 else f"{index:02d}-phase.png"
        target = CACHE / name
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
        pixmap.save(target)
        images.append(str(target))
    overview = ROOT / "bvp-methodology-preview.png"
    page.get_pixmap(matrix=pymupdf.Matrix(1200 / page.rect.width, 1200 / page.rect.width), alpha=False).save(overview)
    svg = ROOT / "bvp-methodology.svg"
    svg_text = page.get_svg_image(text_as_path=True)
    ET.fromstring(svg_text)
    if "<image" in svg_text:
        raise ValueError("The SVG contains an unexpected raster image.")
    svg.write_text(svg_text)
    bad = [span["text"] for span in spans if "\ufffd" in span["text"] or "\x00" in span["text"]]
    if bad:
        raise ValueError(f"PDF text contains missing glyph markers: {bad[:3]}")
    outside = [span for span in spans if not page.rect.contains(pymupdf.Rect(span["bbox"]))]
    if outside:
        raise ValueError("Text lies outside the PDF page.")
    report = {
        "page_count": len(doc), "size_points": list(page.rect),
        "minimum_text_points": min(span["size"] for span in spans if span["text"].strip()),
        "section_images": images, "preview": str(overview),
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    }
    (CACHE / "pdf-check.json").write_text(json.dumps(report, indent=2))
    print(f"PDF TEXT CHECK PASSED: one page, no missing-glyph markers, no out-of-page text; rendered {len(images)} sections.")
    print(json.dumps({key: value for key, value in report.items() if key != "section_images"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and render methodology documents. No physical model is run.")
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument("--pdf", type=Path, default=ROOT / "bvp-methodology.pdf")
    args = parser.parse_args()
    check(ROOT / "bvp-methodology.html")
    if args.source_pdf:
        prepare_pdf(args.source_pdf, args.pdf)
    if args.pdf.exists():
        render(args.pdf)
