"""Build the methodology and architecture diagrams for the five-field BVP.

Both figures follow the style of methodology-flowchart-a4.pdf: one A4 landscape
page, numbered boxes, dashed green groups, yellow decision diamonds and short
labelled arrows. Every node, arrow and line of text is placed by hand below.

The build stops with an error if a line of text does not fit its box, two boxes
overlap, an arrow runs through a box or the text breaks the writing rule.

Run:
    uv run --no-project --with fonttools --with pymupdf python -I tools/build_diagrams.py
    ... python -I tools/build_diagrams.py --only detailed      # the six detailed pages only
    ... python -I tools/build_diagrams.py --only compact       # the one-page compact methodology

The detailed methodology (six pages, steps 1 to 72) goes to docs/methodology/figures/detailed/ and
is merged into docs/methodology/figures/bvp-methodology-detailed.pdf.

Outputs for each figure: NAME.html (editable source with embedded fonts),
NAME.pdf (vector, one A4 page), NAME.svg (vector, text as outlines) and
NAME-preview.png.
"""

from pathlib import Path
from html import escape
import argparse
import base64
import io
import re
import subprocess
import tempfile

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1200, 848
FONT_DIR = Path("/usr/share/fonts/opentype/stix-word")
FONT_FILES = {
    (False, False): FONT_DIR / "STIX-Regular.otf",
    (True, False): FONT_DIR / "STIX-Bold.otf",
    (False, True): FONT_DIR / "STIX-Italic.otf",
    (True, True): FONT_DIR / "STIX-BoldItalic.otf",
}
BANNED = ["delve", "moreover", "furthermore", "underscore", "leverage", "utilize", "seamless",
          "robust", "comprehensive", "pivotal", "transformative", "tapestry", "synergy"]

INK = "#1f2328"
ARROW = "#3b4049"
GROUP = "#3d7a58"
TONES = {
    "peach": ("#fcebe1", "#e3a98a", "#b9532b"),
    "blue": ("#e9effb", "#8ea3dc", "#2f4a9a"),
    "purple": ("#f1ebfa", "#a993d6", "#6b4bb3"),
    "green": ("#e3f3ea", "#82c3a1", "#23735a"),
    "cyan": ("#e4f2f7", "#86bcd0", "#2e6f8e"),
    "gray": ("#f1f2f4", "#a4a9b1", "#4b5361"),
    "orange": ("#fef1e3", "#e2a256", "#c4691b"),
    "red": ("#fbe7e3", "#e2a092", "#b0443a"),
    "white": ("#ffffff", "#8ea3dc", "#2f4a9a"),
}
DIAMOND = ("#fdf3c7", "#d8b443")


# ---------------------------------------------------------------- text

class Metrics:
    def __init__(self):
        self.fonts = {}
        for key, path in FONT_FILES.items():
            font = TTFont(path)
            self.fonts[key] = (font.getBestCmap(), font["hmtx"], font["head"].unitsPerEm)

    def width(self, text, size, bold, italic):
        cmap, hmtx, upem = self.fonts[(bold, italic)]
        total = 0
        for char in text:
            glyph = cmap.get(ord(char))
            if glyph is None:
                raise SystemExit(f"glyph missing for {char!r} (U+{ord(char):04X})")
            total += hmtx[glyph][0]
        return total * size / upem


METRICS = None


def parse(markup, italic=False, shift=0):
    """Split markup into runs. *x* is italic, _{x} is a subscript, ^{x} a superscript.

    SVG collapses runs of spaces, so every space after the first becomes a no-break space: a gap
    written as three spaces is drawn as three spaces, as measured."""
    markup = re.sub(r" {2,}", lambda m: " " + " " * (len(m.group()) - 1), markup)
    runs, buffer, i = [], "", 0
    while i < len(markup):
        char = markup[i]
        if char == "*":
            if buffer:
                runs.append((buffer, italic, shift))
                buffer = ""
            italic = not italic
            i += 1
        elif char in "_^" and markup[i + 1:i + 2] == "{":
            if buffer:
                runs.append((buffer, italic, shift))
                buffer = ""
            depth, j = 1, i + 2
            while depth:
                depth += {"{": 1, "}": -1}.get(markup[j], 0)
                j += 1
            runs += parse(markup[i + 2:j - 1], italic, 1 if char == "_" else -1)
            i = j
        else:
            buffer += char
            i += 1
    if buffer:
        runs.append((buffer, italic, shift))
    return runs


def plain(markup):
    return "".join(run[0] for run in parse(markup))


def measure(markup, size, bold=False):
    return sum(METRICS.width(text, size * (0.72 if shift else 1), bold, italic)
               for text, italic, shift in parse(markup))


def text_svg(markup, x, y, size, bold=False, anchor="middle", fill=INK, extra=""):
    parts, offset = [], 0.0
    for text, italic, shift in parse(markup):
        target = {0: 0.0, 1: size * 0.28, -1: -size * 0.38}[shift]
        attrs = []
        if target != offset:
            attrs.append(f'dy="{target - offset:.2f}"')
            offset = target
        if italic:
            attrs.append('font-style="italic"')
        if shift:
            attrs.append(f'font-size="{size * 0.72:.2f}"')
        parts.append(f'<tspan {" ".join(attrs)}>{escape(text)}</tspan>' if attrs else escape(text))
    weight = ' font-weight="bold"' if bold else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}"{weight}{extra}>{"".join(parts)}</text>')


# ---------------------------------------------------------------- scene

class Scene:
    def __init__(self, name, title, caption):
        self.name, self.title, self.caption = name, title, caption
        self.layers = {"groups": [], "arrows": [], "nodes": [], "labels": []}
        self.nodes, self.label_boxes, self.paths, self.strings = [], [], [], []
        self.errors, self.groups, self.tabs = [], [], []

    # -- checks ----------------------------------------------------------
    def fit(self, markup, size, bold, room, where):
        self.strings.append(plain(markup))
        width = measure(markup, size, bold)
        if width > room:
            self.errors.append(f"{where}: '{plain(markup)}' is {width:.0f}px, room {room:.0f}px")

    def add_node(self, key, x, y, w, h, kind="rect", inner=False):
        if x < 8 or y < 8 or x + w > W - 8 or y + h > H - 8:
            self.errors.append(f"{key}: outside the page")
        self.nodes.append(dict(key=key, x=x, y=y, w=w, h=h, kind=kind, inner=inner))

    # -- shapes ----------------------------------------------------------
    def group(self, x, y, w, h, tab):
        tab = tab.upper()
        tw = measure(tab, 11.5, True) + 0.8 * len(tab) + 30
        self.strings.append(tab)
        self.layers["groups"].append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="none" stroke="{GROUP}" '
            f'stroke-width="1.6" stroke-dasharray="7 5"/>'
            f'<rect x="{x + 16}" y="{y - 12}" width="{tw:.1f}" height="24" rx="12" fill="#2f5d46"/>'
            + text_svg(tab, x + 16 + tw / 2, y + 4.2, 11.5, True, fill="#ffffff", extra=' letter-spacing="0.8"'))
        self.label_boxes.append((f"tab {tab}", x + 16, y - 12, tw, 24))
        self.groups.append((tab, x, y, w, h))
        self.tabs.append(dict(key=f"tab {tab}", x=x + 16, y=y - 12, w=tw, h=24, kind="rect", inner=False))

    def box(self, key, x, y, w, h, tone, number, title, lines=(), size=12.5, thick=False):
        fill, stroke, badge = TONES[tone]
        self.add_node(key, x, y, w, h)
        width = 2 if thick else 1.3
        out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>']
        if number is not None:
            out.append(f'<circle cx="{x + 20}" cy="{y + 20}" r="10.5" fill="{badge}"/>')
            out.append(text_svg(str(number), x + 20, y + 24.2, 11.5, True, fill="#ffffff"))
            self.strings.append(str(number))
        lines = list(lines)
        block = 14 + (21 + (len(lines) - 1) * 17 + 4 if lines else 4)
        top = y + (h - block) / 2 + 13
        if number is not None:
            top = max(top, y + 23)
        self.fit(title, 15, True, w - 2 * 40 if number is not None else w - 28, f"{key} title")
        out.append(text_svg(title, x + w / 2, top, 15, True))
        for i, line in enumerate(lines):
            self.fit(line, size, False, w - 28, f"{key} line {i + 1}")
            out.append(text_svg(line, x + w / 2, top + 21 + i * 17, size))
        if top + 21 + (len(lines) - 1) * 17 + 3 > y + h - 4:
            self.errors.append(f"{key}: text runs past the bottom edge")
        self.layers["nodes"].append("".join(out))

    def cell(self, key, x, y, w, h, title, line, stroke="#8ea3dc"):
        self.add_node(key, x, y, w, h, inner=True)
        self.fit(title, 12.5, True, w - 16, f"{key} title")
        self.fit(line, 11.5, False, w - 16, f"{key} line")
        self.layers["nodes"].append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="{stroke}" stroke-width="1.1"/>'
            + text_svg(title, x + w / 2, y + h / 2 - 3, 12.5, True)
            + text_svg(line, x + w / 2, y + h / 2 + 13, 11.5))

    def frame(self, key, x, y, w, h, tone, number, title):
        """A box whose body is drawn by separate cells."""
        fill, stroke, badge = TONES[tone]
        self.add_node(key, x, y, w, h)
        self.fit(title, 15, True, w - 80, f"{key} title")
        badge_svg = ""
        if number is not None:
            badge_svg = (f'<circle cx="{x + 20}" cy="{y + 20}" r="10.5" fill="{badge}"/>'
                         + text_svg(str(number), x + 20, y + 24.2, 11.5, True, fill="#ffffff"))
        self.layers["nodes"].append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
            + badge_svg + text_svg(title, x + w / 2, y + 26, 15, True))

    def diamond(self, key, cx, cy, w, h, title, lines=()):
        self.add_node(key, cx - w / 2, cy - h / 2, w, h, kind="diamond")
        fill, stroke = DIAMOND
        points = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
        out = [f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>']
        rows = [(title, 13.5, True)] + [(line, 11.5, False) for line in lines]
        heights = [16] + [14] * len(lines)
        y = cy - sum(heights) / 2 + 12
        for (markup, size, bold), step in zip(rows, heights):
            # usable half-width shrinks linearly with distance from the centre line
            far = max(abs(y - size * 0.75 - cy), abs(y + size * 0.2 - cy))
            room = 2 * (w / 2) * (1 - far / (h / 2)) - 14
            self.fit(markup, size, bold, room, f"{key} '{plain(markup)}'")
            out.append(text_svg(markup, cx, y, size, bold))
            y += step
        self.layers["nodes"].append("".join(out))

    def arrow(self, points, dashed=False, inner=False, color=ARROW):
        points = [tuple(map(float, p)) for p in points]
        (x0, y0), (x1, y1) = points[-2], points[-1]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        trimmed = points[:-1] + [(x1 - (x1 - x0) / length * 4, y1 - (y1 - y0) / length * 4)]
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in trimmed)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        marker = "head-green" if color == GROUP else "head"
        self.layers["arrows"].append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#{marker})"/>')
        self.paths.append((points, inner))

    def label(self, markup, x, y, anchor="middle", bold=False, size=10.5, rotate=False, color=ARROW, italic_note=False):
        self.strings.append(plain(markup))
        width = measure(markup, size, bold)
        extra = f' transform="rotate(-90 {x} {y})"' if rotate else ""
        self.layers["labels"].append(text_svg(markup, x, y, size, bold, anchor, color, extra))
        left = {"middle": x - width / 2, "start": x, "end": x - width}[anchor]
        if rotate:
            self.label_boxes.append((plain(markup), x - size * 0.8, y - width / 2, size, width))
        else:
            self.label_boxes.append((plain(markup), left, y - size * 0.75, width, size))

    # -- validation ------------------------------------------------------
    @staticmethod
    def inside(node, px, py, pad=1.5):
        if node["kind"] == "diamond":
            cx, cy = node["x"] + node["w"] / 2, node["y"] + node["h"] / 2
            return abs(px - cx) / (node["w"] / 2) + abs(py - cy) / (node["h"] / 2) < 1 - pad / min(node["w"], node["h"]) * 2
        return (node["x"] + pad < px < node["x"] + node["w"] - pad
                and node["y"] + pad < py < node["y"] + node["h"] - pad)

    def validate(self):
        outer = [n for n in self.nodes if not n["inner"]]
        for i, a in enumerate(outer):
            for b in outer[i + 1:]:
                if (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
                        and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]):
                    self.errors.append(f"{a['key']} overlaps {b['key']}")
        for points, inner in self.paths:
            if inner:
                continue
            samples = []
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) / 2))
                samples += [(x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps) for k in range(steps + 1)]
            for px, py in samples[2:-2]:
                for node in outer + self.tabs:
                    if self.inside(node, px, py):
                        self.errors.append(f"arrow {points[0]}->{points[-1]} crosses {node['key']}")
                        break
                else:
                    continue
                break
        for name, x, y, w, h in self.label_boxes:
            if name.startswith("tab"):
                continue
            for group, gx, gy, gw, gh in self.groups:
                cut_x = any(x < edge < x + w for edge in (gx, gx + gw)) and y < gy + gh and gy < y + h
                cut_y = any(y < edge < y + h for edge in (gy, gy + gh)) and x < gx + gw and gx < x + w
                if cut_x or cut_y:
                    self.errors.append(f"label '{name}' sits on the border of group {group}")
            for node in outer:
                if node["kind"] == "diamond":
                    probes = [(x + w * i / 4, y + h * j / 2) for i in range(5) for j in range(3)]
                    hit = any(self.inside(node, px, py, pad=0) for px, py in probes)
                else:
                    hit = x < node["x"] + node["w"] and node["x"] < x + w and y < node["y"] + node["h"] and node["y"] < y + h
                if hit:
                    self.errors.append(f"label '{name}' overlaps {node['key']}")
        text = " ".join(self.strings + [self.title, self.caption]).lower()
        for word in BANNED:
            if re.search(rf"\b{word}", text):
                self.errors.append(f"banned word: {word}")
        if "—" in text or "–" in text:
            self.errors.append("dash character found; use a period, comma or colon")
        if self.errors:
            raise SystemExit(f"{self.name}: layout check failed\n  " + "\n  ".join(self.errors))

    # -- output ----------------------------------------------------------
    def svg(self):
        defs = (f'<defs>'
                f'<marker id="head" viewBox="0 0 10 10" refX="5.6" refY="5" markerWidth="9" markerHeight="9" '
                f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker>'
                f'<marker id="head-green" viewBox="0 0 10 10" refX="5.6" refY="5" markerWidth="9" markerHeight="9" '
                f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{GROUP}"/></marker>'
                f'</defs>')
        body = "".join(self.layers["groups"] + self.layers["arrows"] + self.layers["nodes"] + self.layers["labels"])
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
                f'aria-label="{escape(self.caption)}" font-family="Diagram Serif, STIX, serif">'
                f'<rect width="{W}" height="{H}" fill="#ffffff"/>{defs}{body}</svg>')


# ---------------------------------------------------------------- figures

def methodology():
    s = Scene("bvp-methodology", "BVP PINN methodology",
              "Methodology for the five-field boundary problem in four gated stages: fix the case and "
              "pass the software tests, build a classical reference for checking only, train four arms "
              "for five seeds each, then test against the reference and report every outcome.")

    # stage 1: case and code
    s.group(46, 24, 1108, 136, "1 · Case and code")
    s.box("case", 66, 46, 250, 76, "peach", 1, "Locked case", [
        "Eqs. 14 to 19 for *f*, *g*, *θ*, *φ*, *χ*", "22 supervisor values, steady form"], size=12)
    s.box("sizes", 340, 46, 250, 76, "purple", 2, "Wall coefficient sizes", [
        "*q* = *h* = 0.2 at *η* = 0: terms reach 250", "fixed residual scales from *η* only"], size=12)
    s.box("tests", 614, 46, 250, 76, "cyan", 3, "Software tests", [
        "pytest: equations against SymPy,", "derivatives, BCs, optimizers"], size=12)
    s.diamond("pass", 980, 84, 200, 76, "All tests pass?", ["every one"])
    s.arrow([(316, 84), (340, 84)])
    s.arrow([(590, 84), (614, 84)])
    s.arrow([(864, 84), (880, 84)])
    s.arrow([(980, 122), (980, 142), (739, 142), (739, 122)])
    s.label("NO: fix the code", 860, 138)
    s.arrow([(1080, 84), (1112, 84), (1112, 178), (290, 178), (290, 218)])
    s.label("YES", 1096, 77, bold=True)

    # stage 2: reference, used only for checking
    s.group(46, 196, 1108, 134, "2 · Reference")
    s.box("system", 66, 218, 250, 76, "cyan", 4, "First-order system", [
        "12 states, same 22 values", "SciPy solve_bvp, mesh dense at the wall"], size=12)
    s.box("converge", 340, 218, 250, 76, "cyan", 5, "Convergence checks", [
        "tolerance 10^{−5} to 10^{−8}, three solvers", "*L* = 20 to 10 240, branch search"], size=12)
    s.diamond("converged", 710, 256, 200, 76, "Converged?", ["every check agrees"])
    s.box("curves", 850, 212, 284, 46, "cyan", None, "Reference curves", [
        "check only, never a training label"], size=12)
    s.box("unverified", 850, 266, 284, 46, "red", None, "Unverified reference", [
        "PINN may still run; no accuracy claim"], size=12)
    s.arrow([(316, 256), (340, 256)])
    s.arrow([(590, 256), (610, 256)])
    s.arrow([(810, 256), (830, 256), (830, 235), (850, 235)])
    s.label("YES", 818, 226, bold=True)
    s.arrow([(710, 294), (710, 304), (850, 304)])
    s.label("NO", 722, 320, anchor="start", bold=True)
    s.arrow([(290, 330), (290, 388)])
    s.label("then train", 298, 352, anchor="start")

    # stage 3: training, per seed
    s.group(46, 366, 1108, 300, "3 · Train 4 arms, 5 seeds")
    s.box("points", 66, 388, 252, 76, "blue", 6, "Point sets", [
        "2000 points, stretched to the wall", "monitor set kept apart"], size=12)
    s.box("net", 342, 388, 252, 76, "purple", 7, "Network and loss", [
        "*η* → 4 tanh layers of 40 → 5 fields", "B1 to B12 built in; loss R14 to R18"], size=12)
    s.box("adam", 618, 388, 252, 76, "green", 8, "Adam training", [
        "rate 10^{−3} to 10^{−4}, 20000 steps", "smooth start profile, log scale"], size=12)
    s.box("lbfgs", 894, 388, 240, 76, "green", 9, "Refine: L-BFGS, LM", [
        "weights frozen; Levenberg-Marquardt", "each stage kept only if it helps"], size=12)
    s.arrow([(318, 426), (342, 426)])
    s.arrow([(594, 426), (618, 426)])
    s.arrow([(870, 426), (894, 426)])
    s.diamond("finite", 1014, 532, 220, 76, "All values finite?", ["in every stage"])
    s.arrow([(1014, 464), (1014, 494)])
    s.box("pick", 618, 494, 252, 76, "blue", 10, "Accept or reject", [
        "Newton-step error estimate, tail", "never the reference; rejects stay"], size=12)
    s.arrow([(904, 532), (870, 532)])
    s.label("YES", 887, 524, bold=True)
    s.box("stop", 894, 596, 240, 52, "red", None, "Stop and save", [
        "first bad term; last good checkpoint kept"], size=12)
    s.arrow([(1014, 570), (1014, 596)])
    s.label("NO", 1024, 587, anchor="start", bold=True)
    s.arrow([(618, 532), (192, 532), (192, 464)], dashed=True, color=GROUP)
    s.label("next run", 405, 525, color=GROUP, bold=True)

    # stage 4: test and report
    s.arrow([(744, 570), (744, 680), (290, 680), (290, 730)])
    s.label("after 20 runs", 752, 600, anchor="start")
    s.arrow([(1134, 235), (1174, 235), (1174, 694), (330, 694), (330, 730)])
    s.label("reference curves, check only", 1000, 689)
    s.group(46, 708, 1108, 124, "4 · Test and report")
    s.box("final", 66, 730, 290, 76, "gray", 11, "Test against the reference", [
        "fields, wall values, every R and B", "time against solve_bvp at equal error"], size=12)
    s.diamond("targets", 486, 768, 220, 76, "Same in all 5 seeds?", ["rule G6 for every claim"])
    s.box("sense", 624, 730, 240, 76, "gray", 12, "Compare the arms", [
        "B1 to B3 against B0, paired seeds", "retention per window, cost"], size=12)
    s.box("report", 888, 730, 246, 76, "orange", 13, "Report", [
        "fields, wall quantities, errors, cost", "or the failure with its cause"], size=12, thick=True)
    s.arrow([(356, 768), (376, 768)])
    s.arrow([(596, 768), (624, 768)])
    s.label("YES", 610, 760, bold=True)
    s.arrow([(864, 768), (888, 768)])
    s.arrow([(486, 806), (486, 820), (1011, 820), (1011, 806)])
    s.label("NO: no consistent effect", 740, 816)
    return s


def architecture():
    s = Scene("bvp-pinn-architecture", "BVP PINN architecture",
              "One training step of the five-field PINN: points pass through one shared network, "
              "Taylor-mode jets give the derivatives, a boundary layer builds in all twelve conditions, "
              "the locked coefficients turn the fields into equation errors, and one scalar loss updates "
              "the weights. Checks and the classical reference stay outside the loss.")

    s.box("setup", 46, 16, 1108, 92, "peach", 1, "One network: equations in the loss, conditions in the output", [
        "the network maps a position *η* to five fields; all 12 boundary conditions are built into the output; "
        "the 22 coefficients are fixed numbers, never trained",
        "same loop as the Lorenz PINN; the bottom row lists the four things that change for this boundary problem"])

    s.group(46, 136, 1108, 432, "One training step, repeated")

    s.box("points", 66, 164, 196, 92, "blue", 2, "Input points", [
        "*N* interior points in (0, *L*)", "plus the two ends *η* = 0, *L*"])
    s.frame("net", 288, 164, 366, 92, "blue", 3, "Shared network")
    cells = [("log map", "log(1 + *η*/0.1)"), ("4 tanh layers", "40 units each"), ("linear", "40 to 5 outputs")]
    for i, (title, line) in enumerate(cells):
        x = 298 + i * 120
        s.cell(f"cell {title}", x, 196, 106, 50, title, line)
        if i:
            s.arrow([(x - 14, 221), (x, 221)], inner=True)
    s.box("fields", 680, 164, 250, 92, "purple", 4, "Fields and derivatives", [
        "*f* to *f* ^{(4)}; *g*, *θ*, *φ*, *χ* to second order",
        "Taylor jets, one forward pass"], size=12)
    s.arrow([(262, 210), (288, 210)])
    s.arrow([(654, 210), (680, 210)])

    s.box("bcs", 380, 296, 260, 72, "cyan", 6, "Boundary layer", [
        "B1 to B12 exact for any weights", "end values set fixed shapes"], size=12)
    s.box("case", 666, 296, 234, 72, "peach", 7, "Locked case", [
        "22 coefficients from the supervisor", "fixed: not trained, not tuned"], size=12)
    s.box("res", 926, 296, 208, 72, "purple", 5, "Equation errors", [
        "R14 to R18 at interior points", "original Eqs. 14 to 18"], size=12)
    s.arrow([(930, 210), (1030, 210), (1030, 296)])
    s.label("interior", 980, 203)
    s.arrow([(720, 256), (720, 276), (510, 276), (510, 296)])
    s.label("ends *η* = 0, *L*", 615, 270)
    s.arrow([(666, 332), (640, 332)])
    s.arrow([(900, 332), (926, 332)])

    s.box("update", 66, 296, 274, 72, "green", 9, "Backward pass and update", [
        "*G* = ∇_{*w*}*J* through all derivatives", "Adam, then L-BFGS and LM"], size=12)
    s.arrow([(300, 296), (300, 256)])
    s.label("new weights", 308, 282, anchor="start")

    s.box("loss", 380, 408, 754, 72, "green", 8, "One scalar loss", [
        "*J* = mean over points of Σ_{*i*} (*s*_{*i*}(*η*) *R*_{*i*})^{2}    with *i* = 14, …, 18",
        "fixed scales *s*(*η*) from *η* and the locked values; no reference data in the loss"])
    s.arrow([(1030, 368), (1030, 408)])

    s.diamond("finite", 203, 444, 230, 72, "All finite?", ["errors, loss, gradients"])
    s.arrow([(380, 444), (318, 444)])
    s.arrow([(203, 408), (203, 368)])
    s.label("YES", 211, 393, anchor="start", bold=True)
    s.arrow([(203, 480), (203, 524), (380, 524)])
    s.label("NO", 211, 500, anchor="start", bold=True)
    s.box("stop", 380, 498, 300, 52, "red", None, "Stop and save the failure", [
        "first non-finite term, nothing masked"], size=12)

    s.group(46, 600, 1108, 112, "Checks outside the loss")
    s.arrow([(66, 350), (56, 350), (56, 660), (80, 660)], dashed=True, color=GROUP)
    s.label("saved weights", 64, 584, anchor="start", color=GROUP, bold=True)
    s.box("monitor", 80, 624, 250, 72, "gray", 10, "Monitor points", [
        "separate set, never in a gradient", "a refine stage kept only if it helps"], size=12)
    s.box("pick", 356, 624, 264, 72, "blue", 11, "Accept or reject", [
        "Newton-step error estimate (C++)", "decaying tail; rejects stay in tables"], size=12)
    s.box("final", 646, 624, 254, 72, "gray", 12, "Final test", [
        "fields, wall values, every R and B", "errors against the reference"], size=12)
    s.box("ref", 926, 624, 208, 72, "cyan", 13, "Classical reference", [
        "solve_bvp, 12 first-order states", "comparison only, never a label"], size=12)
    s.arrow([(330, 660), (356, 660)])
    s.arrow([(620, 660), (646, 660)])
    s.arrow([(926, 660), (900, 660)])

    # bottom row: what changes from the Lorenz PINN
    s.frame("lorenz", 46, 732, 1108, 100, "gray", None, "Changed from the Lorenz PINN")
    changes = [("time *t* → position *η*", "one interval [0, *L*], no time windows"),
               ("exact initial value → 12 conditions", "six at each end, all built in"),
               ("first derivatives → up to fourth", "Taylor jets for *f* ^{(4)}"),
               ("3 outputs → 5 outputs", "*f*, *g*, *θ*, *φ*, *χ* from one network")]
    for i, (title, line) in enumerate(changes):
        s.cell(f"change {i + 1}", 60 + i * 274, 764, 258, 56, title, line, stroke="#a4a9b1")
    return s


# ---------------------------------------------------------------- detailed methodology, six pages
#
# The same page size, fonts and shapes as the one-page methodology, one page per part of the code.
# Steps are numbered 1 to 72 across the pages. Every equation is the reduced form with the locked
# values put in (docs/problem/simplified-equations.md); the numbers were checked against
# src/bvp_pinn/reference/equations.py at random states on 2026-11-26.

PAGES = 6


def row(s, y, h, items, x0=66, x1=1134, gap=24, arrows=True, size=12):
    """A row of boxes. items: (key, tone, number, title, lines, width or None); None shares the rest."""
    fixed = sum(item[5] for item in items if item[5])
    free = sum(1 for item in items if not item[5])
    rest = (x1 - x0 - gap * (len(items) - 1) - fixed) / max(free, 1)
    spans, x = [], x0
    for key, tone, number, title, lines, width in items:
        w = round(width or rest)
        s.box(key, round(x), y, w, h, tone, number, title, lines, size=size)
        spans.append((round(x), w))
        x += w + gap
    if arrows:
        for (xa, wa), (xb, _) in zip(spans, spans[1:]):
            s.arrow([(xa + wa, y + h / 2), (xb, y + h / 2)])
    return spans


def page_label(s, k, topic):
    s.label(f"Detailed methodology · page {k} of {PAGES} · {topic}", 1154, 843, anchor="end", size=10)


def detail_problem():
    s = Scene("bvp-detail-1-problem", "Detailed methodology 1: the fixed problem",
              "Page 1 of the detailed methodology: the locked case, Eqs. 14 to 18 and the twelve boundary "
              "conditions with the supervisor values put in, and the one transcription every solver uses.")
    s.group(46, 24, 1108, 120, "1 · Locked case")
    row(s, 46, 76, [
        ("values", "peach", 1, "Locked values", [
            "22 supervisor values; case file SHA-256 pinned", "read-only; checked before every solve and run"], None),
        ("subs", "peach", 2, "Substitutions", [
            "*q* = *h* = *η* + 0.2,    *A* = 1 − 0.8*θ*", "*δ*^{∗} = 0: every unsteady term drops"], None),
        ("domain", "peach", 3, "Unknowns and domain", [
            "*f*, *g*, *θ*, *φ*, *χ* on 0 ≤ *η* ≤ *L* = 20", "*η* → ∞ is replaced by conditions at *L*"], None)])
    s.arrow([(600, 122), (600, 198)])

    s.group(46, 176, 1108, 236, "2 · Equations")
    row(s, 198, 92, [
        ("r14", "purple", None, "Eq. 14 · momentum, fourth order in *f*", [
            "1.5(*f*^{(4)} + 2*f*‴/*q* − *f*″/*q*^{2} + *f*′/*q*^{3}) + 0.2(*ff*‴ − *f*′*f*″)/*q*",
            "+ 0.2(*ff*″ − *f*′^{2})/*q*^{2} − 0.2*ff*′/*q*^{3} − 5(*f*″ + *f*′/*q*)",
            "− 0.5(*g*″ + *g*′/*q*) − 0.4(*θ*′ − 3*θ*/*q*)/*q*^{4} = 0"], 640),
        ("r15", "purple", None, "Eq. 15 · micro-rotation *g*", [
            "1.25(*g*″ + *g*′/*q*) − 0.25(*f*″ + *f*′/*q* + 2*g*)",
            "− 0.2(*f*′*g* − *fg*′)/*q* = 0"], None)], arrows=False)
    row(s, 314, 76, [
        ("r16", "purple", None, "Eq. 16 · temperature *θ*, times Pr = 8.8", [
            "(1 + 0.5*A*^{3})(*θ*″ + *θ*′/*q*) + (0.3 − 1.2*A*^{2})*θ*′^{2} + 0.1*θ*′*φ*′",
            "+ 1.76*fθ*′/*q* + 0.2(*θ* − 0.1)(0.2*f*(1 − 2/*q*^{2}) − 2*f*′)/*q*^{4} = 0"], 560),
        ("r17", "purple", None, "Eq. 17 · *φ*, times 8.8", [
            "*φ*″ + *φ*′/*q* + 3(*θ*″ + *θ*′/*q*)", "+ 1.76*fφ*′/*q* = 0"], 232),
        ("r18", "purple", None, "Eq. 18 · *χ*", [
            "*χ*″ + *χ*′/*q* + 0.2*fφ*/*q*", "+ (5/22)(*χφ*″ + *χ*′*φ*′) = 0"], None)], arrows=False)
    s.arrow([(420, 390), (420, 466)])

    s.group(46, 444, 1108, 236, "3 · Conditions")
    row(s, 466, 92, [
        ("wall", "cyan", None, "B1 to B6 at the wall, *η* = 0", [
            "B1: 1.5*f*′ = 1 + 0.1*f*″    B2: *f* = *φ*′/44    B3: *g* = −0.3*f*″",
            "B4: −*θ*′ = 0.998 − *θ* + 1.25(1 − 0.8*θ*)^{4}",
            "B5: *φ*′ = −3*θ*′    B6: *χ* = 1"], 520),
        ("far", "cyan", None, "B7 to B12 at the far end, *η* = *L* = 20", [
            "B7: *f*′ = 0    B8: *f*″ = 0    B9: *g* = 0    B12: *χ* = 0",
            "B10: *θ*′ = −*mθ*/20.2    B11: *φ*′ = −(*m*_{p}*φ* − 3*mθ*)/20.2",
            "*m* = 1.76*f*(*L*)/(1 + 0.5*A*^{3}),    *m*_{p} = 1.76*f*(*L*)"], None)], arrows=False)
    row(s, 582, 76, [
        ("kind", "gray", None, "Linear or not", [
            "linear: B1 to B3, B5 to B9, B12", "nonlinear: B4; B10, B11 in the form at *L*"], None),
        ("pinn", "gray", None, "In the PINN", [
            "all 12 exact for any weights (hard)", "the loss holds only R14 to R18"], None),
        ("ref", "gray", None, "In the reference", [
            "all 12 as boundary residuals", "B10, B11 keep the decaying modes, *m* > 0"], None)], arrows=False)
    s.arrow([(300, 658), (300, 734)])

    s.group(46, 712, 1108, 120, "4 · Solvers")
    row(s, 734, 76, [
        ("shared", "blue", 4, "One transcription", [
            "equations.py writes Eqs. 14 to 18 and B1 to B12 once,", "for NumPy (reference) and PyTorch (PINN)"], None),
        ("guards", "blue", 5, "Guards", [
            "a SymPy copy and the C++ engine test it; a run refuses", "other values, unknown settings and any other *L*"], None)])
    page_label(s, 1, "the fixed problem")
    return s


def detail_reference():
    s = Scene("bvp-detail-2-reference", "Detailed methodology 2: the classical reference",
              "Page 2 of the detailed methodology: the reference solve with SciPy solve_bvp, its checks at "
              "L = 20, the export, and the independent solvers and error bars.")
    s.group(46, 24, 1108, 220, "5 · Solve the reference")
    row(s, 46, 76, [
        ("first", "cyan", 6, "First-order form", [
            "12 states *f*, *f*′, *f*″, *f*‴, *g*, *g*′, *θ*, *θ*′, *φ*, *φ*′, *χ*, *χ*′",
            "*MH* = −*r*_{0} gives *H* = (*f*^{(4)}, *g*″, *θ*″, *φ*″, *χ*″)"], None),
        ("mesh", "cyan", 7, "Mesh and start", [
            "400 nodes, first step 10^{−3}, dense at the wall",
            "start *c* + *d*e^{−*rη*}: meets B1 to B3, B5, B6"], None),
        ("scan", "cyan", 8, "Branch scan", [
            "*L* = 10, tol 10^{−5}: 4 rates × 12 *θ*(0) = 48 starts",
            "grouped by wall values: branches A and B"], None)])
    s.arrow([(964, 122), (964, 134), (236, 134), (236, 146)])
    row(s, 146, 76, [
        ("tol", "cyan", 9, "Tolerance ladder", [
            "tol 10^{−5} → 10^{−8}, bc_tol = tol, ≤ 10^{5} nodes", "V4: the two tightest agree to 10^{−4}"], None),
        ("lad", "cyan", 10, "Domain ladder", [
            "*L* = 10, 20, 40, 80, 160, each from the last", "V5: change to next *L*; zero-at-*L* ladder for figures"], None),
        ("decay", "cyan", 11, "Far decay", [
            "*m* = 1.76*f*(*L*)/(1 + 0.5*A*_{*L*}^{3}) > 0",
            "A: *m* = 0.239, decays;   B: *m* < 0, no decay"], None)])
    s.arrow([(964, 222), (964, 234), (300, 234), (300, 298)])

    s.group(46, 276, 1108, 284, "6 · Checks at L = 20")
    row(s, 298, 76, [
        ("bc", "cyan", 12, "Conditions", ["V7: max |B_{*i*}| ≤ 10^{−8}", "V9: (1 + 0.5*A*^{3})/8.8 > 0"], None),
        ("core", "cyan", 13, "Core, then *χ*", ["V8: ten states without *χ*,", "then *χ* as a linear problem"], None),
        ("rk", "cyan", 14, "Runge-Kutta shooting", ["V10: 40 segments, DOP853,", "rtol 10^{−11}, Newton on joins"], None),
        ("slope", "cyan", 15, "Slope check", ["V12: spline slope of each curve", "equals the next saved state"], None)])
    s.diamond("g3", 1009, 436, 210, 72, "All pass?", ["gate G3, fixed *L*"])
    s.arrow([(1009, 374), (1009, 400)])
    s.box("export", 620, 398, 260, 76, "cyan", 16, "Export", [
        "4 001 points, first step 5 × 10^{−4}", "*z*, *H*, meta, SHA-256; study, figures"], size=12)
    s.arrow([(904, 436), (880, 436)])
    s.label("YES", 892, 428, bold=True)
    s.box("refA", 66, 398, 340, 76, "orange", 17, "Reference A, *L* = 20", [
        "*θ*(0) = 0.25019,  *θ*′(0) = −1.25944,  *f*″(0) = −4.20330",
        "*φ*(0) = −0.12015,  *φ*′(0) = 3.77831,  *g*′(0) = −2.87130"], size=12)
    s.arrow([(620, 436), (406, 436)])
    s.box("noref", 884, 492, 250, 52, "red", None, "Not a target", ["no accuracy claim against it"], size=12)
    s.arrow([(1009, 472), (1009, 492)])
    s.label("NO", 1019, 486, anchor="start", bold=True)
    s.arrow([(300, 474), (300, 614)])

    s.group(46, 592, 1108, 240, "7 · Cross-checks")
    row(s, 614, 92, [
        ("box", "gray", 18, "C++ box scheme", [
            "(*z*_{*j*+1} − *z*_{*j*})/Δ*η* = *F* at the midpoint", "Newton, banded LU, 5 mesh levels",
            "order 2; Richardson value"], None),
        ("defl", "gray", 19, "Branch search", [
            "deflation: ∏_{*i*}(‖*z* − *r*_{*i*}‖^{−2} + 1)", "52 starts, 105 Newton runs", "only branches A and B"], None),
        ("bars", "gray", 20, "Error bars", [
            "mesh: GCI of the 3 finest levels", "*L*: *v*(*L*) = *v*_{∞} + *cL*^{−*p*}",
            "two far forms, *L* = 20 to 10 240"], None),
        ("lstudy", "gray", 21, "L study", [
            "*L* = 8 to 160, both far forms", "*L* error at 20: 1.6 to 3.3 × 10^{−4}",
            "*φ*(0) 1.8 × 10^{−3};  *χ*′(0) no limit"], None)])
    s.arrow([(1009, 706), (1009, 734)])
    s.box("use", 600, 734, 534, 76, "orange", 22, "Reference for checking only", [
        "A is compared with the PINN after training, never a training label",
        "B only names the nearest branch of a run"], size=12, thick=True)
    page_label(s, 2, "the classical reference")
    return s


def detail_model():
    s = Scene("bvp-detail-3-model", "Detailed methodology 3: the PINN model",
              "Page 3 of the detailed methodology: points, input map, network and start profile; the "
              "boundary layer that builds in all twelve conditions in a fixed order; derivatives, residual "
              "rows, residual scales and the loss.")
    s.group(46, 24, 1108, 136, "8 · Network")
    row(s, 46, 92, [
        ("pts", "blue", 23, "Points", [
            "*η* = 20(e^{5*s*} − 1)/(e^{5} − 1)", "*s*: one random point per cell, by seed",
            "2 000 training, 1 000 monitor"], 254),
        ("map", "blue", 24, "Input map", [
            "*x* = 2 ln(1 + *η*/0.1)/ln 201 − 1", "spreads the wall layer", "B1: one map per window"], 232),
        ("mlp", "blue", 25, "Network", [
            "*x* → 4 tanh layers of 40 → 5", "float64; weights from the seed", "5 205; B1: 4 networks, 20 820"], 212),
        ("start", "blue", 26, "Start profile", [
            "*N* = network + start; last layer zero at start",
            "*f* = 0.0205 + 0.625(1 − e^{−*η*}),  *g* = 0.1875e^{−*η*}",
            "*θ* = 0.3e^{−*η*},  *φ* = −0.9e^{−*η*},  *χ* = e^{−*η*}"], None)])
    s.arrow([(985, 138), (985, 170), (300, 170), (300, 214)])

    s.group(46, 192, 1108, 220, "9 · Exact BCs")
    s.box("rule", 66, 214, 510, 76, "cyan", 27, "Correction rule", [
        "*u* = *N* + Σ_{*j*} *ψ*_{*j*}(*η*)(*t*_{*j*} − *C*_{*j*}[*N*]),    *C*_{*i*}[*ψ*_{*k*}] = *δ*_{*ik*}",
        "shapes *d*^{*n*}e^{−*d*/0.5},  *d* = *η* or 20 − *η*; kept values stay free"], size=12)
    s.box("chain", 600, 214, 534, 76, "cyan", 28, "Wall temperature first", [
        "*θ*_{0} = *N*_{*θ*}(0) kept;   B4: *θ*′(0) = −(1 − *θ*_{0}) − *Q*_{*w*}",
        "*Q*_{*w*} = (1 − *θ*_{0})(0.2 + *A*_{*w*})(0.04 + *A*_{*w*}^{2}),  *A*_{*w*} = 1 − 0.8*θ*_{0}"], size=12)
    s.arrow([(576, 252), (600, 252)])
    s.arrow([(867, 290), (867, 302), (191, 302), (191, 314)])
    row(s, 314, 76, [
        ("b52", "cyan", 29, "B5, then B2", ["*φ*′(0) = −3*θ*′(0)", "*f*(0) = *φ*′(0)/44"], None),
        ("bf", "cyan", 30, "Build *f*", ["B1: 1.5*f*′(0) − 0.1*f*″(0) = 1", "B7, B8: *f*′(20) = *f*″(20) = 0"], None),
        ("bfar", "cyan", 31, "Far *θ* and *φ*", ["*θ*(20), *φ*(20) kept; *f*(20) from *f*", "B10, B11 set *θ*′(20), *φ*′(20)"], None),
        ("bgc", "cyan", 32, "*g* and *χ*", ["B3: *g*(0) = −0.3*f*″(0);  B9: *g*(20) = 0", "B6: *χ*(0) = 1;  B12: *χ*(20) = 0"], None)])
    s.arrow([(1009, 390), (1009, 428), (260, 428), (260, 466)])

    s.group(46, 444, 1108, 136, "10 · Residuals")
    row(s, 466, 92, [
        ("jets", "purple", 33, "Taylor jets", [
            "orders 0 to 4 in one forward pass", "linear layer: *W* on every order",
            "tanh: chain rule to order 4"], 220),
        ("rows", "purple", 34, "Residual rows", [
            "each *R*_{*k*} over its top coefficient:", "1.5, 1.25, (1 + 0.5*A*^{3})/8.8,",
            "1/8.8, 1"], 226),
        ("scale", "purple", 35, "Residual scale *s*_{*k*}(*η*)", [
            "log: (*η* + 0.2)^{*n*}, *n* = 4, 2, 2, 2, 2",
            "balanced: *q*^{4}/(1 + 3.33*q*^{2}) for R14,",
            "*q*^{2}/(1 + 0.4*q*^{2}) for R15, *q*^{2} for the rest"], 300),
        ("loss", "green", 36, "Loss", [
            "*J* = mean over points of Σ_{*k*} (*s*_{*k*}*R*_{*k*})^{2}", "plus the interface term (B1)",
            "or the replay term (B2, B3)"], None)])
    s.arrow([(1022, 558), (1022, 594), (260, 594), (260, 638)])

    s.group(46, 616, 1108, 216, "One pass")
    row(s, 638, 60, [
        ("f1", "gray", None, "Training points", ["*η* in [0, 20), fixed by the seed"], None),
        ("f2", "gray", None, "Network + start", ["raw *N* and its jets"], None),
        ("f3", "gray", None, "End values", ["*N* at 0 and 20, orders 0 to 2"], None),
        ("f4", "gray", None, "Corrections", ["misses in the order 28 to 32"], None)])
    s.arrow([(1009, 698), (1009, 718), (236, 718), (236, 740)])
    row(s, 740, 60, [
        ("f5", "gray", None, "Built fields", ["*f* to *f*^{(4)}; *g*, *θ*, *φ*, *χ* to second order"], None),
        ("f6", "gray", None, "Residual rows", ["R14 to R18, scaled"], None),
        ("f7", "gray", None, "Scalar loss", ["*J*, then backward to the weights"], None)])
    page_label(s, 3, "the PINN model")
    return s


def detail_training():
    s = Scene("bvp-detail-4-training", "Detailed methodology 4: training one run",
              "Page 4 of the detailed methodology: the four arms, the Adam stage with gradient-norm "
              "weights, the refine chain of L-BFGS and Levenberg-Marquardt with frozen weights and "
              "rollback, the reference-free self check and the saved run folder.")
    s.group(46, 24, 1108, 120, "11 · Arms")
    row(s, 46, 76, [
        ("b0", "green", None, "B0 · one network", ["one network on [0, 20]", "one stage"], None),
        ("b1", "green", None, "B1 · window networks", ["4 networks on 0, 0.5, 2, 6, 20", "+ jumps of all 12 states at edges"], None),
        ("b2", "green", None, "B2 · physics replay", ["one network, windows added outward", "old windows keep their residual"], None),
        ("b3", "green", None, "B3 · self replay", ["as B2; old windows held to the", "network's own frozen outputs"], None)],
        arrows=False)
    s.arrow([(260, 122), (260, 198)])

    s.group(46, 176, 1108, 284, "12 · Adam")
    row(s, 198, 92, [
        ("plan", "green", 37, "Stage plan", [
            "B0, B1: one stage", "B2, B3: 4 stages, one per window", "stage *k*: residual on window *k*"], 210),
        ("adam", "green", 38, "Adam", [
            "rate 10^{−3} → 10^{−4}, exponential", "20 000 steps in total,", "split evenly over the stages"], 216),
        ("wts", "green", 39, "Loss weights", [
            "every 1 000 steps: *λ*_{*i*} ← 0.9*λ*_{*i*} + 0.1Σ_{*j*}*G*_{*j*}/*G*_{*i*}",
            "*G*_{*i*} = ‖∇*J*_{*i*}‖ over the weights; cap 1 000",
            "B0, B2, B3: one term, *λ* = 1"], 320),
        ("log", "green", 40, "Log and monitor", [
            "every 100 steps: each *R*, *B*,", "weight and monitor loss", "monitor points: no gradient"], None)])
    s.diamond("fin", 1009, 342, 210, 64, "Loss finite?", ["every step"])
    s.arrow([(1009, 290), (1009, 310)])
    s.box("ckpt", 600, 304, 280, 76, "green", 41, "Stage checkpoint", [
        "state after each stage: stage*k*.pt", "used later for the after-Adam test"], size=12)
    s.arrow([(904, 342), (880, 342)])
    s.label("YES", 892, 334, bold=True)
    s.box("halt", 884, 392, 250, 52, "red", None, "Stop and save", ["recorded as a result, never re-run"], size=12)
    s.arrow([(1009, 374), (1009, 392)])
    s.label("NO", 1019, 388, anchor="start", bold=True)
    s.arrow([(600, 342), (408, 342), (408, 290)], dashed=True, color=GROUP)
    s.label("next stage (B2, B3)", 504, 336, color=GROUP, bold=True)
    s.arrow([(740, 380), (740, 476), (260, 476), (260, 514)])
    s.label("after the last stage", 500, 472)

    s.group(46, 492, 1108, 340, "13 · Refine")
    row(s, 514, 92, [
        ("lbfgs", "green", 42, "L-BFGS, weights frozen", [
            "log scale, 5 000 iterations", "history 50, strong Wolfe search", "≤ 1.25 loss evaluations each"], 250),
        ("lm", "green", 43, "Levenberg-Marquardt", [
            "balanced scale, 260 steps", "(*K*^{T}*K* + *μI*)*h* = −*K*^{T}*r*,  *K* = ∂*r*/∂*w*",
            "one block per network; Woodbury for B1"], 330),
        ("mu", "green", 44, "Damping", [
            "*ρ* = (*J* − *J*_{new})/(*h*^{T}(*μh* − *K*^{T}*r*))",
            "*ρ* > 0: keep, *μ* ← *μ* max(1/3, 1 − (2*ρ* − 1)^{3})",
            "else *μ* ← *νμ*, *ν* ← 2*ν*;  start *μ* = 10^{−3} max diag"], None)])
    s.diamond("mon", 1009, 668, 230, 72, "Monitor loss lower?", ["after each refine stage"])
    s.arrow([(1009, 606), (1009, 632)])
    s.box("roll", 620, 642, 250, 52, "red", None, "Roll back that stage", ["its start state is restored"], size=12)
    s.arrow([(894, 668), (870, 668)])
    s.label("NO", 882, 660, bold=True)
    s.arrow([(1009, 704), (1009, 730)])
    s.label("YES", 1019, 722, anchor="start", bold=True)
    s.box("self", 600, 730, 534, 76, "blue", 45, "Self check, no reference", [
        "wall values; *m*, *m*_{p}; max |B_{*i*}|; share of points with |*R*| > 10^{−3}",
        "tail: *θ* ≈ *C*_{1} + *C*_{2}*q*^{−*m*} on [10, 20], tail share = |*C*_{1}|/max *θ*"], size=12)
    s.box("save", 66, 730, 460, 76, "gray", 46, "Run folder, named by the config hash", [
        "config, history, model.pt, stage*k*.pt, points", "meta: versions, device, GPU, seconds; written last"], size=12)
    s.arrow([(600, 768), (526, 768)])
    page_label(s, 4, "training one run")
    return s


def detail_evaluation():
    s = Scene("bvp-detail-5-evaluation", "Detailed methodology 5: acceptance and evaluation",
              "Page 5 of the detailed methodology: the acceptance rule that judges each run without the "
              "reference, the test of every run against reference A on the same grid, and the report.")
    s.group(46, 24, 1108, 204, "14 · Acceptance")
    row(s, 46, 76, [
        ("newton", "blue", 47, "Newton-step estimate", [
            "PINN states on 3 201 nodes → C++ engine", "one Newton step: *δz* ≈ PINN − solution"], None),
        ("score", "blue", 48, "Score", [
            "max |*δz*(0)|/|*z*(0)| over", "*f*″, *θ*, *θ*′, *φ*′ at the wall"], None),
        ("rule", "blue", 49, "Rule, set before the runs", [
            "finished;  *m* > 0 and *m*_{p} > 0;  tail ≤ 0.05", "max |B_{*i*}| ≤ 10^{−8};  score ≤ 10^{−3}"], None)])
    s.diamond("ok", 600, 172, 200, 64, "All hold?", ["each run"])
    s.arrow([(964, 122), (964, 132), (600, 132), (600, 140)])
    s.box("acc", 66, 142, 334, 60, "green", None, "Accepted", ["verdict written to selection.json"], size=12)
    s.arrow([(500, 172), (400, 172)])
    s.label("YES", 450, 165, bold=True)
    s.box("rej", 800, 142, 334, 60, "red", None, "Rejected or undecided", [
        "stays in every table; a line in failures.md", "never re-run for a better result"], size=12)
    s.arrow([(700, 172), (800, 172)])
    s.label("NO", 750, 165, bold=True)
    s.arrow([(600, 204), (600, 252), (300, 252), (300, 298)])
    s.label("every run, after all 20", 450, 247)

    s.group(46, 276, 1108, 240, "15 · Test")
    row(s, 298, 76, [
        ("load", "gray", 50, "Load the reference", ["hash check; A at *L* = 20, G3, *m* > 0", "run *L* must equal reference *L*"], None),
        ("grid", "gray", 51, "Same grid", ["the PINN on the 4 001 reference points", "17 curves: *z* and *H*"], None),
        ("l2", "gray", 52, "Field errors", ["(∫Δ*u*^{2}d*η*/∫*u*_{ref}^{2}d*η*)^{1/2}, trapezoid", "and max |Δ*u*|, per curve"], None),
        ("wallv", "gray", 53, "Wall values", ["*f*″, *θ*, *θ*′, *φ*, *φ*′, *g*′, *χ*′ at 0", "relative error; largest per run"], None)])
    s.arrow([(1009, 374), (1009, 386), (191, 386), (191, 414)])
    row(s, 414, 76, [
        ("rb", "gray", 54, "Equations, conditions", ["mean *R*_{*k*}^{2} on the grid; each |B_{*i*}|", "estimate against the true error"], None),
        ("near", "gray", 55, "Nearest branch", ["A or B by the *θ* curve", "none if not finite"], None),
        ("stage", "gray", 56, "Stages", ["error per window after each stage", "and after Adam, before refine"], None),
        ("time", "gray", 57, "solve_bvp timing", ["tol 10^{−3} to 10^{−8}, 5 repeats", "time to reach each arm's error"], None)])
    s.arrow([(1009, 490), (1009, 532), (191, 532), (191, 602)])

    s.group(46, 548, 1108, 284, "16 · Report")
    s.diamond("g6", 191, 640, 230, 76, "All 5 paired seeds?", ["rule G6, every claim"])
    s.box("claim", 340, 602, 250, 76, "orange", None, "Claim", ["arm X beats arm Y,", "with figure and run IDs"], size=12)
    s.arrow([(306, 640), (340, 640)])
    s.label("YES", 323, 632, bold=True)
    s.box("none", 66, 716, 250, 52, "red", None, "No consistent effect", ["reported as such"], size=12)
    s.arrow([(191, 678), (191, 716)])
    s.label("NO", 201, 700, anchor="start", bold=True)
    s.box("tabs", 614, 602, 520, 76, "orange", 58, "Tables", [
        "TB2 wall values;  TB3 accuracy after Adam and final, paired seeds",
        "TX1 cost;  summary.csv, failures.md, settings.json"], size=12, thick=True)
    s.box("figs", 614, 716, 520, 76, "orange", 59, "Figures", [
        "FB1 profiles, FB2 pointwise errors, FB3 *R* and *B* per arm",
        "FB4 training curves, FB5 retention per window, FX1 error against time"], size=12, thick=True)
    s.arrow([(590, 640), (614, 640)])
    s.arrow([(874, 678), (874, 716)])
    page_label(s, 5, "acceptance and evaluation")
    return s


def detail_run():
    s = Scene("bvp-detail-6-run", "Detailed methodology 6: the main run on Colab",
              "Page 6 of the detailed methodology: the Colab notebook and tools/main_run.py, which run the "
              "steps of pages 2 to 5 in order, keep everything in Google Drive and resume after a disconnect.")
    s.group(46, 24, 1108, 120, "17 · Notebook")
    row(s, 46, 76, [
        ("set", "peach", 60, "Settings", ["one cell: Drive folder, branch, device,", "arms, seeds, training settings"], None),
        ("gpu", "peach", 61, "GPU and Drive", ["nvidia-smi if present; mount Drive", "make the results folder"], None),
        ("code", "peach", 62, "Code", ["clone, fetch the branch; the commit", "is pinned by the first session"], None),
        ("pkgs", "peach", 63, "Packages", ["Python 3.12.13, versions of uv.lock", "torch 2.15.0: CUDA 13.2 or +cpu"], None)])
    s.arrow([(1009, 122), (1009, 152), (260, 152), (260, 198)])

    s.group(46, 176, 1108, 488, "18 · Main run")
    s.box("pin", 66, 198, 250, 76, "peach", 64, "Pins", ["first start: commit, settings hash", "settings into configs/pinn.json"], size=12)
    s.diamond("same", 456, 236, 230, 72, "Same code, settings?", ["as the first start"])
    s.arrow([(316, 236), (341, 236)])
    s.box("stopc", 614, 198, 250, 76, "red", None, "Stop", ["use the pinned commit and settings,", "or a new Drive folder"], size=12)
    s.arrow([(571, 236), (614, 236)])
    s.label("NO", 592, 228, bold=True)
    s.box("dev", 884, 198, 250, 76, "peach", 65, "Device", ["GPU if PyTorch sees one, else CPU", "runs at once: 1 on GPU, CPUs on CPU"], size=12)
    s.arrow([(456, 272), (456, 288), (1009, 288), (1009, 274)])
    s.label("YES", 466, 285, anchor="start", bold=True)
    s.arrow([(1134, 236), (1144, 236), (1144, 304), (191, 304), (191, 320)])
    s.diamond("done", 191, 352, 230, 64, "Step done before?", ["steps 66 to 69"])
    s.box("skip", 340, 326, 300, 52, "gray", None, "Skip it", ["the reference is restored from Drive"], size=12)
    s.arrow([(306, 352), (340, 352)])
    s.label("YES", 323, 344, bold=True)
    s.arrow([(191, 384), (191, 420)])
    s.label("NO", 201, 404, anchor="start", bold=True)
    row(s, 420, 92, [
        ("tests", "cyan", 66, "Tests", ["pytest, 64 tests", "record check", "once per folder"], None),
        ("refst", "cyan", 67, "Reference", ["steps 6 to 22 from scratch", "stop unless A is a valid target", "copied to Drive"], None),
        ("dry", "cyan", 68, "Dry run", ["all arms, tiny settings,", "on this device, then evaluated:", "the whole chain in a minute"], None),
        ("tim", "cyan", 69, "Timing", ["Adam and LM step measured here", "prints the campaign estimate", "once per device"], None)])
    s.arrow([(1009, 512), (1009, 524), (191, 524), (191, 548)])
    row(s, 548, 92, [
        ("camp", "green", 70, "Campaign", ["steps 23 to 49 per run, 4 arms × 5 seeds", "run folders in Drive; skips finished runs",
                                           "judges runs that lost their verdict"], 400),
        ("eval", "gray", 71, "Evaluate", ["steps 50 to 59", "results in Drive", "rebuilt every start"], None),
        ("pack", "orange", 72, "Package", ["configs, logs, environment", "README of the folder", "rebuilt every start"], None)])

    s.group(46, 696, 1108, 136, "19 · Restarts")
    row(s, 718, 92, [
        ("fail", "red", None, "A step fails", ["the run stops with the step name", "and the last 25 lines of its log;", "logs/<step>.log keeps all"], None),
        ("disc", "gray", None, "Colab disconnects", ["Run all again: finished steps and", "runs are skipped; the run in", "progress starts again"], None),
        ("atom", "gray", None, "Safe files", ["meta.json is written last and whole:", "it marks a run finished;", "selection.json marks it judged"], None),
        ("stopcell", "gray", None, "Cell stopped", ["the whole process group stops,", "so no run keeps going", "in the background"], None)],
        arrows=False)
    page_label(s, 6, "the main run")
    return s


DETAIL_PAGES = (detail_problem, detail_reference, detail_model, detail_training, detail_evaluation, detail_run)


def compact():
    """The six detailed pages on one page: the reduced problem, then 13 steps."""
    s = Scene("bvp-methodology-compact", "BVP PINN compact methodology",
              "Compact methodology on one page: the reduced Eqs. 14 to 18 and B1 to B12 with the locked "
              "values put in, the classical reference for checking, one PINN run per arm and seed with all "
              "12 conditions exact, the acceptance rule without the reference, the test and the report.")
    s.group(46, 24, 1108, 212, "1 · Fixed problem")
    row(s, 46, 168, [
        ("eqs", "purple", None, "Reduced equations,  *q* = *η* + 0.2,  *A* = 1 − 0.8*θ*", [
            "R14:  1.5(*f*^{(4)} + 2*f*‴/*q* − *f*″/*q*^{2} + *f*′/*q*^{3}) + 0.2(*ff*‴ − *f*′*f*″)/*q* + 0.2(*ff*″ − *f*′^{2})/*q*^{2}",
            "− 0.2*ff*′/*q*^{3} − 5(*f*″ + *f*′/*q*) − 0.5(*g*″ + *g*′/*q*) − 0.4(*θ*′ − 3*θ*/*q*)/*q*^{4} = 0",
            "R15:  1.25(*g*″ + *g*′/*q*) − 0.25(*f*″ + *f*′/*q* + 2*g*) − 0.2(*f*′*g* − *fg*′)/*q* = 0",
            "8.8 R16:  (1 + 0.5*A*^{3})(*θ*″ + *θ*′/*q*) + (0.3 − 1.2*A*^{2})*θ*′^{2} + 0.1*θ*′*φ*′ + 1.76*fθ*′/*q*",
            "+ 0.2(*θ* − 0.1)(0.2*f*(1 − 2/*q*^{2}) − 2*f*′)/*q*^{4} = 0",
            "8.8 R17:  *φ*″ + *φ*′/*q* + 3(*θ*″ + *θ*′/*q*) + 1.76*fφ*′/*q* = 0",
            "R18:  *χ*″ + *χ*′/*q* + 0.2*fφ*/*q* + (5/22)(*χφ*″ + *χ*′*φ*′) = 0"], 700),
        ("bcs", "cyan", None, "Conditions, all 12 exact in the PINN", [
            "*η* = 0:  1.5*f*′ = 1 + 0.1*f*″,  *f* = *φ*′/44,  *g* = −0.3*f*″",
            "−*θ*′ = 0.998 − *θ* + 1.25(1 − 0.8*θ*)^{4}",
            "*φ*′ = −3*θ*′,  *χ* = 1",
            "*η* = 20:  *f*′ = *f*″ = *g* = *χ* = 0",
            "*θ*′ = −*mθ*/20.2,  *φ*′ = −(*m*_{p}*φ* − 3*mθ*)/20.2",
            "*m* = 1.76*f*/(1 + 0.5*A*^{3}),  *m*_{p} = 1.76*f*",
            "22 locked values;  *δ*^{∗} = 0;  *L* = 20"], None)], arrows=False)
    s.arrow([(300, 214), (300, 298)])

    s.group(46, 276, 1108, 120, "2 · Reference")
    row(s, 298, 76, [
        ("solve", "cyan", 1, "Solve the reference", [
            "solve_bvp on 12 states, *MH* = −*r*_{0}", "48 starts at *L* = 10; tol 10^{−5} → 10^{−8}; *L* to 160"], None),
        ("check", "cyan", 2, "Check it", [
            "V7 to V12: conditions, core then *χ*, RK, slope", "C++ box scheme, branch search, *L* error bars"], None),
        ("refA", "orange", 3, "Reference A, *L* = 20", [
            "*θ*(0) = 0.25019,  *f*″(0) = −4.20330,  *m* = 0.239", "4 001 points; check only, used in step 11"], None)])
    s.arrow([(964, 374), (964, 412), (250, 412), (250, 458)])
    s.label("then train", 600, 408)

    s.group(46, 436, 1108, 216, "3 · PINN")
    row(s, 458, 76, [
        ("arms", "green", 4, "Arms", ["B0 one network; B1 4 windows", "B2, B3 staged, with replay"], None),
        ("net", "blue", 5, "Network", ["2 000 wall points, log map", "4 × 40 tanh + start profile"], None),
        ("exact", "cyan", 6, "Exact conditions", [
            "*u* = *N* + Σ_{*j*}*ψ*_{*j*}(*t*_{*j*} − *C*_{*j*}[*N*])", "B4, B5, B2, *f*, B10, B11, *g*, *χ*"], None),
        ("loss", "purple", 7, "Loss", [
            "*J* = mean Σ_{*k*}(*s*_{*k*}*R*_{*k*})^{2}, Taylor jets", "*s*: log (Adam, L-BFGS), balanced (LM)"], None)])
    s.arrow([(1009, 534), (1009, 544), (196, 544), (196, 554)])
    row(s, 554, 76, [
        ("adam", "green", 8, "Adam", ["20 000 steps, rate 10^{−3} → 10^{−4}", "gradient-norm weights, cap 1 000"], 260),
        ("refine", "green", 9, "Refine, weights frozen", [
            "L-BFGS 5 000 (log), then LM 260 (balanced)",
            "(*K*^{T}*K* + *μI*)*h* = −*K*^{T}*r*;  kept only if better"], 330),
        ("accept", "blue", 10, "Accept or reject, no reference", [
            "Newton-step score ≤ 10^{−3};  *m*, *m*_{p} > 0",
            "tail share ≤ 0.05;  max |B_{*i*}| ≤ 10^{−8};  rejects stay"], None)])
    s.arrow([(919, 630), (919, 668), (206, 668), (206, 714)])
    s.label("after all 20 runs", 929, 646, anchor="start")

    s.group(46, 692, 1108, 130, "4 · Report")
    s.box("test", 66, 714, 280, 76, "gray", 11, "Test against A", [
        "rel. *L*_{2}, wall values, *R*, *B*", "per window, after Adam; cost"], size=12)
    s.diamond("g6", 446, 752, 160, 72, "All 5 seeds?", ["rule G6"])
    s.arrow([(346, 752), (366, 752)])
    s.box("report", 556, 714, 270, 76, "orange", 12, "Report", [
        "TB2, TB3, TX1; FB1 to FB5, FX1", "and every failure with its cause"], size=12, thick=True)
    s.arrow([(526, 752), (556, 752)])
    s.label("YES", 541, 744, bold=True)
    s.arrow([(446, 788), (446, 804), (691, 804), (691, 790)])
    s.label("NO: no consistent effect", 568, 800)
    s.box("colab", 850, 714, 284, 76, "peach", 13, "Colab main run", [
        "tests, reference, dry run, timing,", "campaign, evaluation; resumes in Drive"], size=12)
    return s


# ---------------------------------------------------------------- export

def font_css(strings):
    text = "".join(set("".join(strings))) + "0123456789 "
    rules = []
    for (bold, italic), path in FONT_FILES.items():
        options = subset.Options()
        options.flavor = "woff"
        options.layout_features = ["*"]
        font = subset.load_font(str(path), options)
        subsetter = subset.Subsetter(options)
        subsetter.populate(text=text)
        subsetter.subset(font)
        buffer = io.BytesIO()
        subset.save_font(font, buffer, options)
        data = base64.b64encode(buffer.getvalue()).decode()
        rules.append(f"@font-face{{font-family:'Diagram Serif';src:url(data:font/woff;base64,{data}) format('woff');"
                     f"font-weight:{700 if bold else 400};font-style:{'italic' if italic else 'normal'}}}")
    return "\n".join(rules)


def html(scene):
    svg = scene.svg()
    css = font_css(scene.strings + [scene.title, scene.caption])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(scene.title)}</title>
<!-- Generated by tools/build_diagrams.py. Edit that file, not this one. -->
<style>
{css}
@page {{ size: 297mm 210mm; margin: 0 }}
:root {{ color-scheme: light }}
html, body {{ margin: 0; background: #ffffff; color: {INK}; font-family: 'Diagram Serif', STIX, serif }}
figure {{ margin: 0 auto; padding: 16px; max-width: 1400px; box-sizing: border-box }}
figure svg {{ display: block; width: 100%; height: auto }}
figcaption {{ font-size: 15px; line-height: 1.45; margin: 10px 4px 0; color: #3b4049 }}
@media print {{
  figure {{ padding: 0; max-width: none }}
  figure svg {{ width: 297mm; height: 210mm }}
  figcaption {{ display: none }}
}}
</style>
</head>
<body>
<figure>
{svg}
<figcaption>{escape(scene.caption)} Source: tools/build_diagrams.py; style follows methodology-flowchart-a4.pdf.</figcaption>
</figure>
</body>
</html>
"""


def export(scene, html_path):
    import pymupdf

    pdf_path = html_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", "--virtual-time-budget=5000",
                        f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                       check=True, capture_output=True)
    document = pymupdf.open(pdf_path)
    if document.page_count != 1:
        raise SystemExit(f"{pdf_path.name}: expected one page, got {document.page_count}")
    document.set_metadata({"title": scene.title, "subject": scene.caption,
                           "creator": "tools/build_diagrams.py", "producer": "Chrome and PyMuPDF"})
    document.saveIncr()
    page = document[0]
    html_path.with_suffix(".svg").write_text(page.get_svg_image(text_as_path=True))
    zoom = 2400 / page.rect.width
    page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(
        html_path.with_name(html_path.stem + "-preview.png"))


def merge(paths, target, title, caption):
    """One PDF with the given one-page PDFs in order."""
    import pymupdf

    book = pymupdf.open()
    for path in paths:
        with pymupdf.open(path) as page:
            book.insert_pdf(page)
    book.set_metadata({"title": title, "subject": caption, "creator": "tools/build_diagrams.py",
                       "producer": "Chrome and PyMuPDF"})
    book.save(target)


def main():
    global METRICS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-export", action="store_true", help="write HTML only, skip PDF, SVG and PNG")
    parser.add_argument("--only", choices=("summary", "compact", "detailed"), help="build one set of figures only")
    args = parser.parse_args()
    METRICS = Metrics()
    figures = ROOT / "docs/methodology/figures"
    sets = {"summary": (figures, (methodology, architecture)), "compact": (figures, (compact,)),
            "detailed": (figures / "detailed", DETAIL_PAGES)}
    for name, (folder, builds) in sets.items():
        if args.only and args.only != name:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        for build in builds:
            scene = build()
            scene.validate()
            path = folder / f"{scene.name}.html"
            path.write_text(html(scene))
            if not args.no_export:
                export(scene, path)
            print(f"built {scene.name}: {len(scene.nodes)} nodes, {len(scene.paths)} arrows")
        if name == "detailed" and not args.no_export:
            merge([folder / f"{build().name}.pdf" for build in builds], figures / "bvp-methodology-detailed.pdf",
                  "BVP PINN detailed methodology",
                  "Six pages, steps 1 to 72: the fixed problem with the reduced equations, the classical reference, "
                  "the PINN model, training one run, acceptance and evaluation, and the main run on Colab.")
            print("built bvp-methodology-detailed.pdf: 6 pages")


if __name__ == "__main__":
    main()
