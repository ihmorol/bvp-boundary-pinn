"""Build the methodology and architecture diagrams for the five-field BVP.

Both figures follow the style of methodology-flowchart-a4.pdf: one A4 landscape
page, numbered boxes, dashed green groups, yellow decision diamonds and short
labelled arrows. Every node, arrow and line of text is placed by hand below.

The build stops with an error if a line of text does not fit its box, two boxes
overlap, an arrow runs through a box or the text breaks the writing rule.

Run:
    uv run --no-project --with fonttools --with pymupdf python -I tools/build_diagrams.py

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
    """Split markup into runs. *x* is italic, _{x} is a subscript, ^{x} a superscript."""
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
              "pass the software tests, build a classical reference for checking only, train the PINN "
              "for three seeds, then test against the reference and report every outcome.")

    # stage 1: case and code
    s.group(46, 24, 1108, 136, "1 · Case and code")
    s.box("case", 66, 46, 250, 76, "peach", 1, "Locked case", [
        "Eqs. 14 to 19 for *f*, *g*, *θ*, *φ*, *χ*", "22 supervisor values, steady form"], size=12)
    s.box("sizes", 340, 46, 250, 76, "purple", 2, "Wall coefficient sizes", [
        "*q* = *h* = 0.2 at *η* = 0: terms reach 250", "fix loss scales *s*, *t* from these sizes"], size=12)
    s.box("tests", 614, 46, 250, 76, "cyan", 3, "Software tests", [
        "13 tests on synthetic cases", "derivatives, signs, BCs, gradients"], size=12)
    s.diamond("pass", 980, 84, 200, 76, "All tests pass?", ["13 of 13"])
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
        "tolerance 10^{−5} to 10^{−7}, finer mesh", "*L* = 10, 15, 20, several guesses"], size=12)
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
    s.group(46, 366, 1108, 300, "3 · Train, each seed 0, 1, 2")
    s.box("points", 66, 388, 252, 76, "blue", 6, "Point sets", [
        "1024 interior, half near the wall", "monitor and test sets kept apart"], size=12)
    s.box("net", 342, 388, 252, 76, "purple", 7, "Network and loss", [
        "*η* → 4 tanh layers of 64 → 5 fields", "R14 to R18, B1 to B12, scaled"], size=12)
    s.box("adam", 618, 388, 252, 76, "green", 8, "Adam training", [
        "trial rate 10^{−3} to 10^{−4}, 20000 steps", "checkpoint at fixed intervals"], size=12)
    s.box("lbfgs", 894, 388, 240, 76, "green", 9, "L-BFGS refinement", [
        "points, scales, weights frozen", "strong Wolfe line search"], size=12)
    s.arrow([(318, 426), (342, 426)])
    s.arrow([(594, 426), (618, 426)])
    s.arrow([(870, 426), (894, 426)])
    s.diamond("finite", 1014, 532, 220, 76, "All values finite?", ["in Adam and L-BFGS"])
    s.arrow([(1014, 464), (1014, 494)])
    s.box("pick", 618, 494, 252, 76, "blue", 10, "Pick the checkpoint", [
        "best rank on monitor points", "never on reference or test error"], size=12)
    s.arrow([(904, 532), (870, 532)])
    s.label("YES", 887, 524, bold=True)
    s.box("stop", 894, 596, 240, 52, "red", None, "Stop and save", [
        "first bad term; last good checkpoint kept"], size=12)
    s.arrow([(1014, 570), (1014, 596)])
    s.label("NO", 1024, 587, anchor="start", bold=True)
    s.arrow([(618, 532), (192, 532), (192, 464)], dashed=True, color=GROUP)
    s.label("next seed", 405, 525, color=GROUP, bold=True)

    # stage 4: test and report
    s.arrow([(744, 570), (744, 680), (290, 680), (290, 730)])
    s.label("after 3 seeds", 752, 600, anchor="start")
    s.arrow([(1134, 235), (1174, 235), (1174, 694), (330, 694), (330, 730)])
    s.label("reference curves, check only", 1000, 689)
    s.group(46, 708, 1108, 124, "4 · Test and report")
    s.box("final", 66, 730, 290, 76, "gray", 11, "Final test vs reference", [
        "fresh points: fields, *f* ′, every R and B", "proposed targets: BC 10^{−4}, field 10^{−3}"], size=12)
    s.diamond("targets", 486, 768, 220, 76, "Targets met?", ["reference error far below"])
    s.box("sense", 624, 730, 240, 76, "gray", 12, "Sensitivity, variants", [
        "*L*, point count, seed spread", "then B1 to B6, one change at a time"], size=12)
    s.box("report", 888, 730, 246, 76, "orange", 13, "Report", [
        "fields, wall quantities, errors, cost", "or the failure with its cause"], size=12, thick=True)
    s.arrow([(356, 768), (376, 768)])
    s.arrow([(596, 768), (624, 768)])
    s.label("YES", 610, 760, bold=True)
    s.arrow([(864, 768), (888, 768)])
    s.arrow([(486, 806), (486, 820), (1011, 820), (1011, 806)])
    s.label("NO: reported as failed", 740, 816)
    return s


def architecture():
    s = Scene("bvp-pinn-architecture", "BVP PINN architecture",
              "One training step of the five-field PINN: points pass through one shared network, "
              "automatic differentiation gives the derivatives, the locked coefficients turn them into "
              "equation and boundary errors, and one scalar loss updates the weights. Checks and the "
              "classical reference stay outside the loss.")

    s.box("setup", 46, 16, 1108, 92, "peach", 1, "One network, physics only in the loss", [
        "the network maps a position *η* to five fields; automatic differentiation gives their derivatives; "
        "the 22 coefficients are fixed numbers, never trained",
        "same loop as the Lorenz PINN; the bottom row lists the four things that change for this boundary problem"])

    s.group(46, 136, 1108, 432, "One training step, repeated")

    s.box("points", 66, 164, 196, 92, "blue", 2, "Input points", [
        "*N* interior points in (0, *L*)", "plus the two ends *η* = 0, *L*"])
    s.frame("net", 288, 164, 366, 92, "blue", 3, "Shared network")
    cells = [("scale", "2*η*/*L* − 1"), ("4 tanh layers", "64 units each"), ("linear", "64 to 5 outputs")]
    for i, (title, line) in enumerate(cells):
        x = 298 + i * 120
        s.cell(f"cell {title}", x, 196, 106, 50, title, line)
        if i:
            s.arrow([(x - 14, 221), (x, 221)], inner=True)
    s.box("fields", 680, 164, 250, 92, "purple", 4, "Fields and derivatives", [
        "*f* to *f* ^{(4)}; *g*, *θ*, *φ*, *χ* to second order",
        "autograd, one output at a time"], size=12)
    s.arrow([(262, 210), (288, 210)])
    s.arrow([(654, 210), (680, 210)])

    s.box("bcs", 380, 296, 260, 72, "cyan", 6, "Boundary errors", [
        "B1 to B6 at *η* = 0, B7 to B12 at *η* = *L*", "soft penalties, no extra conditions"], size=12)
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
        "*G* = ∇_{*w*}*J* through all derivatives", "Adam, then L-BFGS on frozen points"], size=12)
    s.arrow([(300, 296), (300, 256)])
    s.label("new weights", 308, 282, anchor="start")

    s.box("loss", 380, 408, 754, 72, "green", 8, "One scalar loss", [
        "*J* = Σ_{*i*} *λ*_{*i*} mean(*R*_{*i*}/*s*_{*i*})^{2} + Σ_{*j*} *μ*_{*j*} (*B*_{*j*}/*t*_{*j*})^{2}"
        "    with *i* = 1, …, 5 and *j* = 1, …, 12",
        "fixed scales *s*, *t* and weights *λ*, *μ*; no reference data in the loss"])
    s.arrow([(510, 368), (510, 408)])
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
        "fixed dense grid, every R and B", "ranks the saved checkpoints"], size=12)
    s.box("pick", 356, 624, 264, 72, "blue", 11, "Pick and lock", [
        "best monitor rank, weights frozen", "no retuning after this point"], size=12)
    s.box("final", 646, 624, 254, 72, "gray", 12, "Final test", [
        "fresh points: fields, *f* ′, every R and B", "errors against the reference"], size=12)
    s.box("ref", 926, 624, 208, 72, "cyan", 13, "Classical reference", [
        "solve_bvp, 12 first-order states", "comparison only, never a label"], size=12)
    s.arrow([(330, 660), (356, 660)])
    s.arrow([(620, 660), (646, 660)])
    s.arrow([(926, 660), (900, 660)])

    # bottom row: what changes from the Lorenz PINN
    s.frame("lorenz", 46, 732, 1108, 100, "gray", None, "Changed from the Lorenz PINN")
    changes = [("time *t* → position *η*", "one interval [0, *L*], no time windows"),
               ("exact initial value → 12 errors", "six conditions at each end, soft"),
               ("first derivatives → up to fourth", "nested autograd for *f* ^{(4)}"),
               ("3 outputs → 5 outputs", "*f*, *g*, *θ*, *φ*, *χ* from one network")]
    for i, (title, line) in enumerate(changes):
        s.cell(f"change {i + 1}", 60 + i * 274, 764, 258, 56, title, line, stroke="#a4a9b1")
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


def main():
    global METRICS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-export", action="store_true", help="write HTML only, skip PDF, SVG and PNG")
    args = parser.parse_args()
    METRICS = Metrics()
    for build in (methodology, architecture):
        scene = build()
        scene.validate()
        path = ROOT / "docs/methodology/figures" / f"{scene.name}.html"
        path.write_text(html(scene))
        if not args.no_export:
            export(scene, path)
        print(f"built {scene.name}: {len(scene.nodes)} nodes, {len(scene.paths)} arrows")


if __name__ == "__main__":
    main()
