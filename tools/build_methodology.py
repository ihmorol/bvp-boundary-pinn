from pathlib import Path
from html import escape
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import re
import subprocess

import latex2mathml.converter
import markdown

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bvp_guide_math", ROOT / "tools/math_markup.py")
MATH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATH)
WIDTH = 4200


def formula(source, block=False):
    source = source.strip()
    aligned = re.fullmatch(r"\\begin\{aligned\}(.*?)\\end\{aligned\}", source, flags=re.S)
    if aligned:
        lines = [line.replace("&", "").strip() for line in aligned.group(1).split(r"\\") if line.strip()]
        rows = []
        for line in lines:
            rendered = latex2mathml.converter.convert(line)
            inner = re.search(r"<math[^>]*>(.*)</math>", rendered, flags=re.S).group(1)
            rows.append(f'<mtr><mtd columnalign="left">{inner}</mtd></mtr>')
        display = "block" if block else "inline"
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{display}"><mtable rowspacing="0.45em">{"".join(rows)}</mtable></math>'
    return latex2mathml.converter.convert(source, display="block" if block else "inline")


def markdown_math(text):
    math_fragments = []

    def capture(match):
        block = match.group(1) is not None
        source = match.group(1) if block else match.group(2)
        token = f"BVPFORMULA{len(math_fragments)}TOKEN"
        math_fragments.append(formula(source, block))
        return f"\n\n{token}\n\n" if block else token

    text = re.sub(r"\$\$(.*?)\$\$|(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)", capture, text, flags=re.S)
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    for index, markup in enumerate(math_fragments):
        token = f"BVPFORMULA{index}TOKEN"
        if 'display="block"' in markup:
            html = html.replace(f"<p>{token}</p>", f'<div class="display-math">{markup}</div>')
        html = html.replace(token, markup)
    html = re.sub(r'href="(?!https?:|#)([^\"]+)"', lambda m: f'href="{escape(str(Path("docs") / m.group(1)))}"', html)
    return html


def symbol_tex(name):
    return {
        "Delta": r"\Delta", "kappa": r"\kappa", "delta_star": r"\delta_\star",
        "theta_w": r"\theta_w", "lambda_m": r"\lambda_m", "beta_m": r"\beta_m",
        "epsilon": r"\epsilon", "ell": r"\ell", "N1": "N_1", "fw": "f_w",
    }.get(name, name)


def symbol(name):
    endpoint = None
    if name.endswith(("_0", "_L")):
        name, endpoint = name.rsplit("_", 1)
    derivative = re.fullmatch(r"(f|g|th|ph|ch)([1-4])", name)
    if derivative:
        field, order = derivative.groups()
        order = int(order)
        upper = "′" * order if order < 4 else "(4)"
        value = f'<msup><mi>{MATH.DERIVATIVES[field]}</mi><mo>{upper}</mo></msup>'
    else:
        value = MATH.name_markup(name)
    if endpoint is not None:
        value = f'<mrow>{value}<mo>(</mo><mi>{endpoint}</mi><mo>)</mo></mrow>'
    return value


def precedence(node):
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return 1
        if isinstance(node.op, (ast.Mult, ast.Div)):
            return 2
        return 4
    return 3 if isinstance(node, ast.UnaryOp) else 5


def compact_math(node, parent=0):
    if isinstance(node, ast.Name):
        return symbol(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if node.value == 0.5:
            return "<mfrac><mn>1</mn><mn>2</mn></mfrac>"
        return f"<mn>{node.value}</mn>"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = f"<mrow><mo>−</mo>{compact_math(node.operand, 3)}</mrow>"
    elif isinstance(node, ast.BinOp):
        level = precedence(node)
        if isinstance(node.op, ast.Div):
            value = f"<mfrac>{compact_math(node.left)}{compact_math(node.right)}</mfrac>"
        elif isinstance(node.op, ast.Pow):
            value = f"<msup><mrow>{compact_math(node.left, 4)}</mrow><mrow>{compact_math(node.right)}</mrow></msup>"
        elif isinstance(node.op, ast.Mult):
            value = f"<mrow>{compact_math(node.left, 2)}<mo>·</mo>{compact_math(node.right, 2)}</mrow>"
        elif isinstance(node.op, (ast.Add, ast.Sub)):
            right_level = level + 1 if isinstance(node.op, ast.Sub) else level
            operator = "−" if isinstance(node.op, ast.Sub) else "+"
            value = f"<mrow>{compact_math(node.left, level)}<mo>{operator}</mo>{compact_math(node.right, right_level)}</mrow>"
        else:
            raise ValueError(f"Unsupported operator: {ast.dump(node)}")
    else:
        raise ValueError(f"Unsupported equation: {ast.dump(node)}")
    if precedence(node) < parent:
        return f"<mrow><mo>(</mo>{value}<mo>)</mo></mrow>"
    return value


def equation_lines(name, expression, groups):
    terms = MATH.signed_terms(ast.parse(expression, mode="eval").body)
    if [i for group in groups for i in group] != list(range(len(terms))):
        raise ValueError(f"Line groups do not cover {name} exactly once.")
    rows = []
    for row_index, indices in enumerate(groups):
        label = f'<msub><mi>{name[0]}</mi><mn>{name[1:]}</mn></msub><mo>=</mo>' if row_index == 0 else ""
        parts = [label]
        for index in indices:
            sign, term = terms[index]
            if sign < 0:
                parts.append("<mo>−</mo>")
            elif index:
                parts.append("<mo>+</mo>")
            parts.append(compact_math(term))
        if row_index == len(groups) - 1:
            parts.append("<mo>=</mo><mn>0</mn>")
        rows.append('<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow>' + "".join(parts) + "</mrow></math>")
    return f'<div class="equation" data-equation="{name}" data-source="{escape(expression, quote=True)}">' + "".join(rows) + "</div>"


def equation_panel(expressions):
    groups = {
        "R14": [[0], [1, 2, 3], [4], [5, 6, 7]],
        "R15": [[0, 1], [2, 3]],
        "R16": [[0, 1], [2, 3, 4], [5], [6]],
        "R17": [[0, 1], [2, 3]],
        "R18": [[0, 1], [2, 3]],
    }
    body = "".join(equation_lines(name, expressions[name], groups[name]) for name in groups)
    return '<section class="math-panel" id="equations"><h2>Confirmed interior equations</h2><p class="panel-lead">Substitute all fields and eta derivatives at the same interior point. Every residual has target zero.</p>' + body + '<p class="source-note">Source E: Eqs. 14 to 18. The Eq. 14 convective and unsteady brackets are the corrected ones. Eq. 18 retains the printed f·φ and final φ′ terms.</p></section>'


def dependency_panel():
    return '''<section class="dependency-panel"><h2>Derivatives and numerical checks</h2><table><thead><tr><th>Equation</th><th>Fields used</th><th>Highest derivatives</th></tr></thead><tbody><tr><td>R14</td><td>f, g, θ</td><td>f⁽⁴⁾, g″</td></tr><tr><td>R15</td><td>f, g</td><td>f″, g″</td></tr><tr><td>R16</td><td>f, θ, φ</td><td>θ″, φ′</td></tr><tr><td>R17</td><td>f, θ, φ</td><td>θ″, φ″</td></tr><tr><td>R18</td><td>f, φ, χ</td><td>φ″, χ″</td></tr></tbody></table><p>Differentiate each physical output separately. Preserve each component as N × 1 before stacking. No output summation, detached loss term or extra endpoint condition.</p><p><b>Locked-case wall coefficients:</b> at η = 0, 1/q = 5, 1/q³ = 125, 1/Da = 5 and 2βₘ/h⁴ = 250. Source P: arithmetic from the locked coefficients, not solution values.</p></section>'''


def boundary_panel(expressions):
    walls = []
    for index in range(1, 7):
        name = f"B{index}"
        terms = MATH.signed_terms(ast.parse(expressions[name], mode="eval").body)
        groups = [[i] for i in range(len(terms))] if name == "B4" else [list(range(len(terms)))]
        if name == "B4":
            groups = [[0, 1], [2]]
        walls.append(equation_lines(name, expressions[name], groups))
    far = [equation_lines(f"B{i}", expressions[f"B{i}"], [[0]]) for i in range(7, 13)]
    stable = formula(r"A_w=1+c\theta(0),\qquad Q_w=[1-\theta(0)](\theta_w+A_w)(\theta_w^2+A_w^2)", True)
    return '<section class="math-panel" id="boundaries"><h2>Exactly twelve boundary conditions</h2><div class="boundary-grid"><div><h3>Wall: η = 0</h3>' + "".join(walls) + '</div><div class="far-boundaries"><h3>Far endpoint: η = L</h3>' + "".join(far) + '</div></div><p class="small-note">The zero-flux wall condition is a cancellation, not two zero slopes. Do not add θ(0) = 1, φ(0) = 1, f(0) = 0, f(L) = 0 or f‴(L) = 0.</p><h3>Stable evaluation of the wall radiation term</h3>' + stable + formula(r"B_4=-\theta'(0)-Bi[1-\theta(0)]-Rd\,Q_w", True) + '<p class="small-note">For c ≠ 0 this is algebraically the printed quartic difference divided by c. Test the two forms. A removable limit at c = 0 does not confirm the physical temperature normalization.</p><p class="source-note">Source E: Eq. 19 and its checked radiation rearrangement.</p></section>'


def reference_matrix(equation_text):
    match = re.search(r"M = (\[.*?\n\])", equation_text, flags=re.S)
    matrix = ast.parse(match.group(1), mode="eval").body
    rows = "".join("<mtr>" + "".join(f"<mtd>{compact_math(cell)}</mtd>" for cell in row.elts) + "</mtr>" for row in matrix.elts)
    math = '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>M</mi><mo>=</mo><mo>[</mo><mtable columnspacing="1.1em" rowspacing="0.55em">' + rows + '</mtable><mo>]</mo></mrow></math>'
    return '<div class="reference-matrix"><h3>Highest-derivative matrix from the confirmed equations</h3>' + math + '<p class="small-note">Use the z and H mapping above. The local matrix M and vector H are not the received scalars M = 5 and H = 1. Solve M H = −r₀, never form an inverse. Verify the rearrangement independently before using this reference RHS.</p></div>'


def get_sections(text):
    sections = {}
    pattern = re.compile(r"^## (\d+)\. (.+)$", re.M)
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else text.find("\n## Sources and build notes", match.end())
        if end < 0:
            end = len(text)
        sections[int(match.group(1))] = (match.group(2).strip(), text[match.end():end].strip())
    if sorted(sections) != list(range(1, 11)):
        raise ValueError("The method needs exactly ten numbered stages.")
    return sections


def flow_labels(text):
    block = re.search(r"## Deterministic methodology overview.*?```mermaid\n(.*?)```", text, flags=re.S).group(1)
    labels = dict(re.findall(r"S(\d+)\[(\d+\. [^\]]+)\]", block))
    if sorted(map(int, labels)) != list(range(1, 11)):
        raise ValueError("The editable overview needs the ten method stage labels.")
    return labels


def font_data(family):
    path = subprocess.check_output(["fc-match", "-f", "%{file}", family], text=True).strip()
    suffix = Path(path).suffix.lower()
    mime, format_name = ("font/otf", "opentype") if suffix == ".otf" else ("font/ttf", "truetype")
    return f"url(data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}) format('{format_name}')", path


def stage_tasks(number, body):
    matches = list(re.finditer(rf"^### {number}\.(\d+) (.+)$", body, flags=re.M))
    if [int(match.group(1)) for match in matches] != [1, 2, 3, 4, 5]:
        raise ValueError(f"Stage {number} needs five small tasks in order.")
    tasks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index < 4 else body.find("\n#### ", match.end())
        if end < 0:
            end = len(body)
        text = body[match.end():end].strip()
        if re.search(r"^[-*] |\$\$", text, flags=re.M):
            raise ValueError(f"Task {number}.{index + 1} contains a list or display equation.")
        tasks.append((f"{number}.{index + 1}", match.group(2), text))
    notes = dict(re.findall(r"^#### ([^\n]+)\n(.*?)(?=^#### |\Z)", body, flags=re.M | re.S))
    if "Gate and output" not in notes:
        raise ValueError(f"Stage {number} has no gate and output.")
    return tasks, notes


def stage_card(number, title, body, equation_text):
    tasks, notes = stage_tasks(number, body)
    boxes = []
    for index, (task_id, task_title, text) in enumerate(tasks):
        boxes.append(f'<article class="task" id="task-{task_id}" data-task="{task_id}"><p class="task-id">{task_id}</p><h3>{escape(task_title)}</h3>{markdown_math(text)}</article>')
        if index < 4:
            boxes.append('<div class="task-arrow" aria-label="next task">→</div>')
    mathematics = markdown_math(notes.get("Mathematics", ""))
    if number == 4:
        mathematics += reference_matrix(equation_text)
    math_strip = f'<div class="math-strip">{mathematics}</div>' if mathematics else ""
    loop = ""
    if number == 6:
        loop = '<p class="loop-note">↶ Repeat 6.2 to 6.5 for each declared Adam step and seed. A nonfinite term stops the update before weights change.</p>'
    elif number == 7:
        loop = '<p class="loop-note">↶ Repeat 7.2 to 7.4 within the fixed refinement budget. Do not resample or change the objective inside a closure.</p>'
    gate = markdown_math(notes["Gate and output"])
    decisions = {
        1: ("Locked case valid?", "No: record invalid input; never tune physical values."),
        2: ("Contract complete?", "No: save pending numerical choices; no physical run."),
        3: ("All tests pass?", "No: repair software, then repeat these tests."),
        4: ("Reference reliable?", "No: diagnose; keep reference accuracy unverified."),
        5: ("Point sets valid?", "No: fix the declared sampling recipe and save a new record."),
        6: ("Finite checkpoint?", "No: stop the update and record the first bad term."),
        7: ("Eligible result?", "No refinement result: keep eligible Adam; otherwise report failure."),
        8: ("Final tests pass?", "No: retain failed tests; no hidden model retuning."),
        9: ("Sensitivity passes?", "No: retain failure; no verified accuracy claim."),
        10: ("Record complete?", "No: mark missing evidence or output as pending."),
    }
    question, failure = decisions[number]
    diamond = f'<div class="decision"><svg viewBox="0 0 540 180" aria-hidden="true"><polygon points="270,3 537,90 270,177 3,90"/></svg><b>{question}</b></div>'
    decision_row = f'<div class="phase-gate">{diamond}<div class="gate-arrow">YES →</div><div class="gate-output">{gate}</div><div class="gate-failure"><b>NO / FAILED</b><p>{failure}</p><p>Save outcome in phase 10.</p></div></div>'
    return f'<section class="phase phase-{number}" id="stage-{number}" data-stage="{number}"><header class="phase-heading"><span class="stage-number">{number:02d}</span><h2>{escape(title)}</h2></header><div class="task-row">{"".join(boxes)}</div>{loop}{math_strip}{decision_row}</section>'


def build(output):
    equation_text = (ROOT / "docs/equations.md").read_text()
    expressions = MATH.equation_expressions(equation_text)
    method_text = (ROOT / "docs/methodology.md").read_text()
    sections = get_sections(method_text)
    labels = flow_labels((ROOT / "docs/flowcharts.md").read_text())
    case = json.loads((ROOT / "configs/physical-case-pending.json").read_text())
    for number, (title, _) in sections.items():
        if labels[str(number)] != f"{number}. {title}":
            raise ValueError(f"Stage {number} differs between the method and the editable flow.")
    regular, regular_path = font_data("Tinos")
    bold, bold_path = font_data("Tinos:style=Bold")
    italic, italic_path = font_data("Tinos:style=Italic")
    math_font, math_path = font_data("Latin Modern Math")
    rows = []
    for number in range(1, 11):
        rows.append(stage_card(number, *sections[number], equation_text))
        if number < 10:
            rows.append('<div class="phase-transition"><span>↓</span> Pass the gate before the next phase. Failed or blocked outcomes go to phase 10.</div>')
    sources = ["docs/equations.md", "docs/parameters.md", "docs/strategy.md", "docs/flowcharts.md", "docs/methodology.md", "docs/literature-synthesis.md", "docs/questions-for-supervisor.md", "configs/physical-case-pending.json", "configs/training-proposal.json"]
    provenance = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}
    missing = case["missing_parameters"]
    if missing or any(value is None for value in case["physical_parameters"].values()):
        raise ValueError("The current diagram requires the complete locked physical case.")
    parameter_cells = "".join(
        f'<div class="parameter">{formula(symbol_tex(name))}<b>{value:g}</b></div>'
        for name, value in case["physical_parameters"].items()
    )
    parameter_band = '<section class="parameter-band"><h2>Locked target: supervisor receipt of 16 November 2026</h2><div class="parameter-grid">' + parameter_cells + '</div><p>Source P: the dated receipt and approved notation mappings. Da = 1/M = 1/5 = 0.2 is derived. Received L = 0.5 means λₘ, not the far endpoint. H = 1 and Prm = 0.1 are kept outside Eqs. 14 to 19.</p></section>'
    math_header = formula(r"\eta\in[0,L],\quad U_w(\eta)=[f,g,\theta,\phi,\chi],\quad q=\eta+\kappa,\ h=\eta+b,\ c=\theta_w-1,\ A=1+c\theta", True)
    source_block = '''<footer class="sheet-footer"><h2>How to read and use this method</h2><div class="footer-grid"><div><h3>Source key</h3><p><b>E</b>: confirmed equation sheet, Eqs. 14 to 19. <b>P</b>: parameter record and supervisor questions. <b>T</b>: strategy and training proposal. <b>L</b>: literature synthesis, optional comparisons only. <b>M</b>: procedural rules proposed in this methodology.</p><p>All numbered steps are a method to implement. They are not evidence of a completed solver, a trained model or achieved accuracy.</p></div><div><h3>Meaning of a repeatable run</h3><p>The same approved case, source hashes, point coordinates, seeds, schedules, scales, thresholds, libraries and device settings define a repeatable procedure. They do not prove uniqueness, temporal stability or optimizer convergence. Cross-hardware bitwise equality is not promised.</p><p>Use original equations for acceptance. Never hide a failed boundary, a different branch or an unavailable reference inside an average loss.</p></div><div><h3>Scope and limits</h3><p>This is a fixed-parameter forward BVP, not a diffusion PDE, inverse fit, time rollout or parametric surrogate. No reference value enters the baseline training loss. A later parametric study needs separate held-out parameter cases and reference checks.</p><p>Numerical choices and reported evidence stay separate. Eq. 23 physical wall-output definitions need their own source check and supervisor decision.</p></div></div></footer>'''
    font_css = f'''@font-face{{font-family:'Method Serif';src:{regular};font-style:normal;font-weight:400}}
@font-face{{font-family:'Method Serif';src:{bold};font-style:normal;font-weight:700}}
@font-face{{font-family:'Method Serif';src:{italic};font-style:italic;font-weight:400}}
@font-face{{font-family:'Method Math';src:{math_font};font-style:normal;font-weight:400}}'''
    font_stylesheet = "data:text/css;base64," + base64.b64encode(font_css.encode()).decode()
    css = f'''
:root{{--page-bg:#ffffff;--panel:#eef3ff;--band:#f5f7fc;--line:#4568a8;--ink:#20252b;--heading:#263243;--accent:#236447;--warning:#ad532b;--green:#256447}}
@page{{size:{WIDTH}px 8000px;margin:0}}
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0;width:{WIDTH}px;min-height:100vh;background:var(--page-bg);color:var(--ink)}}
body{{font:24px/1.36 'Method Serif',serif}}
.poster{{width:{WIDTH}px;padding:58px 64px 52px;background:var(--page-bg)}}
h1,h2,h3,h4,p,ul,ol{{margin:0}}
h1{{font-size:76px;line-height:1.08;letter-spacing:-.7px;max-width:3350px}}
h2{{font-size:37px;line-height:1.16;color:var(--heading)}}
h3{{font-size:28px;line-height:1.22;color:var(--heading);margin:20px 0 10px}}
h4{{font-size:25px;margin:14px 0 8px}}
p,li,td{{overflow-wrap:break-word}}
p{{margin:10px 0}}li{{margin:5px 0}}
ul,ol{{padding-left:31px;margin:10px 0}}
a{{color:var(--heading);text-decoration:underline;text-decoration-color:var(--line)}}
code{{font-family:'Method Serif',serif;font-size:.98em;overflow-wrap:anywhere}}
pre{{white-space:pre-wrap;margin:14px 0;padding:18px 22px;background:var(--band);line-height:1.3}}
.masthead{{display:flex;align-items:flex-start;justify-content:space-between;gap:48px;margin-bottom:28px}}
.kicker{{font-size:23px;letter-spacing:2px;font-weight:bold;color:var(--accent);margin-bottom:14px}}
.subtitle{{font-size:29px;max-width:3300px;margin-top:16px}}
.method-tag{{font-size:24px;line-height:1.3;border-left:5px solid var(--accent);padding:12px 22px;min-width:410px;margin-top:12px}}
.status-band{{border:2px solid var(--warning);border-radius:12px;padding:20px 27px;margin-bottom:25px;display:grid;grid-template-columns:1fr 1.1fr;gap:40px}}
.status-band strong{{color:var(--warning)}}
.reading-route{{display:grid;grid-template-columns:1fr 75px 1fr 75px 1fr 75px 1fr;gap:8px;align-items:center;margin:24px 0 30px}}
.route-item{{padding:15px 24px;border-top:3px solid var(--heading);font-size:26px}}
.route-item b{{display:block;font-size:29px;margin-bottom:4px}}
.route-arrow{{text-align:center;font-size:38px;color:var(--accent)}}
.spec-band{{padding:20px 26px;background:var(--band);border-radius:12px;margin-bottom:26px}}
.spec-band p{{margin:6px 0;text-align:center;font-size:24px}}
.math-grid{{display:grid;grid-template-columns:1.12fr 1fr;gap:42px;margin-bottom:40px;align-items:stretch}}
.math-panel{{border:2px solid var(--line);background:var(--panel);border-radius:12px;padding:28px 32px}}
.panel-lead{{margin:10px 0 18px}}
math{{font-family:'Method Math',serif;font-size:1.07em}}
math[display=block]{{display:block;max-width:100%;margin:15px 0;text-align:left;font-size:29px}}
math[display=inline]{{vertical-align:baseline}}
.equation{{padding:11px 0;border-bottom:1px solid var(--line)}}
.equation:last-of-type{{border-bottom:0}}
.equation math{{margin:9px 0;font-size:31px;line-height:1.35}}
.equation math:not(:first-child){{padding-left:75px}}
.source-note{{font-size:21px;line-height:1.3;color:var(--heading);margin-top:18px}}
.small-note{{font-size:23px;line-height:1.31}}
.boundary-grid{{display:grid;grid-template-columns:1.72fr .85fr;gap:25px}}
.boundary-grid h3{{margin:15px 0 4px}}
.boundary-grid .equation{{border-bottom:0;padding:4px 0}}
.boundary-grid .equation math{{font-size:29px;margin:12px 0}}
.far-boundaries{{border-left:1px solid var(--line);padding-left:25px}}
.workflow-title{{display:flex;justify-content:space-between;align-items:baseline;margin:36px 0 25px;border-top:3px solid var(--heading);padding-top:24px}}
.workflow-title p{{font-size:24px;margin:0;color:var(--heading)}}
.parameter-band{{margin:24px 0 28px;padding:24px 28px;background:var(--band);border-radius:12px}}
.parameter-grid{{display:grid;grid-template-columns:repeat(11,1fr);gap:12px 20px;margin:18px 0}}
.parameter{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:12px 14px;font-size:28px}}
.parameter b{{font-size:28px}}
.parameter-band>p{{font-size:23px;margin-top:14px}}
.phase{{padding:44px 28px 25px;border:2px dashed var(--green);border-radius:15px;margin-top:28px;position:relative}}
.phase-heading{{display:inline-flex;align-items:center;gap:18px;margin-top:-76px;margin-bottom:25px;padding:9px 25px;background:var(--green);border-radius:55px;color:#fff}}
.phase-heading h2{{font-size:38px;color:#fff}}
.stage-number{{display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;width:57px;height:57px;background:#fff;color:var(--green);border-radius:50%;font-size:30px;font-weight:bold}}
.task-row{{display:grid;grid-template-columns:1fr 50px 1fr 50px 1fr 50px 1fr 50px 1fr;align-items:stretch}}
.task{{border:2px solid var(--line);background:var(--panel);border-radius:12px;padding:22px 25px;min-width:0}}
.task-id{{display:inline-flex;align-items:center;justify-content:center;min-width:63px;height:48px;border-radius:50%;background:var(--line);color:#fff;padding:0 8px;font-size:22px;font-weight:bold;margin:0 0 13px}}
.task h3{{font-size:31px;line-height:1.18;margin:0 0 13px}}
.task>p:last-child{{font-size:25px;line-height:1.35;margin:0}}
.task-arrow{{display:flex;align-items:center;justify-content:center;font-size:40px;color:var(--accent)}}
.math-strip{{padding:22px 28px 14px;margin-top:20px;background:var(--band);border-radius:10px;column-count:2;column-gap:55px;column-rule:1px solid #d3dbe4}}
.math-strip .display-math,.math-strip p,.math-strip .reference-matrix{{break-inside:avoid}}
.math-strip p{{font-size:24px;line-height:1.3;margin:8px 0}}
.math-strip .display-math{{margin:12px 0}}
.math-strip math[display=block]{{font-size:31px;margin:14px 0}}
.math-strip .reference-matrix{{margin:0;padding:0;border:0}}
.phase-gate{{display:grid;grid-template-columns:520px 95px minmax(0,1fr) 650px;align-items:center;gap:24px;margin:22px 0 0}}
.decision{{position:relative;display:flex;align-items:center;justify-content:center;width:520px;height:174px}}
.decision svg{{position:absolute;inset:0;width:100%;height:100%}}
.decision polygon{{fill:#fff9dd;stroke:#b09d37;stroke-width:3}}
.decision b{{position:relative;font-size:29px;text-align:center;max-width:310px;line-height:1.2}}
.gate-arrow{{font-size:25px;font-weight:bold;text-align:center;color:var(--heading)}}
.gate-output{{border:2px solid #a4adba;border-radius:12px;background:#f5f6f8;padding:17px 25px}}
.gate-output p{{font-size:24px;line-height:1.3;margin:6px 0}}
.gate-failure{{border:2px solid #b7623d;border-radius:12px;background:#fdf0e9;padding:17px 24px}}
.gate-failure>b{{font-size:24px;color:#93481f}}
.gate-failure p{{font-size:23px;line-height:1.3;margin:7px 0}}
.phase-transition{{height:66px;display:flex;align-items:center;justify-content:center;gap:22px;font-size:22px;color:var(--heading)}}
.phase-transition span{{font-size:42px;color:var(--accent)}}
.loop-note{{font-size:23px;color:var(--heading);margin:15px 0 4px}}
.display-math{{margin:16px 0;padding:0 4px}}
.reference-matrix math{{font-size:29px}}
.sheet-footer{{margin-top:34px;border-top:3px solid var(--heading);padding-top:26px}}
.footer-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:42px;font-size:23px}}
.phase-1,.phase-2{{--panel:#fcf0e8;--line:#b7623d;--band:#fff8f3}}
.phase-3{{--panel:#f3edfd;--line:#7954be;--band:#faf7ff}}
.phase-4,.phase-5{{--panel:#eaf5fa;--line:#3e829e;--band:#f5fbfe}}
.phase-6{{--panel:#edf2ff;--line:#4668ad;--band:#f7f9ff}}
.phase-7{{--panel:#e9f7f3;--line:#3d8879;--band:#f4fcf9}}
.phase-8,.phase-9{{--panel:#f2f4f8;--line:#667182;--band:#fafbfe}}
.phase-10{{--panel:#fff3e4;--line:#bc7b26;--band:#fffaf2}}
.task-arrow,.route-arrow,.phase-transition span{{color:#3b4552}}
.masthead{{border:2px solid #b7623d;border-radius:12px;background:#fdf1e8;padding:28px 32px}}
.status-band{{background:#f5faf6;border-color:var(--green)}}
.status-band strong{{color:var(--green)}}
.parameter-band{{background:#fcf5ed;border:2px solid #bd875e}}
.parameter{{border-bottom-color:#d9c3b2}}
.math-panel:first-child{{background:#f0f3fd;border-color:#5474b2}}
.math-panel#boundaries{{background:#edf6fa;border-color:#5089a0}}
.right-math{{display:grid;gap:25px}}
.dependency-panel{{border:2px solid #5482a4;border-radius:12px;background:#f4f8fc;padding:26px 32px}}
.dependency-panel h2{{font-size:32px}}
.dependency-panel table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:25px}}
.dependency-panel th,.dependency-panel td{{text-align:left;padding:10px 16px;border-bottom:1px solid #cbd8e3}}
.dependency-panel p{{font-size:23px;line-height:1.3}}
.math-grid{{align-items:start}}
.math-panel .equation{{border-bottom-color:#c9d4e5}}
.reference-matrix{{display:block}}
.footer-grid h3{{margin-top:18px}}
@media screen{{html{{background:var(--page-bg)}}body{{transform-origin:top left}}@media(max-width:{WIDTH}px){{body{{zoom:calc(100vw / {WIDTH}px)}}}}}}
@media print{{html,body{{min-height:0;transform:none;zoom:1}}.poster{{margin:0}}}}
'''
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="author" content="Boundary BVP research project"><meta name="description" content="Mathematical method for a fixed-case five-field boundary PINN, from input gates through testing and reporting."><title>Five-field boundary-value problem: mathematical methodology</title><link rel="stylesheet" href="{font_stylesheet}"><style>{css}</style></head><body><main class="poster">
<header class="masthead"><div><p class="kicker">BOUNDARY-VALUE PROBLEM · PHYSICS-INFORMED NEURAL NETWORK</p><h1>From confirmed equations to a checked solution</h1><p class="subtitle">A complete mathematical workflow for one fixed physical case: source checks, reference validation, training, final testing and honest reporting.</p></div><div class="method-tag"><b>Method, not results</b><br>Five fields · twelve conditions<br>Read the numbered stages in order</div></header>
<div class="status-band"><div><strong>PHYSICAL PARAMETERS COMPLETE AND LOCKED</strong><br>All 22 equation coefficients are fixed. No automatic physical tuning. Software tests, the classical reference, training and final evaluation are still pending.</div><div><b>Separate choices:</b> numerical scales, budgets, grids and acceptance rules must be fixed before a run. Open physical definitions are recorded, not used to overwrite supplied values. This sheet is a method, not a result.</div></div>
{parameter_band}
<nav class="reading-route" aria-label="method phases"><div class="route-item"><b>1 to 4 · Establish the case</b>Source, contract, tests and checked reference</div><div class="route-arrow">→</div><div class="route-item"><b>5 to 7 · Fit and select</b>Separate point sets; Adam; frozen refinement</div><div class="route-arrow">→</div><div class="route-item"><b>8 to 9 · Test and compare</b>Locked final test; sensitivity and controlled arms</div><div class="route-arrow">→</div><div class="route-item"><b>10 · Report every outcome</b>Passed, failed, blocked or unverified</div></nav>
<div class="spec-band">{math_header}<p>Primes mean eta derivatives. Orders (4, 2, 2, 2, 2) give total order 12. L approximates infinity and must pass a domain study. Network weights w are not the physical temperature θ.</p></div>
<div class="math-grid">{equation_panel(expressions)}<div class="right-math">{boundary_panel(expressions)}{dependency_panel()}</div></div>
<div class="workflow-title"><h2>The deterministic procedure</h2><p>Inputs → exact operation → gate → saved record. Arrows show order, not guaranteed success.</p></div>
{"".join(rows)}
{source_block}
</main><script id="method-provenance" type="application/json">{json.dumps(provenance, indent=2)}</script></body></html>'''
    output.write_text(html)
    inventory = {
        "width_css_px": WIDTH,
        "stages": {str(n): title for n, (title, _) in sections.items()},
        "equations": expressions,
        "source_hashes": provenance,
        "fonts": [regular_path, bold_path, italic_path, math_path],
    }
    cache = ROOT / ".cache/methodology"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "content-inventory.json").write_text(json.dumps(inventory, indent=2))
    print(f"Built {output.name}: ten stages, five equations, twelve boundary conditions and a source-derived reference matrix.")
    print(f"Canvas width: {WIDTH} CSS pixels; height follows the content. Fonts: Tinos and Latin Modern Math.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the local mathematical methodology HTML. No BVP is solved.")
    parser.add_argument("--output", type=Path, default=ROOT / "bvp-methodology.html")
    args = parser.parse_args()
    build(args.output)
