from pathlib import Path
from html import escape
import json
import importlib.util
import markdown

ROOT = Path(__file__).resolve().parents[1]
math_spec = importlib.util.spec_from_file_location("bvp_guide_math", ROOT / "tools/math_markup.py")
math_module = importlib.util.module_from_spec(math_spec)
math_spec.loader.exec_module(math_module)
case = json.loads((ROOT / "configs/physical-case-pending.json").read_text())
missing = ", ".join(case["missing_parameters"])
status = (
    f"<strong>Physical values are incomplete.</strong> Missing: {escape(missing)}. No automatic defaults."
    if missing else
    "<strong>All 22 physical coefficients are complete and locked.</strong> The supervisor receipt dated 2026-11-16 supplies the active values; Da=0.2 is derived from M=5. Software validation, reference, training and evaluation are pending. Numerical settings and definition questions stay separate."
)
sections = [
    ("parameters", "Parameter record", "docs/parameters.md"),
    ("equations", "Equation calculation sheet", "docs/equations.md"),
    ("strategy", "Architecture and training strategy", "docs/strategy.md"),
    ("papers", "Papers, evidence and research comparisons", "docs/reading-list.md"),
    ("flows", "Editable flowchart source", "docs/flowcharts.md"),
]
nav = "".join(f'<a href="#{key}">{escape(title)}</a>' for key, title, _ in sections)
parts = []
for key, title, relative in sections:
    text = (ROOT / relative).read_text()
    content = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    content = content.replace('href="../', 'href="')
    for other in ["parameters", "equations", "strategy", "reading-list", "flowcharts"]:
        content = content.replace(f'href="{other}.md"', f'href="docs/{other}.md"')
    parts.append(f'<section id="{key}">{content}</section>')
body = math_module.rendered_equations((ROOT / "docs/equations.md").read_text()) + "\n" + "\n".join(parts)
html = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Five-variable boundary PINN: parameters and strategy</title>
<style>
:root{color-scheme:light dark;--bg:light-dark(#fafafa,#181a1c);--fg:light-dark(#20252c,#ecedf0);--muted:light-dark(#56616e,#b8c1cb);--line:light-dark(#d6dce2,#414951);--panel:light-dark(#eef2f5,#232a30);--link:light-dark(#1557aa,#9fc7ff);--warn:light-dark(#fff0ca,#413318)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:28px 20px 60px}h1{font-size:1.8rem;line-height:1.25}h2{font-size:1.4rem;line-height:1.3}h3{font-size:1.15rem;line-height:1.4}h1,h2,h3{margin:1.6em 0 .6em}a{color:var(--link)}nav{display:flex;gap:8px 18px;flex-wrap:wrap;margin:24px 0}section{border-top:1px solid var(--line);padding-top:12px;margin-top:42px;scroll-margin-top:16px}.status{padding:16px 20px;background:var(--warn);border-left:4px solid var(--link)}.chain{display:grid;gap:10px;margin:18px 0}.node{background:var(--panel);padding:14px 18px;border-left:3px solid var(--link)}.arrow{text-align:center;color:var(--muted)}.branch{margin-top:8px;color:var(--muted)}.calcs{display:grid;grid-template-columns:1fr 1fr;gap:12px}.calcs .node{height:100%}pre{background:var(--panel);padding:16px;overflow:auto;font-size:14px;line-height:1.55;border-radius:4px}code{font-family:ui-monospace,monospace}p code,li code{background:var(--panel);padding:1px 4px}table{width:100%;border-collapse:collapse;font-size:15px;display:block;overflow-x:auto}th,td{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}th{font-weight:600}math{display:block;font-size:1.15rem;margin:18px 0;overflow-x:auto}.equation-stack math{margin:10px 0}details{margin:12px 0;padding:10px 0;border-bottom:1px solid var(--line)}summary{font-weight:600;cursor:pointer}li{margin:.3em 0}.read{font-weight:600}.small{font-size:14px;color:var(--muted)}@media(max-width:650px){.calcs{grid-template-columns:1fr}main{padding-inline:14px}body{font-size:16px}pre{font-size:13px}h1{font-size:1.5rem}table{font-size:14px}}
</style></head><body><main>
<h1>Five-variable boundary PINN</h1>
<p>The locked supervisor values, source equations and proposed numerical settings are kept separate below. This guide contains no trained result. Read the <a href="bvp-methodology.html">complete small-task methodology diagram</a> or its <a href="bvp-methodology.pdf">PDF</a>.</p>
<div class="status">CASE_STATUS</div>
<nav>NAVIGATION</nav>
<h2>Architecture: what goes in and what comes out</h2>
<div class="chain">
<div class="node"><strong>Input:</strong> one coordinate eta. The complete physical case is fixed outside the network.</div>
<div class="arrow">↓</div>
<div class="node"><strong>One smooth network:</strong> five outputs f, g, theta, phi, chi. Automatic differentiation gives their eta derivatives, including four for f.</div>
<div class="arrow">↓</div>
<div class="calcs"><div class="node"><strong>Interior calculation:</strong> substitute predictions and derivatives into the five original equations. One error per equation per point.</div><div class="node"><strong>Boundary calculation:</strong> evaluate at zero and L. Calculate all twelve original value, slope and coupled errors.</div></div>
<div class="arrow">↓</div>
<div class="node"><strong>One loss:</strong> scale, square, average equation errors, and add all weighted boundary errors. Backpropagation updates only the NN weights.</div>
</div>
<h2>The loss, rendered</h2>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>J</mi><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mn>5</mn></munderover><msub><mi>λ</mi><mi>i</mi></msub><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo>∑</mo><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><msup><mrow><mo>[</mo><mfrac><mrow><msub><mi>R</mi><mi>i</mi></msub><mo>(</mo><msub><mi>η</mi><mi>n</mi></msub><mo>)</mo></mrow><msub><mi>s</mi><mi>i</mi></msub></mfrac><mo>]</mo></mrow><mn>2</mn></msup><mo>+</mo><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mn>12</mn></munderover><msub><mi>μ</mi><mi>j</mi></msub><msup><mrow><mo>[</mo><mfrac><msub><mi>B</mi><mi>j</mi></msub><msub><mi>t</mi><mi>j</mi></msub></mfrac><mo>]</mo></mrow><mn>2</mn></msup></mrow></math>
<p><span class="read">Read aloud:</span> J equals the sum of five weighted average squared equation errors, divided by their scales, plus the sum of twelve weighted squared boundary errors, divided by their scales.</p>
<p>N is the number of interior positions. R is an original equation error; B is an original boundary error. Lambda and mu are positive training weights; s and t are fixed scales. Squaring applies after dividing by each scale. These are not additional physical parameters.</p>
<h2>Full flow, with the stopping points</h2>
<div class="chain">
<div class="node"><strong>1. Preserve and transcribe sources.</strong> Keep source conditions separate from selected values. Test the corrected Eq. 14 grouping.</div><div class="arrow">↓</div>
<div class="node"><strong>2. Are all physical values known?</strong><div class="branch">No: save a blocked case and wait. Yes: check ranges and denominators. Invalid: stop with the exact reason.</div></div><div class="arrow">↓</div>
<div class="node"><strong>3. Run software tests.</strong> Derivatives, normalization, equation signs, BCs, shapes, gradients and manufactured cases.<div class="branch">Failure: fix software and repeat. No physical simulation yet.</div></div><div class="arrow">↓</div>
<div class="node"><strong>4. Build the reference.</strong> Twelve-state first-order BVP; change mesh, tolerance, initial guess and L.<div class="branch">Failure: diagnose and keep reference accuracy unverified. Never manufacture labels or conclude nonexistence from a solver failure.</div></div><div class="arrow">↓</div>
<div class="node"><strong>5. Train Adam for one seed.</strong> Interior and boundary calculations together; finite checks; independent monitoring and checkpoints.<div class="branch">NaN or broken gradients: stop and diagnose. Continue until the declared budget or transition rule.</div></div><div class="arrow">↓</div>
<div class="node"><strong>6. Freeze the objective and refine.</strong> Same points, coefficients, scales and weights for every L-BFGS closure evaluation. Keep the Adam checkpoint if refinement fails.</div><div class="arrow">↓</div>
<div class="node"><strong>7. Test fresh positions.</strong> Original equation errors, every BC, five profiles, f-prime and approved wall quantities. Compare the same reference branch.</div><div class="arrow">↓</div>
<div class="node"><strong>8. Check sensitivity and acceptance.</strong> Larger L, denser points, more seeds.<div class="branch">Does not pass: record failure and launch a separately logged diagnosis/run. Pass: report only the tested-case result.</div></div><div class="arrow">↓</div>
<div class="node"><strong>9. Save and compare.</strong> Settings, sources, weights, errors, budgets and every failure. Change one training method at a time.</div>
</div>
<p class="small">The classical reference is separate from training labels. Parameter arrival, software tests, reference confidence and final accuracy are different gates.</p>
BODY_CONTENT
</main></body></html>'''
html = html.replace("CASE_STATUS", status).replace("NAVIGATION", nav).replace("BODY_CONTENT", body)
(ROOT / "strategy-guide.html").write_text(html)
print("Built strategy-guide.html with local links, rendered equations, five documentation sections and labeled training flows.")
