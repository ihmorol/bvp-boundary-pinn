from pathlib import Path
from html import escape
import argparse
import base64
import hashlib
import importlib.util
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bvp_method_diagram", ROOT / "tools/build_methodology.py")
METHOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METHOD)
WIDTH = 3900


def box(key, title, body, x, y, width, height, tone="blue", number=None):
    badge = f'<span class="badge">{number}</span>' if number else ""
    return f'<article id="{key}" class="node {tone}" data-node="{key}" style="left:{x}px;top:{y}px;width:{width}px;min-height:{height}px">{badge}<h3>{title}</h3><div>{body}</div></article>'


def arrow(points, dashed=False):
    path = "M " + " L ".join(f"{x},{y}" for x, y in points)
    return f'<path class="{"feedback" if dashed else "flow"}" d="{path}" marker-end="url(#arrow)"/>'


def graph():
    nodes = []
    nodes.append(box("interior", "Fixed interior coordinates", '<p>Full interval + near-wall coverage</p><p>η: N × 1, gradient-enabled leaves</p>', 65, 110, 465, 190, number="1"))
    nodes.append(box("normalize", "Differentiable coordinate map", METHOD.formula(r"x=2\eta/L-1", True) + '<p>Normalize inside the model once.</p>', 600, 110, 485, 190, number="2"))
    layers = [('Linear','1 → 64'),('tanh',''),('3 × hidden','64 → 64 + tanh'),('Linear','64 → 5')]
    layer_html = '<div class="layers">' + '<span class="layer-arrow">→</span>'.join(f'<div class="layer"><b>{a}</b><p>{b}</p></div>' for a,b in layers) + '</div>'
    nodes.append(box("network", "One shared physical-output MLP", layer_html + '<p>Four width-64 tanh layers; linear outputs. Float64, Xavier-uniform weights, zero biases: proposed baseline.</p>', 1160, 95, 1890, 245, "peach", "3"))
    nodes.append(box("fields", "Physical predictions", METHOD.formula(r"u_w=(f,g,\theta,\phi,\chi)^T", True) + '<p>N × 5. No hard initial-value lift.</p>', 3130, 110, 530, 190, "teal", "4"))
    nodes.append(box("endpoints", "Two endpoint coordinates", METHOD.formula(r"\eta_b=[0,L]^T", True) + '<p>Separate 2 × 1 coordinate batch.</p>', 65, 490, 465, 185, "cyan", "5"))
    nodes.append(box("endpoint-eval", "Same weights, endpoint traces", '<p>Evaluate the same MLP at 0 and L.</p><p>Keep required endpoint derivative graphs.</p>', 600, 470, 485, 220, "cyan", "6"))
    nodes.append(box("parameters", "Locked physical case", '<p>All 22 coefficients from the supervisor receipt.</p><p>Used by R and B arithmetic. Not optimizer parameters.</p>', 1160, 450, 765, 195, "peach", "P"))
    nodes.append(box("derivatives", "Componentwise nested autograd", METHOD.formula(r"f',f'',f''',f^{(4)}", True) + '<p>g, θ, φ, χ: first and second derivatives. Each component N × 1.</p>', 2250, 430, 615, 240, "purple", "7"))
    nodes.append(box("residuals", "Five original equation errors", '<p>R14, R15, R16, R17, R18</p><p>Fields + derivatives + η + locked coefficients → N × 5.</p>', 3130, 430, 530, 240, "purple", "8"))
    nodes.append(box("boundaries", "Exactly twelve boundary errors", '<p>B1 to B6 at η = 0; B7 to B12 at η = L.</p><p>Values, slopes, slip, flux and radiation. Soft penalties, no extra BCs.</p>', 1160, 815, 765, 220, "cyan", "9"))
    nodes.append(box("boundary-loss", "Boundary contribution", METHOD.formula(r"J_B=\sum_{j=1}^{12}\mu_j(B_j/t_j)^2", True) + '<p>Fixed positive scales and weights.</p>', 2250, 815, 615, 220, "teal", "10b"))
    nodes.append(box("equation-loss", "Equation contribution", METHOD.formula(r"J_R=\sum_{i=1}^{5}\lambda_i\,\mathrm{mean}(r_i^2)", True) + '<p>rᵢ = Rᵢ/sᵢ. Save original errors too.</p>', 3130, 815, 530, 220, "teal", "10a"))
    nodes.append(box("total-loss", "One scalar physics-only loss", METHOD.formula(r"J(w)=J_R+J_B", True) + '<p>No reference labels, initial-value loss, causal weights or time rollout.</p>', 2500, 1190, 1160, 190, "gray", "11"))
    nodes.append(box("gradients", "Loss-to-weight backward pass", METHOD.formula(r"G=\nabla_w J", True) + '<p>Backpropagate through nested derivatives. Check finite weight gradients before updating; stop and save any failure.</p>', 1100, 1190, 530, 190, "teal", "12"))
    nodes.append(box("optimizer", "Adam → frozen-objective L-BFGS", '<p>Only NN weights and biases change.</p><p>Freeze points, case, L, scales and weights for refinement. Rebuild the graph in every closure.</p>', 1100, 1530, 1200, 210, "blue", "13"))
    nodes.append(box("failed-update", "Stop and save the exact failure", '<p>NaN, infinite derivative, disconnected expected gradient path or no eligible checkpoint.</p><p>Keep a valid Adam fallback if refinement fails.</p>', 2500, 1530, 1160, 180, "failure"))
    nodes.append(box("monitor", "Independent monitoring", '<p>Fixed dense coordinates, original/scaled R and every B.</p><p>Rank eligible checkpoints, never by reference or final-test errors.</p>', 65, 1910, 960, 215, "gray", "14"))
    nodes.append(box("lock", "Select and lock the model", '<p>Use the declared monitor rank.</p><p>Restore selected weights; retain failure history and Adam fallback.</p>', 1120, 1910, 700, 215, "blue", "15"))
    nodes.append(box("final-test", "Open withheld final coordinates", '<p>Dense + seeded random sets.</p><p>Fields, f′, derivatives, R and every B. No hidden retuning.</p>', 1920, 1910, 770, 215, "gray", "16"))
    nodes.append(box("report", "Sensitivity and reporting", '<p>Common-domain, point-count and seed checks.</p><p>Then decide acceptance; report uncertainty, time, memory and all outcomes.</p>', 2780, 1910, 880, 215, "peach", "17"))
    nodes.append(box("reference", "Checked classical reference", '<p>Same-case twelve-state solve_bvp.</p><p>Mesh, L, tolerance, guess, conditioning and branch checks. Comparison only.</p>', 65, 2250, 1050, 180, "cyan", "REF"))
    nodes.append('<div class="decision" style="left:1725px;top:1190px;width:590px;height:190px"><svg viewBox="0 0 590 190"><polygon points="295,3 587,95 295,187 3,95"/></svg><b>All loss terms<br>and derivatives finite?</b></div>')
    edges = [
        arrow([(530,205),(600,205)]), arrow([(1085,205),(1160,205)]), arrow([(3050,205),(3130,205)]),
        arrow([(3395,300),(3395,375),(2557,375),(2557,430)]),
        arrow([(3660,230),(3695,230),(3695,550),(3660,550)]),
        arrow([(2865,550),(3130,550)]),
        arrow([(1925,545),(2085,545),(2085,745),(3080,745),(3080,610),(3130,610)]),
        arrow([(530,580),(600,580)]), arrow([(1085,580),(1125,580),(1125,925),(1160,925)]),
        arrow([(1430,340),(1430,390),(845,390),(845,470)], True),
        arrow([(1542,645),(1542,815)]), arrow([(1925,925),(2250,925)]), arrow([(3395,670),(3395,815)]),
        arrow([(2557,1035),(2557,1120),(2810,1120),(2810,1190)]),
        arrow([(3395,1035),(3395,1190)]), arrow([(2500,1285),(2315,1285)]),
        arrow([(1725,1285),(1630,1285)]), arrow([(1365,1380),(1365,1530)]),
        arrow([(2020,1380),(2020,1450),(3070,1450),(3070,1530)]),
        arrow([(1100,1635),(30,1635),(30,15),(2105,15),(2105,95)], True),
        arrow([(1700,1740),(1700,1820),(545,1820),(545,1910)]),
        arrow([(1025,2017),(1120,2017)]), arrow([(1820,2017),(1920,2017)]), arrow([(2690,2017),(2780,2017)]),
        arrow([(1115,2340),(2305,2340),(2305,2125)], True),
    ]
    labels = '<text x="1750" y="1250" class="svg-label">YES</text><text x="2060" y="1440" class="svg-label">NO</text><text x="430" y="1610" class="svg-label">update w only</text><text x="930" y="375" class="svg-label">same NN weights</text><text x="1350" y="2315" class="svg-label">checked reference: scoring only, never loss</text>'
    svg = '<svg class="connectors" viewBox="0 0 3730 2480"><defs><marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0,0 9,3.5 0,7" fill="#3b4552"/></marker></defs>' + ''.join(edges) + labels + '</svg>'
    groups = '<div class="group train"><span>TRAINING: ONE DIFFERENTIABLE COMPUTATION</span></div><div class="group evaluate"><span>MONITOR, LOCK, TEST AND REPORT</span></div>'
    return '<section class="graph" aria-label="BVP PINN computational graph">' + groups + svg + ''.join(nodes) + '</section>'


def build(output):
    equation_text = (ROOT / 'docs/equations.md').read_text()
    expressions = METHOD.MATH.equation_expressions(equation_text)
    case = json.loads((ROOT / 'configs/physical-case-pending.json').read_text())
    if case['missing_parameters'] or not case.get('locked'):
        raise ValueError('This architecture requires the locked complete physical case.')
    method = METHOD.get_sections((ROOT / 'docs/methodology.md').read_text())
    panels = []
    for number, heading in [(5, 'Point sets and coordinate norms'), (6, 'Network, loss and Adam update'), (7, 'Refinement and checkpoint selection'), (8, 'Final error and acceptance measures'), (9, 'Sensitivity and fair comparisons')]:
        _, notes = METHOD.stage_tasks(number, method[number][1])
        panels.append(f'<section class="technical-panel"><h2>{heading}</h2>{METHOD.markdown_math(notes.get("Mathematics", ""))}</section>')
    z_math = METHOD.formula(r"z=[f,f',f'',f''',g,g',\theta,\theta',\phi,\phi',\chi,\chi']^T", True)
    h_math = METHOD.formula(r"H=[f^{(4)},g'',\theta'',\phi'',\chi'']^T,\quad M H=-r_0", True)
    panels.append('<section class="technical-panel"><h2>Reference is a check, not training data</h2>' + z_math + h_math + METHOD.reference_matrix(equation_text) + '<p>Reference regularity, branch identity and uncertainty must pass their declared checks. No physical accuracy claim follows from low training loss alone.</p></section>')
    fonts = []
    for family, weight, style in [('Tinos',400,'normal'),('Tinos:style=Bold',700,'normal'),('Tinos:style=Italic',400,'italic'),('Latin Modern Math',400,'normal')]:
        data,_ = METHOD.font_data(family)
        name = 'Method Math' if family == 'Latin Modern Math' else 'Method Serif'
        fonts.append(f"@font-face{{font-family:'{name}';src:{data};font-weight:{weight};font-style:{style}}}")
    font_sheet = 'data:text/css;base64,' + base64.b64encode('\n'.join(fonts).encode()).decode()
    params = ''.join(f'<span>{METHOD.formula(METHOD.symbol_tex(k))} = <b>{v:g}</b></span>' for k,v in case['physical_parameters'].items())
    css = f'''
@page{{size:{WIDTH}px 7500px;margin:0}}*{{box-sizing:border-box}}html,body{{margin:0;padding:0;width:{WIDTH}px;background:#fff;color:#20252b}}body{{font:25px/1.35 'Method Serif',serif}}.poster{{width:{WIDTH}px;padding:45px 60px}}h1,h2,h3,p{{margin:0}}h1{{font-size:65px;line-height:1.1}}h2{{font-size:34px;line-height:1.15}}h3{{font-size:30px;line-height:1.2}}p{{margin:9px 0}}a{{color:#2e5974}}math{{font-family:'Method Math',serif}}math[display=block]{{display:block;margin:12px 0;font-size:31px;text-align:left}}.masthead{{padding:28px 32px;background:#fcf0e7;border:2px solid #b96739;border-radius:12px}}.masthead p{{font-size:27px}}.status{{display:flex;gap:45px;padding:20px 27px;margin:22px 0;border:2px solid #28674b;background:#f4faf6;border-radius:10px}}.status div{{flex:1}}.parameters{{display:flex;flex-wrap:wrap;gap:14px 34px;background:#fff7ef;border:1px solid #c28a61;padding:20px 28px;margin:20px 0 28px;border-radius:10px}}.parameters span{{white-space:nowrap}}.parameter-note{{font-size:23px;margin-bottom:24px}}.graph{{position:relative;width:3730px;height:2480px;margin:28px auto 40px}}.node{{position:absolute;z-index:2;border:2px solid var(--edge,#5376b8);border-radius:12px;background:var(--fill,#edf2ff);padding:23px 25px}}.node h3{{margin:0 0 12px}}.node p{{font-size:25px;line-height:1.3}}.badge{{float:left;margin:0 15px 8px 0;display:inline-flex;align-items:center;justify-content:center;min-width:46px;height:43px;padding:0 7px;border-radius:50%;background:var(--edge,#5376b8);color:#fff;font-size:21px;font-weight:bold}}.blue{{--fill:#edf2ff;--edge:#4a6fb7}}.peach{{--fill:#fff2e5;--edge:#bc7632}}.teal{{--fill:#e9f7f2;--edge:#448c79}}.cyan{{--fill:#eaf5fa;--edge:#4b8ba6}}.purple{{--fill:#f1eafb;--edge:#8054be}}.gray{{--fill:#f3f5f8;--edge:#6c7684}}.failure{{--fill:#fdf0e8;--edge:#b86540}}.layers{{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:15px 0 12px}}.layer{{padding:12px 23px;background:#fffaf5;border:1px solid #c6946b;border-radius:6px;text-align:center;flex:1}}.layer p{{font-size:26px;margin:5px 0 0}}.layer-arrow{{font-size:34px}}.connectors{{position:absolute;inset:0;width:100%;height:100%;z-index:1;overflow:visible}}.flow,.feedback{{fill:none;stroke:#3b4552;stroke-width:3.5}}.feedback{{stroke-dasharray:12 9}}.svg-label{{font:23px 'Method Serif',serif;fill:#344353}}.group{{position:absolute;border:2px dashed #28674b;border-radius:15px;z-index:0}}.group span{{position:absolute;left:35px;top:-25px;background:#28674b;color:#fff;border-radius:35px;padding:10px 24px;font-size:24px;font-weight:bold}}.group.train{{left:0;top:60px;width:3730px;height:1730px}}.group.evaluate{{left:0;top:1860px;width:3730px;height:600px}}.decision{{position:absolute;z-index:2;display:flex;align-items:center;justify-content:center}}.decision svg{{position:absolute;inset:0;width:100%;height:100%}}.decision polygon{{fill:#fff9dd;stroke:#b6a03d;stroke-width:3}}.decision b{{position:relative;font-size:28px;text-align:center;max-width:360px}}.math-grid{{display:grid;grid-template-columns:1.05fr 1fr;gap:30px;align-items:start}}.math-panel,.dependency-panel,.technical-panel{{border:2px solid #567dad;border-radius:12px;padding:25px 29px;background:#f4f7ff;min-width:0}}.right-math{{display:grid;gap:25px}}#boundaries{{background:#edf6fa;border-color:#528da4}}.panel-lead,.source-note,.small-note{{font-size:24px;line-height:1.3;margin:11px 0}}.equation{{padding:10px 0;border-bottom:1px solid #cbd6e8}}.equation math{{font-size:29px;margin:10px 0}}.equation math:not(:first-child){{padding-left:65px}}.boundary-grid{{display:grid;grid-template-columns:1.7fr .8fr;gap:20px}}.boundary-grid .equation{{border:0;padding:3px 0}}.boundary-grid h3{{font-size:28px;margin:15px 0 8px}}.far-boundaries{{border-left:1px solid #cbd6e8;padding-left:22px}}.dependency-panel table{{width:100%;border-collapse:collapse;font-size:25px;margin:14px 0}}th,td{{padding:9px 14px;border-bottom:1px solid #cad7e5;text-align:left}}.technical-title{{margin:35px 0 22px;padding-top:22px;border-top:2px solid #28674b}}.technical-grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start}}.technical-column{{display:grid;gap:28px;align-content:start}}.technical-panel{{background:#f7fbf9;border-color:#77978d}}.technical-panel h2{{margin-bottom:16px}}.technical-panel p{{font-size:24px;line-height:1.3}}.technical-panel math{{font-size:29px}}.technical-panel .display-math{{margin:13px 0}}.reference-matrix h3{{font-size:29px;margin:18px 0 12px}}.footer{{margin:28px 0 0;padding:23px 28px;background:#fff6e9;border:2px solid #bd8743;border-radius:12px;font-size:24px}}@media screen{{html{{background:#fff}}body{{transform-origin:top left}}@media(max-width:{WIDTH}px){{body{{zoom:calc(100vw / {WIDTH}px)}}}}}}@media print{{body{{zoom:1;transform:none}}}}
'''
    right_panels = ''.join(panels[index] for index in [1, 3, 4])
    left_panels = ''.join(panels[index] for index in [0, 2, 5])
    panels_html = f'<div class="technical-column">{left_panels}</div><div class="technical-column">{right_panels}</div>'
    paths = ['docs/equations.md','docs/methodology.md','configs/physical-case-pending.json','configs/training-proposal.json']
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BVP PINN architecture and training methodology</title><link rel="stylesheet" href="{font_sheet}"><style>{css}</style></head><body><main class="poster"><header class="masthead"><h1>PINN methodology for the five-field BVP</h1><p>Lorenz-style computational graph, adapted to one two-ended boundary-value problem. Network → derivatives and physics → loss → weight update.</p></header><div class="status"><div><b>Physical case complete and locked.</b> The 22 coefficients come from the supervisor receipt. Da = 1/5 is derived. No physical parameter is trained.</div><div><b>Implementation and validation pending.</b> Network settings, grids, scales, budgets and acceptance rules are proposals or pending contract choices. This figure does not claim a solved case.</div></div><div class="parameters">{params}</div><p class="parameter-note">Received L = 0.5 means λₘ, not the endpoint L. Received scalars M = 5 and H = 1 are not the reference matrix/vector. H and Prm remain outside Eqs. 14 to 19.</p>{graph()}<div class="math-grid">{METHOD.equation_panel(expressions)}<div class="right-math">{METHOD.boundary_panel(expressions)}{METHOD.dependency_panel()}</div></div><h2 class="technical-title">Mathematical contract behind the graph</h2><div class="technical-grid">{panels_html}</div><footer class="footer"><b>How this differs from Lorenz:</b> a whole spatial interval replaces time windows; twelve soft boundary errors replace the hard initial-value trial; nested eta derivatives replace first-order time derivatives; solve_bvp replaces solve_ivp. No causal weighting, endpoint handoff, conservation-loss term or solution-data loss is imported. Training and checkpoint ranking never use reference labels or final-test errors. All failures and pending outputs remain in the report.</footer></main><script type="application/json" id="source-hashes">{json.dumps(hashes)}</script></body></html>'''
    output.write_text(html)
    cache=ROOT/'.cache/architecture'
    cache.mkdir(parents=True,exist_ok=True)
    (cache/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
    print(f'Built {output.name}: shared MLP, derivative/physics branches, twelve-BC branch, optimizer feedback and check-only reference.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Build the BVP PINN architecture document. No solver or training runs.')
    parser.add_argument('--output',type=Path,default=ROOT/'bvp-pinn-architecture.html')
    args=parser.parse_args()
    build(args.output)
