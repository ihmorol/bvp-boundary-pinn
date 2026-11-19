import ast
from html import escape
import re

SYMBOLS = {
    "Delta": "Δ", "delta_star": "δ*", "kappa": "κ", "eta": "η",
    "theta": "θ", "phi": "φ", "chi": "χ", "epsilon": "ε", "ell": "ℓ",
    "q": "q", "h": "h", "A": "A", "Pr": "Pr", "Le": "Le",
    "Lb": "Lb", "Pe": "Pe", "Nr": "Nr", "Nt": "Nt", "Nb": "Nb",
    "Da": "Da", "Bi": "Bi", "Rd": "Rd", "B": "B", "b": "b",
}
SUBSCRIPTS = {
    "theta_w": ("θ", "w"), "lambda_m": ("λ", "m"),
    "beta_m": ("β", "m"), "N1": ("N", "1"), "fw": ("f", "w"),
}
DERIVATIVES = {"f": "f", "g": "g", "th": "θ", "ph": "φ", "ch": "χ"}


def name_markup(name):
    endpoint = None
    if name.endswith("_0") or name.endswith("_L"):
        name, endpoint = name.rsplit("_", 1)
    match = re.fullmatch(r"(f|g|th|ph|ch)([1-4])", name)
    if match:
        field, order = match.groups()
        token = f'<msup><mi>{DERIVATIVES[field]}</mi><mrow><mo>(</mo><mn>{order}</mn><mo>)</mo></mrow></msup>'
    elif name in SUBSCRIPTS:
        base, sub = SUBSCRIPTS[name]
        token = f'<msub><mi>{base}</mi><mi>{sub}</mi></msub>'
    else:
        token = f'<mi>{escape(SYMBOLS.get(name, name))}</mi>'
    if endpoint is not None:
        token = f'<mrow>{token}<mo>(</mo><mi>{endpoint}</mi><mo>)</mo></mrow>'
    return token


def markup(node):
    if isinstance(node, ast.Name):
        return name_markup(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return f'<mn>{node.value}</mn>'
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f'<mrow><mo>−</mo>{markup(node.operand)}</mrow>'
    if isinstance(node, ast.BinOp):
        left, right = markup(node.left), markup(node.right)
        if isinstance(node.op, ast.Div):
            return f'<mfrac>{left}{right}</mfrac>'
        if isinstance(node.op, ast.Pow):
            return f'<msup><mrow><mo>(</mo>{left}<mo>)</mo></mrow>{right}</msup>'
        if isinstance(node.op, ast.Mult):
            return f'<mrow><mo>(</mo>{left}<mo>)</mo><mo>·</mo><mo>(</mo>{right}<mo>)</mo></mrow>'
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = "+" if isinstance(node.op, ast.Add) else "−"
            return f'<mrow><mo>(</mo>{left}<mo>{sign}</mo>{right}<mo>)</mo></mrow>'
    raise ValueError(f"Unsupported equation syntax: {ast.dump(node)}")


def signed_terms(node, sign=1):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return signed_terms(node.left, sign) + signed_terms(node.right, sign)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return signed_terms(node.left, sign) + signed_terms(node.right, -sign)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return signed_terms(node.operand, -sign)
    return [(sign, node)]


def equation_expressions(text):
    expressions = {}
    for block in re.findall(r"```text\n(.*?)```", text, flags=re.S):
        current, fragments = None, []
        for line in block.splitlines() + [""]:
            assignment = re.match(r"^(R1[4-8]|B(?:[1-9]|1[0-2]))\s*=\s*(.*)", line)
            if assignment or (not line.strip()):
                if current:
                    expressions.setdefault(current, " ".join(fragments))
                current, fragments = None, []
            if assignment:
                current, fragments = assignment.group(1), [assignment.group(2)]
            elif current:
                fragments.append(line.strip())
    expected = [f"R{i}" for i in range(14, 19)] + [f"B{i}" for i in range(1, 13)]
    if any(name not in expressions for name in expected):
        raise ValueError("An original equation or boundary residual is missing.")
    return expressions


def rendered_equations(text):
    expressions = equation_expressions(text)
    expected = [f"R{i}" for i in range(14, 19)] + [f"B{i}" for i in range(1, 13)]
    output = ['<section id="rendered-equations"><h2>Original equation and boundary errors, rendered</h2>',
              '<p>Each row is an error expression whose target is zero. Multi-line expressions are read together. Parentheses show the exact arithmetic grouping; the shorter q, h and A definitions are in the calculation sheet.</p>',
              '<p>Superscript (1), (2), (3) and (4) mean derivative order. A plain superscript 2 or 3 means a power. All derivatives are with respect to eta. An endpoint (0) or (L) means evaluation at that position.</p>']
    for name in expected:
        tree = ast.parse(expressions[name], mode="eval").body
        label = "Equation " + name[1:] if name.startswith("R") else "Boundary error " + name[1:]
        output.append(f'<details><summary>{escape(label)}: click to read its exact error</summary><div class="equation-stack">')
        terms = signed_terms(tree)
        for index, (sign, node) in enumerate(terms):
            prefix = f'<mi>{name}</mi><mo>=</mo>' if index == 0 else ""
            prefix += '<mo>−</mo>' if sign < 0 else ('<mo>+</mo>' if index else "")
            output.append('<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow>' + prefix + markup(node) + '</mrow></math>')
        output.append('</div><p><strong>Read:</strong> ' + ("assemble the signed terms above at the same eta; the complete Equation " + name[1:] + " error should be zero." if name.startswith("R") else "evaluate this expression at the endpoint shown; boundary error " + name[1:] + " should be zero.") + '</p></details>')
    output.append('</section>')
    return "\n".join(output)
