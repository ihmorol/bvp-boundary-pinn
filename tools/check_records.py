from pathlib import Path
import json
import math
import hashlib
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "Delta", "B", "Da", "kappa", "delta_star", "theta_w", "lambda_m",
    "beta_m", "epsilon", "b", "Pr", "Le", "N1", "Bi", "Rd", "Nr",
    "Nt", "Nb", "Lb", "Pe", "ell", "fw",
}
# Sorted, compact JSON of the supervisor values recorded on 2026-11-16.
SUPERVISOR_VALUES_SHA256 = "8a798b4a19c8ceff9870cff7ebc217a5141533b2eb6950b8a93acdb130d025b5"


def is_finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


case = json.loads((ROOT / "configs/physical-case-pending.json").read_text())
pinn = json.loads((ROOT / "configs/pinn.json").read_text())
reference_settings = json.loads((ROOT / "configs/reference.json").read_text())
p = case["physical_parameters"]
errors = []
if set(p) != REQUIRED:
    errors.append("Physical coefficient names do not match the 22-name specification.")
missing = sorted(k for k, v in p.items() if v is None)
if sorted(case["missing_parameters"]) != missing:
    errors.append("The missing-parameter list does not match null values.")
for key, value in p.items():
    if value is not None and not is_finite_number(value):
        errors.append(f"{key} is not a finite numeric value.")

confirmation = case.get("supervisor_confirmation", {})
received = confirmation.get("received_values", {})
explicit_mappings = {
    "bm": "beta_m", "dt": "delta_star", "Dt": "Delta",
    "L": "lambda_m", "tw": "theta_w", "M": "1/Da",
}
notation_aliases = {"k": "kappa", "l": "ell", "ep": "epsilon"}
unmapped = {"H", "Prm"}
if case.get("locked") is not True:
    errors.append("The supervisor case must stay locked.")
if confirmation.get("received_date") != "2026-11-16" or case.get("user_decisions_date") != "2026-11-16":
    errors.append("The lock must refer to the supervisor values received on 2026-11-16.")
expected_status = "blocked_missing_parameters" if missing else "parameters_complete_validation_pending"
if case.get("status") != expected_status:
    errors.append("The case status does not match parameter completeness and pending validation.")
if confirmation.get("explicit_mappings") != explicit_mappings or confirmation.get("notation_aliases") != notation_aliases:
    errors.append("The received symbol mappings do not match the dated parameter record.")
if set(confirmation.get("unmapped_symbols", {})) != unmapped:
    errors.append("H and Prm must stay recorded outside the equation coefficients.")
if not all(is_finite_number(value) for value in received.values()):
    errors.append("Every received supervisor value must be a finite number, including the known zero dt.")
else:
    encoded = json.dumps(received, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(encoded.encode()).hexdigest() != SUPERVISOR_VALUES_SHA256:
        errors.append("The locked supervisor values changed. A new set needs explicit confirmation and a dated record.")
aliases = {**explicit_mappings, **notation_aliases}
for symbol, value in received.items():
    if not is_finite_number(value):
        continue
    if symbol == "M":
        if value <= 0 or p.get("Da") != 1 / value:
            errors.append("Da must equal 1/M from the supervisor record; M=5 gives Da=0.2.")
    elif symbol not in unmapped:
        key = aliases.get(symbol, symbol)
        if p.get(key) != value:
            errors.append(f"{key} does not match the supervisor value for {symbol}.")

for key in ["Da", "kappa", "b", "Pr", "Le", "Nb", "Lb"]:
    if is_finite_number(p.get(key)) and p[key] <= 0:
        errors.append(f"{key} must be positive for the regular baseline.")
for key in ["Delta", "B", "beta_m", "N1", "Bi", "Rd"]:
    if is_finite_number(p.get(key)) and p[key] < 0:
        errors.append(f"{key} violates the recorded nonnegative/positive source condition.")
if is_finite_number(p.get("theta_w")) and (p["theta_w"] <= 0 or p["theta_w"] == 1):
    errors.append("theta_w must be positive and avoid the original formula's unit-ratio division.")
if p.get("Le") not in [1, 2, 3, 4] or p.get("delta_star") not in [0, 1, 2]:
    errors.append("Le or delta_star is outside the recorded starting-case choices.")
if is_finite_number(p.get("ell")) and not 0 <= p["ell"] <= 1:
    errors.append("ell is outside the recorded interval.")
if is_finite_number(p.get("fw")) and not -1 <= p["fw"] <= 1:
    errors.append("fw is outside the recorded interval.")
# The PINN config (configs/pinn.json): same domain as the reference, the boundary mode frozen by
# pilot P2a, every campaign arm defined, five paired seeds (gate G6), a complete acceptance rule.
common = pinn["common"]
if common["L"] != reference_settings["L_reference"]:
    errors.append("The PINN domain end L must equal the reference L_reference.")
if common["windows"][0] != 0 or common["windows"][-1] != common["L"]:
    errors.append("The PINN window edges must run from 0 to L.")
if common["bc_mode"] != "exact":
    errors.append("bc_mode was frozen to exact by pilot P2a (IMPLEMENTATION_NOTES section 7).")
if not set(pinn["campaign"]["arms"]) <= set(pinn["arms"]) or len(set(pinn["campaign"]["seeds"])) != 5:
    errors.append("The campaign needs defined arms and five paired seeds (gate G6).")
rule = pinn.get("selection", {})
if not all(is_finite_number(rule.get(k)) and rule[k] > 0 for k in ("score_max", "tail_share_max", "boundary_max")):
    errors.append("The acceptance rule in configs/pinn.json is incomplete.")
# Every working doc: README, AGENTS and docs/ with its subfolders. docs/archive keeps
# superseded files as they were, so it is left out of these checks.
DOCS = [path for path in sorted((ROOT / "docs").rglob("*.md")) if "archive" not in path.relative_to(ROOT / "docs").parts]
for path in [ROOT / "README.md", *DOCS]:
    text = path.read_text()
    if "<!--" in text:
        errors.append(f"Unfilled marker in {path.relative_to(ROOT)}.")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http:", "https:", "#")):
            continue
        local = target.split("#", 1)[0]
        if local and not (path.parent / local).exists():
            errors.append(f"Broken local link in {path.relative_to(ROOT)}: {target}")
BANNED_WORDS = [
    "delve", "moreover", "furthermore", "underscore", "leverage", "utilize",
    "seamless", "robust", "comprehensive", "pivotal", "transformative",
    "tapestry", "synergy",
]
PROTECTED_PHRASES = [
    # Exact paper titles; the writing rule never rewrites a real title.
    "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks",
    "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs",
]
for path in [ROOT / "README.md", ROOT / "AGENTS.md", *DOCS]:
    text = path.read_text()
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned = re.sub(r"`[^`]*`", " ", cleaned)
    for phrase in PROTECTED_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    for word in BANNED_WORDS:
        if re.search(rf"\b{word}\b", cleaned, flags=re.I):
            errors.append(f"Writing rule: banned word '{word}' in {path.relative_to(ROOT)}")
    if "\u2014" in cleaned:
        errors.append(f"Writing rule: em dash in {path.relative_to(ROOT)}")
manifest_path = ROOT / "sources/manifest.json"
if manifest_path.exists():
    for entry in json.loads(manifest_path.read_text())["files"]:
        source = ROOT / entry["path"]
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(f"Source hash mismatch: {entry['path']}")
if errors:
    print("RECORD CHECK FAILED")
    print("\n".join(errors))
    sys.exit(1)
print("RECORD CHECK PASSED: names, values, supervisor lock, mappings, weights, links and available source hashes.")
if missing:
    print("PHYSICAL CASE BLOCKED: missing " + ", ".join(missing))
else:
    print("PHYSICAL PARAMETERS COMPLETE: 22 equation coefficients from the locked supervisor record, with Da derived from M.")
print("Numerical settings and the acceptance rule are our choices, with dated notes in configs/. Definition questions closed by the supervisor on 2026-11-19: values are used as given.")
print("This tool checks records only. Software checks: uv run pytest. What has run: docs/status-log.md.")
