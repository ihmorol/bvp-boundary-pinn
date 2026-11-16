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
training = json.loads((ROOT / "configs/training-proposal.json").read_text())
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
if training["loss"]["equation_components"] != 5 or training["loss"]["boundary_components"] != 12:
    errors.append("The loss component counts are incorrect.")
for name in ["initial_equation_weights", "initial_boundary_weights"]:
    expected = 5 if name == "initial_equation_weights" else 12
    values = training["loss"][name]
    if len(values) != expected or any(v <= 0 for v in values):
        errors.append(f"Invalid positive {name}.")
if training["reference"]["used_as_training_labels"]:
    errors.append("Baseline reference cannot be used as training labels.")
if training["model"]["output_fields"] != ["f", "g", "theta", "phi", "chi"]:
    errors.append("Output order does not match the equation sheet.")
for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
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
for path in [ROOT / "README.md", ROOT / "AGENTS.md", *sorted((ROOT / "docs").glob("*.md"))]:
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
print("Numerical settings and acceptance scales remain proposals. Definition questions remain open.")
print("SOFTWARE VALIDATION PENDING: no solver, reference simulation or training test was run.")
