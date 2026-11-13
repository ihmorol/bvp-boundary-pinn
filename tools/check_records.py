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
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
        errors.append(f"{key} is not a finite numeric value.")
for key in ["Da", "kappa", "b", "Pr", "Le", "Nb", "Lb"]:
    if p.get(key) is not None and p[key] <= 0:
        errors.append(f"{key} must be positive for the regular baseline.")
for key in ["Delta", "B", "beta_m", "N1", "Bi", "Rd"]:
    if p.get(key) is not None and p[key] < 0:
        errors.append(f"{key} violates the recorded nonnegative/positive source condition.")
if p["theta_w"] is not None and (p["theta_w"] <= 0 or p["theta_w"] == 1):
    errors.append("theta_w must be positive and avoid the original formula's unit-ratio division.")
if p["Pr"] != 6.8 or p["Le"] not in [1, 2, 3, 4] or p["delta_star"] not in [0, 1, 2]:
    errors.append("Fixed/listed source choices were changed outside this pending pilot specification.")
if p["ell"] is not None and not 0 <= p["ell"] <= 1:
    errors.append("ell is outside the recorded interval.")
if p["fw"] is not None and not -1 <= p["fw"] <= 1:
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
print("RECORD CHECK PASSED: names, values, source choices, weights, links and available source hashes.")
print("PHYSICAL CASE BLOCKED: missing " + ", ".join(missing))
print("Numerical settings and acceptance scales remain proposals. No solver or training test was run.")
