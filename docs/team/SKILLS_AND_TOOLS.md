# Skills and tools

Date: 2026-11-19. Checked by listing `~/.claude/skills` and the skill list offered to the agent in this Claude Code session. Descriptions below are the installed skills' own one-line descriptions, shortened, not guesses.

## 1. How to use a skill

In Claude Code, type `/<skill-name>` in the message box, or ask the agent to use the skill by name (it calls its Skill tool). Plugin skills have a prefix, for example `anthropic-skills:deep-research`.

## 2. Skills you asked about

| You asked for | Installed? | Use it for | Invoke |
|---|---|---|---|
| Spec or PRD skill | **Not available.** No skill named spec or PRD is installed. Closest: `doc-coauthoring`, "a structured workflow for co-authoring documentation", including "technical specs" | Interactive drafting of a new spec with the user | `/doc-coauthoring` |
| caveman | Yes: `caveman`, `caveman-help`, `caveman-review` | Short, terse answers to save tokens; compressed one-line code review | `/caveman`, stop with "stop caveman"; `/caveman-review` |
| ponytail | Yes: `ponytail`, `ponytail-review`, `ponytail-audit`, `ponytail-debt`, `ponytail-gain`, `ponytail-help` | "the smallest change that fully solves the task" while coding; quality review of a change; whole-repo audit | `/ponytail`, `/ponytail-review`, `/ponytail-audit` |
| Implementation skill | **Not available** under that name. Installed coding skills: `ponytail` (above), `code-reviewer` (review local changes or PRs), and the built-in commands `/code-review` and `/simplify` | Writing and reviewing code | As named |
| Academic research skills | Yes, see section 3 | | |

## 3. Research and numerical skills that fit this project

| Skill | Use it for here | Invoke |
|---|---|---|
| `benchmark-and-mms-planner` | Planning manufactured-solution and benchmark verification with pass/fail rules. Already used for gates G1 to G6 | `/benchmark-and-mms-planner` |
| `convergence-study` | Observed order, Richardson extrapolation and grid convergence index for the reference study (SPEC V4 and V5) | `/convergence-study` |
| `differentiation-schemes` | Choosing derivative stencils, for example in an optional finite-difference cross-check | `/differentiation-schemes` |
| `numerical-stability` | Stiffness and conditioning checks (the large wall coefficients) | `/numerical-stability` |
| `linear-solvers` | Conditioning of the highest-derivative matrix M | `/linear-solvers` |
| `sympy` | The independent SymPy transcription for residual tests | `/sympy` |
| `statistical-analysis` | Comparing arms over 5 paired seeds; effect sizes | `/statistical-analysis` |
| `scientific-critical-thinking` | Checking claims and experiment design before writing them up | `/scientific-critical-thinking` |
| `scientific-visualization`, `matplotlib`, `data-visualization` | Publication figures FB1 to FB5, FX1 | As named |
| `methodology-flowchart` | One-page methodology diagrams in our house style | `/methodology-flowchart` |
| `markitdown` | Reading PDFs and DOCX files cheaply (for example the supervisor's draft) | `/markitdown` |
| `anthropic-skills:deep-research` | Multi-source literature research. It runs research subagents, so it costs many tokens; ask before using it | `/anthropic-skills:deep-research` |
| `agent-tutor` | Learning a topic with quizzes and spaced repetition (for team members) | `/learn` |

Other installed skills (`humanizer`, `teach`, `socrates`, `scientific-slides`, `presentation-design` and others) are outside this implementation work.

## 4. Core technologies and official documentation

All links returned HTTP 200 on 2026-11-19.

| Tool | Role here | Documentation |
|---|---|---|
| Python 3.12 | Language | https://docs.python.org/3.12/ |
| uv | Environment and lock file | https://docs.astral.sh/uv/ |
| NumPy | Arrays, reference evaluator | https://numpy.org/doc/stable/ |
| SciPy `solve_bvp` | Primary reference solver | https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html |
| PyTorch | PINN, autograd derivatives | https://docs.pytorch.org/docs/stable/index.html |
| SymPy | Independent residual transcription for tests | https://docs.sympy.org/latest/index.html |
| Matplotlib | Figures | https://matplotlib.org/stable/ |
| pytest | Tests | https://docs.pytest.org/en/stable/ |
