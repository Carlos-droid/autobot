"""single_agent_swarm.py — AutoResearch loop, todas las correcciones incorporadas.
Usage: python single_agent_swarm.py [--rounds N]   (requires ANTHROPIC_API_KEY)

Correcciones aplicadas:
  1. Robustez como segunda métrica (además de pass_rate)
  2. GOAL_BINARY / GOAL_UNARY — aridad explícita en el prompt
  3. regenerate_calculator() — calculator.py generado automáticamente
  4. MIN_PASS = 0.9 — umbral mínimo de calidad
  5. Terminación por convergencia (idle_rounds), no por lista fija
"""
from typing import Any
import ast, json, shutil, subprocess, sys, textwrap, requests, time
from pathlib import Path

# ── Cliente Ollama con Soporte para DeepSeek ────────────────────────────────
class OllamaClient:
    def __init__(self, model="qwen2.5:7b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
    
    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            # Aumentamos a 10 min (600s) para modelos grandes que piensan mucho
            resp = requests.post(self.url, json=payload, timeout=600)
            if resp.status_code != 200:
                print(f"  [ERR] Ollama API error: {resp.status_code} - {resp.text}")
                return ""
            
            data = resp.json()
            # Algunos modelos localizados pueden usar 'message' en lugar de 'response'
            content = data.get("response", "")
            if not content and "message" in data:
                content = data["message"].get("content", "")
            
            return content
        except Exception as e:
            print(f"  [ERR] Error de conexión con Ollama: {e}")
            return ""

USE_LOCAL = "--local" in sys.argv

# ── Corrección 2: separar por aridad, no por "nivel" ─────────────────────────
GOAL_BINARY = ["add","subtract","multiply","divide","modulo","power"]
GOAL_UNARY  = ["negate","absolute","sqrt","log","sin","cos","tan"]
GOAL        = GOAL_BINARY + GOAL_UNARY

SRC  = Path("core/operations.py")
TEST = Path("tests/test_basic.py")
CALC = Path("core/calculator.py")
EXPS = Path("experiments")

RUNS     = int(sys.argv[sys.argv.index("--rounds")+1]) if "--rounds" in sys.argv else 20
MIN_PASS = 0.9          # corrección 4: umbral mínimo de calidad
MAX_IDLE = 2            # corrección 5: rondas sin mejora antes de terminar

TOOL = {
    "name": "propose_change",
    "description": "One change per cycle. Both fields must be complete Python files, not snippets.",
    "input_schema": {
        "type": "object",
        "required": ["reasoning","new_file_content","new_test_content","function_name","improvement_type"],
        "properties": {
            "reasoning":        {"type":"string"},
            "new_file_content": {"type":"string", "description":"Complete operations.py"},
            "new_test_content": {"type":"string", "description":"Complete test_basic.py"},
            "function_name":    {"type":"string"},
            "improvement_type": {
                "type":"string",
                "enum":["new_function","robustness"],
                "description":"new_function when adding an op; robustness when improving an existing one",
            },
        },
    },
}

# ── Bootstrap ─────────────────────────────────────────────────────────────────
def ensure_files() -> None:
    """Create baseline files so the script runs from a clean directory."""
    for d in ("core","tests"): Path(d).mkdir(exist_ok=True)
    EXPS.mkdir(exist_ok=True)
    if not SRC.exists():
        SRC.write_text(
            "import math\n\n"
            "def add(a, b):\n    return a + b\n\n"
            "def subtract(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
    if not TEST.exists():
        TEST.write_text(
            "import pytest\nfrom core.operations import *\n\n"
            "def test_add():      assert add(1,2) == 3\n"
            "def test_subtract(): assert subtract(5,3) == 2\n",
            encoding="utf-8",
        )
    regenerate_calculator()

# ── Corrección 3: calculator.py generado automáticamente ──────────────────────
def regenerate_calculator() -> None:
    """Build calculator.py from whatever ops are implemented — no agent touches this."""
    try:
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
    except (FileNotFoundError, SyntaxError):
        return
    implemented = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    sym = {"add":"+","subtract":"-","multiply":"*","divide":"/","modulo":"%","power":"**"}
    binary_entries = "\n".join(
        f'    "{sym[f]}": {f},' for f in GOAL_BINARY if f in implemented and f in sym
    )
    unary_entries = "\n".join(
        f'    "{f}": {f},' for f in GOAL_UNARY if f in implemented
    )
    CALC.write_text(textwrap.dedent(f"""\
        from core.operations import *

        BINARY = {{
        {binary_entries}
        }}
        UNARY = {{
        {unary_entries}
        }}

        def evaluate(expr: str) -> float:
            t = expr.strip().split()
            if len(t) == 3: return BINARY[t[1]](float(t[0]), float(t[2]))
            if len(t) == 2: return UNARY[t[0]](float(t[1]))
            raise ValueError(f"Cannot parse: {{expr!r}}")
    """), encoding="utf-8")

# ── Cambio 1: robustez multi-dimensión (4 señales, pesos distintos) ──────────
def robustness_score() -> float:
    """Score 0.0–1.0 averaged across all functions in SRC.

    Per-function breakdown (weights sum to 1.0):
      0.4  ast.Try   — try/except block handles runtime errors
      0.3  isinstance call — validates input types at entry
      0.2  type hints — annotated args signal design intent
      0.1  if+raise guard — domain guard (e.g. b==0, a<0)
      0.0  PENALTY if eval/exec present — security invalidation
    """
    try:
        src  = SRC.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (FileNotFoundError, SyntaxError):
        return 0.0

    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not functions:
        return 1.0  # no functions → nothing to break

    total_score = 0.0
    for fn in functions:
        fs = 0.0

        # 0.4 — try/except block
        if any(isinstance(n, ast.Try) for n in ast.walk(fn)):
            fs += 0.4

        # 0.3 — isinstance() call in body
        if any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "isinstance"
            for n in ast.walk(fn)
        ):
            fs += 0.3

        # 0.2 — at least one annotated argument
        if any(arg.annotation is not None for arg in fn.args.args):
            fs += 0.2

        # 0.1 — if-block that contains a raise (domain guard)
        if any(
            isinstance(n, ast.If)
            and any(isinstance(s, ast.Raise) for s in ast.walk(n))
            for n in ast.walk(fn)
        ):
            fs += 0.1

        # PENALTY — eval/exec invalidates the function entirely
        if any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in {"eval", "exec"}
            for n in ast.walk(fn)
        ):
            fs = 0.0

        total_score += fs

    return round(total_score / len(functions), 2)

def composite_score(rate: float, robust: float) -> float:
    return 0.7 * rate + 0.3 * robust

def run_tests() -> tuple[float, str]:
    """Return (pass_rate 0.0–1.0, failure_reason str).

    failure_reason is the pytest short-traceback output when tests fail,
    empty string when all pass. This is stored in state["failed"][fn]["last_reason"]
    so the agent receives semantic feedback, not just a failure count.

    Primary path: pytest-json-report for exact structured counts.
    Fallback: verbose output marker counting. Never estimates.
    """
    report = Path("report.json")
    report.unlink(missing_ok=True)          # never read stale report from prior round

    # Run once for JSON report (quiet, no traceback noise)
    subprocess.run(
        ["python", "-m", "pytest", "tests/test_basic.py", "-q", "--tb=no",
         "--json-report", f"--report-file={report}"],
        capture_output=True, timeout=30,
    )
    if report.exists():
        try:
            s = json.loads(report.read_text(encoding="utf-8")).get("summary", {})
            total  = s.get("total", 0)
            passed = s.get("passed", 0)
            rate   = passed / total if total else 0.0
            # If failed, run again with short tb to capture reason
            reason = ""
            if rate < 1.0:
                tb = subprocess.run(
                    ["python", "-m", "pytest", "tests/test_basic.py", "--tb=short", "--no-header", "-q"],
                    capture_output=True, text=True, timeout=30,
                ).stdout
                reason = tb[:600]   # cap at 600 chars — enough for one traceback
            return rate, reason
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: verbose markers
    out = subprocess.run(
        ["python", "-m", "pytest", "tests/test_basic.py", "-v", "--tb=short", "--no-header"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    ok  = out.count(" PASSED")
    bad = out.count(" FAILED") + out.count(" ERROR")
    rate = ok / (ok + bad) if (ok + bad) else 0.0
    reason = out[:600] if rate < 1.0 else ""
    return rate, reason

# ── Snapshot ──────────────────────────────────────────────────────────────────
def save_snapshot(fn: str, rate: float, robust: float, itype: str = "new_function") -> None:
    """Copy source files into experiments/exp_NNN/ with full metadata for diffable history."""
    snap = EXPS / f"exp_{state['cycle']:03d}"
    snap.mkdir(exist_ok=True)
    shutil.copy(SRC,  snap / SRC.name)
    shutil.copy(TEST, snap / TEST.name)
    (snap / "meta.json").write_text(json.dumps({
        "cycle":            state["cycle"],
        "function":         fn,
        "improvement_type": itype,
        "pass_rate":        rate,
        "robustness":       robust,
        "score":            composite_score(rate, robust),
        "ops_done":         state["ops_done"],
        "improved":         state["improved"],
    }, indent=2), encoding="utf-8")

# ── State ─────────────────────────────────────────────────────────────────────
# failed: {fn: {"count": int, "last_reason": str}}
# "last_reason" carries the pytest traceback so the agent avoids repeating
# the same error — upgrades Results Store from amnesic to basic (AutoResearch doc §2.5)
state = {"cycle":0, "score":0.0, "pass_rate":0.0, "robustness":0.0,
         "ops_done":[], "improved":[], "failed":{}}

# ── Cambio 3: roles Builder / Fixer con autonomía híbrida ────────────────────
ROBUSTNESS_FIXER_THRESHOLD = 0.7   # below → Fixer mode; at or above → Builder mode

PERSONAS = {
    "builder": (
        "You are a PRODUCT ENGINEER. Your goal is to implement new math functions. "
        "Focus on correctness, passing all tests, and choosing from the remaining ops list."
    ),
    "fixer": (
        "You are a RELIABILITY ENGINEER (SRE). Your goal is to make existing functions "
        "INDESTRUCTIBLE. Add isinstance checks, try/except blocks, type hints, and "
        "if-raise guards. Do NOT add new functions — improve robustness of existing ones."
    ),
}

def _decide_role() -> str:
    """Autonomy rule (design question 2): Fixer when robustness below threshold,
    Builder otherwise. Returns 'fixer' or 'builder'."""
    remaining = [op for op in GOAL if op not in state["ops_done"]]
    if not remaining:
        return "fixer"   # all ops done — only improvement left
    return "fixer" if state["robustness"] < ROBUSTNESS_FIXER_THRESHOLD else "builder"

def _failure_memory() -> str:
    """Format the last failure reason for each function into a memory block.
    Passes semantic traceback to agent — upgrades Generator from amnesic to basic."""
    if not state["failed"]:
        return ""
    lines = ["Do NOT repeat these failed proposals (with reasons):"]
    for fn, info in state["failed"].items():
        count  = info["count"]
        reason = info["last_reason"][:200] if info["last_reason"] else "score did not improve"
        lines.append(f"  - {fn} (failed {count}x): {reason}")
    return "\n".join(lines)

# ── Agent ─────────────────────────────────────────────────────────────────────
def ask(client: Any) -> dict | None:
    """Call the LLM with a role-specific persona and current state as context."""
    role         = _decide_role()
    persona      = PERSONAS[role]
    remaining_b  = [op for op in GOAL_BINARY if op not in state["ops_done"]]
    remaining_u  = [op for op in GOAL_UNARY  if op not in state["ops_done"]]
    memory_block = _failure_memory()

    if role == "builder":
        task_section = textwrap.dedent(f"""
            Remaining binary ops (two args: a, b) : {remaining_b}
            Remaining unary  ops (one  arg: a)    : {remaining_u}

            Pick ONE op (binary first). Rules:
            - Binary tests: fn(a, b) — Unary tests: fn(a)
            - divide/modulo raise ZeroDivisionError when b==0
            - sqrt/log raise ValueError for invalid domain inputs
            - Use pytest.approx for float comparisons
            - Include `import math` at top of operations.py
            - Set improvement_type to "new_function"
        """).strip()
    else:  # fixer
        task_section = textwrap.dedent(f"""
            Current robustness: {state['robustness']:.2f} (target ≥ {ROBUSTNESS_FIXER_THRESHOLD})
            Ops already implemented: {state['ops_done']}

            Pick ONE existing function and improve its robustness. Add:
            - isinstance() input validation (+0.3 to that function's score)
            - try/except block (+0.4)
            - type hints on args (+0.2)
            - if-raise guard for invalid domain (+0.1)
            The robustness improvement must be ≥ 0.05 to be accepted.
            - Set improvement_type to "robustness"
        """).strip()

    # Guía estructural para modelos locales (Ollama)
    local_guide = textwrap.dedent("""
        IMPORTANT: Return ONLY a valid JSON object. 
        Do NOT include any explanations, markdown code blocks, or text before/after the JSON.
        Format: {"reasoning":"...", "function_name":"...", "new_file_content":"...", "new_test_content":"...", "improvement_type":"..."}
    """).strip()

    prompt = textwrap.dedent(f"""
        {persona}

        ### CURRENT STATE
        Pass rate : {state['pass_rate']:.0%}
        Robustness: {state['robustness']:.2f}
        Score     : {state['score']:.0%}  (0.7×pass + 0.3×robust)
        Ops done  : {state['ops_done']}

        {task_section}

        {memory_block}

        {local_guide if USE_LOCAL else ""}

        ### FILES
        {SRC}:
        ```python
{SRC.read_text(encoding="utf-8")}```

        {TEST}:
        ```python
{TEST.read_text(encoding="utf-8")}```

        {"Return JSON now." if USE_LOCAL else "Call propose_change now."}
    """).strip()

    if USE_LOCAL:
        raw = client.generate(prompt)
        if not raw:
            print("  [DEBUG] Ollama returned an empty response.")
            return None
            
        try:
            clean = raw.strip()
            # Log de depuración para ver qué está intentando enviar el modelo
            print(f"  [DEBUG] Raw response length: {len(clean)} chars")
            
            if "```" in clean:
                parts = clean.split("```")
                for p in parts:
                    if p.strip().startswith("{") or "{" in p:
                        clean = p.strip()
                        if clean.startswith("json"): clean = clean[4:].strip()
                        break
            
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start == -1 or end == 0: 
                print(f"  [DEBUG] No JSON found in raw: {clean[:100]}...")
                return None
            
            json_str = clean[start:end]
            return json.loads(json_str, strict=False)
        except Exception as e:
            print(f"  [DEBUG] JSON Parse Error: {e}")
            return None

    import anthropic
    # Anthropic implementation below
    client_anthropic = anthropic.Anthropic()
    resp = client_anthropic.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=2500,
        tools=[TOOL], tool_choice={"type":"any"},
        messages=[{"role":"user","content":prompt}],
    )
    blk = next((b for b in resp.content if b.type == "tool_use"), None)
    return dict(blk.input) if blk else None

# ── Loop ──────────────────────────────────────────────────────────────────────
ROBUSTNESS_DELTA_MIN = 0.05   # cambio 4: Fixer must improve robustness by at least this

def main() -> None:
    ensure_files()
    
    if USE_LOCAL:
        client = OllamaClient()
    else:
        import anthropic
        client = anthropic.Anthropic()

    idle   = 0
    print(f"single_agent_swarm_2  rounds={RUNS}  min_pass={MIN_PASS:.0%}  "
          f"idle_limit={MAX_IDLE}  rob_threshold={ROBUSTNESS_FIXER_THRESHOLD}  "
          f"backend={'ollama' if USE_LOCAL else 'anthropic'}\n")

    for r in range(1, RUNS + 1):
        role     = _decide_role()
        proposal = ask(client)

        if not proposal:
            print(f"[{r:02d}] no proposal returned"); idle += 1
            if idle >= MAX_IDLE: break
            continue

        fn      = proposal.get("function_name", "?")
        itype   = proposal.get("improvement_type", "new_function")
        reason  = proposal.get("reasoning", "")[:55]
        print(f"[{r:02d}] {role:7s}  {fn:12s}  {itype:14s}  \"{reason}\"", end="  ")

        old_src, old_test = SRC.read_text(encoding="utf-8"), TEST.read_text(encoding="utf-8")
        SRC.write_text(proposal["new_file_content"], encoding="utf-8")
        TEST.write_text(proposal["new_test_content"], encoding="utf-8")

        rate,   fail_reason = run_tests()
        robust = robustness_score()
        score  = composite_score(rate, robust)

        # ── Cambio 4: dual acceptance paths (option C) ────────────────────────
        if itype == "new_function":
            # Builder path: composite score must not regress, pass rate above floor,
            # and the function must be genuinely new
            accepted = (
                score  >= state["score"]
                and rate   >= MIN_PASS
                and fn not in state["ops_done"]
            )
            reject_reason = (
                "below MIN_PASS"         if rate < MIN_PASS else
                "already implemented"    if fn in state["ops_done"] else
                "no score improvement"
            )
        else:
            # Fixer path: robustness delta must be meaningful; pass rate must not drop
            rob_delta = robust - state["robustness"]
            accepted  = (
                rob_delta >= ROBUSTNESS_DELTA_MIN
                and rate  >= state["pass_rate"] - 0.001   # allow rounding noise
            )
            reject_reason = (
                f"rob delta {rob_delta:+.2f} < {ROBUSTNESS_DELTA_MIN}"
                if rob_delta < ROBUSTNESS_DELTA_MIN
                else f"pass rate dropped {state['pass_rate']:.0%}→{rate:.0%}"
            )

        if accepted:
            state["cycle"]     += 1
            state["score"]      = score
            state["pass_rate"]  = rate
            state["robustness"] = robust

            if itype == "new_function":
                state["ops_done"].append(fn)
                state["failed"].pop(fn, None)
                label = "ACCEPTED-NEW"
            else:
                if fn not in state["improved"]:
                    state["improved"].append(fn)
                state["failed"].pop(fn, None)
                label = "ACCEPTED-ROB"

            regenerate_calculator()
            save_snapshot(fn, rate, robust, itype)
            idle = 0
            print(f"{label}  rate={rate:.0%}  robust={robust:.0%}  score={score:.0%}")

        else:
            SRC.write_text(old_src, encoding="utf-8")
            TEST.write_text(old_test, encoding="utf-8")
            # Cambio 2: store semantic reason, not just count
            prev = state["failed"].get(fn, {"count": 0, "last_reason": ""})
            state["failed"][fn] = {
                "count":       prev["count"] + 1,
                "last_reason": fail_reason or reject_reason,
            }
            idle += 1
            print(f"REJECTED  [{reject_reason}]  rate={rate:.0%}  robust={robust:.0%}")

        if idle >= MAX_IDLE:
            print(f"\nConverged — {idle} consecutive rounds without improvement."); break

    remaining = [op for op in GOAL if op not in state["ops_done"]]
    print(f"\nops_done   : {state['ops_done']}")
    print(f"improved   : {state['improved']}")
    print(f"remaining  : {remaining}")
    print(f"final score: {state['score']:.0%}  "
          f"(pass={state['pass_rate']:.0%}  robust={state['robustness']:.0%})")
    print(f"cycles     : {state['cycle']}")

if __name__ == "__main__": main()
