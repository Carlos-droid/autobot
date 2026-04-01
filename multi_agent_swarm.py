"""
multi_agent_swarm.py — Orquestador evolutivo con QueryEngine, Compactación y KAIROS.
Evolución de single_agent_swarm.py integrando los patrones de Claude Code.

Uso: python multi_agent_swarm.py [--rounds N] [--local] [--threshold TOKENS]

Mejoras vs single_agent_swarm.py:
1. Autocompactación de 3 niveles (pruning + summarize + KAIROS)
2. Memoria persistente inyectada en System Prompt
3. QueryEngine con soporte multi-backend
4. Consolidación periódica (cada N ciclos)
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

from core.compactor import Compactor
from core.memory_manager import MemoryManager
from core.query_engine import QueryEngine
from core.token_utils import estimate_tokens

# ── Configuración CLI ──────────────────────────────────────────────────────

def get_arg(name: str, default: Any = None) -> Any:
    """Extrae un argumento CLI por nombre."""
    if name in sys.argv:
        idx = sys.argv.index(name) + 1
        if idx < len(sys.argv):
            return sys.argv[idx]
    return default

RUNS = int(get_arg("--rounds", 20))
USE_LOCAL = "--local" in sys.argv
CONTEXT_THRESHOLD = int(get_arg("--threshold", 80000))
CONSOLIDATION_EVERY = int(get_arg("--consolidate-every", 10))

# ── Goals (heredados de single_agent_swarm.py) ─────────────────────────────

GOAL_BINARY = ["add", "subtract", "multiply", "divide", "modulo", "power"]
GOAL_UNARY = ["negate", "absolute", "sqrt", "log", "sin", "cos", "tan"]
GOAL = GOAL_BINARY + GOAL_UNARY

SRC = Path("core/operations.py")
TEST = Path("tests/test_basic.py")
CALC = Path("core/calculator.py")
EXPS = Path("experiments")

MIN_PASS = 0.9
MAX_IDLE = 2
MIN_ROBUST_DELTA = 0.05
ROBUST_THRESHOLD = 0.7

# ── State ──────────────────────────────────────────────────────────────────

state = {
    "cycle": 0, "score": 0.0, "pass_rate": 0.0, "robustness": 0.0,
    "elapsed": 0.0, "ops_done": [], "improved": [], "failed": {},
}

# ── Funciones heredadas (sin cambios) ──────────────────────────────────────

def ensure_files() -> None:
    """Bootstrap: crea archivos iniciales si no existen."""
    for d in ("core", "tests"):
        Path(d).mkdir(exist_ok=True)
    EXPS.mkdir(exist_ok=True)
    if not SRC.exists():
        SRC.write_text(
            "import math\n\ndef add(a, b):\n    return a + b\n\n"
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


def regenerate_calculator() -> None:
    """Mantiene calculator.py sincronizado con operations.py."""
    try:
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
    except (FileNotFoundError, SyntaxError):
        return
    implemented = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    sym = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "modulo": "%", "power": "**"}
    binary_entries = "\n".join(f'    "{sym[f]}": {f},' for f in GOAL_BINARY if f in implemented and f in sym)
    unary_entries = "\n".join(f'    "{f}": {f},' for f in GOAL_UNARY if f in implemented)
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


def robustness_score() -> float:
    """Score 0.0–1.0 averaged across all functions in SRC.

    Per-function breakdown (weights sum to 1.0):
      0.4  ast.Try   — try/except block handles runtime errors
      0.3  isinstance call — validates input types at entry
      0.2  type hints — annotated args signal design intent
      0.1  if+raise guard — domain guard (e.g. b==0, a<0)
    """
    try:
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
    except (FileNotFoundError, SyntaxError):
        return 0.0
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not functions:
        return 0.0
    
    total_score = 0.0
    for fn in functions:
        fs = 0.0
        if any(isinstance(x, ast.Try) for x in ast.walk(fn)): fs += 0.4
        if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "isinstance" for x in ast.walk(fn)): fs += 0.3
        if any(arg.annotation is not None for arg in fn.args.args): fs += 0.2
        if any(isinstance(x, ast.If) and any(isinstance(s, ast.Raise) for s in ast.walk(x)) for x in ast.walk(fn)): fs += 0.1
        total_score += fs
    return round(total_score / len(functions), 2)


def composite_score(rate: float, robust: float, elapsed: float) -> float:
    """Score compuesto: 60% correctness + 20% robustness + 20% speed."""
    perf = 1.0 / (1.0 + elapsed)
    return 0.6 * rate + 0.2 * robust + 0.2 * perf


def run_tests() -> tuple:
    """Ejecuta tests y retorna (pass_rate, elapsed_time)."""
    start_time = time.perf_counter()
    report = Path("report.json")
    report.unlink(missing_ok=True)

    subprocess.run(
        ["python", "-m", "pytest", "tests/test_basic.py", "-q", "--tb=no",
         "--json-report", f"--report-file={report}"],
        capture_output=True, timeout=30,
    )
    elapsed = time.perf_counter() - start_time

    if report.exists():
        try:
            s = json.loads(report.read_text(encoding="utf-8")).get("summary", {})
            total = s.get("total", 0)
            return (s.get("passed", 0) / total if total else 0.0, elapsed)
        except (json.JSONDecodeError, KeyError):
            pass

    out = subprocess.run(
        ["python", "-m", "pytest", "tests/test_basic.py", "-v", "--tb=no", "--no-header"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    ok = out.count(" PASSED")
    bad = out.count(" FAILED") + out.count(" ERROR")
    rate = ok / (ok + bad) if (ok + bad) else 0.0
    return rate, elapsed


def save_snapshot(fn: str, rate: float, robust: float, elapsed: float,
                  change_type: str = "new_function") -> None:
    """Guarda un snapshot del estado actual en experiments/."""
    snap = EXPS / f"exp_{state['cycle']:03d}"
    snap.mkdir(exist_ok=True)
    shutil.copy(SRC, snap / SRC.name)
    shutil.copy(TEST, snap / TEST.name)
    (snap / "meta.json").write_text(json.dumps({
        "cycle": state["cycle"], "function": fn,
        "type": change_type,
        "pass_rate": rate, "robustness": robust, "elapsed": elapsed,
        "score": composite_score(rate, robust, elapsed),
        "ops_done": state["ops_done"],
        "improved": state["improved"],
    }, indent=2), encoding="utf-8")

# ── Backends de LLM ───────────────────────────────────────────────────────

class OllamaClient:
    """Cliente Ollama local para modelos pequeños."""
    def __init__(self, model="deepseek-v3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        import requests
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        resp = requests.post(self.url, json=payload, timeout=120)
        return resp.json().get("response", "")


def create_summarize_fn(client: Any, use_local: bool) -> callable:
    """Crea la función de summarización para el Compactor (Nivel 2)."""
    if use_local:
        return lambda prompt: client.generate(prompt)
    else:
        def anthropic_summarize(prompt: str) -> str:
            resp = client.messages.create(
                model="claude-haiku-4-20250414",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else ""
        return anthropic_summarize


def create_consolidation_fn(client: Any, use_local: bool) -> callable:
    """Crea la función KAIROS para el MemoryManager (Nivel 3)."""
    if use_local:
        return lambda prompt: client.generate(prompt)
    else:
        def anthropic_consolidate(prompt: str) -> str:
            resp = client.messages.create(
                model="claude-haiku-4-20250414",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else ""
        return anthropic_consolidate

# ── Prompt Builder ─────────────────────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "propose_change",
    "description": "One change per cycle. Both fields must be complete Python files.",
    "input_schema": {
        "type": "object",
        "required": ["reasoning", "new_file_content", "new_test_content", "function_name"],
        "properties": {
            "reasoning":        {"type": "string"},
            "new_file_content": {"type": "string", "description": "Complete operations.py"},
            "new_test_content": {"type": "string", "description": "Complete test_basic.py"},
            "function_name":    {"type": "string"},
        },
    },
}


def build_task_prompt(memory_section: str) -> str:
    """Construye el prompt de tarea con contexto actual."""
    remaining_b = [op for op in GOAL_BINARY if op not in state["ops_done"]]
    remaining_u = [op for op in GOAL_UNARY if op not in state["ops_done"]]
    improvable = [op for op in state["ops_done"] if op not in state["improved"]]
    failures = "\n".join(f"  - {f}: failed {n}x" for f, n in state["failed"].items())

    if state["robustness"] < ROBUST_THRESHOLD and improvable:
        priority = f"PRIORITY: robustness={state['robustness']:.0%} < {ROBUST_THRESHOLD:.0%}. IMPROVE existing: {improvable}"
    elif remaining_b or remaining_u:
        priority = f"PRIORITY: robustness OK ({state['robustness']:.0%}). ADD new function."
    else:
        priority = f"All implemented. IMPROVE robustness: {improvable or 'all done'}"

    return textwrap.dedent(f"""
        You are basic_agent. Add or improve ONE function per cycle.

        Remaining binary ops (a, b): {remaining_b}
        Remaining unary ops (a):     {remaining_u}
        Already implemented:         {state['ops_done']}
        Already improved:            {state['improved']}

        {priority}

        Rules:
        - Pick ONE op — new or existing to improve
        - Binary: (a, b); Unary: (a)
        - Return COMPLETE file contents — not snippets
        - Preserve ALL existing functions and tests
        - Error guards rewarded: divide/modulo raise ZeroDivisionError for b==0
        - Use pytest.approx for float comparisons
        - Include `import math` at top of operations.py
        {"Do NOT repeat these failures:" + chr(10) + failures if failures else ""}

        pass_rate: {state['pass_rate']:.0%} | robustness: {state['robustness']:.0%}
        speed: {state['elapsed']:.3f}s | score: {state['score']:.0%}

        {memory_section}

        {SRC}:
        ```python
{SRC.read_text(encoding="utf-8")}```

        {TEST}:
        ```python
{TEST.read_text(encoding="utf-8")}```

        {"Return JSON: {{reasoning, function_name, new_file_content, new_test_content}}" if USE_LOCAL else "Call propose_change now."}
    """).strip()

# ── Ask Function (adaptada para ambos backends) ───────────────────────────

def ask(client: Any, memory_section: str) -> Optional[dict]:
    """Genera una propuesta del agente."""
    # Guía estructural reforzada para modelos locales (Ollama) que no soportan Tool Calling nativo
    local_guide = textwrap.dedent("""
        IMPORTANT: Return ONLY a valid JSON object. 
        Do NOT include any explanations, markdown code blocks, or text before/after the JSON.
        Format: {"reasoning":"...", "function_name":"...", "new_file_content":"...", "new_test_content":"..."}
    """).strip()

    prompt = build_task_prompt(memory_section)
    if USE_LOCAL:
        prompt += f"\n\n{local_guide}"

    if USE_LOCAL:
        raw = client.generate(prompt)
        try:
            # Limpieza agresiva de bloques de código markdown y ruido
            clean = raw.strip()
            if "```" in clean:
                # Extraer contenido entre los bloques de código si existen
                parts = clean.split("```")
                for p in parts:
                    if p.strip().startswith("{") or "{" in p:
                        clean = p.strip()
                        if clean.startswith("json"): clean = clean[4:].strip()
                        break
            
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start == -1 or end == 0:
                print(f"  [ERR] No se encontró estructura JSON en la respuesta.")
                return None
            
            json_str = clean[start:end]
            # Limpiar caracteres de control que rompen JSON (new lines no escapadas)
            # El modelo a veces envía saltos de línea literales dentro de strings de JSON
            return json.loads(json_str, strict=False)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  [ERR] Error parseando JSON de modelo local: {e}")
            if len(raw) > 50:
                print(f"  [DEBUG] Inicio de respuesta: {raw[:100]}...")
            return None

    import anthropic
    resp = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=2500,
        tools=[TOOL_SCHEMA], tool_choice={"type": "any"},
        messages=[{"role": "user", "content": prompt}],
    )
    blk = next((b for b in resp.content if b.type == "tool_use"), None)
    return dict(blk.input) if blk else None

# ── Main Loop ──────────────────────────────────────────────────────────────

def main() -> None:
    ensure_files()

    # Inicializar backend
    if USE_LOCAL:
        client = OllamaClient()
    else:
        import anthropic
        client = anthropic.Anthropic()

    # Inicializar componentes del motor
    summarize_fn = create_summarize_fn(client, USE_LOCAL)
    consolidation_fn = create_consolidation_fn(client, USE_LOCAL)

    compactor = Compactor(
        threshold=CONTEXT_THRESHOLD,
        summarize_fn=summarize_fn,
    )

    # Migrar memoria legacy si existe
    MemoryManager.migrate_legacy_memory()

    memory = MemoryManager(
        consolidation_interval=CONSOLIDATION_EVERY,
        consolidate_fn=consolidation_fn,
    )

    # Historial global para consolidación
    full_history: list = []
    idle = 0

    print(
        f"🚀 multi_agent_swarm  rounds={RUNS}  threshold={CONTEXT_THRESHOLD}  "
        f"consolidate_every={CONSOLIDATION_EVERY}  backend={'ollama' if USE_LOCAL else 'anthropic'}\n"
    )

    for r in range(1, RUNS + 1):
        # Inyectar memoria en el contexto del agente
        memory_section = memory.build_system_prompt_section()

        proposal = ask(client, memory_section)
        if not proposal:
            print(f"[{r}] no proposal returned")
            idle += 1
        else:
            fn = proposal.get("function_name", "?")
            reason = proposal.get("reasoning", "")[:55]
            print(f"[{r}] {fn:12s}  \"{reason}\"", end="  ")

            # Guardar estado actual para rollback
            old_src = SRC.read_text(encoding="utf-8")
            old_test = TEST.read_text(encoding="utf-8")
            SRC.write_text(proposal["new_file_content"], encoding="utf-8")
            TEST.write_text(proposal["new_test_content"], encoding="utf-8")

            rate, elapsed = run_tests()
            robust = robustness_score()
            score = composite_score(rate, robust, elapsed)
            is_new = fn not in state["ops_done"]

            # Registrar en historial para KAIROS
            full_history.append({
                "role": "assistant",
                "content": f"[{fn}] rate={rate:.0%} robust={robust:.0%} score={score:.0%} — {reason}",
            })

            if is_new:
                if score >= state["score"] and rate >= MIN_PASS:
                    state.update(cycle=state["cycle"] + 1, score=score,
                                 pass_rate=rate, robustness=robust, elapsed=elapsed)
                    state["ops_done"].append(fn)
                    state["failed"].pop(fn, None)
                    regenerate_calculator()
                    save_snapshot(fn, rate, robust, elapsed, "new_function")
                    idle = 0
                    print(f"ACCEPTED  rate={rate:.0%}  robust={robust:.0%}  score={score:.0%}")
                else:
                    SRC.write_text(old_src, encoding="utf-8")
                    TEST.write_text(old_test, encoding="utf-8")
                    state["failed"][fn] = state["failed"].get(fn, 0) + 1
                    idle += 1
                    rej = "below MIN_PASS" if rate < MIN_PASS else "no score improvement"
                    print(f"REJECTED  [{rej}]  rate={rate:.0%}  score={score:.0%}")
                    full_history.append({"role": "system", "content": f"REJECTED {fn}: {rej}"})
            else:
                robust_delta = robust - state["robustness"]
                if rate >= MIN_PASS and robust_delta >= MIN_ROBUST_DELTA:
                    old_robust = state["robustness"]
                    state.update(cycle=state["cycle"] + 1, score=score,
                                 pass_rate=rate, robustness=robust, elapsed=elapsed)
                    if fn not in state["improved"]:
                        state["improved"].append(fn)
                    state["failed"].pop(fn, None)
                    regenerate_calculator()
                    save_snapshot(fn, rate, robust, elapsed, "robustness_improvement")
                    idle = 0
                    print(f"IMPROVED  robust={old_robust:.0%}→{robust:.0%}  rate={rate:.0%}")
                else:
                    SRC.write_text(old_src, encoding="utf-8")
                    TEST.write_text(old_test, encoding="utf-8")
                    state["failed"][fn] = state["failed"].get(fn, 0) + 1
                    idle += 1
                    rej = ("below MIN_PASS" if rate < MIN_PASS
                           else f"robustness Δ={robust_delta:+.2f} < {MIN_ROBUST_DELTA}")
                    print(f"REJECTED  [{rej}]  rate={rate:.0%}  robust={robust:.0%}")

        # ── Compactación del historial global ──────────────────────────────
        full_history = compactor.compact_if_needed(full_history)

        # ── Consolidación KAIROS ───────────────────────────────────────────
        if memory.should_consolidate():
            memory.consolidate(full_history)
            full_history = []  # Purga post-consolidación

        # ── Convergencia ───────────────────────────────────────────────────
        if idle >= MAX_IDLE:
            print(f"\nConverged — {idle} consecutive rounds without improvement.")
            break

    # ── Resumen Final ──────────────────────────────────────────────────────
    remaining = [op for op in GOAL if op not in state["ops_done"]]
    print(f"\nops_done    : {state['ops_done']}")
    print(f"improved    : {state['improved']}")
    print(f"remaining   : {remaining}")
    print(f"final score : {state['score']:.0%}  "
          f"(pass={state['pass_rate']:.0%}  robust={state['robustness']:.0%})")
    print(f"cycles      : {state['cycle']}")

    # ── Smoke test ─────────────────────────────────────────────────────────
    print("\n── calculator.py smoke test ──")
    try:
        import importlib
        if "core.calculator" in sys.modules:
            importlib.reload(sys.modules["core.calculator"])
        from core.calculator import evaluate, BINARY, UNARY

        passed = 0
        if "+" in BINARY:
            result = evaluate("2 + 3")
            assert result == 5.0, f"expected 5.0, got {result}"
            passed += 1
            print(f"  ✓ evaluate('2 + 3') = {result}")
        if UNARY:
            first_unary = next(iter(UNARY))
            result = evaluate(f"{first_unary} 4")
            passed += 1
            print(f"  ✓ evaluate('{first_unary} 4') = {result}")
        if passed == 0:
            print("  ⚠ no operations available to test")
        else:
            print(f"  smoke test PASSED ({passed} expressions evaluated)")
    except Exception as e:
        print(f"  ✗ smoke test FAILED — {e}")


if __name__ == "__main__":
    main()
