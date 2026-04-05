"""
AutoResearch v4 — Integración de Playwright-MCP, PydanticAI y Browser-Use
==========================================================================

ANÁLISIS DE REPOS Y QUÉ SE EXTRAE:

┌──────────────────────┬──────────────────────────────────────────────────────────────────┐
│ REPO                 │ PATRÓN EXTRAÍDO                                                  │
├──────────────────────┼──────────────────────────────────────────────────────────────────┤
│ playwright-mcp       │ Accessibility tree (NO screenshots) → token-efficient            │
│ microsoft/           │ MCP server via stdio / HTTP/SSE en puerto 8931                   │
│                      │ Usado para agentes de larga duración con estado persistente      │
│                      │ → Ideal para: self-healing tests, exploración web autónoma       │
│                      │ CLI+SKILLS más eficiente para coding agents (token cost)         │
│                      │ browser_snapshot() devuelve refs de elementos para clicks       │
├──────────────────────┼──────────────────────────────────────────────────────────────────┤
│ pydantic-ai          │ Agent(model, output_type=PydanticModel, instructions=...)        │
│                      │ @agent.tool / @agent.tool_plain decorators                       │
│                      │ Structured output con JSON Schema auto-generado desde Pydantic   │
│                      │ Dependency injection via RunContext[Deps]                        │
│                      │ ModelRetry para retry automático desde tools                     │
│                      │ run_sync() y run() (async)                                       │
│                      │ output_type = BaseModel → el LLM DEBE devolver ese schema       │
├──────────────────────┼──────────────────────────────────────────────────────────────────┤
│ browser-use          │ Agent(task, llm, controller) con Controller personalizable       │
│                      │ @controller.action('descripción') → tool custom                 │
│                      │ Loop: Observe DOM → Decide → Act → Evaluate                     │
│                      │ DOM + Vision: extrae HTML + screenshot simultáneamente           │
│                      │ Multi-tab management integrado                                   │
│                      │ use_vision=True/False (False = solo DOM, más barato)             │
│                      │ 89.1% en WebVoyager benchmark                                    │
└──────────────────────┴──────────────────────────────────────────────────────────────────┘

DECISIÓN ARQUITECTURAL: QUÉ USAR EN AutoResearch

  Playwright-MCP:  → tool: browser_snapshot() para extraer estado de página
                     tool: browser_navigate() + browser_click() para scraping profundo
                     MEJOR QUE Crawl4AI para páginas JS-heavy (SPAs, dashboards)
                     NO reemplaza Crawl4AI para contenido estático (más lento)

  PydanticAI:     → REEMPLAZA LLMClient para llamadas donde necesitamos
                     output estructurado validado (BuilderProposal, FixerProposal)
                     Ventaja: el LLM DEBE devolver JSON válido o falla con retry
                     Dependency injection limpia para pasar memoria/tools a agents

  Browser-Use:    → tool: browser_agent_task() para tareas web complejas
                     "Find Python calculator implementation on GitHub, extract code"
                     Autónomo: planifica multi-step, maneja login, forms, etc.
                     MEJOR QUE scrape_url para tareas que requieren navegación activa

CUÁNDO USAR CADA BROWSER TOOL:
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │ HERRAMIENTA     │ CASO DE USO ÓPTIMO                                   │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │ Crawl4AI        │ Scraping estático rápido, contenido HTML/Markdown    │
  │ Playwright-MCP  │ Páginas JS-heavy, accessibility tree, estado largo   │
  │ Browser-Use     │ Tareas multi-step complejas, navegación autónoma     │
  └─────────────────┴──────────────────────────────────────────────────────┘

DEPS:
    pip install pydantic-ai browser-use anthropic mem0ai requests duckduckgo-search pytest
    pip install playwright && playwright install chromium
    # Para Playwright-MCP (Node.js requerido):
    npx @playwright/mcp@latest --port 8931 --headless  # en proceso separado

ENV:
    ANTHROPIC_API_KEY  — obligatorio
    OPENAI_API_KEY     — Mem0 embeddings
    GITHUB_TOKEN       — GitHub API
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import requests
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def _fh(name: str) -> logging.FileHandler:
    h = logging.FileHandler(LOG_DIR / name, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    return h

log      = logging.getLogger("ar4");        log.setLevel(logging.DEBUG);     log.addHandler(logging.StreamHandler(sys.stdout)); log.addHandler(_fh("ar4.log"))
llm_log  = logging.getLogger("ar4.llm");   llm_log.setLevel(logging.DEBUG);  llm_log.addHandler(_fh("llm_raw.log"))
tool_log = logging.getLogger("ar4.tools"); tool_log.setLevel(logging.DEBUG);  tool_log.addHandler(_fh("tools.log"))
test_log = logging.getLogger("ar4.test");  test_log.setLevel(logging.DEBUG);  test_log.addHandler(_fh("pytest.log"))
mem_log  = logging.getLogger("ar4.mem");   mem_log.setLevel(logging.DEBUG);   mem_log.addHandler(_fh("memory.log"))


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS  (PydanticAI: structured output garantizado por schema)
# ─────────────────────────────────────────────────────────────────────────────

class CodeProposal(BaseModel):
    """
    PydanticAI output_type: el LLM DEBE devolver este schema.
    Si falla la validación → ModelRetry automático.
    """
    code: str = Field(description="Código Python completo de calculator.py")
    reasoning: str = Field(description="Explicación de las decisiones de diseño")
    functions_implemented: list[str] = Field(description="Lista de funciones implementadas")
    edge_cases_handled: list[str] = Field(description="Edge cases manejados explícitamente")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza en la implementación")

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        # Extraer código si viene en bloque markdown
        m = re.search(r"```python\s*(.*?)```", v, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Validación mínima: debe tener al menos una función
        if "def " not in v:
            raise ValueError("El código debe contener al menos una función Python")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return round(v, 2)


class FixProposal(BaseModel):
    """Output estructurado para el Fixer. El LLM indica root cause + fix."""
    fixed_code: str = Field(description="Código Python completo corregido")
    root_cause: str = Field(description="Causa raíz del error identificado")
    changes_made: list[str] = Field(description="Lista de cambios específicos realizados")
    preserved_passing: list[str] = Field(description="Funciones que ya pasaban y se preservaron")

    @field_validator("fixed_code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        m = re.search(r"```python\s*(.*?)```", v, re.DOTALL)
        if m:
            return m.group(1).strip()
        if "def " not in v:
            raise ValueError("El código debe contener funciones Python")
        return v.strip()


class BrowserTaskResult(BaseModel):
    """Output de Browser-Use task."""
    success: bool
    extracted_content: str
    url_visited: str = ""
    actions_taken: int = 0
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC-AI AGENTS  (structured output + dependency injection)
# ─────────────────────────────────────────────────────────────────────────────

def _build_pydantic_builder_agent():
    """
    PydanticAI: Agent con output_type=CodeProposal.
    El LLM es FORZADO a devolver JSON válido matching CodeProposal.
    Si falla → retry automático hasta max_retries.
    """
    try:
        from pydantic_ai import Agent as PydanticAgent

        builder_agent = PydanticAgent(
            "anthropic:claude-opus-4-5",
            output_type=CodeProposal,
            instructions=textwrap.dedent("""
                Eres un experto Python generando implementaciones de calculadora.
                Debes devolver JSON estructurado con el código completo.

                REGLAS DE CÓDIGO:
                - Maneja TypeError para tipos no numéricos
                - Maneja ValueError para divisiones/módulos por cero y sqrt negativo
                - Docstrings en cada función
                - Maneja overflow con float('inf') donde aplique
                - NO incluyas imports de test ni código de test

                Devuelve SIEMPRE el código completo en el campo 'code'.
            """).strip(),
            retries=3,  # PydanticAI: retry automático si validación falla
        )
        return builder_agent
    except ImportError:
        return None


def _build_pydantic_fixer_agent():
    """PydanticAI Fixer: output estructurado con root cause analysis."""
    try:
        from pydantic_ai import Agent as PydanticAgent

        fixer_agent = PydanticAgent(
            "anthropic:claude-opus-4-5",
            output_type=FixProposal,
            instructions=textwrap.dedent("""
                Eres un experto en debugging Python.
                Analiza el error exacto e identifica la causa raíz.
                Devuelve el código COMPLETO corregido.

                REGLAS:
                - NO elimines funciones que ya pasan tests
                - Fix mínimo: cambia solo lo necesario
                - Identifica exactamente qué línea causó el error
            """).strip(),
            retries=3,
        )
        return fixer_agent
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PLAYWRIGHT-MCP CLIENT  (accessibility tree, no screenshots)
# ─────────────────────────────────────────────────────────────────────────────

class PlaywrightMCPClient:
    """
    Cliente para Playwright-MCP server.

    Playwright-MCP usa accessibility tree (NO screenshots):
    - Token-efficient: no visión models necesarios
    - Determinístico: refs de elementos para clicks precisos
    - Ideal para: páginas JS-heavy, SPAs, dashboards

    Arranque del servidor (una sola vez, proceso separado):
        npx @playwright/mcp@latest --port 8931 --headless --browser chromium

    Documentación Microsoft:
    "MCP remains relevant for specialized agentic loops that benefit from
     persistent state, rich introspection, and iterative reasoning over
     page structure — such as exploratory automation or self-healing tests."
    """

    MCP_PORT = int(os.getenv("PLAYWRIGHT_MCP_PORT", "8931"))
    BASE_URL = f"http://localhost:{MCP_PORT}"

    def __init__(self):
        self._server_proc: subprocess.Popen | None = None
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Verifica si el servidor MCP está corriendo."""
        try:
            resp = requests.get(f"{self.BASE_URL}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def start_server(self) -> bool:
        """
        Arranca servidor Playwright-MCP en background.
        Requiere: npm/npx disponible.
        """
        if self._available:
            return True
        try:
            self._server_proc = subprocess.Popen(
                ["npx", "-y", "@playwright/mcp@latest",
                 "--port", str(self.MCP_PORT),
                 "--headless", "--browser", "chromium"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            # Esperar arranque
            for _ in range(15):
                time.sleep(1)
                if self._check_available():
                    log.info(f"[PLAYWRIGHT-MCP] Servidor disponible en :{self.MCP_PORT}")
                    self._available = True
                    return True
            log.warning("[PLAYWRIGHT-MCP] Timeout esperando servidor")
            return False
        except FileNotFoundError:
            log.warning("[PLAYWRIGHT-MCP] npx no disponible. Instala Node.js")
            return False

    def call_tool(self, tool_name: str, params: dict) -> dict:
        """
        Llama a una tool de Playwright-MCP via HTTP.
        Playwright-MCP expone ~70 tools: browser_navigate, browser_click,
        browser_snapshot, browser_type, browser_evaluate, etc.
        """
        if not self._available:
            return {"error": "Playwright-MCP no disponible"}
        try:
            resp = requests.post(
                f"{self.BASE_URL}/tools/{tool_name}",
                json=params, timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def navigate_and_snapshot(self, url: str) -> str:
        """
        Navega a URL y devuelve accessibility tree como texto.
        Playwright-MCP: structured data, NO screenshots, token-efficient.
        """
        nav = self.call_tool("browser_navigate", {"url": url})
        if "error" in nav:
            return f"Navigation error: {nav['error']}"
        snap = self.call_tool("browser_snapshot", {})
        # El snapshot devuelve texto con refs de elementos: [ref=1] Button "Submit"
        return snap.get("content", snap.get("text", str(snap)))[:4000]

    def stop_server(self):
        if self._server_proc:
            self._server_proc.terminate()
            self._server_proc = None


# ─────────────────────────────────────────────────────────────────────────────
# BROWSER-USE INTEGRATION  (DOM + Vision, multi-step autónomo)
# ─────────────────────────────────────────────────────────────────────────────

class BrowserUseClient:
    """
    Browser-Use: agente web autónomo.
    Arquitectura: Agent(task, llm, controller)
    Loop: Observe DOM → Decide → Act → Evaluate

    DIFERENCIA vs Crawl4AI:
    - Crawl4AI: extrae contenido de URL → rápido, estático
    - Browser-Use: navega activamente → formularios, login, multi-step

    DIFERENCIA vs Playwright-MCP:
    - Playwright-MCP: bajo nivel, tú controlas cada step
    - Browser-Use: alto nivel, el LLM decide los steps

    Benchmark: 89.1% en WebVoyager (vs 73.1% Agent-E sin visión)
    """

    def __init__(self):
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            import browser_use  # noqa: F401
            return True
        except ImportError:
            return False

    def run_task(self, task: str, use_vision: bool = False,
                 max_steps: int = 20) -> BrowserTaskResult:
        """
        Ejecuta tarea web autónoma con Browser-Use.

        PydanticAI + Browser-Use: podemos definir output_type=BrowserTaskResult
        para obtener resultado estructurado.

        Args:
            task: Descripción en lenguaje natural de la tarea
            use_vision: Si True envía screenshots al LLM (más potente pero costoso)
            max_steps: Límite de pasos de navegación
        """
        if not self._available:
            return BrowserTaskResult(
                success=False,
                extracted_content="",
                error="browser-use no instalado. pip install browser-use && playwright install"
            )

        try:
            from browser_use import Agent as BrowserAgent, Controller, ActionResult
            from langchain_anthropic import ChatAnthropic

            # Controller custom: podemos añadir acciones personalizadas
            # Browser-Use: @controller.action('descripción')
            controller = Controller()

            @controller.action("Extract code blocks from page")
            def extract_code_blocks() -> ActionResult:
                """Acción custom: extrae todos los bloques de código Python de la página."""
                # Esta función será llamada por el agente cuando lo necesite
                return ActionResult(extracted_content="[code extraction requested]")

            llm = ChatAnthropic(
                model="claude-haiku-4-5-20251001",  # haiku para tareas web (cost)
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )

            agent = BrowserAgent(
                task=task,
                llm=llm,
                controller=controller,
                use_vision=use_vision,
                max_steps=max_steps,
            )

            # Browser-Use es async
            result = asyncio.run(agent.run())

            return BrowserTaskResult(
                success=True,
                extracted_content=str(result)[:3000],
                actions_taken=max_steps,
            )

        except ImportError as e:
            return BrowserTaskResult(
                success=False, extracted_content="",
                error=f"Dependencia faltante: {e}. pip install browser-use langchain-anthropic"
            )
        except Exception as e:
            return BrowserTaskResult(
                success=False, extracted_content="",
                error=f"Browser-Use error: {e}"
            )

    def find_code_on_github(self, query: str) -> str:
        """
        Uso real: Browser-Use para buscar código en GitHub.
        Playwright-MCP sería mejor aquí si se necesita autenticación.
        Browser-Use lo hace autónomamente.
        """
        task = (
            f"Go to github.com and search for: '{query}'. "
            f"Open the most relevant Python repository. "
            f"Find and extract the main Python implementation code. "
            f"Return the raw code content."
        )
        result = self.run_task(task, use_vision=False, max_steps=15)
        return result.extracted_content if result.success else f"Error: {result.error}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY ACTUALIZADO con los 3 nuevos repos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool: str
    success: bool
    data: Any
    error: str = ""
    elapsed_ms: float = 0.0

    def to_context(self, max_chars: int = 2500) -> str:
        if not self.success:
            return f"[{self.tool} ERROR] {self.error}"
        s = json.dumps(self.data, ensure_ascii=False) if not isinstance(self.data, str) else self.data
        return f"[{self.tool}]\n{s[:max_chars]}"


class ToolRegistry:
    """
    Tool registry centralizado (Hermes-style).
    Nuevas tools añadidas:
    - playwright_snapshot: Playwright-MCP accessibility tree
    - browser_task: Browser-Use tarea autónoma
    """
    _tools: dict = {}
    _playwright: PlaywrightMCPClient | None = None
    _browser_use: BrowserUseClient | None = None

    @classmethod
    def init_browser_tools(cls):
        """Inicializa clientes de browser lazy."""
        if cls._playwright is None:
            cls._playwright = PlaywrightMCPClient()
        if cls._browser_use is None:
            cls._browser_use = BrowserUseClient()

    @classmethod
    def register(cls, name: str, fn):
        cls._tools[name] = fn

    @classmethod
    def call(cls, name: str, **kwargs) -> ToolResult:
        t0 = time.time()
        if name not in cls._tools:
            return ToolResult(name, False, {}, f"Tool desconocida: {name}")
        try:
            result = cls._tools[name](**kwargs)
            elapsed = round((time.time() - t0) * 1000, 1)
            tool_log.debug(json.dumps({
                "tool": name, "elapsed_ms": elapsed,
                "preview": str(result)[:200]
            }))
            return ToolResult(name, True, result, elapsed_ms=elapsed)
        except Exception as e:
            elapsed = round((time.time() - t0) * 1000, 1)
            tool_log.error(json.dumps({"tool": name, "error": str(e), "elapsed_ms": elapsed}))
            return ToolResult(name, False, {}, str(e), elapsed_ms=elapsed)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._tools.keys())


# ─── Registrar tools ────────────────────────────────────────────────────────

# Web search
def _web_search(query: str, max_results: int = 5) -> list[dict]:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]}
                for r in ddgs.text(query, max_results=max_results)]

ToolRegistry.register("web_search", _web_search)


# GitHub code search
def _github_search(query: str, language: str = "python", max_results: int = 5) -> list[dict]:
    h = {"Accept": "application/vnd.github.v3+json"}
    if os.getenv("GITHUB_TOKEN"):
        h["Authorization"] = f"token {os.getenv('GITHUB_TOKEN')}"
    r = requests.get("https://api.github.com/search/code",
                     params={"q": f"{query} language:{language}", "per_page": max_results},
                     headers=h, timeout=10)
    r.raise_for_status()
    return [{"name": i["name"], "url": i["html_url"], "repo": i["repository"]["full_name"]}
            for i in r.json().get("items", [])]

ToolRegistry.register("github_search", _github_search)


# Crawl4AI (estático rápido)
def _crawl4ai_scrape(url: str, max_chars: int = 3000) -> str:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        async def _fetch():
            async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as c:
                result = await c.arun(
                    url=url,
                    config=CrawlerRunConfig(
                        markdown_generator=DefaultMarkdownGenerator(
                            content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed")
                        ),
                        word_count_threshold=10,
                        exclude_external_links=True,
                    )
                )
                return result.markdown[:max_chars] if result.markdown else ""

        return asyncio.run(_fetch())
    except ImportError:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "AutoResearch/4.0"})
        text = re.sub(r"<[^>]+>", " ", resp.text)
        return re.sub(r"\s+", " ", text).strip()[:max_chars]

ToolRegistry.register("scrape_url", _crawl4ai_scrape)


# ── NUEVO: Playwright-MCP (accessibility tree, JS-heavy pages) ──────────────
def _playwright_snapshot(url: str) -> str:
    """
    Playwright-MCP: navega y devuelve accessibility tree.
    Token-efficient: NO screenshots, NO vision model necesario.
    Refs de elementos incluidos: [ref=42] Button "Submit"
    Útil para: SPAs, dashboards, páginas con login preservado.
    """
    ToolRegistry.init_browser_tools()
    if not ToolRegistry._playwright:
        return "Playwright-MCP no inicializado"
    client = ToolRegistry._playwright
    if not client._available:
        # Intentar arrancar servidor
        started = client.start_server()
        if not started:
            return "Playwright-MCP no disponible. Ejecuta: npx @playwright/mcp@latest --port 8931 --headless"
    return client.navigate_and_snapshot(url)

ToolRegistry.register("playwright_snapshot", _playwright_snapshot)


# ── NUEVO: Browser-Use (tareas web autónomas, multi-step) ───────────────────
def _browser_task(task: str, use_vision: bool = False, max_steps: int = 15) -> str:
    """
    Browser-Use: agente web autónomo.
    Describe la tarea en lenguaje natural, Browser-Use la ejecuta.
    Loop autónomo: Observe → Decide → Act → Evaluate.
    use_vision=False: solo DOM (barato). True: DOM + screenshots (más potente).
    """
    ToolRegistry.init_browser_tools()
    if not ToolRegistry._browser_use:
        return "Browser-Use no inicializado"
    result = ToolRegistry._browser_use.run_task(task, use_vision=use_vision, max_steps=max_steps)
    if result.success:
        return result.extracted_content
    return f"Error: {result.error}"

ToolRegistry.register("browser_task", _browser_task)


# Python sandbox
def _run_python(code: str, timeout: int = 15) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write(code); tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=timeout)
        return {"stdout": r.stdout[:2000], "stderr": r.stderr[:500], "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "returncode": -1}
    finally:
        Path(tmp).unlink(missing_ok=True)

ToolRegistry.register("run_python", _run_python)


# PR-Agent
def _pr_agent(pr_url: str, command: str = "review") -> str:
    env = os.environ.copy()
    if os.getenv("ANTHROPIC_API_KEY"):
        env["ANTHROPIC.KEY"] = os.getenv("ANTHROPIC_API_KEY")
        env["CONFIG__MODEL"] = "anthropic/claude-haiku-4-5-20251001"
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pr_agent.cli", f"--pr_url={pr_url}", command],
            capture_output=True, text=True, timeout=120, env=env
        )
        return (r.stdout + r.stderr)[:3000]
    except Exception as e:
        return f"pr-agent error: {e}"

ToolRegistry.register("pr_review", _pr_agent)


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGER (Mem0 + SQLite FTS5 fallback — igual que v3)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    def __init__(self, agent_id: str = "autoresearch_v4"):
        self.agent_id = agent_id
        self._mem0 = None
        self._sqlite = LOG_DIR / "memory_v4.db"
        self._init_mem0()
        self._init_sqlite()

    def _init_mem0(self):
        try:
            from mem0 import Memory
            self._mem0 = Memory()
            log.info("[MEM0] Inicializado")
        except Exception as e:
            log.warning(f"[MEM0] fallback SQLite: {e}")

    def _init_sqlite(self):
        import sqlite3
        c = sqlite3.connect(str(self._sqlite))
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS mem USING fts5(content, agent_id, mtype)")
        c.execute("CREATE TABLE IF NOT EXISTS mem_raw (id INTEGER PRIMARY KEY, content TEXT, agent_id TEXT, mtype TEXT, ts TEXT)")
        c.commit(); c.close()

    def add(self, text: str, mtype: str = "episodic"):
        mem_log.info(json.dumps({"type": mtype, "text": text[:200]}))
        if self._mem0:
            try:
                self._mem0.add([{"role": "user", "content": text}],
                               user_id=self.agent_id, metadata={"type": mtype})
                return
            except Exception as e:
                log.warning(f"[MEM0] add error: {e}")
        import sqlite3
        c = sqlite3.connect(str(self._sqlite))
        c.execute("INSERT INTO mem_raw (content, agent_id, mtype, ts) VALUES (?,?,?,?)",
                  (text, self.agent_id, mtype, datetime.utcnow().isoformat()))
        c.execute("INSERT INTO mem (content, agent_id, mtype) VALUES (?,?,?)",
                  (text, self.agent_id, mtype))
        c.commit(); c.close()

    def search(self, query: str, limit: int = 5) -> list[str]:
        if self._mem0:
            try:
                r = self._mem0.search(query, user_id=self.agent_id, limit=limit)
                items = r.get("results", r) if isinstance(r, dict) else r
                return [x.get("memory", str(x)) for x in items[:limit]]
            except Exception as e:
                log.warning(f"[MEM0] search error: {e}")
        import sqlite3
        c = sqlite3.connect(str(self._sqlite))
        try:
            rows = c.execute("SELECT content FROM mem WHERE mem MATCH ? LIMIT ?",
                             (query, limit)).fetchall()
            return [r[0] for r in rows]
        except Exception:
            kw = query.split()[:2]
            out = []
            for k in kw:
                rows = c.execute("SELECT content FROM mem_raw WHERE content LIKE ? LIMIT ?",
                                 (f"%{k}%", limit)).fetchall()
                out.extend(r[0] for r in rows)
            return list(dict.fromkeys(out))[:limit]
        finally:
            c.close()

    def add_episode(self, ep: dict): self.add(json.dumps(ep), "episodic")
    def add_semantic(self, pattern: str, solution: str):
        self.add(json.dumps({"pattern": pattern, "solution": solution}), "semantic")


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT HÍBRIDO: PydanticAI para structured output, Anthropic SDK para texto libre
# ─────────────────────────────────────────────────────────────────────────────

class HybridLLMClient:
    """
    Combina dos modos:
    1. PydanticAI: cuando se necesita output estructurado validado (CodeProposal, FixProposal)
    2. Anthropic SDK directo: para prompts de texto libre (más control)

    PydanticAI ventaja clave: retry automático si el LLM no genera JSON válido.
    """

    def __init__(self):
        self._raw = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._pai_builder = _build_pydantic_builder_agent()
        self._pai_fixer = _build_pydantic_fixer_agent()
        pai_status = "✓ PydanticAI" if self._pai_builder else "✗ fallback SDK"
        log.info(f"[LLM] {pai_status}")

    def generate_code(self, prompt: str, attempt: int) -> CodeProposal:
        """
        PydanticAI: output garantizado como CodeProposal.
        Si el LLM devuelve JSON malformado → retry automático (max 3).
        """
        if self._pai_builder:
            try:
                result = self._pai_builder.run_sync(prompt)
                code_proposal = result.output
                llm_log.debug(json.dumps({
                    "mode": "pydantic_ai", "attempt": attempt,
                    "functions": code_proposal.functions_implemented,
                    "confidence": code_proposal.confidence,
                }))
                return code_proposal
            except Exception as e:
                log.warning(f"[PydanticAI] builder error: {e}, usando fallback SDK")

        # Fallback: Anthropic SDK + parse manual
        resp = self._raw_call(
            system="Genera código Python. Responde SOLO con ```python ... ```",
            user=prompt, label=f"builder_sdk_{attempt}"
        )
        code = self._extract_code(resp)
        return CodeProposal(
            code=code,
            reasoning="Generated via fallback SDK",
            functions_implemented=re.findall(r"def (\w+)\(", code),
            edge_cases_handled=["fallback mode"],
            confidence=0.5
        )

    def fix_code(self, prompt: str, attempt: int) -> FixProposal:
        """PydanticAI: output garantizado como FixProposal con root cause."""
        if self._pai_fixer:
            try:
                result = self._pai_fixer.run_sync(prompt)
                fix = result.output
                llm_log.debug(json.dumps({
                    "mode": "pydantic_ai_fixer", "attempt": attempt,
                    "root_cause": fix.root_cause,
                    "changes": fix.changes_made,
                }))
                return fix
            except Exception as e:
                log.warning(f"[PydanticAI] fixer error: {e}, usando fallback SDK")

        resp = self._raw_call(
            system="Corrige el código Python. Responde SOLO con ```python ... ```",
            user=prompt, label=f"fixer_sdk_{attempt}"
        )
        code = self._extract_code(resp)
        return FixProposal(
            fixed_code=code,
            root_cause="Identified via fallback SDK",
            changes_made=["fallback mode"],
            preserved_passing=[]
        )

    def _raw_call(self, system: str, user: str, label: str = "",
                  model: str = "claude-opus-4-5") -> str:
        t0 = time.time()
        msg = self._raw.messages.create(
            model=model, max_tokens=4096, system=system,
            messages=[{"role": "user", "content": user}]
        )
        text = msg.content[0].text
        llm_log.debug(json.dumps({
            "label": label, "model": model,
            "elapsed_ms": round((time.time()-t0)*1000),
            "tokens_in": msg.usage.input_tokens, "tokens_out": msg.usage.output_tokens,
        }))
        return text

    @staticmethod
    def _extract_code(raw: str) -> str:
        m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# BUILDER AGENT v4  (usa PydanticAI + herramientas browser inteligentes)
# ─────────────────────────────────────────────────────────────────────────────

class BuilderAgent:
    """
    Estrategia de herramientas por intento:

    Intento 1-2: web_search + github_search (rápido, sin browser)
    Intento 3-4: scrape_url (Crawl4AI, páginas estáticas)
    Intento 5-6: playwright_snapshot (JS-heavy, accessibility tree)
    Intento 7+:  browser_task (Browser-Use autónomo, último recurso)
    """

    def __init__(self, llm: HybridLLMClient, memory: MemoryManager):
        self.llm = llm
        self.memory = memory

    def build(self, spec: str, attempt: int) -> CodeProposal:
        memories = self.memory.search(f"calculator python implementation {spec[:60]}", limit=3)
        mem_ctx = "\n".join(memories) or "Sin memoria previa."

        external_ctx = self._get_external_context(attempt)

        prompt = textwrap.dedent(f"""
            ESPECIFICACIÓN:
            {spec}

            MEMORIA RELEVANTE:
            {mem_ctx}

            CONTEXTO EXTERNO (intento {attempt}):
            {external_ctx[:1200] if external_ctx else 'N/A'}

            Genera una implementación completa y robusta de calculator.py.
            Asegúrate de incluir TODAS las funciones del spec.
        """).strip()

        return self.llm.generate_code(prompt, attempt)

    def _get_external_context(self, attempt: int) -> str:
        """
        Estrategia de herramientas escalonada:
        - Web/GitHub: rápido, sin browser
        - Crawl4AI: páginas estáticas, contenido Markdown
        - Playwright-MCP: accessibility tree para JS-heavy
        - Browser-Use: autónomo multi-step (costoso, último recurso)
        """
        if attempt <= 2:
            # Fase 1: web + GitHub (rápido)
            r = ToolRegistry.call("web_search",
                                  query="python calculator safe implementation TypeError ValueError")
            if r.success and r.data:
                return "\n".join(x["snippet"][:200] for x in r.data[:3])

        elif attempt <= 4:
            # Fase 2: Crawl4AI para documentación Python
            r = ToolRegistry.call("scrape_url",
                                  url="https://docs.python.org/3/library/math.html")
            if r.success:
                return r.data[:1200]

        elif attempt <= 6:
            # Fase 3: Playwright-MCP para páginas JS-heavy (GitHub, StackOverflow)
            r = ToolRegistry.call("playwright_snapshot",
                                  url="https://stackoverflow.com/questions/tagged/python+calculator")
            if r.success:
                return r.data[:1500]

        else:
            # Fase 4: Browser-Use autónomo (último recurso)
            r = ToolRegistry.call(
                "browser_task",
                task=(
                    "Search GitHub for 'python safe calculator implementation TypeError ValueError'. "
                    "Find a good implementation and extract the Python code."
                ),
                use_vision=False, max_steps=10
            )
            if r.success:
                return r.data[:1500]

        return ""


# ─────────────────────────────────────────────────────────────────────────────
# FIXER AGENT v4  (PydanticAI structured output + Playwright para debugging)
# ─────────────────────────────────────────────────────────────────────────────

class FixerAgent:
    def __init__(self, llm: HybridLLMClient, memory: MemoryManager):
        self.llm = llm
        self.memory = memory

    def fix(self, code: str, test_output: str, attempt: int) -> FixProposal:
        error_key = self._extract_error(test_output)
        fixes = self.memory.search(f"fix error {error_key}", limit=3)
        fix_ctx = "\n".join(fixes) or "Sin fixes previos."

        prompt = textwrap.dedent(f"""
            CÓDIGO ACTUAL:
            ```python
            {code}
            ```

            ERRORES DE PYTEST:
            {test_output[-2500:]}

            ERROR PRINCIPAL DETECTADO: {error_key}

            FIXES ANTERIORES EN MEMORIA:
            {fix_ctx}

            Identifica la causa raíz y devuelve el código COMPLETO corregido.
            Preserva todas las funciones que ya pasan sus tests.
        """).strip()

        return self.llm.fix_code(prompt, attempt)

    @staticmethod
    def _extract_error(out: str) -> str:
        for p in [r"(TypeError[^\n]{0,150})", r"(ValueError[^\n]{0,150})",
                  r"(AssertionError[^\n]{0,150})", r"(FAILED[^\n]{0,150})"]:
            m = re.search(p, out)
            if m:
                return m.group(1)
        return out[-150:]


# ─────────────────────────────────────────────────────────────────────────────
# PYTEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    passed: int = 0; failed: int = 0; errors: int = 0; total: int = 0; output: str = ""

    @property
    def score(self) -> float: return self.passed / self.total if self.total > 0 else 0.0
    @property
    def is_perfect(self) -> bool: return self.passed == self.total and self.total > 0
    @property
    def has_progress(self) -> bool: return self.passed > 0


def run_pytest(code: str, test_file: str) -> TestResult:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "calculator.py").write_text(code, encoding="utf-8")
        (Path(tmp) / "test_calculator.py").write_text(test_file, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "test_calculator.py", "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, cwd=tmp, timeout=30
        )
    output = r.stdout + r.stderr
    test_log.debug(output)
    tr = TestResult(output=output)
    for pat, attr in [(r"(\d+) passed", "passed"), (r"(\d+) failed", "failed"), (r"(\d+) error", "errors")]:
        m = re.search(pat, output)
        if m: setattr(tr, attr, int(m.group(1)))
    tr.total = tr.passed + tr.failed + tr.errors
    return tr


# ─────────────────────────────────────────────────────────────────────────────
# JUDGE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class JudgeAgent:
    def __init__(self, memory: MemoryManager):
        self.memory = memory
        self._best_score = 0.0
        self._best_code = ""

    def evaluate(self, attempt: int, code: str, proposal: CodeProposal | FixProposal,
                 tr: TestResult, prev_score: float) -> dict:
        improved = tr.score > prev_score
        if tr.score > self._best_score:
            self._best_score = tr.score
            self._best_code = code

        # Guardar episodio con metadata de PydanticAI
        episode = {
            "attempt": attempt, "score": tr.score,
            "passed": tr.passed, "total": tr.total,
            "improved": improved,
        }
        if isinstance(proposal, CodeProposal):
            episode["functions"] = proposal.functions_implemented
            episode["confidence"] = proposal.confidence
        elif isinstance(proposal, FixProposal):
            episode["root_cause"] = proposal.root_cause
            episode["changes"] = proposal.changes_made

        self.memory.add_episode(episode)

        # Patrón semántico si mejoró
        if improved and tr.score >= 0.6:
            if isinstance(proposal, CodeProposal):
                self.memory.add_semantic(
                    pattern=f"calculator builder attempt {attempt}",
                    solution=f"score={tr.score:.2f} confidence={proposal.confidence} functions={proposal.functions_implemented}"
                )
            elif isinstance(proposal, FixProposal):
                self.memory.add_semantic(
                    pattern=f"fix: {proposal.root_cause[:80]}",
                    solution=f"changes: {proposal.changes_made}"
                )

        return {
            "attempt": attempt, "score": tr.score,
            "improved": improved,
            "action": "converged" if tr.is_perfect else "continue"
        }

    @property
    def best_code(self) -> str: return self._best_code


# ─────────────────────────────────────────────────────────────────────────────
# SWARM STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwarmState:
    current_code: str = ""; current_score: float = 0.0
    best_code: str = ""; best_score: float = 0.0
    attempt: int = 0; max_attempts: int = 20
    stall_count: int = 0; max_stall: int = 6
    history: list = field(default_factory=list)
    status: str = "running"

    def update(self, score: float, code: str):
        self.attempt += 1
        if score > self.best_score:
            self.best_score = score; self.best_code = code; self.stall_count = 0
        else:
            self.stall_count += 1
        self.current_score = score; self.current_code = code
        self.history.append({"attempt": self.attempt, "score": round(score, 4), "stall": self.stall_count})

    def should_stop(self) -> bool:
        if self.attempt >= self.max_attempts: self.status = "max_attempts"; return True
        if self.stall_count >= self.max_stall: self.status = "stalled"; return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR v4
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    v4: PydanticAI structured outputs + Playwright-MCP + Browser-Use
    OpenClaw loop pattern preservado.
    """

    def __init__(self, spec: str, test_file: str, max_attempts: int = 20):
        self.spec = spec
        self.test_file = test_file
        self.state = SwarmState(max_attempts=max_attempts)
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        self.memory = MemoryManager()
        self.llm = HybridLLMClient()
        self.builder = BuilderAgent(self.llm, self.memory)
        self.fixer = FixerAgent(self.llm, self.memory)
        self.judge = JudgeAgent(self.memory)

    def run(self) -> SwarmState:
        log.info("=" * 65)
        log.info(f"AutoResearch v4 | session={self.session_id}")
        log.info(f"Tools: {ToolRegistry.available()}")
        log.info("=" * 65)

        last_test_output = ""
        last_proposal: CodeProposal | FixProposal | None = None

        while not self.state.should_stop():
            attempt = self.state.attempt + 1
            log.info(f"\n{'─' * 50}")
            log.info(f"INTENTO {attempt} | best={self.state.best_score:.1%} | stall={self.state.stall_count}")

            # ── GENERAR / CORREGIR ──
            if attempt == 1 or self.state.current_score == 0.0:
                proposal = self.builder.build(self.spec, attempt)
                code = proposal.code
                log.info(f"[BUILDER] confidence={proposal.confidence:.2f} funcs={proposal.functions_implemented}")
            else:
                fix_proposal = self.fixer.fix(self.state.current_code, last_test_output, attempt)
                code = fix_proposal.fixed_code
                proposal = fix_proposal
                log.info(f"[FIXER] root_cause={fix_proposal.root_cause[:60]}")

            # ── EVALUAR ──
            try:
                tr = run_pytest(code, self.test_file)
            except Exception as e:
                log.error(f"[PYTEST] crash: {e}")
                self.state.attempt += 1; self.state.stall_count += 1
                continue

            last_test_output = tr.output
            log.info(f"[JUDGE] {tr.passed}/{tr.total} — {tr.score:.1%}")

            # ── ACEPTAR SI HAY PROGRESO (bug fix clave) ──
            if tr.score >= self.state.best_score or tr.has_progress:
                self.state.update(tr.score, code)
            else:
                self.state.attempt += 1; self.state.stall_count += 1
                self.state.history.append({"attempt": attempt, "score": round(tr.score, 4), "stall": self.state.stall_count})

            verdict = self.judge.evaluate(attempt, code, proposal, tr, self.state.current_score)
            last_proposal = proposal

            self._log_session(attempt, tr, verdict, proposal)

            if verdict["action"] == "converged":
                self.state.status = "converged"
                log.info("✅ CONVERGENCIA TOTAL")
                break

        self._finalize()
        return self.state

    def _log_session(self, attempt: int, tr: TestResult, verdict: dict,
                     proposal: CodeProposal | FixProposal | None):
        entry = {
            "session": self.session_id, "attempt": attempt,
            "score": tr.score, "passed": tr.passed, "total": tr.total,
            "verdict": verdict["action"], "ts": datetime.utcnow().isoformat(),
        }
        if isinstance(proposal, CodeProposal):
            entry["confidence"] = proposal.confidence
        p = LOG_DIR / f"session_{self.session_id}.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _finalize(self):
        log.info("\n" + "=" * 65)
        log.info(f"FINAL: {self.state.status.upper()} | {self.state.best_score:.1%} | {self.state.attempt} intentos")
        if self.state.best_code:
            Path("calculator_final.py").write_text(self.state.best_code, encoding="utf-8")
        (LOG_DIR / "run_summary.json").write_text(json.dumps({
            "session": self.session_id, "status": self.state.status,
            "best_score": self.state.best_score, "attempts": self.state.attempt,
            "history": self.state.history, "ts": datetime.utcnow().isoformat(),
        }, indent=2))
        # Cleanup browser tools
        if ToolRegistry._playwright:
            ToolRegistry._playwright.stop_server()


# ─────────────────────────────────────────────────────────────────────────────
# SPEC Y TESTS
# ─────────────────────────────────────────────────────────────────────────────

CALCULATOR_SPEC = """
Implementar calculator.py con:
  add(a, b), subtract(a, b), multiply(a, b), divide(a, b),
  power(base, exp), sqrt(n), modulo(a, b)

Requisitos de robustez:
  - TypeError si los argumentos no son int/float
  - ValueError: divide/modulo por cero, sqrt de negativo
  - Docstring en cada función
  - Overflow → float('inf')
"""

CALCULATOR_TESTS = """
import pytest, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from calculator import add, subtract, multiply, divide, power, sqrt, modulo

class TestBasic:
    def test_add(self):           assert add(2, 3) == 5
    def test_subtract(self):      assert subtract(5, 3) == 2
    def test_multiply(self):      assert multiply(4, 3) == 12
    def test_divide(self):        assert divide(10, 2) == 5.0
    def test_power(self):         assert power(2, 10) == 1024
    def test_sqrt(self):          assert sqrt(9) == 3.0
    def test_modulo(self):        assert modulo(10, 3) == 1
    def test_floats(self):        assert abs(add(0.1, 0.2) - 0.3) < 1e-9
    def test_approx(self):        assert divide(1, 3) == pytest.approx(0.333, rel=1e-2)

class TestEdgeCases:
    def test_div_zero(self):   pytest.raises(ValueError, divide, 5, 0)
    def test_sqrt_neg(self):   pytest.raises(ValueError, sqrt, -1)
    def test_mod_zero(self):   pytest.raises(ValueError, modulo, 5, 0)
    def test_sqrt_zero(self):  assert sqrt(0) == 0.0
    def test_negatives(self):  assert subtract(-5, -3) == -2

class TestRobustness:
    def test_type_add(self):    pytest.raises(TypeError, add, "a", 1)
    def test_type_div(self):    pytest.raises(TypeError, divide, None, 2)
    def test_large_pow(self):   assert power(2, 100) > 0
"""

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoResearch v4")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--spec",  default=None)
    parser.add_argument("--tests", default=None)
    args = parser.parse_args()

    spec      = Path(args.spec).read_text()  if args.spec  else CALCULATOR_SPEC
    test_file = Path(args.tests).read_text() if args.tests else CALCULATOR_TESTS

    orch = Orchestrator(spec=spec, test_file=test_file, max_attempts=args.max_attempts)
    state = orch.run()
    sys.exit(0 if state.status in ("converged", "max_attempts") else 1)
