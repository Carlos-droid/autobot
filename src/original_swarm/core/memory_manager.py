"""
core/memory_manager.py — Sistema de memoria persistente KAIROS (Auto-Dream).
Inspirado en autoDream.ts y consolidationPrompt.ts de Claude Code.

Responsabilidades:
1. Cargar el ADN del agente (MEMORY.md) al inicio de cada sesión.
2. Consolidar periódicamente el historial en reglas permanentes.
3. Resolver contradicciones entre memoria existente y señales nuevas.

Razón: sin memoria persistente, cada sesión empieza desde cero. Con ella,
el agente acumula conocimiento y no repite errores documentados.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── Configuración ──────────────────────────────────────────────────────────

DEFAULT_MEMORY_PATH = Path("memory/MEMORY.md")

# Cada cuántos ciclos disparar la consolidación automática.
# Basado en autoDream.ts:DEFAULTS — 24h mínimo entre consolidaciones,
# pero para nuestro bucle usamos ciclos en vez de horas.
CONSOLIDATION_INTERVAL = 10

# Máximo de líneas del archivo de memoria — si crece más, la fase de
# "Prune" del prompt de consolidación lo recorta.
MAX_MEMORY_LINES = 150

# ── Prompt de Consolidación ────────────────────────────────────────────────
# Adaptado de consolidationPrompt.ts:buildConsolidationPrompt (líneas 10-64)
# 4 fases: Orient → Gather → Consolidate → Prune

CONSOLIDATION_PROMPT = """# Dream: Consolidación de Memoria

Eres el Arquitecto de Memoria del sistema. Sintetiza lo aprendido recientemente en memoria durable para que futuras sesiones se orienten rápido.

## Fase 1 — Orientar
Lee la memoria existente para entender qué ya se sabe.

<memoria_actual>
{current_memory}
</memoria_actual>

## Fase 2 — Recoger señal reciente
Analiza estos mensajes recientes del agente:

<historial_reciente>
{recent_history}
</historial_reciente>

## Fase 3 — Consolidar
Para cada cosa que vale la pena recordar:
1. Identifica HECHOS INMUTABLES: decisiones técnicas, stack, arquitectura.
2. Extrae PREFERENCIAS: estilos de código, flujos de trabajo.
3. Registra CAMINOS MUERTOS: errores encontrados y cómo se resolvieron. NO repetir.
4. Actualiza ESTADO: qué se completó y qué queda pendiente.
5. Resuelve CONTRADICCIONES: si lo nuevo invalida lo antiguo, sobrescribe.

## Fase 4 — Podar
- Mantén el archivo conciso (máximo {max_lines} líneas).
- Elimina entradas obsoletas o redundantes.
- Convierte fechas relativas en absolutas.
- Fusiona entradas duplicadas.

FORMATO DE SALIDA OBLIGATORIO (Markdown puro, sin texto introductorio):

### 🏗️ Arquitectura y Stack
- ...

### ⚙️ Preferencias y Reglas
- ...

### 🛑 Caminos Muertos (No repetir)
- ...

### 📝 Estado del Proyecto
- ...
"""


class MemoryManager:
    """Gestiona la memoria persistente del enjambre — carga, consolida, guarda."""

    def __init__(
        self,
        memory_path: Path = DEFAULT_MEMORY_PATH,
        consolidation_interval: int = CONSOLIDATION_INTERVAL,
        consolidate_fn: Optional[Callable[[str], str]] = None,
    ):
        self.memory_path = memory_path
        self.consolidation_interval = consolidation_interval
        # Función inyectada para llamar al LLM consolidador (testabilidad)
        self._consolidate_fn = consolidate_fn
        self._cycle_counter = 0

        # Crear directorio y archivo si no existen
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self._write_initial_memory()

    def _write_initial_memory(self) -> None:
        """Inicializa el archivo de memoria con la estructura base."""
        initial = (
            f"# Memoria del Proyecto\n"
            f"Generado: {datetime.now().isoformat()}\n\n"
            f"### 🏗️ Arquitectura y Stack\n"
            f"- Proyecto: Swarm Evolutivo Multi-Agente\n"
            f"- Stack: Python + Anthropic/Ollama\n\n"
            f"### ⚙️ Preferencias y Reglas\n"
            f"- Sin preferencias registradas aún.\n\n"
            f"### 🛑 Caminos Muertos (No repetir)\n"
            f"- Sin errores registrados aún.\n\n"
            f"### 📝 Estado del Proyecto\n"
            f"- Inicio del enjambre.\n"
        )
        self.memory_path.write_text(initial, encoding="utf-8")

    # ── Lectura ────────────────────────────────────────────────────────────

    def load(self) -> str:
        """
        Carga la memoria persistente del disco.
        Se inyecta al inicio del System Prompt de cada agente.
        Patrón de context.ts:getUserContext — cachear el claudeMd al inicio.
        """
        try:
            return self.memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._write_initial_memory()
            return self.memory_path.read_text(encoding="utf-8")

    def build_system_prompt_section(self) -> str:
        """
        Formatea la memoria como sección inyectable en el system prompt.
        Patrón de context.ts líneas 182-187: el claudeMd se prepende al context.
        """
        memory = self.load()
        return f"\n# MEMORIA DEL PROYECTO (Persistente)\n{memory}\n"

    # ── Consolidación (KAIROS) ─────────────────────────────────────────────

    def should_consolidate(self) -> bool:
        """
        Determina si es momento de consolidar.
        Basado en autoDream.ts:isGateOpen (línea 95) — comprueba intervalo
        y disponibilidad de la función consolidadora.
        """
        self._cycle_counter += 1
        return (
            self._consolidate_fn is not None
            and self._cycle_counter >= self.consolidation_interval
        )

    def consolidate(self, history: List[Dict[str, Any]]) -> None:
        """
        Ejecuta el ciclo KAIROS: destila historial → actualiza MEMORY.md.
        Basado en autoDream.ts:runAutoDream (líneas 125-271).

        Filtra mensajes de tool para reducir ruido antes de enviar al LLM.
        """
        if not self._consolidate_fn:
            print("⚠️  [KAIROS] Sin función de consolidación configurada.")
            return

        print("💤 [KAIROS] Iniciando consolidación de memoria...")

        current_memory = self.load()

        # Filtrar noise: excluir raw tool outputs (igual que en autoDream.ts:216)
        condensed = []
        for msg in history:
            role = msg.get("role", "")
            if role in ("tool", "function"):
                continue  # Ignorar volcados crudos de herramientas
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)[:300]
            elif isinstance(content, str) and len(content) > 300:
                content = content[:300] + "..."
            condensed.append(f"[{role.upper()}]: {content}")

        recent_text = "\n".join(condensed[-50:])  # Últimos 50 mensajes máx

        prompt = CONSOLIDATION_PROMPT.format(
            current_memory=current_memory,
            recent_history=recent_text,
            max_lines=MAX_MEMORY_LINES,
        )

        try:
            result = self._consolidate_fn(prompt)
        except Exception as e:
            print(f"❌ [KAIROS] Consolidación falló: {e}")
            return

        # Limpiar formato markdown residual del LLM
        clean = result.strip()
        if clean.startswith("```markdown"):
            clean = clean.replace("```markdown\n", "", 1)
        if clean.startswith("```"):
            clean = clean.replace("```\n", "", 1)
        if clean.endswith("```"):
            clean = clean[:-3]

        # Validar que el resultado tiene estructura mínima
        if "###" not in clean:
            print("⚠️  [KAIROS] Resultado sin estructura válida. No se actualiza memoria.")
            return

        # Guardar en disco
        self.memory_path.write_text(clean.strip(), encoding="utf-8")
        self._cycle_counter = 0  # Reset del contador
        print("✨ [KAIROS] Memoria consolidada y guardada.")

    # ── Migración de failed_patterns.md ────────────────────────────────────

    @staticmethod
    def migrate_legacy_memory(
        legacy_path: Path = Path("memory/failed_patterns.md"),
        target_path: Path = DEFAULT_MEMORY_PATH,
    ) -> None:
        """
        Migra el archivo de memoria legacy al nuevo formato, si existe.
        Razón: preservar el conocimiento acumulado durante la transición.
        """
        if not legacy_path.exists() or target_path.exists():
            return

        legacy_content = legacy_path.read_text(encoding="utf-8")

        # Extraer failures del formato antiguo
        failures = []
        for line in legacy_content.splitlines():
            if line.startswith("## Failure:"):
                failures.append(f"- {line.replace('## Failure: ', '')}")
            elif line.startswith("- Reason:"):
                failures.append(f"  {line}")

        migrated = (
            f"# Memoria del Proyecto\n"
            f"Migrado desde failed_patterns.md: {datetime.now().isoformat()}\n\n"
            f"### 🏗️ Arquitectura y Stack\n"
            f"- Proyecto: Swarm Evolutivo Multi-Agente\n"
            f"- Stack: Python + Anthropic/Ollama\n\n"
            f"### ⚙️ Preferencias y Reglas\n"
            f"- Sin preferencias registradas aún.\n\n"
            f"### 🛑 Caminos Muertos (No repetir)\n"
            + ("\n".join(failures) if failures else "- Sin errores registrados aún.")
            + f"\n\n### 📝 Estado del Proyecto\n"
            f"- Migración completada. Continuando evolución.\n"
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(migrated, encoding="utf-8")
        print(f"📦 [KAIROS] Memoria legacy migrada: {legacy_path} → {target_path}")
