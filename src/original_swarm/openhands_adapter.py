"""
OpenHands Adapter — core de integración AG2-Coder v6.

Reemplaza la implementación placeholder con orquestación real entre:
- Planner (backed by AutoAgent LLMBackend)
- Coder (genera código Python real)
- Validator (ejecuta pytest/mypy/ruff/bandit via AutoAgent ToolExecutor)
- Critic (analiza errores reales con LLM)
- Memory (persiste en disco via AutoAgent StateManager)

El loop implement el ciclo plan → code → execute → validate → criticize
con corrección iterativa hasta max_correction_rounds o hasta que pase.
"""

import logging
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from core.config import AgentConfig
from core.state_manager import StateManager
from core.state_manager import StepStatus

logger = logging.getLogger(__name__)


@dataclass
class AdapterResult:
    """Resultado completo de una ejecución del adapter."""

    task: str
    task_id: str
    success: bool
    steps_completed: int
    steps_total: int
    correction_rounds: int
    generated_files: list[str] = field(default_factory=list)
    final_validation_summary: str = ""
    error: str = ""


class OpenHandsAdapter:
    """
    Adapter que conecta el runtime de OpenHands con el sistema AG2.

    Traduce el ciclo completo de ejecución entre:
    - OpenHands (workspace, acciones de archivo, ejecución CLI)
    - AG2 (planner, coder, validator, critic, memory)

    A diferencia del placeholder, este adapter:
    - Ejecuta código real via ToolExecutor de AutoAgent.
    - Lee salidas reales de pytest/mypy/ruff.
    - Itera hasta que todos los checks pasan o se agota el límite.
    - Persiste estado en disco para resumir entre sesiones.

    Example:
        >>> config = AgentConfig()
        >>> adapter = OpenHandsAdapter.from_config(config)
        >>> result = adapter.handle_task(
        ...     "Crea un módulo de hashing de contraseñas con bcrypt"
        ... )
        >>> print(result.success)
        True
    """

    MAX_CORRECTION_ROUNDS: int = 3

    def __init__(
        self,
        planner: "Planner",     # type: ignore[name-defined]  # noqa: F821
        coder: "Coder",         # type: ignore[name-defined]  # noqa: F821
        validator: "Validator", # type: ignore[name-defined]  # noqa: F821
        critic: "Critic",       # type: ignore[name-defined]  # noqa: F821
        memory: "Memory",       # type: ignore[name-defined]  # noqa: F821
        config: AgentConfig | None = None,
    ) -> None:
        """
        Inicializar el Adapter con todos los componentes AG2.

        Args:
            planner: Instancia real de Planner (no placeholder).
            coder: Instancia real de Coder (no placeholder).
            validator: Instancia real de Validator (no placeholder).
            critic: Instancia real de Critic (no placeholder).
            memory: Instancia real de Memory (no placeholder).
            config: AgentConfig compartida. Si None, usa defaults.
        """
        self.planner = planner
        self.coder = coder
        self.validator = validator
        self.critic = critic
        self.memory = memory
        self._config = config or AgentConfig()
        self._state_manager = StateManager(self._config)

    @classmethod
    def from_config(cls, config: AgentConfig) -> "OpenHandsAdapter":
        """
        Factory method que construye todos los componentes desde config.

        Args:
            config: AgentConfig con todos los settings del sistema.

        Returns:
            OpenHandsAdapter completamente inicializado.
        """
        # Import aquí para evitar circular imports al nivel de módulo
        from coder.coder import Coder
        from critic.critic import Critic
        from memory.memory import Memory
        from planner.planner import Planner
        from validation.validator import Validator

        workspace = config.workspace_dir.parent

        return cls(
            planner=Planner(config),
            coder=Coder(config, workspace_dir=workspace / "src"),
            validator=Validator(
                config,
                target_path=workspace / "src",
                tests_path=workspace / "tests",
            ),
            critic=Critic(config),
            memory=Memory(config),
            config=config,
        )

    def handle_task(self, task: str) -> AdapterResult:
        """
        Orquestar el ciclo completo de ejecución del agente.

        Implementa el loop real:
        1. Planner descompone el task en pasos.
        2. Para cada paso: Coder genera código real.
        3. Código se ejecuta en el workspace (OpenHands action).
        4. Validator corre todos los checks reales.
        5. Si falla: Critic analiza, Planner actualiza, Coder corrige.
        6. Memory persiste cada resultado.

        Args:
            task: Instrucción en lenguaje natural.

        Returns:
            AdapterResult con estado final y métricas de ejecución.
        """
        logger.info(f"[AG2-Coder] Iniciando tarea: {task}")
        self.critic.reset_iteration_count()
        generated_files: list[str] = []
        correction_rounds = 0

        # ── 1. PLAN ──────────────────────────────────────────────────
        plan = self.planner.create_plan(task)
        task_id = self.planner.task_id
        state = self.planner.state

        if state is None:
            return AdapterResult(
                task=task,
                task_id=task_id,
                success=False,
                steps_completed=0,
                steps_total=0,
                correction_rounds=0,
                error="Planner no generó estado — verificar LLM backend.",
            )

        self.memory.bind_state(state, task_id)
        logger.info(f"[{task_id}] Plan: {len(plan)} pasos")

        # ── 2. EXECUTE EACH STEP ─────────────────────────────────────
        for step_idx, step in enumerate(plan):
            logger.info(f"[{task_id}] Paso {step_idx + 1}/{len(plan)}: {step}")

            self._state_manager.update_step(
                state, step_idx, StepStatus.IN_PROGRESS
            )

            # Código generado y escrito a disco real
            memory_context = self.memory.to_context_string(query=step, limit=2)
            file_path = self.coder.generate(step, context=memory_context)
            generated_files.append(file_path)

            # ── 3. VALIDATE + CORRECTION LOOP ────────────────────────
            step_passed = False
            for round_num in range(self.MAX_CORRECTION_ROUNDS):
                report = self.validator.run_all()

                if report.all_passed:
                    step_passed = True
                    logger.info(
                        f"[{task_id}] Paso {step_idx + 1} ✅ "
                        f"(round {round_num + 1})"
                    )
                    break

                correction_rounds += 1
                feedback = self.critic.analyze(report)

                logger.warning(
                    f"[{task_id}] Paso {step_idx + 1} ❌ round {round_num + 1}: "
                    f"{feedback.failed_checks}"
                )

                if not feedback.has_errors:
                    # Critic dice que está bien a pesar de exit codes
                    step_passed = True
                    break

                if round_num < self.MAX_CORRECTION_ROUNDS - 1:
                    # Enviar correcciones al Planner y Coder
                    self.planner.update_plan(feedback.to_planner_message())
                    file_path = self.coder.generate_fix(
                        step, feedback.to_coder_context()
                    )
                    generated_files.append(file_path)

            # ── 4. STORE IN MEMORY ────────────────────────────────────
            final_report = self.validator.run_all()
            self.memory.store(
                step=step,
                result={
                    "file": file_path,
                    "passed": step_passed,
                    "failed_checks": final_report.failed_checks,
                    "correction_rounds": correction_rounds,
                },
                tags=["code", "validation", step.split(":")[0].lower()],
            )

            new_status = StepStatus.COMPLETED if step_passed else StepStatus.FAILED
            self._state_manager.update_step(
                state,
                step_idx,
                new_status,
                f"rounds={correction_rounds}, pass={step_passed}",
            )

        # ── 5. FINALIZE ───────────────────────────────────────────────
        final_report = self.validator.run_all()
        steps_completed = sum(
            1 for s in state.steps if s.status == StepStatus.COMPLETED
        )
        success = steps_completed == len(plan)
        summary = final_report.to_summary()

        self._state_manager.mark_complete(
            state,
            f"{steps_completed}/{len(plan)} pasos. "
            f"{'PASS' if success else 'FAIL'}",
        )

        logger.info(
            f"[{task_id}] Tarea {'completada ✅' if success else 'fallida ❌'}: "
            f"{steps_completed}/{len(plan)} pasos"
        )

        return AdapterResult(
            task=task,
            task_id=task_id,
            success=success,
            steps_completed=steps_completed,
            steps_total=len(plan),
            correction_rounds=correction_rounds,
            generated_files=list(dict.fromkeys(generated_files)),  # dedup
            final_validation_summary=summary,
        )

    def execute_in_openhands(self, code: str) -> dict:
        """
        Escribir y ejecutar código en el workspace de OpenHands.

        En la versión actual usa el workspace local de AutoAgent.
        Cuando OpenHands esté disponible, este método hace la llamada
        real a su Actions API sin cambiar la firma del Adapter.

        Args:
            code: Código Python a ejecutar.

        Returns:
            Dict con status, output, y exit_code reales.
        """
        sandbox_file = self._config.workspace_dir.parent / "src" / "_sandbox_exec.py"
        sandbox_file.write_text(code, encoding="utf-8")

        from tools.executor import ToolExecutor
        executor = ToolExecutor(self._config)
        result = executor.execute_raw(["python", str(sandbox_file)])

        return {
            "status": "executed" if result.succeeded else "failed",
            "output": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
        }

    def _is_valid(self, validation: dict) -> bool:
        """
        Evaluar si el resultado es válido.

        Interfaz compatible con el contrato original del documento.

        Args:
            validation: Dict {check_name: bool} del ValidationReport.

        Returns:
            True sólo si todos los valores son True.
        """
        return all(validation.values())
