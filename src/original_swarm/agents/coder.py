"""
Coder module para AG2-Coder v6.

Reemplaza el placeholder `return f"# Código generado para: {step}"`
con generación real de código Python usando el LLMBackend de AutoAgent.

El Coder genera código completo, con type hints, docstrings PEP 257,
y lo escribe en el workspace para que el Validator lo pueda ejecutar.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from core.config import AgentConfig
from llm.backend import LLMBackend
from llm.backend import LLMMessage

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Eres un ingeniero Python senior. Generas código de producción que:
- Sigue PEP 8 y PEP 257 estrictamente.
- Usa type hints en TODAS las funciones.
- Tiene docstrings en todos los módulos, clases y funciones.
- Usa f-strings, nunca format() ni %.
- Máximo 88 caracteres por línea (Black standard).
- NUNCA usa pass, ..., o placeholders.
- Genera código COMPLETO y EJECUTABLE, no fragmentos.

Responde ÚNICAMENTE con el código Python listo para guardar en un archivo.
No incluyas explicaciones, markdown fences (```), ni comentarios fuera del código.
"""


@dataclass
class GeneratedCode:
    """Resultado de una generación de código por el Coder."""

    step: str
    filename: str
    content: str
    language: str = "python"

    @property
    def line_count(self) -> int:
        """Número de líneas de código generado."""
        return len(self.content.splitlines())


class Coder:
    """
    Genera código Python real usando el LLMBackend de AutoAgent.

    A diferencia del placeholder que retornaba un comentario,
    este Coder llama al LLM con contexto estructurado y escribe
    el código resultante en el workspace para ejecución real.

    Example:
        >>> config = AgentConfig()
        >>> coder = Coder(config, workspace_dir=Path("./workspace"))
        >>> generated = coder.generate("Crear función de hashing de contraseñas")
        >>> print(generated.filename)   # "hasher.py"
        >>> print(generated.line_count) # 45
    """

    def __init__(
        self,
        config: AgentConfig,
        workspace_dir: Path | None = None,
    ) -> None:
        """
        Inicializar el Coder con LLMBackend y directorio de trabajo.

        Args:
            config: AgentConfig con settings de LLM y workspace.
            workspace_dir: Directorio donde se escriben los archivos generados.
        """
        self._config = config
        self._llm = LLMBackend(config)
        self._workspace = workspace_dir or config.workspace_dir.parent / "src"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._generation_history: list[GeneratedCode] = []

    def generate(self, step: str, context: str = "") -> str:
        """
        Generar código Python para el paso dado y escribirlo al disco.

        Args:
            step: Descripción del paso a implementar.
            context: Contexto adicional (findings, errores previos).

        Returns:
            Ruta absoluta del archivo escrito, como string.
            Compatible con la firma original del Adapter.
        """
        filename = self._infer_filename(step)
        existing = self._read_existing(filename)

        user_content = self._build_prompt(step, context, existing)

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            response = self._llm.complete(messages)
            code = self._clean_code(response.content)
        except Exception as exc:
            logger.error(f"Coder LLM falló para paso '{step}': {exc}")
            code = self._generate_fallback(step)

        generated = GeneratedCode(
            step=step,
            filename=filename,
            content=code,
        )
        self._generation_history.append(generated)

        output_path = self._workspace / filename
        output_path.write_text(code, encoding="utf-8")
        logger.info(
            f"Código generado: {filename} ({generated.line_count} líneas)"
        )
        return str(output_path)

    def generate_fix(self, step: str, error_output: str) -> str:
        """
        Re-generar código incorporando errores del Validator.

        Args:
            step: El paso original que produjo el error.
            error_output: Salida de pytest/mypy/ruff con los errores.

        Returns:
            Ruta absoluta del archivo corregido.
        """
        fix_context = (
            f"El código anterior produjo estos errores:\n\n"
            f"{error_output}\n\n"
            f"Corrige TODOS los errores manteniendo la funcionalidad."
        )
        return self.generate(step, context=fix_context)

    @property
    def history(self) -> list[GeneratedCode]:
        """Retornar historial de todos los archivos generados."""
        return list(self._generation_history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        step: str,
        context: str,
        existing_code: str,
    ) -> str:
        """Construir prompt estructurado con XML tags para el LLM."""
        parts = [f"<task>\n{step}\n</task>"]
        if existing_code:
            parts.append(f"<existing_code>\n{existing_code}\n</existing_code>")
        if context:
            parts.append(f"<context>\n{context}\n</context>")
        parts.append(
            "<requirements>\n"
            "- Genera un archivo Python completo y ejecutable.\n"
            "- Incluye todos los imports necesarios.\n"
            "- Incluye type hints y docstrings PEP 257.\n"
            "- El archivo debe pasar mypy --strict sin errores.\n"
            "</requirements>"
        )
        return "\n\n".join(parts)

    def _infer_filename(self, step: str) -> str:
        """
        Inferir un nombre de archivo snake_case del paso.

        Args:
            step: Descripción del paso.

        Returns:
            Nombre de archivo snake_case con extensión .py.
        """
        # Extract meaningful words, ignore verbs like "crear", "implementar"
        skip = {"crear", "implementar", "generar", "escribir", "añadir", "para"}
        words = [
            w.lower()
            for w in re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", step)
            if w.lower() not in skip and len(w) > 2
        ]
        base = "_".join(words[:3]) if words else "generated_module"
        # Sanitize to valid Python identifier
        base = re.sub(r"[^a-z0-9_]", "_", base)
        return f"{base}.py"

    def _read_existing(self, filename: str) -> str:
        """
        Leer código existente si el archivo ya fue generado antes.

        Args:
            filename: Nombre del archivo a buscar en el workspace.

        Returns:
            Contenido del archivo existente, o cadena vacía.
        """
        path = self._workspace / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _clean_code(raw: str) -> str:
        """
        Limpiar el output del LLM eliminando markdown fences.

        Args:
            raw: Respuesta cruda del LLM.

        Returns:
            Código Python limpio listo para escribir a disco.
        """
        clean = re.sub(r"```(?:python)?\s*", "", raw).strip().rstrip("```").strip()
        return clean

    @staticmethod
    def _generate_fallback(step: str) -> str:
        """
        Generar código mínimo estructuralmente válido cuando el LLM falla.

        Garantiza que el Validator siempre tenga algo que ejecutar,
        aunque sea un módulo stub con type hints correcto.

        Args:
            step: El paso original para documentación.

        Returns:
            Código Python stub válido y ejecutable.
        """
        return (
            f'"""\nMódulo generado para: {step}\n\nGenerado por AG2-Coder v6.\n"""\n\n\n'
            "def main() -> None:\n"
            f'    """Implementación pendiente para: {step}"""\n'
            '    raise NotImplementedError(\n'
            f'        "Implementación pendiente: {step}"\n'
            "    )\n\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
