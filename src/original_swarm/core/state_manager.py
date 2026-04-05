"""
Memory module para AG2-Coder v6.

Reemplaza el placeholder `pass` con persistencia real usando el
StateManager de AutoAgent para escribir findings.md y state JSON.

Además añade un índice en memoria para búsqueda por contenido,
que sirve como base para el sistema RAG del siguiente paso.
"""

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path

from core.config import AgentConfig
from core.state_manager import AgentState
from core.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Una entrada individual en la memoria del agente."""

    step: str
    result: dict
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tags: list[str] = field(default_factory=list)

    def matches(self, query: str) -> bool:
        """True si el query aparece en step, tags, o valores del result."""
        query_lower = query.lower()
        if query_lower in self.step.lower():
            return True
        if any(query_lower in tag.lower() for tag in self.tags):
            return True
        result_str = json.dumps(self.result).lower()
        return query_lower in result_str


class Memory:
    """
    Persiste decisiones, resultados y código generado del agente.

    A diferencia del placeholder `pass`, esta implementación:
    - Escribe cada entrada en findings.md vía StateManager (AutoAgent).
    - Mantiene un índice en memoria para búsqueda rápida.
    - Persiste un archivo memory_index.json para recuperación entre sesiones.
    - Expone retrieve() que el RAG retriever puede usar como fuente.

    Example:
        >>> config = AgentConfig()
        >>> memory = Memory(config)
        >>> memory.bind_state(state, task_id)
        >>> memory.store("Generar hasher.py", {"file": "hasher.py", "lines": 45})
        >>> entries = memory.retrieve("hasher")
        >>> print(len(entries))  # 1
    """

    def __init__(
        self,
        config: AgentConfig,
        index_file: str = "memory_index.json",
    ) -> None:
        """
        Inicializar Memory con StateManager de AutoAgent.

        Args:
            config: AgentConfig con settings de workspace.
            index_file: Nombre del archivo JSON de índice persistente.
        """
        self._config = config
        self._state_manager = StateManager(config)
        self._state: AgentState | None = None
        self._task_id: str = ""
        self._entries: list[MemoryEntry] = []
        self._index_path = config.workspace_dir / index_file
        self._load_index()

    def bind_state(self, state: AgentState, task_id: str) -> None:
        """
        Vincular la Memory al estado activo de una tarea.

        Debe llamarse después de Planner.create_plan() y antes de store().

        Args:
            state: AgentState inicializado por el Planner.
            task_id: Task ID activo para encontrar los archivos en disco.
        """
        self._state = state
        self._task_id = task_id
        logger.debug(f"Memory vinculada a task_id={task_id}")

    def store(self, step: str, result: dict, tags: list[str] | None = None) -> None:
        """
        Persistir un paso y su resultado en disco y en el índice.

        A diferencia del placeholder `pass`, este método:
        - Escribe en findings.md vía StateManager.
        - Actualiza memory_index.json.
        - Mantiene el índice en memoria para retrieve().

        Args:
            step: Descripción del paso ejecutado.
            result: Resultado del paso (dict con cualquier estructura).
            tags: Tags opcionales para búsqueda posterior.
        """
        entry = MemoryEntry(step=step, result=result, tags=tags or [])
        self._entries.append(entry)

        # Persistir en findings.md vía StateManager
        if self._state is not None:
            content = (
                f"**Paso:** {step}\n"
                f"**Resultado:** {json.dumps(result, indent=2, ensure_ascii=False)}\n"
                f"**Tags:** {', '.join(tags or [])}"
            )
            self._state_manager.add_finding(
                self._state,
                source=f"memory/{self._task_id}",
                content=content,
            )

        # Persistir en el índice JSON
        self._save_index()
        logger.debug(f"Memory: stored entry for step='{step[:50]}'")

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """
        Buscar entradas en la memoria por contenido.

        Implementación simple de búsqueda textual sobre el índice
        en memoria. El RAG retriever puede extender esto con embeddings.

        Args:
            query: Texto a buscar en pasos, tags y resultados.
            limit: Número máximo de entradas a retornar.

        Returns:
            Lista de MemoryEntry que coinciden con el query, más recientes primero.
        """
        matches = [e for e in reversed(self._entries) if e.matches(query)]
        return matches[:limit]

    def get_all(self) -> list[MemoryEntry]:
        """Retornar todas las entradas en orden cronológico."""
        return list(self._entries)

    def to_context_string(self, query: str = "", limit: int = 3) -> str:
        """
        Generar string de contexto para inyectar en prompts del LLM.

        Args:
            query: Query para filtrar entradas relevantes.
            limit: Máximo de entradas a incluir.

        Returns:
            Texto formateado con las entradas más relevantes.
        """
        entries = self.retrieve(query, limit) if query else self._entries[-limit:]
        if not entries:
            return "Sin entradas en memoria para este contexto."

        lines = ["=== Contexto de memoria ==="]
        for i, entry in enumerate(entries, start=1):
            lines.append(f"\n[{i}] Paso: {entry.step}")
            lines.append(f"    Timestamp: {entry.timestamp}")
            result_preview = json.dumps(entry.result)[:200]
            lines.append(f"    Resultado: {result_preview}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        """Serializar el índice de memoria a JSON en disco."""
        payload = [
            {
                "step": e.step,
                "result": e.result,
                "timestamp": e.timestamp,
                "tags": e.tags,
            }
            for e in self._entries
        ]
        self._index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_index(self) -> None:
        """Cargar entradas previas del índice JSON si existe."""
        if not self._index_path.exists():
            return
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._entries = [
                MemoryEntry(
                    step=item["step"],
                    result=item["result"],
                    timestamp=item["timestamp"],
                    tags=item.get("tags", []),
                )
                for item in raw
            ]
            logger.info(f"Memory: cargadas {len(self._entries)} entradas del índice.")
        except Exception as exc:
            logger.warning(f"Memory: no se pudo cargar el índice: {exc}")
            self._entries = []
