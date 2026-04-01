"""
tests/test_memory_manager.py — Tests unitarios para core/memory_manager.py.
Verifica carga, consolidación, migración y protección de estructura.
"""

import pytest
from pathlib import Path
from core.memory_manager import MemoryManager


@pytest.fixture
def temp_memory(tmp_path):
    """Fixture que crea un MemoryManager con path temporal."""
    mem_path = tmp_path / "MEMORY.md"
    return MemoryManager(memory_path=mem_path, consolidation_interval=3)


class TestMemoryLoad:
    """Tests de carga de memoria."""

    def test_creates_initial_file(self, temp_memory):
        """Si el archivo no existe, se crea con estructura inicial."""
        content = temp_memory.load()
        assert "Arquitectura" in content
        assert "Preferencias" in content
        assert "Caminos Muertos" in content

    def test_load_existing_file(self, tmp_path):
        """Si el archivo ya existe, lo lee sin modificar."""
        mem_path = tmp_path / "MEMORY.md"
        mem_path.write_text("# Custom Memory\n- fact 1\n", encoding="utf-8")
        mm = MemoryManager(memory_path=mem_path)
        content = mm.load()
        assert "Custom Memory" in content


class TestSystemPromptSection:
    """Tests de la sección inyectable en el system prompt."""

    def test_contains_header(self, temp_memory):
        section = temp_memory.build_system_prompt_section()
        assert "MEMORIA DEL PROYECTO" in section

    def test_contains_memory_content(self, temp_memory):
        section = temp_memory.build_system_prompt_section()
        assert "Arquitectura" in section


class TestConsolidation:
    """Tests del ciclo KAIROS."""

    def test_should_consolidate_respects_interval(self, tmp_path):
        """La consolidación solo se activa tras N ciclos."""
        mem_path = tmp_path / "MEMORY.md"
        mm = MemoryManager(
            memory_path=mem_path,
            consolidation_interval=3,
            consolidate_fn=lambda p: "### 🏗️ Arquitectura\n- test",
        )
        assert mm.should_consolidate() is False  # cycle 1
        assert mm.should_consolidate() is False  # cycle 2
        assert mm.should_consolidate() is True   # cycle 3

    def test_consolidate_updates_file(self, tmp_path):
        """La consolidación escribe el resultado en disco."""
        mem_path = tmp_path / "MEMORY.md"
        new_content = "### 🏗️ Arquitectura\n- Python 3.12\n### ⚙️ Preferencias\n- tabs"
        mm = MemoryManager(
            memory_path=mem_path,
            consolidate_fn=lambda p: new_content,
        )
        history = [
            {"role": "user", "content": "use Python 3.12"},
            {"role": "assistant", "content": "done"},
        ]
        mm.consolidate(history)
        assert "Python 3.12" in mem_path.read_text(encoding="utf-8")

    def test_consolidate_resets_counter(self, tmp_path):
        """Después de consolidar, el contador se reinicia a 0."""
        mem_path = tmp_path / "MEMORY.md"
        mm = MemoryManager(
            memory_path=mem_path,
            consolidation_interval=2,
            consolidate_fn=lambda p: "### 🏗️ Arquitectura\n- ok",
        )
        mm.should_consolidate()  # 1
        mm.should_consolidate()  # 2 → True
        mm.consolidate([{"role": "user", "content": "test"}])
        assert mm._cycle_counter == 0

    def test_consolidate_without_fn_does_nothing(self, tmp_path):
        """Sin función de consolidación, no crashea."""
        mem_path = tmp_path / "MEMORY.md"
        mm = MemoryManager(memory_path=mem_path, consolidate_fn=None)
        mm.consolidate([])  # Should not raise

    def test_consolidate_rejects_bad_format(self, tmp_path):
        """Si el LLM retorna formato inválido, no sobrescribe la memoria."""
        mem_path = tmp_path / "MEMORY.md"
        original = mem_path.read_text(encoding="utf-8") if mem_path.exists() else ""
        mm = MemoryManager(
            memory_path=mem_path,
            consolidate_fn=lambda p: "just plain text without headers",
        )
        mm.consolidate([{"role": "user", "content": "test"}])
        # Should not have overwritten with bad format
        current = mem_path.read_text(encoding="utf-8")
        assert "###" in current  # Still has the initial headers

    def test_consolidate_filters_tool_messages(self, tmp_path):
        """Los mensajes de herramientas se filtran antes de enviar al LLM."""
        mem_path = tmp_path / "MEMORY.md"
        received_prompt = []
        mm = MemoryManager(
            memory_path=mem_path,
            consolidate_fn=lambda p: (received_prompt.append(p), "### 🏗️ Arquitectura\n- ok")[1],
        )
        history = [
            {"role": "user", "content": "do something"},
            {"role": "tool", "content": "massive output " * 1000},
            {"role": "assistant", "content": "done"},
        ]
        mm.consolidate(history)
        # Tool message should not appear in the prompt sent to LLM
        assert "massive output" not in received_prompt[0]


class TestLegacyMigration:
    """Tests de migración de failed_patterns.md."""

    def test_migrates_legacy_file(self, tmp_path):
        """Si failed_patterns.md existe y MEMORY.md no, migra."""
        legacy = tmp_path / "failed_patterns.md"
        target = tmp_path / "MEMORY.md"
        legacy.write_text(
            "# Project Memory\n\n## Failure: multiply (Cycle 0)\n- Reason: below MIN_PASS\n",
            encoding="utf-8",
        )
        MemoryManager.migrate_legacy_memory(legacy_path=legacy, target_path=target)
        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "multiply" in content
        assert "Caminos Muertos" in content

    def test_no_migration_if_target_exists(self, tmp_path):
        """Si MEMORY.md ya existe, no sobrescribe."""
        legacy = tmp_path / "failed_patterns.md"
        target = tmp_path / "MEMORY.md"
        legacy.write_text("legacy content", encoding="utf-8")
        target.write_text("existing memory", encoding="utf-8")
        MemoryManager.migrate_legacy_memory(legacy_path=legacy, target_path=target)
        assert target.read_text(encoding="utf-8") == "existing memory"
