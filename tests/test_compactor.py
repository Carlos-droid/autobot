"""
tests/test_compactor.py — Tests unitarios para core/compactor.py.
Verifica los 3 niveles de compactación individualmente y en combinación.
"""

import pytest
from core.compactor import Compactor, TRUNCATED_MARKER


class TestLevel1Pruning:
    """Tests del Nivel 1: Pruning Quirúrgico de tool outputs."""

    def test_short_tool_output_unchanged(self):
        """Tool outputs bajo el umbral no se tocan."""
        compactor = Compactor(max_tool_chars=100)
        msgs = [{"role": "tool", "content": "short output"}]
        result, saved = compactor.prune_tool_results(msgs)
        assert result[0]["content"] == "short output"
        assert saved == 0

    def test_long_tool_output_truncated(self):
        """Tool outputs sobre el umbral se truncan conservando inicio+final."""
        compactor = Compactor(max_tool_chars=100)
        long_content = "x" * 500
        msgs = [{"role": "tool", "content": long_content}]
        result, saved = compactor.prune_tool_results(msgs)

        assert "TRUNCATED" in result[0]["content"]
        assert len(result[0]["content"]) < len(long_content)
        assert saved > 0

    def test_preserves_head_and_tail(self):
        """El truncamiento conserva el inicio y final del contenido."""
        compactor = Compactor(max_tool_chars=100)
        content = "HEAD_MARKER " + "x" * 500 + " TAIL_MARKER"
        msgs = [{"role": "tool", "content": content}]
        result, _ = compactor.prune_tool_results(msgs)

        assert "HEAD_MARKER" in result[0]["content"]
        assert "TAIL_MARKER" in result[0]["content"]

    def test_user_messages_untouched(self):
        """Mensajes de usuario no se truncan."""
        compactor = Compactor(max_tool_chars=10)
        msgs = [{"role": "user", "content": "x" * 1000}]
        result, saved = compactor.prune_tool_results(msgs)
        assert result[0]["content"] == "x" * 1000
        assert saved == 0

    def test_multiblock_tool_result_truncated(self):
        """Tool results en formato multi-bloque también se truncan."""
        compactor = Compactor(max_tool_chars=100)
        msgs = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "y" * 500, "tool_use_id": "123"},
            ],
        }]
        result, saved = compactor.prune_tool_results(msgs)
        assert "TRUNCATED" in result[0]["content"][0]["content"]
        assert saved > 0

    def test_tokens_saved_positive(self):
        """La cantidad de tokens ahorrados es positiva para contenido largo."""
        compactor = Compactor(max_tool_chars=100)
        msgs = [
            {"role": "tool", "content": "a" * 5000},
            {"role": "tool", "content": "b" * 3000},
        ]
        _, saved = compactor.prune_tool_results(msgs)
        assert saved > 100


class TestLevel2Summarization:
    """Tests del Nivel 2: Colapso de Turnos."""

    def test_no_summarize_fn_returns_original(self):
        """Sin función de summarización, retorna mensajes sin cambios."""
        compactor = Compactor(summarize_fn=None, sticky_count=2)
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = compactor.summarize_old_turns(msgs)
        assert len(result) == 10

    def test_summarize_replaces_old_messages(self):
        """Con summarize_fn, los mensajes antiguos se reemplazan por resumen."""
        mock_summary = "Technical summary of conversation."
        compactor = Compactor(
            summarize_fn=lambda p: mock_summary,
            sticky_count=3,
        )
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = compactor.summarize_old_turns(msgs)

        # Debería tener 1 resumen + 3 sticky
        assert len(result) == 4
        assert "CONTEXT SUMMARY" in result[0]["content"]
        assert mock_summary in result[0]["content"]

    def test_sticky_latches_preserved(self):
        """Los últimos N mensajes se preservan intactos."""
        compactor = Compactor(
            summarize_fn=lambda p: "summary",
            sticky_count=3,
        )
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = compactor.summarize_old_turns(msgs)

        # Los últimos 3 deben ser los originales
        assert result[-1]["content"] == "msg 9"
        assert result[-2]["content"] == "msg 8"
        assert result[-3]["content"] == "msg 7"

    def test_too_few_messages_no_summarize(self):
        """Con muy pocos mensajes, no se activa la summarización."""
        compactor = Compactor(
            summarize_fn=lambda p: "summary",
            sticky_count=5,
        )
        msgs = [{"role": "user", "content": "hello"}]
        result = compactor.summarize_old_turns(msgs)
        assert len(result) == 1

    def test_circuit_breaker_after_failures(self):
        """Después de N fallos consecutivos, deja de intentar."""
        call_count = 0

        def failing_fn(prompt):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("API error")

        compactor = Compactor(summarize_fn=failing_fn, sticky_count=2)
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(10)]

        # Fallar 3 veces (circuit breaker default)
        for _ in range(3):
            compactor.summarize_old_turns(msgs)

        # Después del circuit breaker, compact_if_needed no debería llamar L2
        assert compactor._consecutive_failures >= 3


class TestCompactIfNeeded:
    """Tests del orquestador de compactación."""

    def test_below_threshold_no_action(self):
        """Bajo el umbral no se compacta nada."""
        compactor = Compactor(threshold=100_000)
        msgs = [{"role": "user", "content": "short"}]
        result = compactor.compact_if_needed(msgs)
        assert len(result) == 1

    def test_above_threshold_triggers_pruning(self):
        """Sobre el umbral, al menos Nivel 1 se activa."""
        compactor = Compactor(threshold=100, max_tool_chars=50)
        msgs = [
            {"role": "tool", "content": "x" * 1000},
            {"role": "user", "content": "continue"},
        ]
        result = compactor.compact_if_needed(msgs)
        assert "TRUNCATED" in result[0]["content"]
