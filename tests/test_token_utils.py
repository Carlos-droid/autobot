"""
tests/test_token_utils.py — Tests unitarios para core/token_utils.py.
Patrón AAA: Arrange → Act → Assert. Aislados e independientes.
"""

import pytest
from core.token_utils import estimate_tokens, estimate_message_tokens, is_above_threshold


class TestEstimateTokens:
    """Tests de la estimación rápida de tokens."""

    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        result = estimate_tokens("hello")
        # 5 chars / 4 * 1.33 ≈ 1.66 → 1
        assert result >= 1

    def test_long_string_scales_linearly(self):
        short = estimate_tokens("a" * 100)
        long = estimate_tokens("a" * 1000)
        # Long should be ~10x short
        assert 8 <= long / short <= 12

    def test_none_like_empty(self):
        # Should not crash on empty
        assert estimate_tokens("") == 0

    def test_unicode_text(self):
        result = estimate_tokens("こんにちは世界！")
        assert result > 0


class TestEstimateMessageTokens:
    """Tests de estimación de tokens en historiales completos."""

    def test_empty_history(self):
        assert estimate_message_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        result = estimate_message_tokens(msgs)
        assert result > 0

    def test_multiple_messages_additive(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = estimate_message_tokens(msgs)
        single = estimate_message_tokens([msgs[0]])
        # Total should be greater than one message
        assert result > single

    def test_tool_calls_counted(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/test.py"}',
                        }
                    }
                ],
            }
        ]
        result = estimate_message_tokens(msgs)
        assert result > 0

    def test_multiblock_content(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_result", "content": "Result here"},
                ],
            }
        ]
        result = estimate_message_tokens(msgs)
        assert result > 0


class TestIsAboveThreshold:
    """Tests del chequeo de umbral."""

    def test_below_threshold(self):
        msgs = [{"role": "user", "content": "short"}]
        assert is_above_threshold(msgs, 1000) is False

    def test_above_threshold(self):
        msgs = [{"role": "user", "content": "x" * 10000}]
        assert is_above_threshold(msgs, 100) is True

    def test_exact_threshold(self):
        # At threshold should return True (>= comparison)
        msgs = [{"role": "user", "content": "x" * 400}]
        tokens = estimate_message_tokens(msgs)
        assert is_above_threshold(msgs, tokens) is True
