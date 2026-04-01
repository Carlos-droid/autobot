import pytest
from core.operations import *
class TestMultiply:
    def test_multiply_positive_numbers(self): 
        assert multiply(4, 3) == 12.0
    def test_multiply_negative_numbers(self): 
        assert multiply(-7, -2) == 14.0
    def test_multiply_mixed_signs(self):
        assert multiply(5, -3) == -15.0
    def test_multiply_with_zero(self):
        assert multiply(9, 0) == 0.0
def test_multiply_invalid_input():
    with pytest.raises(ValueError):
        multiply('a', 'b')