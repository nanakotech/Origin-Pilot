import pytest
import sympy as sp
import numpy as np
from pyqpanda_alg.QUBO import QUBO


class Test_QUBO_function_value:

    def test_function_value_basic(self):
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        test0 = QUBO.QuadraticBinary(function)

        value = test0.function_value([0, 1, 0])
        value_rel = -0.5 * 0 * 1 - 0.7 * 0 * 1 + 0.9 * 1 * 0 + 1.3 * 0 - 1 - 0.5 * 0
        assert value_rel == value
        assert value is not None, "function_value应该返回非None结果"
        assert isinstance(value, (int, float, np.number)), f"函数值应该是数值类型，实际是{type(value)}"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])