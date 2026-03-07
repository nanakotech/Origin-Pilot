import pytest
import sympy as sp
from pyqpanda_alg.QUBO import QUBO


class Test_QUBO_qubobytraversal:

    def test_qubobytraversal_basic(self):
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        test0 = QUBO.QuadraticBinary(function)

        api_solution, api_value = test0.qubobytraversal()
        print(api_solution)
        print(api_value)
        all_combinations = [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ]

        all_values = []
        for comb in all_combinations:
            value = test0.function_value(comb)
            all_values.append((comb, value))

        theoretical_optimal_comb, theoretical_optimal_value = min(all_values, key=lambda x: x[1])

        assert api_value == theoretical_optimal_value, \
            f"最优值不匹配: 接口返回{api_value}, 理论值{theoretical_optimal_value}"

        api_solution_value = test0.function_value(api_solution[0])
        assert abs(api_solution_value - theoretical_optimal_value) < 1e-10, \
            f"接口返回的解计算值{api_solution_value}与最优值{theoretical_optimal_value}不匹配"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
