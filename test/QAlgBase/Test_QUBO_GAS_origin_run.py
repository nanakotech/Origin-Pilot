import pytest
import sympy as sp
import numpy as np
from pyqpanda_alg.QUBO import QUBO


class Test_QUBO_GAS_origin_run:

    @pytest.fixture
    def setup_qubo_instance(self):
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        qubo_instance = QUBO.QUBO_GAS_origin(function)
        return qubo_instance

    @pytest.fixture
    def setup_exact_solution(self):
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        qubo_exact = QUBO.QuadraticBinary(function)
        exact_solution, exact_value = qubo_exact.qubobytraversal()
        return exact_solution, exact_value

    def test_qubo_gas_basic_functionality(self, setup_qubo_instance, setup_exact_solution):
        test_instance = setup_qubo_instance
        exact_solution, exact_value = setup_exact_solution

        gas_result = test_instance.run(init_value=0, continue_times=20, process_show=False)

        gas_solution, gas_value = gas_result
        assert gas_solution[0] in exact_solution, \
            f"GAS未找到全局最优解: GAS解={gas_solution[0]}, 最优解集合={exact_solution}"

        # 验证最优值相等（允许浮点数误差）
        assert abs(gas_value - exact_value) < 1e-10, \
            f"GAS最优值不准确: GAS值={gas_value}, 最优值={exact_value}"




if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])