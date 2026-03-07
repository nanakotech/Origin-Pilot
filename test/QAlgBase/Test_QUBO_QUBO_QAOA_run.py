import pytest
import sympy as sp
from pyqpanda_alg.QUBO import QUBO


class Test_QUBO_QUBO_QAOA_run:
    @pytest.fixture
    def setup_qubo_instance(self):
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        qubo_instance = QUBO.QUBO_QAOA(function)
        return qubo_instance

    def test_qubo_qaoa_basic_functionality(self, setup_qubo_instance):
        test_instance = setup_qubo_instance
        result = test_instance.run(layer=5, optimizer='SLSQP',
                                   optimizer_option={'options': {'eps': 1e-3}})

        print(f"计算结果: {result}")

        assert result['010'] >= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])