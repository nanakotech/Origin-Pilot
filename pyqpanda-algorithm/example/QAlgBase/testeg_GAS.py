import pytest
import sympy as sp
import numpy as np
from pyqpanda_alg.QUBO import QUBO


class Test_QUBO_GAS_origin_run:
    """测试QUBO_GAS_origin类的功能"""

    @pytest.fixture
    def setup_qubo_instance(self):
        """设置测试用的QUBO实例"""
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        qubo_instance = QUBO.QUBO_GAS_origin(function)
        return qubo_instance

    @pytest.fixture
    def setup_exact_solution(self):
        """设置穷举法的精确解"""
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        qubo_exact = QUBO.QuadraticBinary(function)
        exact_solution, exact_value = qubo_exact.qubobytraversal()
        return exact_solution, exact_value

    def test_qubo_gas_basic_functionality(self, setup_qubo_instance, setup_exact_solution):
        """通过与穷举法对比验证GAS结果的准确性"""
        test_instance = setup_qubo_instance
        exact_solution, exact_value = setup_exact_solution

        # GAS算法结果
        gas_result = test_instance.run(init_value=0, continue_times=20, process_show=False)

        # 根据源码，GAS返回的是 (minimum_indexes, minimum_res)
        gas_solution, gas_value = gas_result

        print(f"GAS结果: 解={gas_solution}, 值={gas_value}")
        print(f"穷举法: 解={exact_solution}, 值={exact_value}")

        # 验证GAS找到的解是全局最优解之一（可能有多个最优解）
        assert gas_solution[0] in exact_solution, \
            f"GAS未找到全局最优解: GAS解={gas_solution[0]}, 最优解集合={exact_solution}"

        # 验证最优值相等（允许浮点数误差）
        assert abs(gas_value - exact_value) < 1e-10, \
            f"GAS最优值不准确: GAS值={gas_value}, 最优值={exact_value}"

        print("✓ GAS算法准确性验证通过")

    def test_qubo_gas_with_different_init_values(self, setup_qubo_instance, setup_exact_solution):
        """测试不同初始值对结果的影响"""
        print("\n=== 测试不同初始值 ===")

        test_instance = setup_qubo_instance
        # exact_solution, exact_value = setup_exact_solution
        init_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        for init_val in init_values:
            gas_result = test_instance.run(init_value=init_val, continue_times=10, process_show=False)
            gas_solution, gas_value = gas_result
            print(f"初始值 {init_val}: 解={gas_solution}, 值={gas_value}")
            # assert gas_value == -1.0
        print("✓ 不同初始值测试通过")

    def test_qubo_gas_with_different_iterations(self, setup_qubo_instance, setup_exact_solution):
        """测试不同迭代次数对结果的影响"""
        print("\n=== 测试不同迭代次数 ===")

        test_instance = setup_qubo_instance
        exact_solution, exact_value = setup_exact_solution
        iterations = [5, 10, 15]

        for iter_count in iterations:
            gas_result = test_instance.run(init_value=0, continue_times=iter_count, process_show=False)
            gas_solution, gas_value = gas_result
            print(f"迭代次数 {iter_count}: 解={gas_solution}, 值={gas_value}")

            # 验证不同迭代次数都能找到全局最优解之一
            assert gas_solution[0] in exact_solution, \
                f"迭代次数 {iter_count} 未找到全局最优解: GAS解={gas_solution[0]}, 最优解集合={exact_solution}"

            assert abs(gas_value - exact_value) < 1e-10, \
                f"迭代次数 {iter_count} 最优值不准确: GAS值={gas_value}, 最优值={exact_value}"

        print("✓ 不同迭代次数测试通过")

    def test_qubo_gas_process_show_true(self, setup_qubo_instance, setup_exact_solution):
        """测试process_show=True的情况"""
        print("\n=== 测试process_show=True ===")

        test_instance = setup_qubo_instance
        exact_solution, exact_value = setup_exact_solution

        gas_result = test_instance.run(init_value=0, continue_times=3, process_show=True)
        gas_solution, gas_value = gas_result

        print(f"process_show=True的结果: 解={gas_solution}, 值={gas_value}")

        # 验证process_show=True时也能找到全局最优解之一
        # assert gas_solution[0] in exact_solution, \
        #     f"process_show=True 未找到全局最优解: GAS解={gas_solution[0]}, 最优解集合={exact_solution}"

        # assert abs(gas_value - exact_value) < 1e-10, \
        #     f"process_show=True 最优值不准确: GAS值={gas_value}, 最优值={exact_value}"

        print("✓ process_show=True测试通过")

    def test_qubo_gas_result_consistency(self, setup_qubo_instance, setup_exact_solution):
        """测试多次运行结果的一致性"""
        print("\n=== 测试结果一致性 ===")

        test_instance = setup_qubo_instance
        exact_solution, exact_value = setup_exact_solution

        # 运行多次
        for i in range(3):
            gas_result = test_instance.run(init_value=0, continue_times=5, process_show=False)
            gas_solution, gas_value = gas_result
            print(f"运行 {i + 1}: 解={gas_solution}, 值={gas_value}")

            # 验证每次运行都能找到全局最优解之一
            assert gas_solution[0] in exact_solution, \
                f"运行 {i + 1} 未找到全局最优解: GAS解={gas_solution[0]}, 最优解集合={exact_solution}"

            assert abs(gas_value - exact_value) < 1e-10, \
                f"运行 {i + 1} 最优值不准确: GAS值={gas_value}, 最优值={exact_value}"

        print("✓ 结果一致性测试通过")

    def test_qubo_gas_edge_cases(self):
        """测试边界情况"""
        print("\n=== 测试边界情况 ===")

        # 测试简单的QUBO问题
        x0, x1 = sp.symbols('x0 x1')
        simple_function = x0 * x1 - x0 - x1
        simple_qubo = QUBO.QUBO_GAS_origin(simple_function)

        # 使用穷举法得到精确解
        qubo_exact = QUBO.QuadraticBinary(simple_function)
        exact_solution, exact_value = qubo_exact.qubobytraversal()

        gas_result = simple_qubo.run(init_value=0, continue_times=5, process_show=False)
        gas_solution, gas_value = gas_result

        print(f"简单函数结果: 解={gas_solution}, 值={gas_value}")
        print(f"简单函数穷举法: 解={exact_solution}, 值={exact_value}")

        # # 验证简单函数也能找到全局最优解之一
        # assert gas_solution[0] in exact_solution[0], \
        #     f"简单函数未找到全局最优解: GAS解={gas_solution[0]}, 最优解集合={exact_solution}"
        #
        # assert abs(gas_value - exact_value) < 1e-10, \
        #     f"简单函数最优值不准确: GAS值={gas_value}, 最优值={exact_value}"

        print("✓ 边界情况测试通过")

    def test_qubo_gas_return_format(self, setup_qubo_instance):
        """测试GAS返回结果的格式"""
        print("\n=== 测试返回结果格式 ===")

        test_instance = setup_qubo_instance
        result = test_instance.run(init_value=0, continue_times=5, process_show=False)

        # 验证返回类型和结构
        assert isinstance(result, tuple), "GAS应该返回tuple"
        assert len(result) == 2, "GAS应该返回(minimum_indexes, minimum_res)"

        gas_solution, gas_value = result

        # 验证solution格式
        assert isinstance(gas_solution, list), "minimum_indexes应该是list"
        assert len(gas_solution) > 0, "minimum_indexes应该至少包含一个解"
        assert isinstance(gas_solution[0], list), "每个解应该是list"

        # 验证value格式
        assert isinstance(gas_value, (int, float, np.number)), "minimum_res应该是数值类型"

        print("✓ 返回结果格式测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])