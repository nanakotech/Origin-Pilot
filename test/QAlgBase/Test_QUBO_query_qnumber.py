
import pytest
import sympy as sp
import numpy as np
from pyqpanda_alg.QUBO import QUBO


class Test_QUBO_query_qnumber:

    @staticmethod
    def calculate_function_range_and_nres(function, variables):
        quadratic, linear, constant = Test_QUBO_query_qnumber._quadratic_func_to_coeff(function, variables)

        def pos(x): return x > 0

        def neg(x): return x < 0

        max_val = 0
        max_val += sum(sum(q_ij for q_ij in q_i if pos(q_ij)) for q_i in quadratic)
        max_val += sum(l_i for l_i in linear if pos(l_i))
        max_val += constant if pos(constant) else 0

        min_val = 0
        min_val += sum(sum(q_ij for q_ij in q_i if neg(q_ij)) for q_i in quadratic)
        min_val += sum(l_i for l_i in linear if neg(l_i))
        min_val += constant if neg(constant) else 0

        pos_bits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0
        neg_bits = int(np.ceil(np.log2(abs(min_val)))) + 1 if min_val < 0 else 0
        n_res_rel = max(pos_bits, neg_bits)

        return n_res_rel

    @staticmethod
    def _quadratic_func_to_coeff(quadratic_func, variables):
        quadratic_func = sp.Poly(quadratic_func, variables)
        monoms = quadratic_func.monoms()
        coeffs = quadratic_func.coeffs()

        n = len(variables)
        constant = 0
        linear = np.zeros(n)
        quadratic = np.zeros((n, n))

        for i, monom in enumerate(monoms):
            coeff = coeffs[i]
            # 计算单项式中每个变量的指数
            exp_dict = dict(zip(variables, monom))

            # 统计非零指数的变量
            nonzero_vars = [var for var in variables if exp_dict.get(var, 0) > 0]
            nonzero_exps = [exp_dict[var] for var in nonzero_vars]

            if len(nonzero_vars) == 0:
                # 常数项
                constant = coeff
            elif len(nonzero_vars) == 1 and nonzero_exps[0] == 1:
                # 线性项
                var_index = variables.index(nonzero_vars[0])
                linear[var_index] += coeff
            elif len(nonzero_vars) == 1 and nonzero_exps[0] == 2:
                # 二次项 (x_i^2 = x_i for binary variables)
                var_index = variables.index(nonzero_vars[0])
                linear[var_index] += coeff  # x_i^2 = x_i
            elif len(nonzero_vars) == 2 and all(exp == 1 for exp in nonzero_exps):
                # 交叉二次项
                var1_index = variables.index(nonzero_vars[0])
                var2_index = variables.index(nonzero_vars[1])
                quadratic[var1_index][var2_index] += coeff
            else:
                # 处理其他情况（理论上不应该出现，因为这是二次函数）
                raise ValueError(f"不支持的单项式: {monom}")

        return quadratic, linear, constant

    def test_query_qnumber_basic_function(self):

        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        test0 = QUBO.QuadraticBinary(function)
        n_key, n_res = test0.query_qnumber()
        n_res_rel = Test_QUBO_query_qnumber.calculate_function_range_and_nres(function, [x0, x1, x2])
        assert n_key == 3, f"对于3个变量的问题，n_key应该为3，实际是{n_key}"
        assert n_res > 0, f"n_res应该大于0，实际是{n_res}"
        assert n_res == n_res_rel, f"n_res计算错误: 实际值{n_res}, 理论值{n_res_rel}"



if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])