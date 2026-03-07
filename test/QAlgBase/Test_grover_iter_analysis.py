import pytest
import math
from pyqpanda_alg.Grover import iter_analysis, iter_num


class Test_grover_iter_analysis:

    def calculate_theoretical_probability(self, q_num, sol_num, iternum):
        N = 2 ** q_num
        M = sol_num

        if M == 0 or M == N:
            return 1.0 if M == N else 0.0
        theta = math.asin(math.sqrt(M / N))
        probability = math.sin((2 * iternum + 1) * theta) ** 2

        return probability

    def test_iter_analysis_basic_case(self):
        q_num = 3
        sol_num = 2
        iternum = 1

        prob, angle = iter_analysis(q_num=q_num, sol_num=sol_num, iternum=iternum)
        theoretical_prob = self.calculate_theoretical_probability(q_num, sol_num, iternum)
        assert isinstance(prob, float), "概率应该是浮点数类型"
        assert isinstance(angle, float), "角度应该是浮点数类型"
        assert 0 <= prob <= 1, "概率应该在0到1之间"
        assert angle >= 0, "角度应该大于等于0"
        assert abs(prob - theoretical_prob) < 1e-10, \
            f"计算概率 {prob} 与理论值 {theoretical_prob} 差异过大"


if __name__ == "__main__":
    # 直接运行所有测试
    pytest.main([__file__, "-v"])
