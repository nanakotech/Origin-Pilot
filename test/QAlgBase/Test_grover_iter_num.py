import pytest
import math
from pyqpanda_alg.Grover import iter_num


class Test_grover_iter_num:

    def test_iter_num_basic_case(self):
        q_num = 3
        sol_num = 2

        # 执行计算
        result = iter_num(q_num=q_num, sol_num=sol_num)

        assert isinstance(result, int), "迭代次数应该是整数类型"
        assert result >= 0, "迭代次数应该大于等于0"
        N = 2 ** q_num
        M = sol_num
        theoretical_r = round((math.pi / 4) * math.sqrt(N / M))
        assert abs(result - theoretical_r) <= 1, f"迭代次数 {result} 与理论值 {theoretical_r} 差异过大"


if __name__ == "__main__":
    # 直接运行所有测试
    pytest.main([__file__, "-v"])
