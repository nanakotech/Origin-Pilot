import pytest
import sys
from pathlib import Path
import sympy as sp
import numpy as np

# 添加项目路径
sys.path.append((Path.cwd().parent.parent).__str__())

from pyqpanda_alg.QAOA.qaoa import QAOA


class TestRunQAOACircuit:
    """测试run_qaoa_circuit接口"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建测试问题：f = 2*x0*x1 + 3*x2 - 1
        self.vars = sp.symbols('x0:3')
        self.f = 2*self.vars[0]*self.vars[1] + 3*self.vars[2] - 1
        self.qaoa_instance = QAOA(self.f)
        
        # 通用的测试参数
        self.gammas = [1.0, 2.0]
        self.betas = [2.0, 1.0]
    
    def test_basic_functionality_probability_measurement(self):
        """测试基本功能：概率测量模式 (shot=-1)"""
        # 运行QAOA电路
        result = self.qaoa_instance.run_qaoa_circuit(self.gammas, self.betas, -1)
        
        # 验证返回结果格式
        assert isinstance(result, dict), "结果应该是字典类型"
        assert len(result) > 0, "结果字典不应为空"
        
        # 验证键的格式（应该是二进制字符串）
        for key in result.keys():
            assert isinstance(key, str), "键应该是字符串类型"
            assert all(bit in '01' for bit in key), "键应该是二进制字符串"
            assert len(key) == 3, "二进制字符串长度应该等于变量数(3)"
        
        # 验证概率值
        total_probability = sum(result.values())
        assert abs(total_probability - 1.0) < 1e-10, f"概率总和应该为1，实际为{total_probability}"
        
        # 验证所有概率值在合理范围内
        for prob in result.values():
            assert 0 <= prob <= 1, f"概率值应该在[0,1]范围内，实际为{prob}"
    


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])