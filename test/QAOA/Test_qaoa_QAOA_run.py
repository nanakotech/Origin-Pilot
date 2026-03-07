import pytest
import sys
from pathlib import Path
import sympy as sp
import numpy as np

# 添加项目路径
sys.path.append((Path.cwd().parent.parent).__str__())

from pyqpanda_alg.QAOA import qaoa


class TestQAOARun:
    """测试QAOA的run接口"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.vars = sp.symbols('x0:3')
        self.f = 2*self.vars[0]*self.vars[1] + 3*self.vars[2] - 1
        self.qaoa_instance = qaoa.QAOA(self.f)
    
    def test_basic_functionality_default_parameters(self):
        """测试基本功能：使用默认参数"""
        # 运行QAOA优化
        result = self.qaoa_instance.run(layer=2)
        
        # 验证返回结果格式
        assert isinstance(result, tuple), "结果应该是元组"
        assert len(result) >= 2, "结果应该至少包含两个元素"
        
        # 验证第一个元素是概率分布字典
        prob_dist = result[0]
        assert isinstance(prob_dist, dict), "第一个元素应该是字典"
        assert len(prob_dist) > 0, "概率分布字典不应为空"
        
        # 验证概率总和
        total_prob = sum(prob_dist.values())
        assert abs(total_prob - 1.0) < 1e-10, f"概率总和应该为1，实际为{total_prob}"
        
        # 验证概率值范围
        for prob in prob_dist.values():
            assert 0 <= prob <= 1, f"概率值应该在[0,1]范围内，实际为{prob}"
        
        # 验证二进制字符串格式
        for key in prob_dist.keys():
            assert isinstance(key, str), "键应该是字符串类型"
            assert all(bit in '01' for bit in key), "键应该是二进制字符串"
            assert len(key) == 3, "二进制字符串长度应该等于变量数(3)"
    

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])