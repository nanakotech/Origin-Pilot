import pytest
import numpy as np
from pyqpanda_alg.QSEncode import QSpare_Code
import warnings
import os

class Test_class_basic_sparecode_QSpare_Code:
    """QSpare_Code模块测试类"""

    def setup_method(self):
        """测试方法前置设置"""
        warnings.filterwarnings("ignore")
        np.random.seed(42)  # 设置随机种子保证结果可重现
        # 确保测试输出目录存在
        os.makedirs('test_outputs', exist_ok=True)


    def test_interface12_walsh_mode_basic_1(self):
        mu = 0
        sigma = 1
        x = np.linspace(-3, 3, 2 ** 10)
        pdf_normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        ini = pdf_normal / np.linalg.norm(pdf_normal)
        res = QSpare_Code(ini ** 2, mode='walsh', cut_length=20).Quantum_Res()
        res1 = QSpare_Code(ini ** 2, mode='fourier', cut_length=20).Quantum_Res()

        prob_list = ini ** 2
        original_peak = x[np.argmax(prob_list)]
        result_peak = x[np.argmax(res)]
        result_peak1 = x[np.argmax(res1)]
        assert abs(original_peak - result_peak) < 0.5
        assert abs(original_peak - result_peak1) < 0.5

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])