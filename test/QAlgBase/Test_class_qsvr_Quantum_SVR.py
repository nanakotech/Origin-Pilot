import pytest
import numpy as np
import os
from pyqpanda_alg.QSVR import Quantum_SVR
import warnings
import os


class Test_class_qsvr_Quantum_SVR:

    def setup_method(self):
        """测试方法前置设置"""
        warnings.filterwarnings("ignore")
        np.random.seed(42)  # 设置随机种子保证结果可重现
        os.makedirs('test_outputs', exist_ok=True)

    def test_interface11_show_res_basic(self):
        print("\n" + "=" * 60)
        print("测试接口11: Quantum_SVR.show_res() - 基本功能")
        print("=" * 60)

        # 创建测试数据（与示例相同）
        n_samples = 100
        n_features = 2
        X = np.random.rand(n_samples, n_features) * 10
        y = 2 * np.sin(X[:, 0]) + 1.5 * np.cos(X[:, 1])

        # 创建Quantum_SVR实例
        qsvr = Quantum_SVR(X, y)
        assert qsvr is not None, "Quantum_SVR实例创建失败"
        print("✓ Quantum_SVR实例创建成功")

        # 调用show_res()方法
        try:
            result = qsvr.show_res()
            print(result)

        except Exception as e:
            pytest.fail(f"show_res()方法执行失败: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])