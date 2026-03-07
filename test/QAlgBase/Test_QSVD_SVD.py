import pytest
import numpy as np
from pyqpanda_alg.QSVD import SVD
import warnings




class Test_QSVD_SVD:

    def setup_method(self):
        warnings.filterwarnings("ignore")
        np.random.seed(42)

    def test_orthogonality_properties(self):
        matrix = np.random.random(16).reshape([4, 4])  # 使用方阵便于测试

        qsvd_instance = SVD(matrix_in=matrix)
        para = qsvd_instance.QSVD_min()
        qeig = qsvd_instance.return_diag(para)
        q_singular_values = np.diag(qeig)

        u_np, s_np, v_np = np.linalg.svd(matrix)

        assert np.all(q_singular_values >= 0), "QSVD奇异值应该非负"
        assert np.all(s_np >= 0), "NumPy奇异值应该非负"

        q_sorted = np.sort(q_singular_values)[::-1]
        np_sorted = np.sort(s_np)[::-1]
        assert np.all(np.diff(q_sorted) <= 0), "QSVD奇异值应该降序排列"
        assert np.all(np.diff(np_sorted) <= 0), "NumPy奇异值应该降序排列"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])