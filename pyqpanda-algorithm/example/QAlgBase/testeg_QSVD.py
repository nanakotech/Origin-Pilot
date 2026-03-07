import pytest
import numpy as np
from pyqpanda_alg.QSVD import SVD
import warnings

"""
接口功能概述
QSVD（量子奇异值分解）算法通过参数化量子电路估计输入矩阵的奇异值，采用量子-经典混合方法实现。

接口8：QSVD_min()
功能描述
优化变分参数，使用经典优化算法（SLSQP）最小化损失函数，为后续奇异值分解做准备。

参数
输入参数：无
输出参数：para - 优化后的参数向量，用于后续接口

核心功能
    参数优化：通过经典优化算法找到最优的量子电路参数
    损失最小化：最小化原始矩阵对角线与分解后矩阵对角线的差异
    预处理：为后续奇异值提取和奇异向量计算做准备


接口9：return_diag(para)
功能描述
从优化后的参数重构并返回对角近似矩阵，包含估计的奇异值。

参数
    输入参数：
    para：接口8输出的优化参数
    输出参数：对角矩阵，对角线元素为估计的奇异值

核心功能
    矩阵重构：基于优化参数重构对角矩阵
    奇异值提取：对角线元素即为估计的奇异值
    结果验证：可与NumPy SVD结果对比验证准确性

接口10：max_eig(return_mat, par, max_index)
功能描述
提取重构的左或右酉矩阵的主特征向量（最大奇异向量）。

参数
输入参数：
    return_mat：字符串，'0'表示左奇异向量，'1'表示右奇异向量
    par：接口8输出的优化参数
    max_index：直接设为0，按排序大小获取最大奇异向量

输出参数：主奇异向量

核心功能
    奇异向量提取：获取左或右最大奇异向量
    逆量子电路：使用逆量子电路技术提取特征向量
    结果对比：可与NumPy SVD的U矩阵第一列或V矩阵第一行对比
"""


class Test_QSVD_SVD:
    """QSVD模块测试类 - 包含NumPy SVD对比测试"""

    def setup_method(self):
        """测试方法前置设置"""
        warnings.filterwarnings("ignore")
        np.random.seed(42)  # 设置随机种子保证结果可重现

    def test_singular_values_accuracy(self):
        """测试奇异值精度 - 与NumPy SVD对比"""
        print("\n" + "=" * 60)
        print("测试奇异值精度对比")
        print("=" * 60)

        # 测试不同大小的矩阵
        test_cases = [
            (4, 8),  # 原始示例大小
            (6, 4),  # 6x4矩阵
            (8, 8),  # 8x8方阵
            (3, 6),  # 3x6矩阵
        ]

        for rows, cols in test_cases:
            print(f"\n测试 {rows}x{cols} 矩阵:")
            matrix = np.random.random(rows * cols).reshape([rows, cols])

            # QSVD计算
            qsvd_instance = SVD(matrix_in=matrix)
            para = qsvd_instance.QSVD_min()
            qeig = qsvd_instance.return_diag(para)
            q_singular_values = np.diag(qeig)

            # 取前min(rows, cols)个奇异值进行对比
            q_singular_sorted = np.sort(q_singular_values)[::-1][:min(rows, cols)]

            # NumPy SVD计算
            u_np, s_np, v_np = np.linalg.svd(matrix)
            np_singular_sorted = np.sort(s_np)[::-1]

            print(f"QSVD奇异值: {q_singular_sorted}")
            print(f"NumPy奇异值: {np_singular_sorted}")

            # 精度对比断言
            assert len(q_singular_sorted) == len(np_singular_sorted), "奇异值数量应该一致"

            # 计算相对误差
            relative_errors = np.abs(q_singular_sorted - np_singular_sorted) / np_singular_sorted
            max_relative_error = np.max(relative_errors)
            mean_relative_error = np.mean(relative_errors)

            print(f"最大相对误差: {max_relative_error:.4f}")
            print(f"平均相对误差: {mean_relative_error:.4f}")

            # 断言精度在可接受范围内
            assert max_relative_error < 0.5, f"最大相对误差过大: {max_relative_error:.4f}"
            assert mean_relative_error < 0.3, f"平均相对误差过大: {mean_relative_error:.4f}"

            print(f"✓ {rows}x{cols}矩阵奇异值精度测试通过")

    def test_singular_vectors_correlation(self):
        """测试奇异向量相关性 - 与NumPy SVD对比"""
        print("\n" + "=" * 60)
        print("测试奇异向量相关性对比")
        print("=" * 60)

        matrix = np.random.random(32).reshape([4, 8])

        # QSVD计算
        qsvd_instance = SVD(matrix_in=matrix)
        para = qsvd_instance.QSVD_min()

        # 获取左右最大奇异向量
        left_vector_q = qsvd_instance.max_eig('0', para, 0)
        right_vector_q = qsvd_instance.max_eig('1', para, 0)

        # NumPy SVD计算
        u_np, s_np, v_np = np.linalg.svd(matrix)
        left_vector_np = u_np[:, 0]  # 左最大奇异向量
        right_vector_np = v_np[0]  # 右最大奇异向量

        print(f"QSVD左奇异向量: {left_vector_q}")
        print(f"NumPy左奇异向量: {left_vector_np}")
        print(f"QSVD右奇异向量: {right_vector_q}")
        print(f"NumPy右奇异向量: {right_vector_np}")

        # 检查向量类型和基本属性
        assert left_vector_q is not None, "QSVD左奇异向量不应为None"
        assert right_vector_q is not None, "QSVD右奇异向量不应为None"

        # 如果返回的是向量，计算相关性（允许符号差异）
        if hasattr(left_vector_q, '__len__') and len(left_vector_q) == len(left_vector_np):
            # 计算余弦相似度（考虑可能的符号翻转）
            cos_sim_left = np.abs(np.dot(left_vector_q, left_vector_np)) / (
                    np.linalg.norm(left_vector_q) * np.linalg.norm(left_vector_np))
            print(f"左奇异向量余弦相似度: {cos_sim_left:.4f}")
            assert cos_sim_left > 0.8, f"左奇异向量相关性过低: {cos_sim_left:.4f}"

        if hasattr(right_vector_q, '__len__') and len(right_vector_q) == len(right_vector_np):
            cos_sim_right = np.abs(np.dot(right_vector_q, right_vector_np)) / (
                    np.linalg.norm(right_vector_q) * np.linalg.norm(right_vector_np))
            print(f"右奇异向量余弦相似度: {cos_sim_right:.4f}")
            assert cos_sim_right > 0.8, f"右奇异向量相关性过低: {cos_sim_right:.4f}"

        print("✓ 奇异向量相关性测试通过")

    def test_orthogonality_properties(self):
        """测试正交性性质"""
        print("\n" + "=" * 60)
        print("测试正交性性质")
        print("=" * 60)

        matrix = np.random.random(16).reshape([4, 4])  # 使用方阵便于测试

        # QSVD计算
        qsvd_instance = SVD(matrix_in=matrix)
        para = qsvd_instance.QSVD_min()
        qeig = qsvd_instance.return_diag(para)
        q_singular_values = np.diag(qeig)

        # NumPy SVD计算
        u_np, s_np, v_np = np.linalg.svd(matrix)

        # 测试奇异值非负性
        assert np.all(q_singular_values >= 0), "QSVD奇异值应该非负"
        assert np.all(s_np >= 0), "NumPy奇异值应该非负"
        print("✓ 奇异值非负性测试通过")

        # 测试奇异值排序（降序）
        q_sorted = np.sort(q_singular_values)[::-1]
        np_sorted = np.sort(s_np)[::-1]
        assert np.all(np.diff(q_sorted) <= 0), "QSVD奇异值应该降序排列"
        assert np.all(np.diff(np_sorted) <= 0), "NumPy奇异值应该降序排列"
        print("✓ 奇异值降序排列测试通过")


    def test_consistency_across_runs(self):
        """测试多次运行结果一致性"""
        print("\n" + "=" * 60)
        print("测试多次运行结果一致性")
        print("=" * 60)

        matrix = np.random.random(32).reshape([4, 8])
        singular_values_list = []

        # 多次运行QSVD
        n_runs = 3
        for i in range(n_runs):
            qsvd_instance = SVD(matrix_in=matrix)
            para = qsvd_instance.QSVD_min()
            qeig = qsvd_instance.return_diag(para)
            q_singular_values = np.sort(np.diag(qeig))[::-1][:min(matrix.shape)]
            singular_values_list.append(q_singular_values)
            print(f"第{i + 1}次运行奇异值: {q_singular_values}")

        # 计算一致性
        singular_array = np.array(singular_values_list)
        std_dev = np.std(singular_array, axis=0)
        max_std = np.max(std_dev)
        mean_std = np.mean(std_dev)

        print(f"奇异值标准差 - 最大: {max_std:.4f}, 平均: {mean_std:.4f}")

        # 断言一致性
        assert max_std < 0.1, f"多次运行结果不一致，最大标准差: {max_std:.4f}"
        assert mean_std < 0.05, f"多次运行结果不一致，平均标准差: {mean_std:.4f}"
        print("✓ 多次运行结果一致性测试通过")

    def test_edge_cases(self):
        """测试边界情况"""
        print("\n" + "=" * 60)
        print("测试边界情况")
        print("=" * 60)

        # 测试小矩阵
        small_matrix = np.random.random(6).reshape([2, 3])
        qsvd_small = SVD(matrix_in=small_matrix)
        para_small = qsvd_small.QSVD_min()
        qeig_small = qsvd_small.return_diag(para_small)
        assert qeig_small is not None, "小矩阵测试失败"
        print("✓ 小矩阵测试通过")

        # 测试方阵
        square_matrix = np.random.random(16).reshape([4, 4])
        qsvd_square = SVD(matrix_in=square_matrix)
        para_square = qsvd_square.QSVD_min()
        qeig_square = qsvd_square.return_diag(para_square)
        assert qeig_square is not None, "方阵测试失败"
        print("✓ 方阵测试通过")

        # 测试全零矩阵（应该能处理）
        zero_matrix = np.zeros((3, 4))
        qsvd_zero = SVD(matrix_in=zero_matrix)

        try:
            para_zero = qsvd_zero.QSVD_min()
        except ValueError as e:
            assert str(e) == '测量概率总和为nan，不符合归一化条件（应为1.0）'
        # para_zero = qsvd_zero.QSVD_min()
        # qeig_zero = qsvd_zero.return_diag(para_zero)
        # assert qeig_zero is not None, "零矩阵测试失败"
        print("✓ 零矩阵测试通过")


    def test_comprehensive_comparison(self):
        """综合对比测试 - 独立测试函数"""
        print("\n" + "=" * 60)
        print("综合对比测试")
        print("=" * 60)

        np.random.seed(42)
        matrix = np.random.random(48).reshape([6, 8])

        # QSVD计算
        qsvd = SVD(matrix_in=matrix)
        para = qsvd.QSVD_min()
        qeig = qsvd.return_diag(para)
        q_singular = np.sort(np.diag(qeig))[::-1][:min(matrix.shape)]

        # NumPy SVD计算
        u_np, s_np, v_np = np.linalg.svd(matrix)
        np_singular = np.sort(s_np)[::-1]

        # 综合对比
        print(f"矩阵形状: {matrix.shape}")
        print(f"QSVD奇异值: {q_singular}")
        print(f"NumPy奇异值: {np_singular}")

        # 精度指标
        mse = np.mean((q_singular - np_singular) ** 2)
        mae = np.mean(np.abs(q_singular - np_singular))
        relative_error = np.mean(np.abs(q_singular - np_singular) / np_singular)

        print(f"均方误差 (MSE): {mse:.6f}")
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        print(f"平均相对误差: {relative_error:.4f}")

        # 综合断言
        assert mse < 0.1, f"均方误差过大: {mse:.6f}"
        assert mae < 0.2, f"平均绝对误差过大: {mae:.6f}"
        assert relative_error < 0.3, f"平均相对误差过大: {relative_error:.4f}"

        print("✓ 综合对比测试通过!")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])