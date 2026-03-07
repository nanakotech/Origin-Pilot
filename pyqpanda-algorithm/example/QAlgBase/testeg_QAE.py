import pytest
import numpy as np
from pyqpanda_alg.QAE.QAE import IQAE
from pyqpanda3.core import QCircuit, RY, X, RZ

"""
接口功能概述: QAE.IQAE 是基于迭代量子振幅估计（Iterative Quantum Amplitude Estimation）算法的接口，专门用于估计单个量子比特的振幅。
与之前的QAE接口不同，它针对的是单个量子比特而非特定的量子态。
参数详解

参数	类型	说明	约束条件
operator_in	function	需要振幅估计的量子线路函数	必须返回 QCircuit 对象
qnumber	int	量子线路的比特数	必须 ≥ 1
epsilon	float	估计误差范围	值越小精度越高，但计算时间越长
res_index	int	要估计振幅的量子比特索引	只能是整数，支持负索引

核心功能
1. 单比特振幅估计
功能：估计指定量子比特处于 |1⟩ 状态的振幅
数学表达：对于量子态 |ψ⟩，估计 ⟨ψ|Π₁|ψ⟩，其中 Π₁ 是目标比特的 |1⟩⟨1| 投影算符
输出：返回一个在 [0, 1] 范围内的浮点数

2. 索引灵活性
支持正索引：res_index=0（第一个比特），res_index=1（第二个比特）
支持负索引：res_index=-1（最后一个比特），res_index=-2（倒数第二个比特）

3. 精度控制
通过 epsilon 参数控制估计精度
典型值：0.1（快速粗略），0.01（标准精度），0.001（高精度）
"""


class Test_QAE_IQAE:
    """测试 IQAE().run() 接口"""

    def create_cir_basic(self, qlist):
        """基础测试量子线路"""
        cir = QCircuit()
        cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
        return cir

    def create_cir_simple(self, qlist):
        """简单量子线路"""
        cir = QCircuit()
        cir << RY(qlist[0], np.pi / 4)
        return cir

    def create_cir_deterministic_0(self, qlist):
        """确定性量子线路 - 创建 |0⟩ 状态"""
        cir = QCircuit()
        # 不应用任何门，保持 |0⟩ 状态
        return cir

    def create_cir_deterministic_1(self, qlist):
        """确定性量子线路 - 创建 |1⟩ 状态"""
        cir = QCircuit()
        cir << X(qlist[0])  # 应用X门得到 |1⟩
        return cir

    def create_cir_multi_qubit(self, qlist):
        """多量子比特线路"""
        cir = QCircuit()
        cir << RY(qlist[0], np.pi / 3)
        cir << X(qlist[1]).control(qlist[0])
        cir << RY(qlist[2], np.pi / 6)
        cir << X(qlist[2]).control(qlist[1])
        return cir

    def create_cir_complex(self, qlist):
        """复杂量子线路"""
        cir = QCircuit()
        # 添加多个量子门操作
        cir << RY(qlist[0], np.pi / 3)
        cir << RZ(qlist[0], np.pi / 4)
        cir << X(qlist[1]).control(qlist[0])
        cir << RY(qlist[1], np.pi / 6)
        if len(qlist) > 2:
            cir << X(qlist[2]).control([qlist[0], qlist[1]])
        return cir

    def test_iqae_basic_functionality(self):
        """测试基本功能 - 估计最后一个量子比特的振幅"""
        print("\n=== 测试1: 基本功能测试 ===")

        W = IQAE(
            operator_in=self.create_cir_basic,
            qnumber=2,
            epsilon=0.01,
            res_index=-1  # 最后一个量子比特
        ).run()

        print(f"量子线路: create_cir_basic")
        print(f"目标量子比特: 最后一个 (索引-1)")
        print(f"估计振幅: {W}")

        # 断言验证
        assert W is not None, "振幅估计结果不应为None"
        assert isinstance(W, (float, np.floating)), f"振幅应为数值类型，实际为{type(W)}"
        assert 0.2 <= W <= 0.3, f"振幅应在[0,1]范围内，实际为{W}"

        print("✓ 基本功能测试通过")

    def test_iqae_different_res_index(self):
        """测试不同的res_index值"""
        print("\n测试2 - 不同res_index测试:")

        # 测试不同的res_index值
        test_cases = [
            (-1, "最后一个比特"),
            (0, "第一个比特"),
            (1, "第二个比特"),
        ]

        for res_index, description in test_cases:
            W = IQAE(
                operator_in=self.create_cir_basic,
                qnumber=2,
                epsilon=0.1,  # 使用较大的epsilon加快测试
                res_index=res_index
            ).run()

            print(f"{description} (res_index={res_index}) 的振幅: {W:.6f}")

            # 断言
            assert W is not None, f"res_index={res_index} 应该返回结果"
            assert isinstance(W, (float, np.floating)), "振幅应该是数值类型"
            assert 0 <= W <= 1, f"振幅应该在[0,1]范围内"

        print("✓ 不同res_index测试通过")

    def test_iqae_different_epsilon_values(self):
        """测试不同的epsilon值"""
        print("\n测试3 - 不同epsilon值测试:")

        epsilon_values = [0.1, 0.05, 0.01]

        for epsilon in epsilon_values:
            W = IQAE(
                operator_in=self.create_cir_basic,
                qnumber=2,
                epsilon=epsilon,
                res_index=-1
            ).run()

            print(f"epsilon={epsilon} 时的振幅: {W:.6f}")

            # 断言
            assert W is not None, f"epsilon={epsilon} 应该返回结果"
            assert isinstance(W, (float, np.floating)), "振幅应该是数值类型"
            assert 0 <= W <= 1, f"振幅应该在[0,1]范围内"

        print("✓ 不同epsilon值测试通过")

    def test_iqae_deterministic_states(self):
        """测试确定性状态的振幅估计"""
        print("\n=== 测试3: 确定性状态测试 ===")

        # 测试 |0⟩ 状态
        W_0 = IQAE(
            operator_in=self.create_cir_deterministic_0,
            qnumber=1,
            epsilon=0.01,
            res_index=-1
        ).run()

        print(f"确定性 |0⟩ 状态振幅: {W_0:.6f}")
        print(f"理论值: 0.000000")

        # 对于确定性|0⟩状态，振幅应接近0
        assert abs(W_0 - 0.0) < 0.1, f"确定性|0⟩状态的振幅应接近0，实际为{W_0}"

        # 测试 |1⟩ 状态
        W_1 = IQAE(
            operator_in=self.create_cir_deterministic_1,
            qnumber=1,
            epsilon=0.01,
            res_index=-1
        ).run()

        print(f"确定性 |1⟩ 状态振幅: {W_1:.6f}")
        print(f"理论值: 1.000000")

        # 对于确定性|1⟩状态，振幅应接近1
        assert abs(W_1 - 1.0) < 0.1, f"确定性|1⟩状态的振幅应接近1，实际为{W_1}"

        print("✓ 确定性状态测试通过")

    def test_iqae_different_qubit_numbers_single(self):
        """测试单量子比特"""
        print("\n测试4 - 单量子比特测试:")

        qubit_cases = [
            (1, -1, "单量子比特"),
        ]

        for qnumber, res_index, description in qubit_cases:
            W = IQAE(
                operator_in=self.create_cir_simple,
                qnumber=qnumber,
                epsilon=0.1,
                res_index=res_index
            ).run()

            print(f"{description} (qnumber={qnumber}, res_index={res_index}) 振幅: {W:.6f}")

            # 断言
            assert W is not None, f"{description} 应该返回结果"
            assert isinstance(W, (float, np.floating)), "振幅应该是数值类型"
            assert 0 <= W <= 1, f"振幅应该在[0,1]范围内"

        print("✓ 不同量子比特数测试通过")

    def test_iqae_different_qubit_numbers(self):
        """测试不同的量子比特数"""
        print("\n测试4 - 不同量子比特数测试:")

        qubit_cases = [
            # (1, -1, "单量子比特"),
            (2, -1, "双量子比特-最后一个"),
            (2, 0, "双量子比特-第一个"),
            (3, -1, "三量子比特-最后一个"),
            (3, 1, "三量子比特-中间"),
        ]

        for qnumber, res_index, description in qubit_cases:
            W = IQAE(
                operator_in=self.create_cir_complex,
                qnumber=qnumber,
                epsilon=0.1,
                res_index=res_index
            ).run()

            print(f"{description} (qnumber={qnumber}, res_index={res_index}) 振幅: {W:.6f}")

            # 断言
            assert W is not None, f"{description} 应该返回结果"
            assert isinstance(W, (float, np.floating)), "振幅应该是数值类型"
            assert 0 <= W <= 1, f"振幅应该在[0,1]范围内"

        print("✓ 不同量子比特数测试通过")

    def test_iqae_different_quantum_circuits(self):
        """测试不同的量子线路"""
        print("\n测试5 - 不同量子线路测试:")

        circuits = [
            (self.create_cir_simple, 1, -1, "简单线路"),
            (self.create_cir_basic, 2, -1, "基础线路"),
            (self.create_cir_multi_qubit, 3, -1, "多量子比特线路"),
            (self.create_cir_complex, 2, 0, "复杂线路-第一个比特"),
        ]

        for circuit_func, qnumber, res_index, desc in circuits:
            W = IQAE(
                operator_in=circuit_func,
                qnumber=qnumber,
                epsilon=0.1,
                res_index=res_index
            ).run()

            print(f"{desc}: 振幅 = {W:.6f}")

            # 断言
            assert W is not None, f"{desc} 应该返回结果"
            assert isinstance(W, (float, np.floating)), "振幅应该是数值类型"
            assert 0 <= W <= 1, f"振幅应该在[0,1]范围内"

        print("✓ 不同量子线路测试通过")

    def test_iqae_negative_index_behavior(self):
        """测试负索引的行为"""
        print("\n测试6 - 负索引行为测试:")

        # 测试各种负索引
        negative_indices = [-1, -2, -3]

        for res_index in negative_indices:
            try:
                W = IQAE(
                    operator_in=self.create_cir_multi_qubit,
                    qnumber=3,
                    epsilon=0.1,
                    res_index=res_index
                ).run()

                print(f"res_index={res_index} 的振幅: {W:.6f}")

                # 断言
                assert W is not None, f"res_index={res_index} 应该返回结果"
                assert isinstance(W, (float, np.floating)), "振幅应该是数值类型"
                assert 0 <= W <= 1, f"振幅应该在[0,1]范围内"

            except Exception as e:
                print(f"res_index={res_index} 出现异常: {e}")
                # 对于超出范围的索引，允许出现异常
                if res_index < -3:  # 对于3量子比特系统，-4就是超出范围
                    continue
                else:
                    raise e

        print("✓ 负索引行为测试通过")

    def test_iqae_amplitude_consistency(self):
        """测试振幅估计的一致性"""
        print("\n测试7 - 振幅估计一致性测试:")

        # 多次运行相同的IQAE，结果应该相近
        results = []
        num_runs = 3

        for i in range(num_runs):
            W = IQAE(
                operator_in=self.create_cir_basic,
                qnumber=2,
                epsilon=0.1,
                res_index=-1
            ).run()

            results.append(W)
            print(f"第{i + 1}次运行: {W:.6f}")

        # 检查结果的一致性（允许一定的误差）
        max_diff = max(results) - min(results)
        avg_amplitude = sum(results) / len(results)
        print(f"平均振幅: {avg_amplitude:.6f}")
        print(f"最大差异: {max_diff:.6f}")

        assert max_diff < 0.2, f"多次运行结果应该相近，最大差异为{max_diff}"

        print("✓ 振幅估计一致性测试通过")

    def test_iqae_edge_cases(self):
        """测试边界情况"""
        print("\n测试8 - 边界情况测试:")

        # 测试边界情况
        edge_cases = [
            (self.create_cir_simple, 1, 0, "单比特第一个"),
            (self.create_cir_basic, 2, -1, "双比特最后一个"),
            (self.create_cir_basic, 2, 1, "双比特第二个"),
        ]

        for circuit_func, qnumber, res_index, desc in edge_cases:
            W = IQAE(
                operator_in=circuit_func,
                qnumber=qnumber,
                epsilon=0.1,
                res_index=res_index
            ).run()

            print(f"{desc}: 振幅 = {W:.6f}")

            # 断言
            assert W is not None, f"{desc} 应该返回结果"
            assert isinstance(W, (float, np.floating)), "振幅应该是数值类型"
            assert 0 <= W <= 1, f"振幅应该在[0,1]范围内"

        print("✓ 边界情况测试通过")

    def test_iqae_parameter_validation(self):
        """测试参数验证"""
        print("\n测试9 - 参数验证测试:")

        # 测试res_index为int类型（这是接口的要求）
        W = IQAE(
            operator_in=self.create_cir_basic,
            qnumber=2,
            epsilon=0.1,
            res_index=-1  # 必须是int，不能是list
        ).run()

        assert W is not None, "参数正确时应该返回结果"
        assert isinstance(W, (float, np.floating)), "振幅应该是数值类型"
        print(f"参数验证测试通过: 振幅 = {W:.6f}")

    def test_iqae_comparison_with_different_indices(self):
        """测试不同索引的振幅比较"""
        print("\n测试10 - 不同索引振幅比较测试:")

        # 对于同一个线路，不同索引的振幅应该不同
        amplitudes = {}

        for res_index in [-1, 0, 1]:
            W = IQAE(
                operator_in=self.create_cir_basic,
                qnumber=2,
                epsilon=0.1,
                res_index=res_index
            ).run()

            amplitudes[res_index] = W
            print(f"res_index={res_index}: 振幅 = {W:.6f}")

        # 验证不同索引的振幅确实不同（允许很小的差异）
        unique_amplitudes = len(set(round(amp, 4) for amp in amplitudes.values()))
        assert unique_amplitudes > 1, "不同索引的振幅应该有差异"

        print("✓ 不同索引振幅比较测试通过")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
