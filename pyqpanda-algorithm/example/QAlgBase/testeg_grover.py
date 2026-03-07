import pytest

from pyqpanda_alg.Grover import Grover
from pyqpanda3.core import CPUQVM, QCircuit, QProg, TOFFOLI, Z


def mark(qubits):
    """标记函数：标记 |11⟩ 状态"""
    cir = QCircuit()
    cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    cir << Z(qubits[2])
    cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    return cir


class Test_grover_cir:
    """测试Grover接口的cir方法"""

    def setup_method(self):
        """每个测试方法前的初始化"""
        self.qvm = CPUQVM()

    def test_cir_basic_functionality(self):
        """测试cir方法的基本功能"""
        print("\n=== 测试基本功能 ===")
        # 准备量子比特
        q_state = QProg(3).qubits()
        print(f"分配的量子比特: {q_state}")

        # 创建Grover实例
        demo_search = Grover(flip_operator=mark)

        # 调用cir接口
        circuit = demo_search.cir(
            q_input=q_state[:2],  # 前2个qubit作为搜索空间
            q_flip=q_state,  # 所有qubit用于标记
            q_zero=q_state[:2],  # 前2个qubit用于振幅放大
            iternum=1
        )
        print(f"电路内容: {circuit}")
        print(circuit.originir())
        assert circuit.originir() == """QINIT 3
CREG 1
H q[0]
H q[1]
CONTROL q[0],q[1]
X q[2]
ENDCONTROL
Z q[2]
CONTROL q[0],q[1]
X q[2]
ENDCONTROL
DAGGER
H q[1]
H q[0]
ENDDAGGER

X q[0]
X q[1]
Z q[1]
ENDCONTROL

X q[0]
X q[1]
H q[0]
H q[1]
"""
        # 验证返回类型
        assert isinstance(circuit, QCircuit), "cir方法应返回QCircuit对象"
        assert circuit is not None, "返回的电路不应为None"

        # 验证电路可以正常添加到程序中
        prog = QProg()
        prog << circuit

        assert isinstance(prog, QProg), "电路应能正常添加到量子程序中"
        assert len(str(prog)) > 0, "量子程序应有内容"

    def test_cir_different_iterations(self):
        """测试不同迭代次数下的cir方法"""
        print("\n=== 测试不同迭代次数 ===")

        q_state = QProg(3).qubits()
        demo_search = Grover(flip_operator=mark)

        # 测试不同迭代次数
        for iterations in [0, 1, 2, 3]:
            circuit = demo_search.cir(
                q_input=q_state[:2],
                q_flip=q_state,
                q_zero=q_state[:2],
                iternum=iterations
            )

            assert isinstance(circuit, QCircuit), f"迭代次数{iterations}时应返回QCircuit"
            assert circuit is not None, f"迭代次数{iterations}时电路不应为None"
            # 验证电路可以转换为字符串
            circuit_str = str(circuit)
            assert len(circuit_str) > 0, f"迭代次数{iterations}时电路字符串不应为空"
            print(f"迭代{iterations}的电路长度: {len(circuit_str)}")

    def test_cir_with_execution(self):
        """测试cir方法生成的电路可以正常执行"""
        print("\n=== 测试电路执行 ===")

        q_state = QProg(3).qubits()
        demo_search = Grover(flip_operator=mark)

        # 生成Grover搜索电路
        circuit = demo_search.cir(
            q_input=q_state[:2],
            q_flip=q_state,
            q_zero=q_state[:2],
            iternum=1
        )

        # 构建量子程序并执行
        prog = QProg()
        prog << circuit
        print(prog.originir())
        assert prog.originir() == """QINIT 3
CREG 1
H q[0]
H q[1]
CONTROL q[0],q[1]
X q[2]
ENDCONTROL
Z q[2]
CONTROL q[0],q[1]
X q[2]
ENDCONTROL
DAGGER
H q[1]
ENDDAGGER
DAGGER
H q[0]
ENDDAGGER
X q[0]
X q[1]
CONTROL q[0]
Z q[1]
ENDCONTROL
X q[0]
X q[1]
H q[0]
H q[1]
"""
        # 执行量子程序
        self.qvm.run(prog, shots=1000)
        result = self.qvm.result().get_prob_dict(q_state[:2])  # 只看前2个qubit的结果
        print(f"执行结果概率分布: {result}")

        # 验证结果
        assert result is not None, "概率字典不应为None"
        assert isinstance(result, dict), "结果应为字典类型"
        assert len(result) > 0, "结果字典不应为空"
        assert result['11'] >= 0.99

    def test_cir_parameter_validation(self):
        """测试cir方法的参数验证"""
        print("\n=== 测试参数验证 ===")

        q_state = QProg(3).qubits()
        demo_search = Grover(flip_operator=mark)

        # 测试缺少必要参数
        with pytest.raises(Exception) as exc_info:
            demo_search.cir()  # 缺少所有参数
            assert '量子比特索引必须是整数' in str(exc_info.value)
        print(f"缺少所有参数时的错误: {exc_info.value}")

    def test_cir_quantum_bit_requirements(self):
        """测试量子比特数量要求"""
        print("\n=== 测试量子比特数量要求 ===")

        demo_search = Grover(flip_operator=mark)

        # 测试量子比特数量不足的情况
        insufficient_qubits = QProg(2).qubits()  # 只有2个量子比特
        print(f"不足的量子比特: {insufficient_qubits}")

        with pytest.raises(Exception) as exc_info:
            demo_search.cir(
                q_input=insufficient_qubits[:2],
                q_flip=insufficient_qubits,  # 需要3个qubit但只有2个
                q_zero=insufficient_qubits[:2],
                iternum=1
            )
            assert 'list index out of range' in str(exc_info.value)
        print(f"量子比特不足时的错误: {exc_info.value}")

    def test_cir_circuit_structure(self):
        """测试生成的电路结构"""
        q_state = QProg(3).qubits()
        demo_search = Grover(flip_operator=mark)

        circuit = demo_search.cir(
            q_input=q_state[:2],
            q_flip=q_state,
            q_zero=q_state[:2],
            iternum=1
        )

        # 验证电路可以转换为字符串表示
        circuit_str = str(circuit)
        assert circuit_str is not None, "电路应有字符串表示"
        assert len(circuit_str) > 0, "电路字符串不应为空"

        originir = circuit.originir()
        print(originir)
        assert originir == """QINIT 3
CREG 1
H q[0]
H q[1]
CONTROL q[0],q[1]
X q[2]
ENDCONTROL
Z q[2]
CONTROL q[0],q[1]
X q[2]
ENDCONTROL
DAGGER
H q[1]
H q[0]
ENDDAGGER

X q[0]
X q[1]
Z q[1]
ENDCONTROL

X q[0]
X q[1]
H q[0]
H q[1]
"""

    @pytest.mark.parametrize("iterations,expected_success", [
        (0, True),  # 0次迭代应该也能工作
        (1, True),
        (2, True),
        (5, True),
    ])
    def test_cir_parameterized(self, iterations, expected_success):
        """参数化测试不同迭代次数"""
        print(f"\n=== 参数化测试: iterations={iterations} ===")

        q_state = QProg(3).qubits()
        demo_search = Grover(flip_operator=mark)

        if expected_success:
            circuit = demo_search.cir(
                q_input=q_state[:2],
                q_flip=q_state,
                q_zero=q_state[:2],
                iternum=iterations
            )
            assert isinstance(circuit, QCircuit)
            assert circuit is not None
            print(f"迭代{iterations}次成功生成电路")
        else:
            with pytest.raises(Exception) as exc_info:
                demo_search.cir(
                    q_input=q_state[:2],
                    q_flip=q_state,
                    q_zero=q_state[:2],
                    iternum=iterations
                )
            print(f"迭代{iterations}次预期失败: {exc_info.value}")

    def test_complete_grover_workflow(self):
        """测试完整的Grover搜索工作流程"""
        # 准备量子比特
        q_state = QProg(3).qubits()

        # 创建Grover实例
        demo_search = Grover(flip_operator=mark)

        # 生成搜索电路
        grover_circuit = demo_search.cir(
            q_input=q_state[:2],
            q_flip=q_state,
            q_zero=q_state[:2],
            iternum=1
        )

        # 构建完整的量子程序
        prog = QProg()
        prog << grover_circuit

        # 执行并获取结果
        self.qvm.run(prog, shots=1000)
        result = self.qvm.result().get_prob_dict(q_state[:2])

        # 验证结果的基本属性
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result.keys())
        assert all(isinstance(value, (int, float)) for value in result.values())

        print(f"完整工作流程结果: {result}")
        print(f"量子程序: {prog}")


if __name__ == "__main__":
    # 直接运行所有测试
    pytest.main([__file__, "-v"])
