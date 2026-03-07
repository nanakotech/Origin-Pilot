import pytest
import numpy as np
from pyqpanda_alg.QmRMR import Feature_Selection
import warnings
import os


class Test_QmRMR_all_code_Feature_Selection:

    def setup_method(self):
        warnings.filterwarnings("ignore")
        np.random.seed(42)
        os.makedirs('test_outputs', exist_ok=True)

    def generate_test_data(self, m=6):
        u = np.random.random(m)

        cor = np.random.random([m, m])
        cor = (cor.T + cor) / 2

        ini_par = np.random.random(int(m / 2) * m) * np.pi

        return u, cor, ini_par

    def test_interface13_basic_functionality(self):
        m = 6
        select_num = 3
        u, cor, ini_par = self.generate_test_data(m)

        try:
            feature_selector = Feature_Selection(
                quadratic=cor,
                linear=u,
                select_num=select_num
            )
            assert feature_selector is not None, "Feature_Selection实例创建失败"
            print("✓ Feature_Selection实例创建成功")

            loss, choice, dic = feature_selector.get_his_res(ini_par)

            assert loss is not None, "loss结果不应为None"
            assert choice is not None, "choice结果不应为None"
            assert dic is not None, "dic结果不应为None"
            assert isinstance(loss, (list, np.ndarray)), "loss应为列表或数组"
            assert len(loss) > 0, "loss应包含数据"
            assert np.all(np.isfinite(loss)), "loss应全部为有限值"
            assert isinstance(choice, (list, np.ndarray)), "choice应为列表或数组"
            assert len(choice) == m, f"choice长度应为特征数量{m}"
            assert np.all(np.isin(choice, [0, 1])), "choice应只包含0和1"

            selected_count = np.sum(choice)
            assert selected_count == select_num, f"选择特征数量应为{select_num}, 实际为{selected_count}"
            assert isinstance(dic, dict), "dic应为字典类型"
            assert len(dic) > 0, "dic应包含数据"

        except Exception as e:
            pytest.fail(f"基本功能测试失败: {e}")



if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])