import unittest
import sys
import os
from pathlib import Path

# --- 路径设置 ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    ist_module_path = os.path.join(project_root, 'src', 'data_preprocessing', 'IST')

    if not os.path.isdir(ist_module_path):
        raise ImportError(f"无法在以下位置找到 IST 模块: {ist_module_path}")

    sys.path.insert(0, ist_module_path)

    from transfer import StyleTransfer as IST

except ImportError as e:
    print(f"路径设置或导入错误: {e}")
    sys.exit(1)


# --- 路径设置结束 ---


class TestDeadcodeTransformations(unittest.TestCase):
    """
    测试 'deadcode' (-1.1 和 -1.2) 风格转换。
    这些转换应在函数体开头插入不可达的死代码。
    """

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，初始化 IST 实例和示例代码。"""
        try:
            cls.ist = IST(language="python")
        except Exception as e:
            raise unittest.SkipTest(f"初始化 IST 失败: {e}")

        # 使用与之前相同的示例代码
        cls.sample_code = """
def add(leftnum, rightnum):
    if leftnum > rightnum:
        return leftnum - rightnum
    sum = 0
    cnt = 0
    for i in range(leftnum, rightnum):
        sum += i
        cnt += 1
    diff = rightnum - leftnum
    return sum / cnt + diff
"""

    def test_style_deadcode_1_1(self):
        """测试风格 -1.1 (deadcode_test_message)"""
        style = "-1.1"
        target_string = "INFO Test message:aaaaa"

        # 1. 检查原始代码不包含该字符串
        self.assertNotIn(target_string, self.sample_code,
                         f"原始代码不应包含 '{target_string}'")

        # 2. 应用转换
        transformed_code, success = self.ist.transfer(styles=[style], code=self.sample_code)

        # 3. 检查转换是否报告成功
        self.assertTrue(success, f"IST.transfer 应为样式 {style} 返回 success=True")

        # 4. 检查代码是否真的被修改了
        self.assertNotEqual(self.sample_code, transformed_code,
                            f"代码应被样式 {style} 修改")

        # 5. 检查转换后的代码是否包含目标字符串
        self.assertIn(target_string, transformed_code,
                      f"转换后的代码应包含 '{target_string}'")

    def test_style_deadcode_1_2(self):
        """测试风格 -1.2 (deadcode_233) - (更新) 预期在 Python 上成功"""
        style = "-1.2"
        target_string = "233"  # 目标字符串

        # 1. 检查原始代码
        self.assertNotIn(target_string, self.sample_code, "原始代码不应包含 '233'")

        # 2. 应用转换
        transformed_code, success = self.ist.transfer(styles=[style], code=self.sample_code)

        # 3. (已修改) 检查转换是否报告成功
        self.assertTrue(success, f"IST.transfer 应为样式 {style} 在 Python 上返回 success=True")

        # 4. (已修改) 检查代码是否被修改
        self.assertNotEqual(self.sample_code, transformed_code,
                            f"代码应被样式 {style} 在 Python 上修改")

        # 5. (已修改) 检查目标字符串是否被插入
        self.assertIn(target_string, transformed_code,
                      f"转换后的代码应包含 '{target_string}'")


if __name__ == "__main__":
    unittest.main()