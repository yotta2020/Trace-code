import unittest
import sys
import os
from pathlib import Path

# --- 路径设置 ---
# 此测试文件位于 .../tests/IST/test_tokensub.py
# 我们需要将 .../src/data_preprocessing/IST 添加到 Python 路径中
try:
    # 获取当前文件 (test_tokensub.py) 的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 向上导航三级到项目根目录 (shulin-li22/causalcode-defender/)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # 构建 IST 模块的路径
    ist_module_path = os.path.join(project_root, 'src', 'data_preprocessing', 'IST')

    # 检查路径是否存在
    if not os.path.isdir(ist_module_path):
        raise ImportError(f"无法在以下位置找到 IST 模块: {ist_module_path}")

    # 将 IST 模块路径添加到 sys.path
    sys.path.insert(0, ist_module_path)

    # 现在导入 StyleTransfer (IST)
    from transfer import StyleTransfer as IST

except ImportError as e:
    print(f"路径设置或导入错误: {e}")
    print("请确保测试文件位于 'tests/IST/' 目录中，并且 'src/data_preprocessing/IST' 路径正确。")
    sys.exit(1)


# --- 路径设置结束 ---


class TestTokenSubTransformations(unittest.TestCase):
    """
    测试 'tokensub' (-3.1 和 -3.2) 风格转换。
    这些转换应在变量名后附加 '_sh' 或 '_rb'。
    """

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，初始化 IST 实例和示例代码。"""
        try:
            cls.ist = IST(language="python")
        except Exception as e:
            raise unittest.SkipTest(f"初始化 IST 失败 (可能是 tree-sitter 库问题): {e}")

        # 使用 src/data_preprocessing/IST/test_cases/test.py 的内容作为测试代码
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
        cls.expected_vars = {"leftnum", "rightnum", "sum", "cnt", "i", "diff"}

    def test_style_tokensub_sh(self):
        """测试风格 -3.1 (tokensub_sh)"""
        style = "-3.1"

        # 1. 应用转换
        transformed_code, success = self.ist.transfer(styles=[style], code=self.sample_code)

        # 2. 检查转换是否报告成功
        self.assertTrue(success, f"IST.transfer 应为样式 {style} 返回 success=True")

        # 3. 检查代码是否真的被修改了
        self.assertNotEqual(self.sample_code, transformed_code, f"代码应被样式 {style} 修改")

        # 4. 使用 get_style 检查是否至少有一个 token 被替换
        # (根据 transform_tokensub.py, 它会随机选一个变量并替换所有实例)
        count_sh = self.ist.get_style(code=transformed_code, styles=[style])[style]
        self.assertGreater(count_sh, 0, "get_style 应该报告至少一个 '_sh' 转换")

        # 5. 检查原始代码中不应存在 _sh 后缀
        original_count_sh = self.ist.get_style(code=self.sample_code, styles=[style])[style]
        self.assertEqual(original_count_sh, 0, "原始代码不应包含 '_sh' 后缀")

        # 6. 确认至少一个预期变量被添加了后缀
        found = False
        for var in self.expected_vars:
            if f"{var}_sh" in transformed_code:
                found = True
                break
        self.assertTrue(found, "转换后的代码必须包含一个带有 '_sh' 后缀的预期变量")

    def test_style_tokensub_rb(self):
        """测试风格 -3.2 (tokensub_rb)"""
        style = "-3.2"

        # 1. 应用转换
        transformed_code, success = self.ist.transfer(styles=[style], code=self.sample_code)

        # 2. 检查转换是否报告成功
        self.assertTrue(success, f"IST.transfer 应为样式 {style} 返回 success=True")

        # 3. 检查代码是否真的被修改了
        self.assertNotEqual(self.sample_code, transformed_code, f"代码应被样式 {style} 修改")

        # 4. 使用 get_style 检查是否至少有一个 token 被替换
        count_rb = self.ist.get_style(code=transformed_code, styles=[style])[style]
        self.assertGreater(count_rb, 0, "get_style 应该报告至少一个 '_rb' 转换")

        # 5. 检查原始代码中不应存在 _rb 后缀
        original_count_rb = self.ist.get_style(code=self.sample_code, styles=[style])[style]
        self.assertEqual(original_count_rb, 0, "原始代码不应包含 '_rb' 后缀")

        # 6. 确认至少一个预期变量被添加了后缀
        found = False
        for var in self.expected_vars:
            if f"{var}_rb" in transformed_code:
                found = True
                break
        self.assertTrue(found, "转换后的代码必须包含一个带有 '_rb' 后缀的预期变量")


if __name__ == "__main__":
    unittest.main()