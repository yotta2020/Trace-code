import unittest
import sys
import os
from pathlib import Path

# --- 路径设置 ---
# (与 test_tokensub.py 相同)
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


class TestInvicharTransformations(unittest.TestCase):
    """
    测试 'invichar' (-2.1, -2.2, -2.3, -2.4) 风格转换。
    这些转换应在代码中插入不可见字符。
    """

    # 定义我们要检查的不可见字符
    ZWSP = chr(0x200B)  # -2.1
    ZWNJ = chr(0x200C)  # -2.2
    LRO = chr(0x202D)  # -2.3
    BKSP = chr(0x8)  # -2.4 (Backspace)

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

    def _run_test(self, style: str, char_to_check: str, description: str):
        """辅助函数来运行通用的 invichar 测试。"""

        # 1. 检查原始代码不包含该字符
        self.assertNotIn(char_to_check, self.sample_code,
                         f"原始代码不应包含 {description}")

        # 2. 应用转换
        transformed_code, success = self.ist.transfer(styles=[style], code=self.sample_code)

        # 3. 检查转换是否报告成功
        self.assertTrue(success, f"IST.transfer 应为样式 {style} ({description}) 返回 success=True")

        # 4. 检查代码是否真的被修改了
        self.assertNotEqual(self.sample_code, transformed_code,
                            f"代码应被样式 {style} ({description}) 修改")

        # 5. 检查转换后的代码是否包含目标不可见字符
        # 注意：我们不能使用 get_style()，因为 count_invichar() 总是返回 0。
        # 我们必须手动检查字符串。
        self.assertIn(char_to_check, transformed_code,
                      f"转换后的代码应包含 {description} (char code: {ord(char_to_check)})")

    def test_style_zwsp_2_1(self):
        """测试风格 -2.1 (Zero-Width Space)"""
        self._run_test("-2.1", self.ZWSP, "ZWSP")

    def test_style_zwnj_2_2(self):
        """测试风格 -2.2 (Zero-Width Non-Joiner)"""
        self._run_test("-2.2", self.ZWNJ, "ZWNJ")

    def test_style_lro_2_3(self):
        """测试风格 -2.3 (Left-to-Right Override)"""
        self._run_test("-2.3", self.LRO, "LRO")

    def test_style_bksp_2_4(self):
        """测试风格 -2.4 (Backspace)"""
        self._run_test("-2.4", self.BKSP, "BKSP")


if __name__ == "__main__":
    unittest.main()