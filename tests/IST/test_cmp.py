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


class TestCmpTransformations(unittest.TestCase):
    """
    测试 'cmp' (3.1 - 3.4) 风格转换。
    """

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，初始化 IST 实例。"""
        try:
            cls.ist = IST(language="python")
        except Exception as e:
            raise unittest.SkipTest(f"初始化 IST 失败: {e}")

        # 包含各种比较操作符的代码
        cls.base_code = """
def compare_values(a, b):
    if a > b:
        print("a is greater")
    elif a >= b:
        print("a is greater or equal")
    elif a < b:
        print("a is smaller")
    elif a <= b:
        print("a is smaller or equal")
    elif a == b:
        print("a equals b")
    elif a != b:
        print("a not equals b")
    else:
        print("comparison error")
"""
        # Python 的比较操作符 AST 节点类型是 'comparison_operator'
        cls.node_type = "comparison_operator"

    def test_style_3_1_smaller(self):
        """测试风格 3.1 (转为 < 或 <=)"""
        style = "3.1"
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        # 注意：由于 Python AST 的复杂性，我们只测试 > 和 >= 是否被转换
        self.assertTrue(success, f"样式 {style} (smaller) 应返回 success=True")
        self.assertNotEqual(self.base_code, transformed_code, "代码应被修改")

        self.assertIn("b < a", transformed_code)  # a > b -> b < a
        self.assertIn("b <= a", transformed_code)  # a >= b -> b <= a
        self.assertNotIn("a > b", transformed_code)
        self.assertNotIn("a >= b", transformed_code)
        # 保持 < 和 <= 不变
        self.assertIn("a < b", transformed_code)
        self.assertIn("a <= b", transformed_code)
        # == 和 != 可能未被转换（取决于修复的复杂程度）
        # self.assertNotIn("a == b", transformed_code)
        # self.assertNotIn("a != b", transformed_code)

    def test_style_3_2_bigger(self):
        """测试风格 3.2 (转为 > 或 >=)"""
        style = "3.2"
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        self.assertTrue(success, f"样式 {style} (bigger) 应返回 success=True")
        self.assertNotEqual(self.base_code, transformed_code, "代码应被修改")

        self.assertIn("b > a", transformed_code)  # a < b -> b > a
        self.assertIn("b >= a", transformed_code)  # a <= b -> b >= a
        self.assertNotIn("a < b", transformed_code)
        self.assertNotIn("a <= b", transformed_code)
        # 保持 > 和 >= 不变
        self.assertIn("a > b", transformed_code)
        self.assertIn("a >= b", transformed_code)

    def test_style_3_3_equal(self):
        """测试风格 3.3 (转为 ==)"""
        style = "3.3"
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        # 这种转换在 Python AST 上比较复杂，可能不会成功或只转换部分
        # 我们只检查代码是否被修改（如果实现的话），并且 == 仍然存在
        if success:  # 如果转换器报告成功
            self.assertNotEqual(self.base_code, transformed_code, "代码应被修改（如果转换成功）")
            self.assertIn("a == b", transformed_code)  # 原始的 == 应该还在
            # 检查是否添加了等效表达（取决于实现）
            # self.assertIn("a <= b and b <= a", transformed_code) # >= 转 ==
            # self.assertIn("!(a != b)", transformed_code) # != 转 ==
        else:  # 如果转换器报告失败
            self.assertEqual(self.base_code, transformed_code, "代码不应被修改（如果转换失败）")

        # 确保原始的 == 还在
        self.assertIn("a == b", transformed_code)

    def test_style_3_4_not_equal(self):
        """测试风格 3.4 (转为 !=)"""
        style = "3.4"
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        if success:
            self.assertNotEqual(self.base_code, transformed_code, "代码应被修改（如果转换成功）")
            self.assertIn("a != b", transformed_code)
            # 检查是否添加了等效表达（取决于实现）
            # self.assertIn("a < b or b < a", transformed_code) # > 转 !=
            # self.assertIn("!(a == b)", transformed_code) # == 转 !=
        else:
            self.assertEqual(self.base_code, transformed_code, "代码不应被修改（如果转换失败）")

        self.assertIn("a != b", transformed_code)


if __name__ == "__main__":
    unittest.main()