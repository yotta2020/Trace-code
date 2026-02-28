# tests/IST/test_for_while.py

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


class TestForWhileTransformations(unittest.TestCase):
    """
    测试 Python 'for_while' (11.1, 11.2, 11.3) 风格转换
    """

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，初始化 IST 实例"""
        try:
            cls.ist = IST(language="python")
        except Exception as e:
            raise unittest.SkipTest(f"初始化 IST 失败: {e}")

    # ==================== Python 11.2 测试 (for → while) ====================

    def test_python_11_2_basic_range(self):
        """测试 Python 11.2: 基础 range(n) 转换为 while"""
        style = "11.2"
        code = """for i in range(10):
    print(i)
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertNotEqual(code, transformed_code, "代码应被修改")

        self.assertIn("i = 0", transformed_code, "应包含初始化语句")
        self.assertIn("while i < 10:", transformed_code, "应包含 while 循环")
        self.assertIn("i += 1", transformed_code, "应包含更新语句")
        self.assertNotIn("for i in range", transformed_code, "不应包含 for 循环")

    def test_python_11_2_range_start_stop(self):
        """测试 Python 11.2: range(start, stop) 转换"""
        style = "11.2"
        code = """for i in range(5, 10):
    print(i)
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("i = 5", transformed_code, "应包含起始值 5")
        self.assertIn("while i < 10:", transformed_code, "应包含正确的条件")
        self.assertIn("i += 1", transformed_code, "应包含更新语句")

    def test_python_11_2_range_with_step(self):
        """测试 Python 11.2: range(start, stop, step) 转换"""
        style = "11.2"
        code = """for i in range(0, 10, 2):
    print(i)
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("i = 0", transformed_code, "应包含初始化")
        self.assertIn("while i < 10:", transformed_code, "应包含条件")
        self.assertIn("i += 2", transformed_code, "应包含步长 2 的更新")

    def test_python_11_2_negative_step(self):
        """测试 Python 11.2: 负步长的 range 转换"""
        style = "11.2"
        code = """for i in range(10, 0, -1):
    print(i)
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("i = 10", transformed_code, "应包含起始值 10")
        self.assertIn("while i > 0:", transformed_code, "负步长应使用 > 比较")
        self.assertIn("i += -1", transformed_code, "应包含负步长更新")

    def test_python_11_2_complex_body(self):
        """测试 Python 11.2: 复杂循环体的转换"""
        style = "11.2"
        code = """for i in range(5):
    x = i * 2
    print(x)
    if x > 5:
        break
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("while i < 5:", transformed_code, "应包含 while 循环")
        self.assertIn("x = i * 2", transformed_code, "应保留循环体内容")
        self.assertIn("if x > 5:", transformed_code, "应保留 if 语句")
        self.assertIn("i += 1", transformed_code, "更新语句应在循环体末尾")

    # ==================== Python 11.1 测试 (while → for) ====================

    def test_python_11_1_basic_while(self):
        """测试 Python 11.1: 基础 while 转换为 for"""
        style = "11.1"
        code = """i = 0
while i < 10:
    print(i)
    i += 1
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertNotEqual(code, transformed_code, "代码应被修改")

        self.assertNotIn("i = 0", transformed_code, "初始化语句应被删除")
        self.assertIn("for i in range(0, 10):", transformed_code, "应包含 for 循环")
        self.assertNotIn("i += 1", transformed_code, "更新语句应被删除")
        self.assertNotIn("while", transformed_code, "不应包含 while")

    def test_python_11_1_while_with_start(self):
        """测试 Python 11.1: 非零起始的 while 转换"""
        style = "11.1"
        code = """i = 5
while i < 10:
    print(i)
    i += 1
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("for i in range(5, 10):", transformed_code, "应包含正确的起始值")
        self.assertNotIn("while", transformed_code, "不应包含 while")

    def test_python_11_1_while_with_step(self):
        """测试 Python 11.1: 带步长的 while 转换"""
        style = "11.1"
        code = """i = 0
while i < 10:
    print(i)
    i += 2
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("for i in range(0, 10, 2):", transformed_code, "应包含步长参数")

    def test_python_11_1_while_less_equal(self):
        """测试 Python 11.1: <= 条件的 while 转换"""
        style = "11.1"
        code = """i = 0
while i <= 9:
    print(i)
    i += 1
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("for i in range(0, 10):", transformed_code, "应调整边界值")

    def test_python_11_1_while_greater(self):
        """测试 Python 11.1: > 条件的 while 转换 (倒序)"""
        style = "11.1"
        code = """i = 10
while i > 0:
    print(i)
    i += -1
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("for i in range(10, 0, -1):", transformed_code, "应包含负步长")

    def test_python_11_1_complex_body(self):
        """测试 Python 11.1: 复杂循环体的转换"""
        style = "11.1"
        code = """i = 0
while i < 5:
    x = i * 2
    print(x)
    i += 1
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        self.assertTrue(success, f"Python 样式 {style} 应返回 success=True")
        self.assertIn("for i in range(0, 5):", transformed_code, "应包含 for 循环")
        self.assertIn("x = i * 2", transformed_code, "应保留循环体")
        self.assertNotIn("i += 1", transformed_code, "更新语句应被删除")

    # ==================== Python 11.3 测试 (do-while 不支持) ====================

    def test_python_11_3_not_supported(self):
        """测试 Python 11.3: do-while 不支持"""
        style = "11.3"
        code = """i = 0
while i < 10:
    print(i)
    i += 1
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        # Python 不支持 do-while，应该返回失败或无变化
        self.assertFalse(success, "Python 不应支持 do-while 转换")

    # ==================== 负面测试用例 ====================

    def test_python_11_2_non_range_for(self):
        """测试 Python 11.2: 非 range 的 for 循环不应转换"""
        style = "11.2"
        code = """for item in [1, 2, 3]:
    print(item)
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        # 应该失败或保持不变
        if success:
            self.assertEqual(code, transformed_code, "非 range 循环不应被转换")
        else:
            self.assertFalse(success, "非 range 循环应返回失败")

    def test_python_11_2_enumerate_for(self):
        """测试 Python 11.2: enumerate 的 for 循环不应转换"""
        style = "11.2"
        code = """for i, val in enumerate([1, 2, 3]):
    print(i, val)
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        if success:
            self.assertEqual(code, transformed_code, "enumerate 循环不应被转换")
        else:
            self.assertFalse(success, "enumerate 循环应返回失败")

    def test_python_11_1_complex_condition(self):
        """测试 Python 11.1: 复杂条件的 while 不应转换"""
        style = "11.1"
        code = """i = 0
while i < 10 and x > 0:
    print(i)
    i += 1
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        # 复杂条件应该失败
        if success:
            self.assertEqual(code, transformed_code, "复杂条件不应被转换")
        else:
            self.assertFalse(success, "复杂条件应返回失败")

    def test_python_11_1_no_initialization(self):
        """测试 Python 11.1: 缺少初始化的 while 不应转换"""
        style = "11.1"
        code = """while i < 10:
    print(i)
    i += 1
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        if success:
            self.assertEqual(code, transformed_code, "缺少初始化不应被转换")
        else:
            self.assertFalse(success, "缺少初始化应返回失败")

    def test_python_11_1_no_update(self):
        """测试 Python 11.1: 缺少更新的 while 不应转换"""
        style = "11.1"
        code = """i = 0
while i < 10:
    print(i)
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        if success:
            self.assertEqual(code, transformed_code, "缺少更新语句不应被转换")
        else:
            self.assertFalse(success, "缺少更新语句应返回失败")

    # ==================== 集成测试 ====================

    def test_python_round_trip_11_1_then_11_2(self):
        """测试 Python: 11.1 然后 11.2 的往返转换"""
        original_code = """i = 0
while i < 10:
    print(i)
    i += 1
"""

        # 第一步: while → for
        step1_code, success1 = self.ist.transfer(styles=["11.1"], code=original_code)
        self.assertTrue(success1, "第一步转换应成功")
        self.assertIn("for i in range", step1_code, "第一步应生成 for 循环")

        # 第二步: for → while
        step2_code, success2 = self.ist.transfer(styles=["11.2"], code=step1_code)
        self.assertTrue(success2, "第二步转换应成功")
        self.assertIn("while i < 10:", step2_code, "第二步应生成 while 循环")

    def test_python_round_trip_11_2_then_11_1(self):
        """测试 Python: 11.2 然后 11.1 的往返转换"""
        original_code = """for i in range(0, 10):
    print(i)
"""

        # 第一步: for → while
        step1_code, success1 = self.ist.transfer(styles=["11.2"], code=original_code)
        self.assertTrue(success1, "第一步转换应成功")
        self.assertIn("while", step1_code, "第一步应生成 while 循环")

        # 第二步: while → for
        step2_code, success2 = self.ist.transfer(styles=["11.1"], code=step1_code)
        self.assertTrue(success2, "第二步转换应成功")
        self.assertIn("for i in range", step2_code, "第二步应生成 for 循环")

    def test_python_nested_loops(self):
        """测试 Python: 嵌套循环的转换"""
        style = "11.2"
        code = """for i in range(3):
    for j in range(5):
        print(i, j)
"""

        transformed_code, success = self.ist.transfer(styles=[style], code=code)

        # 应该能够转换（至少转换外层或内层循环）
        self.assertTrue(success, "嵌套循环转换应成功")
        self.assertIn("while", transformed_code, "应包含 while 循环")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)