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


class TestIdentifierNameTransformations(unittest.TestCase):
    """
    测试 'identifier_name' (0.1 - 0.6) 风格转换。
    """

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，初始化 IST 实例。"""
        try:
            cls.ist = IST(language="python")
        except Exception as e:
            raise unittest.SkipTest(f"初始化 IST 失败: {e}")

        # 我们使用 snake_case 作为转换的“基础”代码
        cls.base_code = """
def my_test_function(param_one, param_two):
    my_var = param_one + param_two
    total_sum = 0
    for i_val in range(my_var):
        total_sum += i_val
    return total_sum
"""
        # 预期会被转换的变量名
        cls.vars_to_check = ["param_one", "param_two", "my_var", "total_sum", "i_val"]

    def test_style_0_1_camel(self):
        """测试风格 0.1 (snake_case -> camelCase)"""
        style = "0.1"
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        self.assertTrue(success, f"样式 {style} (camel) 应返回 success=True")
        self.assertNotEqual(self.base_code, transformed_code, "代码应被修改")

        # 检查变量是否已转换
        self.assertIn("paramOne", transformed_code)
        self.assertIn("paramTwo", transformed_code)
        self.assertIn("myVar", transformed_code)
        self.assertIn("totalSum", transformed_code)
        self.assertIn("iVal", transformed_code)
        # 确保原始名称已不存在
        self.assertNotIn("param_one", transformed_code)
        self.assertNotIn("total_sum", transformed_code)

    def test_style_0_2_pascal(self):
        """测试风格 0.2 (snake_case -> PascalCase)"""
        style = "0.2"
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        self.assertTrue(success, f"样式 {style} (pascal) 应返回 success=True")
        self.assertNotEqual(self.base_code, transformed_code, "代码应被修改")

        self.assertIn("ParamOne", transformed_code)
        self.assertIn("TotalSum", transformed_code)
        self.assertIn("IVal", transformed_code)

    def test_style_0_3_snake(self):
        """测试风格 0.3 (camelCase -> snake_case)"""
        style = "0.3"
        # 使用一个 camelCase 版本的代码来测试转换
        camel_code = """
def myTestFunction(paramOne, paramTwo):
    myVar = paramOne + paramTwo
    return myVar
"""
        transformed_code, success = self.ist.transfer(styles=[style], code=camel_code)

        self.assertTrue(success, f"样式 {style} (snake) 应返回 success=True")
        self.assertNotEqual(camel_code, transformed_code, "代码应被修改")

        self.assertIn("param_one", transformed_code)
        self.assertIn("param_two", transformed_code)
        self.assertIn("my_var", transformed_code)
        self.assertNotIn("paramOne", transformed_code)

    def test_style_0_4_hungarian(self):
        """测试风格 0.4 (hungarian) - 在 Python 上预期失败"""
        style = "0.4"
        # 匈牙利命名法需要类型信息，而 Python 的实现 'find_type' 为空
        # (参见 transform_identifier_name.py, convert_hungarian)
        # 因此，此转换不应修改代码
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        self.assertFalse(success, f"样式 {style} (hungarian) 在 Python 上应返回 success=False")
        self.assertEqual(self.base_code, transformed_code, "代码不应被修改")

    def test_style_0_5_init_underscore(self):
        """测试风格 0.5 (init_underscore)"""
        style = "0.5"
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        self.assertTrue(success, f"样式 {style} (underscore) 应返回 success=True")
        self.assertNotEqual(self.base_code, transformed_code, "代码应被修改")

        # 检查是否所有变量（除了函数名）都被添加了前缀
        self.assertIn("_param_one", transformed_code)
        self.assertIn("_total_sum", transformed_code)
        self.assertIn("_i_val", transformed_code)
        self.assertNotIn(" param_one", transformed_code)  # 确保原始的还在
        self.assertIn("my_test_function", transformed_code)  # 函数名不应改变

    def test_style_0_6_init_dollar(self):
        """测试风格 0.6 (init_dollar)"""
        style = "0.6"
        transformed_code, success = self.ist.transfer(styles=[style], code=self.base_code)

        self.assertTrue(success, f"样式 {style} (dollar) 应返回 success=True")
        self.assertNotEqual(self.base_code, transformed_code, "代码应被修改")

        self.assertIn("$param_one", transformed_code)
        self.assertIn("$total_sum", transformed_code)
        self.assertIn("$i_val", transformed_code)
        self.assertIn("my_test_function", transformed_code)  # 函数名不应改变


if __name__ == "__main__":
    unittest.main()