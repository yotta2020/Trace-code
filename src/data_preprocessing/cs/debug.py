import sys
import os

# 确保能导入您的模块
# 假设脚本放在 src/data_preprocessing/cs/ 目录下
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from docstring_remover import DocstringRemover
except ImportError:
    # 如果运行在根目录或其他位置，尝试调整路径
    sys.path.append(os.path.join(current_dir, "src/data_preprocessing/cs"))
    from src.data_preprocessing.cs.docstring_remover import DocstringRemover

def verify_fix():
    code_sample = """
def my_func():
    \"\"\"
    这是一个Docstring。
    \"\"\"
    x = 1
    # print
    print("Code Body")
    return x #print
"""

    print("=== 正在验证修复效果 ===")
    print("源代码片段：")
    print(code_sample)
    print("-" * 30)

    try:
        remover = DocstringRemover('python')
        cleaned_code = remover.remove_docstrings(code_sample)
        
        print("处理后的代码：")
        print(cleaned_code)
        print("-" * 30)
        
        # 验证核心指标：正文是否还在
        if 'print("Code Body")' in cleaned_code:
            print("✅ 验证通过！代码正文被保留了。")
            if '这是一个Docstring' not in cleaned_code:
                print("✅ Docstring 也成功移除了。")
            else:
                print("❌ 警告：Docstring 没有被移除。")
        else:
            print("❌ 验证失败！代码正文丢失，Bug 仍然存在。")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    verify_fix()