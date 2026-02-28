import sys
import os
from transformers import AutoConfig
import logging

# --- 配置 ---

# 设置日志级别，以便我们能看到 transformers 库的输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 这是从您的训练脚本（如 run.sh）中获取的模型路径
#
# MODEL_PATH = "../models/base/StarCoder-3B"
MODEL_PATH = "/home/nfs/share/backdoor2023/backdoor/base_model/StarCoder-3B-old"


# --- 检查脚本 ---

def check_starcoder_version(model_path: str):
    """
    加载模型配置并检查其 'model_type' 属性以确定版本。
    """
    # 检查路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"错误: 找不到模型路径 '{model_path}'。")
        logger.error("请确保您已将模型下载到该目录。")
        return

    try:
        # 1. 加载模型配置 (这只会读取 config.json，不会加载整个模型)
        logger.info(f"正在加载配置: {model_path}...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 2. 检查 'model_type' 属性
        model_type = getattr(config, 'model_type', None)
        
        logger.info(f"检测到 'model_type': {model_type}")

        # 3. 判断版本
        if model_type == 'starcoder2':
            print(f"\n✅ 结论: '{model_path}' 是 StarCoder 2。")
        elif model_type == 'starcoder':
            print(f"\n✅ 结论: '{model_path}' 是 StarCoder 1。")
        else:
            # 4. 如果 'model_type' 不明确，检查 'architectures' 作为后备
            logger.warning("'model_type' 不明确，正在检查 'architectures' 属性...")
            archs = getattr(config, 'architectures', [])
            if any("StarCoder2ForCausalLM" in arch for arch in archs):
                print(f"\n✅ 结论: '{model_path}' 是 StarCoder 2 (基于 'architectures' 判断)。")
            elif any("GPTBigCodeForCausalLM" in arch for arch in archs):
                print(f"\n✅ 结论: '{model_path}' 是 StarCoder 1 (基于 'architectures' 判断)。")
            else:
                print(f"\n❌ 无法确定: '{model_path}' 似乎不是 StarCoder 1 或 2。")
                print(f"   - model_type: {model_type}")
                print(f"   - architectures: {archs}")

    except OSError:
        logger.error(f"错误: 路径 '{model_path}' 不是一个有效的 Hugging Face 模型目录。")
        logger.error("请确保该目录包含 'config.json' 文件。")
    except Exception as e:
        logger.error(f"加载配置时发生意外错误: {e}")

if __name__ == "__main__":
    check_starcoder_version(MODEL_PATH)