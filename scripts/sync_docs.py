#!/usr/bin/env python3
"""
CLAUDE.md / AGENT.md / AGENT-ZH.md 同步脚本

功能：
1. CLAUDE.md ↔ AGENT.md 双向同步（内容保持一致）
2. CLAUDE.md → AGENT-ZH.md 自动翻译（中文版本）

使用方法：
    python scripts/sync_docs.py [--source CLAUDE.md|AGENT.md] [--translate]
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
AGENT_MD = PROJECT_ROOT / "AGENT.md"
AGENT_ZH_MD = PROJECT_ROOT / "AGENT-ZH.md"


def get_file_hash(filepath: Path) -> str:
    """获取文件的 MD5 哈希值"""
    import hashlib
    if not filepath.exists():
        return ""
    content = filepath.read_text(encoding="utf-8")
    return hashlib.md5(content.encode()).hexdigest()


def get_file_mtime(filepath: Path) -> float:
    """获取文件修改时间"""
    if not filepath.exists():
        return 0
    return filepath.stat().st_mtime


def translate_with_claude_code(content: str) -> str:
    """
    使用 Claude Code CLI 进行翻译（如果可用）
    """
    try:
        # 尝试使用 claude-cli 进行翻译
        prompt = f"""请将以下技术文档从英文翻译成中文。
要求：
1. 保持 Markdown 格式不变（标题、代码块、列表等）
2. 翻译所有英文内容，但保留代码、命令和专有名词
3. 只输出翻译结果，不要额外解释

原文：
{content}"""

        result = subprocess.run(
            ["claude", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def translate_with_curl(content: str, api_url: str, api_key: str) -> str:
    """
    使用 HTTP API 进行翻译
    """
    import urllib.request
    import urllib.error

    prompt = f"Translate the following technical documentation from English to Chinese. Keep Markdown formatting, code blocks, and commands unchanged. Only output the translation.\n\n{content}"

    try:
        data = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096
        }).encode('utf-8')

        req = urllib.request.Request(
            api_url,
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
        )

        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('content', [{}])[0].get('text', content)
    except Exception as e:
        print(f"API 翻译失败：{e}")
        return None


def translate_content_with_llm(content: str) -> str:
    """
    使用 LLM 翻译内容
    按优先级尝试：
    1. Claude Code CLI
    2. HTTP API (如果配置了 ANTHROPIC_API_KEY)
    3. 本地缓存/占位翻译
    """
    # 方案 1: 尝试使用 Claude Code CLI
    print("  尝试使用 Claude Code CLI 翻译...")
    result = translate_with_claude_code(content)
    if result:
        return result

    # 方案 2: 尝试使用 HTTP API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    api_url = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
    if api_key:
        print("  尝试使用 Anthropic API 翻译...")
        result = translate_with_curl(content, api_url, api_key)
        if result:
            return result

    # 方案 3: 返回占位翻译（带翻译标记）
    print("  使用占位翻译（未检测到可用的翻译 API）")
    header = f"""# CLAUDE.md (中文翻译版)

> **注意**: 此文件由 CLAUDE.md 自动翻译生成
> 最后更新：{datetime.now().strftime('%Y-%m-%d')}
>
> 如需更新翻译，请运行：`python scripts/sync_docs.py --translate`
>
> 当前未配置翻译 API，以下为原文内容

---

"""
    return header + content


def sync_files(source: str = "newest", translate: bool = True):
    """
    同步三个文件

    Args:
        source: 源文件选择
            - "newest": 使用最新的文件作为源
            - "CLAUDE.md": 强制使用 CLAUDE.md 作为源
            - "AGENT.md": 强制使用 AGENT.md 作为源
        translate: 是否更新 AGENT-ZH.md
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始同步文档...")

    # 确定源文件
    if source == "newest":
        claude_mtime = get_file_mtime(CLAUDE_MD)
        agent_mtime = get_file_mtime(AGENT_MD)

        if claude_mtime >= agent_mtime:
            source_file = CLAUDE_MD
            source_name = "CLAUDE.md"
        else:
            source_file = AGENT_MD
            source_name = "AGENT.md"
    elif source == "CLAUDE.md":
        source_file = CLAUDE_MD
        source_name = "CLAUDE.md"
    else:
        source_file = AGENT_MD
        source_name = "AGENT.md"

    if not source_file.exists():
        print(f"错误：源文件 {source_file} 不存在")
        return False

    # 读取源文件内容
    content = source_file.read_text(encoding="utf-8")

    # 同步 CLAUDE.md 和 AGENT.md
    target_md = AGENT_MD if source_file == CLAUDE_MD else CLAUDE_MD

    # 检查是否需要更新
    if get_file_hash(source_file) != get_file_hash(target_md):
        target_md.write_text(content, encoding="utf-8")
        print(f"✓ 已同步 {source_name} -> {target_md.name}")
    else:
        print(f"- {target_md.name} 已是最新")

    # 翻译并更新 AGENT-ZH.md
    if translate:
        print("正在生成中文翻译...")
        zh_content = translate_content_with_llm(content)

        # 检查 AGENT-ZH.md 是否存在且内容不同
        current_zh = AGENT_ZH_MD.read_text(encoding="utf-8") if AGENT_ZH_MD.exists() else ""
        if current_zh != zh_content:
            AGENT_ZH_MD.write_text(zh_content, encoding="utf-8")
            print(f"✓ 已更新 {AGENT_ZH_MD.name}")
        else:
            print(f"- {AGENT_ZH_MD.name} 已是最新")

    print("同步完成!")
    return True


def check_status():
    """检查文件同步状态"""
    print("\n=== 文档同步状态 ===\n")

    files = [
        ("CLAUDE.md", CLAUDE_MD),
        ("AGENT.md", AGENT_MD),
        ("AGENT-ZH.md", AGENT_ZH_MD),
    ]

    hashes = {}
    for name, path in files:
        if path.exists():
            hashes[name] = get_file_hash(path)
            mtime = datetime.fromtimestamp(get_file_mtime(path))
            print(f"{name}: 存在 (修改时间：{mtime}, 哈希：{hashes[name][:8]})")
        else:
            hashes[name] = None
            print(f"{name}: 不存在")

    print("\n=== 一致性检查 ===")
    if hashes["CLAUDE.md"] and hashes["AGENT.md"]:
        if hashes["CLAUDE.md"] == hashes["AGENT.md"]:
            print("✓ CLAUDE.md 与 AGENT.md 内容一致")
        else:
            print("✗ CLAUDE.md 与 AGENT.md 内容不一致")
    else:
        print("- 无法检查一致性（文件缺失）")

    return hashes


def main():
    parser = argparse.ArgumentParser(description="同步 CLAUDE.md / AGENT.md / AGENT-ZH.md")
    parser.add_argument(
        "--source",
        choices=["newest", "CLAUDE.md", "AGENT.md"],
        default="newest",
        help="选择源文件（默认：使用最新的文件）"
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="跳过翻译步骤"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="仅检查状态，不同步"
    )

    args = parser.parse_args()

    if args.status:
        check_status()
    else:
        sync_files(
            source=args.source,
            translate=not args.no_translate
        )


if __name__ == "__main__":
    main()
