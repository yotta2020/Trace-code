#!/bin/bash
# 安装 Git Hooks 脚本
# 使用方法：bash scripts/install_hooks.sh

set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

echo "=== 安装 Git Hooks ==="

# 检查 .git 目录是否存在
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "错误：.git 目录不存在，请确认当前目录是 Git 仓库"
    exit 1
fi

# 创建 hooks 目录（如果不存在）
mkdir -p "$HOOKS_DIR"

# 复制 post-commit hook
if [ -f "$SCRIPTS_DIR/post-commit" ]; then
    cp "$SCRIPTS_DIR/post-commit" "$HOOKS_DIR/post-commit"
    chmod +x "$HOOKS_DIR/post-commit"
    echo "✓ 已安装 post-commit hook"
else
    echo "✗ 找不到 scripts/post-commit 文件"
    exit 1
fi

# 复制 pre-commit hook（用于修改文件前的检查）
if [ -f "$SCRIPTS_DIR/pre-commit" ]; then
    cp "$SCRIPTS_DIR/pre-commit" "$HOOKS_DIR/pre-commit"
    chmod +x "$HOOKS_DIR/pre-commit"
    echo "✓ 已安装 pre-commit hook"
fi

echo ""
echo "=== Git Hooks 安装完成 ==="
echo ""
echo "已安装的 Hooks:"
ls -la "$HOOKS_DIR"/*-commit 2>/dev/null || echo "（无）"
echo ""
echo "使用方法:"
echo "1. 修改 CLAUDE.md 或 AGENT.md 后提交"
echo "2. Hook 会自动同步文档并创建新提交"
echo ""
echo "手动同步文档:"
echo "  python3 scripts/sync_docs.py"
echo ""
echo "检查同步状态:"
echo "  python3 scripts/sync_docs.py --status"
