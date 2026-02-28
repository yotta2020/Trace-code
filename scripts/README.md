# 文档自动同步系统

本目录包含 CLAUDE.md / AGENT.md / AGENT-ZH.md 自动同步脚本和 Git Hooks。

## 功能

1. **双向同步**: CLAUDE.md ↔ AGENT.md 保持内容一致
2. **自动翻译**: CLAUDE.md → AGENT-ZH.md (中文版本)

## 快速开始

### 1. 安装 Git Hooks

```bash
# 安装 post-commit hook
bash scripts/install_hooks.sh
```

安装后，每次提交包含 CLAUDE.md 或 AGENT.md 的更改时，会自动同步文档。

### 2. 手动同步

```bash
# 同步所有文档（使用最新的文件作为源）
python3 scripts/sync_docs.py

# 仅检查状态
python3 scripts/sync_docs.py --status

# 强制使用特定文件作为源
python3 scripts/sync_docs.py --source CLAUDE.md

# 跳过翻译
python3 scripts/sync_docs.py --no-translate
```

## 翻译配置

AGNENT-ZH.md 的翻译支持以下三种方式（按优先级）：

### 方案 1: Claude Code CLI

如果已安装 Claude Code CLI，将自动使用：

```bash
# 测试是否可用
claude --help
```

### 方案 2: Anthropic HTTP API

配置环境变量：

```bash
export ANTHROPIC_API_KEY="your-api-key"
export ANTHROPIC_API_URL="https://api.anthropic.com/v1/messages"
```

### 方案 3: 占位翻译

如果未配置翻译 API，脚本会生成带翻译提示的占位文件，可以稍后手动翻译。

## GitHub Actions

项目包含自动同步的 GitHub Actions workflow：

- 触发条件：推送包含 CLAUDE.md / AGENT.md / AGENT-ZH.md 的更改
- 操作：自动运行同步脚本并提交结果

如需禁用，删除 `.github/workflows/sync-agent-docs.yml` 即可。

## 文件说明

```
scripts/
├── sync_docs.py          # 主同步脚本
├── install_hooks.sh      # Git Hooks 安装脚本
└── post-commit           # Git post-commit hook 模板

.github/workflows/
└── sync-agent-docs.yml   # GitHub Actions workflow
```

## 故障排除

### Hook 不工作？

检查 hook 是否有执行权限：

```bash
chmod +x .git/hooks/post-commit
```

### 翻译失败？

检查是否配置了翻译 API：

```bash
python3 scripts/sync_docs.py --status
```

### 同步冲突？

如果同时修改了多个文件，手动解决冲突后运行：

```bash
python3 scripts/sync_docs.py --source CLAUDE.md
```
