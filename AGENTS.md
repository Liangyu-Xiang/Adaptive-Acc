# VGGT-omega Codex 协作与同步规则

本文件适用于所有在本仓库中工作的 Codex 实例。GitHub 仓库是两台服务器之间代码同步的唯一事实来源。

## 远端与分支

- `origin` 必须指向 `git@github.com:Liangyu-Xiang/VGGT-omega.git`。
- `upstream` 指向 `https://github.com/facebookresearch/vggt-omega.git`，仅用于获取上游更新，不得向其推送。
- 默认同步分支为 `main`。
- 开始修改前先执行 `git status -sb` 和 `git fetch origin`。工作区干净时，用 `git pull --rebase origin main` 获取另一台服务器的提交。
- 推送前再次执行 `git fetch origin`。若远端有新提交，先执行 `git rebase origin/main`，解决并验证冲突后再推送。
- 禁止使用 `git push --force`、`git reset --hard` 或其他可能覆盖另一台服务器工作成果的命令。

## 可以同步的内容

- 源代码与包文件。
- 可复用的训练、评测、推理和可视化脚本。
- 配置文件及其示例，但不得包含密钥、令牌或服务器专属凭据。
- README、设计说明、使用文档和许可证。
- 小型测试代码及测试配置。
- `.gitignore`、`AGENTS.md` 和其他开发工具配置。

## 禁止同步的内容

- 数据集及其缓存：`data/`、`dataset/`、`datasets/`。
- 预训练模型和 checkpoint：`pretrained_ckpts/`、`pretrained_models/`、`checkpoint/`、`checkpoints/`，以及 `*.pt`、`*.pth`、`*.ckpt`、`*.safetensors`、`*.onnx`。
- 实验产物：`outputs/`、`demo_outputs/`、`runs/`。
- 实验与服务日志：`log/`、`logs/`、`*.log`。
- 跟踪平台目录：`wandb/`、`wanlab/`、`swanlab/`。
- Python/构建缓存、虚拟环境、安装元数据和临时文件。
- 论文 PDF、新增视频等大型二进制文件，除非用户明确要求。
- `.env`、访问令牌、私钥、账号密码及任何其他秘密信息。

如果出现新的数据或产物目录，先把它加入 `.gitignore`，再进行暂存。不要仅依赖文件大小判断是否属于代码。

## 提交与推送流程

1. 查看 `git status -sb`，区分代码变更和本地产物。
2. 使用明确的文件路径执行 `git add`；工作区混合时禁止直接使用 `git add -A` 或 `git add .`。
3. 用 `git diff --cached --name-only` 和 `git diff --cached --stat` 核对暂存范围。
4. 执行 `git diff --cached --check`，并运行与改动相关的最小测试或语法检查。
5. 检查暂存文件中是否存在秘密信息、绝对数据路径或意外的大文件。
6. 创建内容明确、粒度合理的提交。
7. 获取并 rebase 最新的 `origin/main`，验证后执行 `git push origin main`。
8. 推送后确认本地 `main` 与 `origin/main` 一致且工作区没有意外未跟踪文件。

## 冲突与异常处理

- 同一文件在两台服务器上都被修改时，保留双方有效逻辑并运行验证；不得静默选择一方覆盖另一方。
- 发现已跟踪的数据、模型、日志、秘密或大文件时，停止推送并报告具体文件。
- GitHub 凭据由服务器本地的 SSH agent 或凭据管理器提供；不得把令牌写入仓库、命令参数或聊天内容。
- 除非用户明确要求，不得修改远端历史、删除远端分支或同步上游新版本。
