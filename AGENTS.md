# 仓库指南

## 项目结构与模块组织

Python 包位于 `src/handumi/`，按职责划分为 `calibration`、`dataset`、`retargeting`、`robots`、`real`、`sim` 和 `scripts` 等模块。`handumi` 和 `hu` 命令均通过 `handumi.scripts.cli:main` 进入。测试位于 `tests/`，其中的 `dataset`、`inpainting`、`robots` 和 `real` 等子目录对应功能领域。机器人模型和场景位于 `assets/`，示例配置和机器人配置位于 `configs/`，Sphinx 文档位于 `docs/`。

## 构建、测试与开发命令

- `bash install.sh`：配置工作站；仅使用 Meta Quest 时可加上 `--skip-xrt`。
- `uv sync`：根据 `uv.lock` 安装项目和开发依赖；要求 Python 3.12 或更高版本。
- `uv run handumi --help`：查看可用的命令行工作流程。
- `uv run pytest`：运行测试套件；可指定 `tests/dataset/` 等路径，只运行相关测试。
- `ruff check src tests`：安装 Ruff 后检查 Python 代码和导入顺序。
- 构建文档时，先在独立环境中安装 `docs/requirements.txt`，再运行 `PATH="$PWD/.venv-docs/bin:$PATH" make -C docs html`。环境配置和链接检查方法见 `docs/README.md`。

## 代码风格与命名规范

遵循相邻代码的风格，保持改动聚焦。Python 使用四空格缩进，模块和函数使用 `snake_case`，类使用 `PascalCase`。Ruff 配置位于 `pyproject.toml`，格式化目标行宽为 88 字符；现有代码没有强制行宽检查规则。公共接口应在有助于理解时提供清晰的文档字符串和类型标注。

编写代码时添加准确、简洁的标准中文注释，并遵循项目现有注释风格。Python 文件使用模块级 docstring，函数、方法和类使用中文 docstring 说明职责及必要的参数、返回值和副作用；C/C++ 文件使用 Doxygen 风格，文件头包含 `@file` 和 `@brief`，接口注释说明功能、参数、返回值和重要副作用。变量注释说明用途，避免重复代码含义；C/C++ 变量声明和定义均需添加中文注释。

## 测试指南

使用 pytest，测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。将测试放在对应功能领域的目录中，覆盖主要行为和失败场景。涉及外部服务或硬件时优先使用 fixture 和 mock；手动检查所需的硬件、可选依赖或工具应在说明中列出。项目未配置覆盖率门槛。提交前运行相关测试。

## 提交与拉取请求规范

近期提交采用 Conventional Commits 前缀，通常包含作用域，例如 `feat(visualize): add standalone Rerun sensor viewer` 或 `fix(screening): ...`。每次提交保持主题集中，摘要使用祈使语气。拉取请求应说明改动和动机，在有相关 issue 时提供链接，并列出已执行的测试和硬件验证。可见效果发生变化时，按需附上截图或录屏。

## 配置与安全

不要提交凭据和机器专用配置。`configs/calibration/table/local/` 下的本地桌面标定数据属于私有数据；使用提供的示例作为模板，且不要将本地测量结果包含在发布包中。
