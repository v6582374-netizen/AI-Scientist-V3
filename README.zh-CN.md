<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2：通过智能体树搜索实现</b><br>
    <b>Workshop 级别的自动化科学发现</b>
  </h1>
</div>

<p align="center">
  <a href="README.md">English README</a> |
  中文 README
</p>

<p align="center">
  <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[论文]</a> |
  <a href="https://sakana.ai/ai-scientist-first-publication/">[博客文章]</a> |
  <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment">[ICLR 2025 Workshop 实验]</a>
</p>

完全自主的科学研究系统正在快速发展，AI 正在深刻改变科学发现的方式。
我们很高兴介绍 The AI Scientist-v2：一个通用的端到端智能体系统。它生成了首篇完全由 AI 撰写并通过同行评审接收的 workshop 论文。

该系统能够自主生成假设、运行实验、分析数据，并撰写科学论文。与[前代 AI Scientist-v1](https://github.com/SakanaAI/AI-Scientist) 不同，AI Scientist-v2 不再依赖人工编写的模板，能够泛化到多个机器学习领域，并采用由实验管理智能体引导的渐进式智能体树搜索。

> **说明：**
> AI Scientist-v2 并不一定比 v1 生成更好的论文，尤其是在已有强起始模板的场景下。v1 遵循定义明确的模板，因此成功率更高；v2 则采用更宽泛、更探索式的方法，成功率相对较低。v1 更适合目标清晰、基础扎实的任务，而 v2 面向开放式科学探索。

> **警告：**
> 这个代码库会执行由大语言模型（LLM）编写的代码。这种自主性会带来多种风险和挑战，包括可能使用危险依赖包、不可控的网络访问，以及意外启动进程等。请务必在受控的沙箱环境中运行，例如 Docker 容器。请自行判断并承担使用风险。

## 目录

1. [环境要求](#环境要求)
   - [安装](#安装)
   - [支持的模型与 API Key](#支持的模型与-api-key)
2. [生成研究想法](#生成研究想法)
3. [运行 AI Scientist-v2 论文生成实验](#运行-ai-scientist-v2-论文生成实验)
4. [引用 The AI Scientist-v2](#引用-the-ai-scientist-v2)
5. [常见问题](#常见问题)
6. [致谢](#致谢)
7. [许可证与负责任使用](#许可证与负责任使用)

## 环境要求

本代码设计为在 Linux 环境中运行，并需要配备 NVIDIA GPU、CUDA 和 PyTorch。

### 安装

```bash
# 创建新的 conda 环境
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# 安装支持 CUDA 的 PyTorch（请根据你的环境调整 pytorch-cuda 版本）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 安装 PDF 和 LaTeX 工具
conda install anaconda::poppler
conda install conda-forge::chktex

# 安装 Python 依赖
pip install -r requirements.txt
```

安装通常不超过一小时。

### 支持的模型与 API Key

#### OpenAI 模型

默认情况下，系统会通过 `OPENAI_API_KEY` 环境变量调用 OpenAI 模型。

#### Gemini 模型

默认情况下，系统会通过 `GEMINI_API_KEY` 环境变量，使用 OpenAI API 兼容方式调用 Gemini 模型。

#### 通过 AWS Bedrock 使用 Claude 模型

如果需要使用 Amazon Bedrock 提供的 Claude 模型，请先安装额外依赖：

```bash
pip install anthropic[bedrock]
```

然后配置有效的 [AWS 凭证](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html)和目标 [AWS 区域](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html)，并设置以下环境变量：`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_REGION_NAME`。

#### Semantic Scholar API（文献检索）

代码可以选择性使用 Semantic Scholar API Key（`S2_API_KEY`）来提高文献检索吞吐量。[如果你有可用 Key](https://www.semanticscholar.org/product/api)，建议配置该变量。它会在构思阶段和论文写作阶段使用。

没有该 Key 时系统仍然可以运行，但你可能会遇到速率限制，或在构思阶段得到较弱的新颖性检查。如果 Semantic Scholar 访问出现问题，可以在论文生成时跳过引用阶段。

#### 设置 API Key

请为计划使用的模型提供必要的 API Key，并以环境变量方式配置。例如：

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export S2_API_KEY="YOUR_S2_KEY_HERE"

# 如果使用 Bedrock，请设置 AWS 凭证
# export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
# export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
# export AWS_REGION_NAME="your-aws-region"
```

## 生成研究想法

在运行完整的 AI Scientist-v2 实验流水线之前，需要先使用 `ai_scientist/perform_ideation_temp_free.py` 脚本生成潜在研究想法。该脚本会基于你提供的高层主题描述，让 LLM 进行头脑风暴和反思式改进，并可结合 Semantic Scholar 等工具检查新颖性。

1. **准备主题描述文件**

   创建一个 Markdown 文件，例如 `my_research_topic.md`，描述你希望 AI 探索的研究领域或主题。该文件应包含 `Title`、`Keywords`、`TL;DR` 和 `Abstract` 等部分，用于定义研究范围。

   你可以参考示例文件 `ai_scientist/ideas/i_cant_believe_its_not_better.md` 的结构与内容格式。请将你的文件放在脚本可以访问的位置，例如 `ai_scientist/ideas/` 目录。

2. **运行构思脚本**

   在项目根目录执行脚本，传入主题描述文件路径，并指定要使用的 LLM：

   ```bash
   python ai_scientist/perform_ideation_temp_free.py \
    --workshop-file "ai_scientist/ideas/my_research_topic.md" \
    --model gpt-5.5 \
    --max-num-generations 20 \
    --num-reflections 5
   ```

   参数说明：

   - `--workshop-file`：主题描述 Markdown 文件路径。
   - `--model`：用于生成研究想法的 LLM。请确保已经设置对应 API Key。
   - `--max-num-generations`：尝试生成的不同研究想法数量。
   - `--num-reflections`：每个想法的反思和改进轮数。

3. **查看输出**

   脚本会生成一个与输入 Markdown 同名的 JSON 文件，例如 `ai_scientist/ideas/my_research_topic.json`。该文件会包含结构化研究想法列表，包括假设、拟议实验和相关工作分析。

4. **进入实验阶段**

   得到包含研究想法的 JSON 文件后，即可进入下一节，运行主要实验流水线。

这个构思步骤会把 AI Scientist 引导到你关心的具体研究方向，并生成可在主实验流水线中测试的研究方案。

## 运行 AI Scientist-v2 论文生成实验

拿到上一节生成的 JSON 文件后，可以启动 AI Scientist-v2 主流水线。该流程会通过智能体树搜索运行实验、分析结果，并生成论文草稿。

写作阶段和评审阶段使用的模型可通过命令行参数指定。
最佳优先树搜索（BFTS）的配置位于 `bfts_config.yaml`。你可以根据需要调整其中参数。

`bfts_config.yaml` 中的重要树搜索配置包括：

- `agent` 配置：
  - `num_workers`：并行探索路径数量。
  - `steps`：最多探索的节点数量。例如，如果 `num_workers=3` 且 `steps=21`，树搜索最多会探索 21 个节点，并在每一步并行扩展 3 个节点。
  - `num_seeds`：当 `num_workers` 小于 3 时，通常应与 `num_workers` 相同；否则建议设为 3。
  - 说明：当前版本不使用 `k_fold_validation`、`expose_prediction` 和 `data_preview` 等其他 agent 参数。
- `search` 配置：
  - `max_debug_depth`：在放弃某条搜索路径前，agent 最多尝试调试失败节点的次数。
  - `debug_prob`：尝试调试失败节点的概率。
  - `num_drafts`：Stage 1 中初始根节点数量，也就是要增长的独立树数量。

下面示例使用生成的想法文件运行 AI Scientist-v2，例如 `my_research_topic.json`。请先查看 `bfts_config.yaml` 以了解详细树搜索参数。构思、BFTS 实验、绘图、写作、引用和评审阶段的默认模型均为 `gpt-5.5`。如果你不想用一段代码片段初始化实验，请不要设置 `load_code`。

```bash
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/my_research_topic.json" \
 --load_code \
 --add_dataset_ref \
 --model_writeup gpt-5.5 \
 --model_citation gpt-5.5 \
 --model_review gpt-5.5 \
 --model_agg_plots gpt-5.5 \
 --num_cite_rounds 20
```

初始实验阶段完成后，你会在 `experiments/` 目录下看到一个带时间戳的日志文件夹。进入该文件夹中的 `experiments/"timestamp_ideaname"/logs/0-run/`，可以找到树可视化文件 `unified_tree_viz.html`。

所有实验阶段完成后，系统会进入写作阶段。写作阶段通常总共需要 20 到 30 分钟。完成后，你应该能在 `timestamp_ideaname` 文件夹中看到 `timestamp_ideaname.pdf`。

对于上述示例运行，全部阶段通常会在数小时内完成。

## 引用 The AI Scientist-v2

如果你在研究中使用 **The AI Scientist-v2**，请按如下方式引用：

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## 常见问题

**为什么我的实验没有生成 PDF 或评审结果？**

AI Scientist-v2 完成实验的成功率取决于所选基础模型和想法复杂度。通常，在实验阶段使用 Claude 3.5 Sonnet 等能力较强的模型时，成功率会更高。

**单次实验的预估成本是多少？**

构思步骤的成本取决于所用 LLM 以及生成和反思轮数，但通常较低，大约几美元。对于主实验流水线，如果实验阶段使用 Claude 3.5 Sonnet，通常每次运行成本约为 15 到 20 美元。后续写作阶段如果使用示例命令中的默认模型，大约会额外增加 5 美元。建议将 GPT-4o 用于 `model_citation`，这有助于降低写作成本。

**如何让 The AI Scientist-v2 运行在不同学科领域？**

首先执行[生成研究想法](#生成研究想法)步骤。创建一个新的 Markdown 文件来描述目标学科或主题，并遵循示例文件 `ai_scientist/ideas/i_cant_believe_its_not_better.md` 的结构。然后使用该文件运行 `perform_ideation_temp_free.py` 脚本，生成对应的 JSON 想法文件。最后进入[运行 AI Scientist-v2 论文生成实验](#运行-ai-scientist-v2-论文生成实验)步骤，通过 `--load_ideas` 参数把该 JSON 文件传给 `launch_scientist_bfts.py`。

**如果无法访问 Semantic Scholar API，该怎么办？**

Semantic Scholar API 用于评估生成想法的新颖性，并在论文写作阶段收集引用。如果你没有 API Key，或遇到速率限制，可以考虑跳过这些阶段。

**遇到 "CUDA Out of Memory" 错误怎么办？**

该错误通常发生在 AI Scientist-v2 尝试加载或运行一个需要超过可用 GPU 显存的模型时。可尝试修改构思提示文件，例如 `ai_scientist/ideas/my_research_topic.md`，建议实验使用更小的模型。

## 致谢

`ai_scientist` 目录中的树搜索组件基于 [AIDE](https://github.com/WecoAI/aideml) 项目构建。感谢 AIDE 开发者的贡献，以及他们将项目开源。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist-v2&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist-v2&Date)

## 许可证与负责任使用

本项目采用 **The AI Scientist Source Code License** 许可，这是 Responsible AI License 的衍生版本。

**强制披露：** 使用本代码即表示你在法律上有义务，在任何由此产生的科学手稿或论文中清楚且醒目地披露 AI 的使用。

我们建议在论文的摘要或方法部分加入如下归因：

> "This manuscript was autonomously generated using [The AI Scientist](https://github.com/SakanaAI/AI-Scientist)."
