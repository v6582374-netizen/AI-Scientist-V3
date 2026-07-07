"""Final plot aggregation stage.

这个文件负责把一次实验目录中的 baseline / research / ablation summary，
以及 `research_idea.md`，交给 LLM 生成一个可执行的绘图汇总脚本
`auto_plot_aggregator.py`。生成出来的脚本会读取已有 `.npy` 结果和 JSON
摘要，把最终论文要用的图保存到实验目录下的 `figures/`。

这里的“画图智能”不是内置绘图库逻辑，而是一个 LLM 写脚本、程序执行脚本、
再把执行结果反馈给 LLM 反思修正的闭环。后续 writeup 阶段只会看
`figures/` 里的图和 `auto_plot_aggregator.py`，因此本文件是实验结果到论文
图表之间的桥。

关键耦合点：
- perform_icbinb_writeup.load_idea_text / load_exp_summaries /
  filter_experiment_summaries 负责读取并裁剪实验摘要。
- ai_scientist.llm.create_client / get_response_from_llm 负责实际调用模型。
- 产物 `figures/*.png` 和 `auto_plot_aggregator.py` 会被 writeup 阶段读取。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from rich import print

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.model_defaults import DEFAULT_MODEL
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.perform_icbinb_writeup import (
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)

MAX_FIGURES = 12

# 这是给“绘图脚本生成器”的系统提示。它把 LLM 的任务限定为：
# 只基于已有实验数据和摘要写一个独立 Python 脚本，不凭空编数据，也不要把图
# 放到 `figures/` 之外的位置。
# 注意 MAX_FIGURES 只是 prompt 里的软约束；代码不会强制删除超额图片。
AGGREGATOR_SYSTEM_MSG = f"""You are an ambitious AI researcher who is preparing final plots for a scientific paper submission.
You have multiple experiment summaries (baseline, research, ablation), each possibly containing references to different plots or numerical insights.
There is also a top-level 'research_idea.md' file that outlines the overarching research direction.
Your job is to produce ONE Python script that fully aggregates and visualizes the final results for a comprehensive research paper.

Key points:
1) Combine or replicate relevant existing plotting code, referencing how data was originally generated (from code references) to ensure correctness.
2) Create a complete set of final scientific plots, stored in 'figures/' only (since only those are used in the final paper).
3) Make sure to use existing .npy data for analysis; do NOT hallucinate data. If single numeric results are needed, these may be copied from the JSON summaries.
4) Only create plots where the data is best presented as a figure and not as a table. E.g. don't use bar plots if the data is hard to visually compare.
5) The final aggregator script must be in triple backticks and stand alone so it can be dropped into a codebase and run.
6) If there are plots based on synthetic data, include them in the appendix.

Implement best practices:
- Do not produce extraneous or irrelevant plots.
- Maintain clarity, minimal but sufficient code.
- Demonstrate thoroughness for a final research paper submission.
- Do NOT reference non-existent files or images.
- Use the .npy files to get data for the plots and key numbers from the JSON summaries.
- Demarcate each individual plot, and put them in separate try-catch blocks so that the failure of one plot does not affect the others.
- Make sure to only create plots that are unique and needed for the final paper and appendix. A good number could be around {MAX_FIGURES} plots in total.
- Aim to aggregate multiple figures into one plot if suitable, i.e. if they are all related to the same topic. You can place up to 3 plots in one row.
- Provide well-labeled plots (axes, legends, titles) that highlight main findings. Use informative names everywhere, including in the legend for referencing them in the final paper. Make sure the legend is always visible.
- Make the plots look professional (if applicable, no top and right spines, dpi of 300, adequate ylim, etc.).
- Do not use labels with underscores, e.g. "loss_vs_epoch" should be "loss vs epoch".
- For image examples, select a few categories/classes to showcase the diversity of results instead of showing a single category/class. Some can be included in the main paper, while the rest can go in the appendix.

Your output should be the entire Python aggregator script in triple backticks.
"""


def build_aggregator_prompt(combined_summaries_str, idea_text):
    """把实验摘要和研究 idea 组装成让 LLM 写绘图脚本的用户 prompt。"""
    return f"""
We have three JSON summaries of scientific experiments: baseline, research, ablation.
They may contain lists of figure descriptions, code to generate the figures, and paths to the .npy files containing the numerical results.
Our goal is to produce final, publishable figures.

--- RESEARCH IDEA ---
```
{idea_text}
```

IMPORTANT:
- The aggregator script must load existing .npy experiment data from the "exp_results_npy_files" fields (ONLY using full and exact file paths in the summary JSONs) for thorough plotting.
- It should call os.makedirs("figures", exist_ok=True) before saving any plots.
- Aim for a balance of empirical results, ablations, and diverse, informative visuals in 'figures/' that comprehensively showcase the finalized research outcomes.
- If you need .npy paths from the summary, only copy those paths directly (rather than copying and parsing the entire summary).

Your generated Python script must:
1) Load or refer to relevant data and .npy files from these summaries. Use the full and exact file paths in the summary JSONs.
2) Synthesize or directly create final, scientifically meaningful plots for a final research paper (comprehensive and complete), referencing the original code if needed to see how the data was generated.
3) Carefully combine or replicate relevant existing plotting code to produce these final aggregated plots in 'figures/' only, since only those are used in the final paper.
4) Do not hallucinate data. Data must either be loaded from .npy files or copied from the JSON summaries.
5) The aggregator script must be fully self-contained, and place the final plots in 'figures/'.
6) This aggregator script should produce a comprehensive and final set of scientific plots for the final paper, reflecting all major findings from the experiment data.
7) Make sure that every plot is unique and not duplicated from the original plots. Delete any duplicate plots if necessary.
8) Each figure can have up to 3 subplots using fig, ax = plt.subplots(1, 3).
9) Use a font size larger than the default for plot labels and titles to ensure they are readable in the final PDF paper.


Below are the summaries in JSON:

{combined_summaries_str}

Respond with a Python script in triple backticks.
"""


def extract_code_snippet(text: str) -> str:
    """
    Look for a Python code block in triple backticks in the LLM response.
    Return only that code. If no code block is found, return the entire text.
    """
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[0].strip() if matches else text.strip()


def run_aggregator_script(
    aggregator_code, aggregator_script_path, base_folder, script_name
):
    """把 LLM 生成的 Python 代码落盘并在实验目录中执行。"""
    if not aggregator_code.strip():
        print("No aggregator code was provided. Skipping aggregator script run.")
        return ""

    # 这里写出的脚本是本阶段最重要的可追踪产物：后续 writeup 会把它读进 prompt，
    # 用来理解每张图是怎么由实验数据生成的。
    with open(aggregator_script_path, "w") as f:
        f.write(aggregator_code)

    print(
        f"Aggregator script written to '{aggregator_script_path}'. Attempting to run it..."
    )

    aggregator_out = ""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=base_folder,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        aggregator_out = result.stdout + "\n" + result.stderr
        print("Aggregator script ran successfully.")
    except subprocess.CalledProcessError as e:
        aggregator_out = (e.stdout or "") + "\n" + (e.stderr or "")
        print("Error: aggregator script returned a non-zero exit code.")
        print(e)
    except Exception as e:
        aggregator_out = str(e)
        print("Error while running aggregator script.")
        print(e)

    return aggregator_out


def aggregate_plots(
    base_folder: str, model: str = DEFAULT_MODEL, n_reflections: int = 5
) -> None:
    """为一个实验目录生成最终论文图表。

    流程是：
    1. 清理旧的聚合脚本和 figures 目录；
    2. 读取 idea 与实验 summary，并裁剪出绘图需要的字段；
    3. 让 LLM 写 `auto_plot_aggregator.py`；
    4. 执行脚本生成 figures；
    5. 把脚本输出和图数量反馈给 LLM，最多反思修正 n_reflections 轮。
    """
    filename = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(base_folder, filename)
    figures_dir = os.path.join(base_folder, "figures")

    # 每次重新聚合图表时先清理旧产物，避免 writeup 阶段误读过期图片。
    if os.path.exists(aggregator_script_path):
        os.remove(aggregator_script_path)
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        print(f"Cleaned up previous figures directory")

    idea_text = load_idea_text(base_folder)
    exp_summaries = load_exp_summaries(base_folder)
    filtered_summaries_for_plot_agg = filter_experiment_summaries(
        exp_summaries, step_name="plot_aggregation"
    )
    # plot_aggregation 会保留 plot_code、plot_analyses、npy 路径等字段；这些是 LLM
    # 写汇总绘图脚本时真正需要的上下文。
    combined_summaries_str = json.dumps(filtered_summaries_for_plot_agg, indent=2)

    aggregator_prompt = build_aggregator_prompt(combined_summaries_str, idea_text)

    # 第一次 LLM 调用只负责产出完整 Python 脚本；脚本能否运行由下一步实际执行验证。
    client, model_name = create_client(model)
    response, msg_history = None, []
    try:
        response, msg_history = get_response_from_llm(
            prompt=aggregator_prompt,
            client=client,
            model=model_name,
            system_message=AGGREGATOR_SYSTEM_MSG,
            print_debug=False,
            msg_history=msg_history,
        )
    except Exception:
        traceback.print_exc()
        print("Failed to get aggregator script from LLM.")
        return

    aggregator_code = extract_code_snippet(response)
    if not aggregator_code.strip():
        print(
            "No Python code block was found in LLM response. Full response:\n", response
        )
        return

    # 先执行一次脚本，拿到 stdout/stderr；后面的反思轮会基于真实执行结果修脚本。
    aggregator_out = run_aggregator_script(
        aggregator_code, aggregator_script_path, base_folder, filename
    )

    # 多轮反思不是让 LLM “评价图好不好看”这么简单，而是把脚本运行结果、图数量、
    # 常见绘图要求重新喂给它，让它修正缺图、重复图、文件名/legend/label 等问题。
    for i in range(n_reflections):
        figure_count = 0
        if os.path.exists(figures_dir):
            figure_count = len(
                [
                    f
                    for f in os.listdir(figures_dir)
                    if os.path.isfile(os.path.join(figures_dir, f))
                ]
            )
        print(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        reflection_prompt = f"""We have run your aggregator script and it produced {figure_count} figure(s). The script's output is:
```
{aggregator_out}
```

Please criticize the current script for any flaws including but not limited to:
- Are these enough plots for a final paper submission? Don't create more than {MAX_FIGURES} plots.
- Have you made sure to both use key numbers and generate more detailed plots from .npy files?
- Does the figure title and legend have informative and descriptive names? These plots are the final versions, ensure there are no comments or other notes.
- Can you aggregate multiple plots into one figure if suitable?
- Do the labels have underscores? If so, replace them with spaces.
- Make sure that every plot is unique and not duplicated from the original plots.

If you believe you are done, simply say: "I am done". Otherwise, please provide an updated aggregator script in triple backticks."""

        print("[green]Reflection prompt:[/green] ", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG,
                print_debug=False,
                msg_history=msg_history,
            )

        except Exception:
            traceback.print_exc()
            print("Failed to get reflection from LLM.")
            return

        # 只有已经生成过至少一张图时才接受 “I am done”，避免空 figures 被误判成功。
        if figure_count > 0 and "I am done" in reflection_response:
            print("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code = extract_code_snippet(reflection_response)

        # 如果 LLM 给了新版脚本，就覆盖旧脚本并重新执行；否则保留当前结果。
        # 这里不会在每轮反思前清空 figures/，所以旧图可能残留并影响 figure_count；
        # 读结果时要把它看作“LLM 写脚本 + 轻量执行反馈”的软约束流程。
        if (
            aggregator_new_code.strip()
            and aggregator_new_code.strip() != aggregator_code.strip()
        ):
            aggregator_code = aggregator_new_code
            aggregator_out = run_aggregator_script(
                aggregator_code, aggregator_script_path, base_folder, filename
            )
        else:
            print(
                f"No new aggregator script was provided or it was identical. Reflection step {i+1} complete."
            )


def main():
    """命令行入口：对指定实验目录运行图表聚合阶段。"""
    parser = argparse.ArgumentParser(
        description="Generate and execute a final plot aggregation script with LLM assistance."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the experiment folder with summary JSON files.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Number of reflection steps to attempt (default: 5).",
    )
    args = parser.parse_args()
    aggregate_plots(
        base_folder=args.folder, model=args.model, n_reflections=args.reflections
    )


if __name__ == "__main__":
    main()
