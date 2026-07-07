"""Template-free idea generation entrypoint.

这个文件负责把一个“研究主题 Markdown”变成一组结构化 research idea JSON。

这里的“生成想法”没有隐藏的神秘算法：它本质上是一个受约束的 LLM
交互循环。脚本先把 LLM 设定成 AI 研究员，再要求它每轮只能输出一种
ACTION 和对应 ARGUMENTS。ACTION 要么调用一个外部工具（目前主要是
Semantic Scholar 文献检索），要么用 FinalizeIdea 提交最终 idea JSON。

关键耦合点：
- ai_scientist.llm.create_client / get_response_from_llm 负责适配不同 LLM 后端。
- ai_scientist.tools.semantic_scholar.SemanticScholarSearchTool 提供文献检索。
- BaseTool 是工具分发的统一接口；FinalizeIdea 是脚本内部约定的“伪工具”。
- 输出的 idea JSON 是后续 launch_scientist_bfts.py 通过 --load_ideas 读取的输入。
"""

import argparse
import json
import os.path as osp
import re
import traceback
from typing import Any, Dict, List

import sys

# 允许直接从仓库根目录执行本脚本时仍能导入 ai_scientist 包。
sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    create_client,
    get_response_from_llm,
)
from ai_scientist.model_defaults import DEFAULT_MODEL
from ai_scientist.research_profile.budgets import valid_budget_profile_ids
from ai_scientist.research_profile.domains import valid_domain_ids
from ai_scientist.research_profile.execution_backends import (
    valid_execution_backend_ids,
)
from ai_scientist.research_profile.planner import plan_research_profile
from ai_scientist.research_profile.prompting import build_ideation_system_prompt
from ai_scientist.research_profile.schema import (
    make_idea_envelope,
    validate_idea_envelope,
    validate_research_profile,
)

from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.base_tool import BaseTool

# 实例化真实工具。这个工具会访问 Semantic Scholar API，把论文标题、作者、
# venue、年份、引用量和摘要整理成文本，供下一轮 LLM 反思使用。
semantic_scholar_tool = SemanticScholarSearchTool()

# LLM 在本脚本里不能随意“调用函数”，只能从这里列出的动作里选择。
# SearchSemanticScholar 是真实 BaseTool；FinalizeIdea 是脚本约定的终止动作，
# 用来告诉程序“这个 idea 已经可以写入 JSON 了”。
tools = [
    semantic_scholar_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.""",
    },
]

# 只把真实 BaseTool 放入分发表。FinalizeIdea 不在这里，因为它不需要外部
# API 调用，而是在 generate_temp_free_idea 中直接解析并保存。
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# 把工具说明拼进 system prompt。这样 LLM 知道自己有哪些动作可选，以及每个
# 动作应该提交什么参数。
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# 工具名同时用于提示 LLM 和在解析失败时打印可选动作，避免提示词与校验逻辑
# 发生漂移。
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

def build_system_prompt(research_profile: Dict[str, Any]) -> str:
    # system_prompt 是整个 agent 循环的“协议层”：
    # 1. 规定 LLM 的角色和 idea 质量标准；
    # 2. 注入 Research Profile 中的领域、执行和证据约束；
    # 3. 告诉它可以使用哪些工具；
    # 4. 强制它返回 ACTION / ARGUMENTS，方便程序用正则和 JSON 自动解析。
    return build_ideation_system_prompt(
        research_profile,
        tool_descriptions,
        tool_names_str,
    )

# 第一轮 prompt 给 LLM 两类上下文：
# - workshop_description：用户提供的研究范围；
# - prev_ideas_string：已经生成过的 idea，帮助它避免重复。
idea_generation_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""

# 后续轮次不再重复完整 workshop 文本，而是让 LLM 基于同一段对话历史反思：
# idea 是否新颖、可行、清晰，以及上一轮工具调用返回了什么文献信息。
idea_reflection_prompt = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}
"""


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    research_profile: Dict[str, Any],
    max_num_generations: int = 20,
    num_reflections: int = 5,
    reload_ideas: bool = True,
) -> List[Dict]:
    """生成一批结构化研究想法，并写入 idea_fname 指向的 JSON 文件。

    工作方式可以理解为“外层生成多个 proposal，内层打磨单个 proposal”：
    - 外层循环最多尝试 max_num_generations 个独立 idea；
    - 内层循环给每个 idea 最多 num_reflections 次行动/反思机会；
    - 每一轮都让 LLM 在 SearchSemanticScholar 和 FinalizeIdea 之间选择；
    - 只有 FinalizeIdea 成功解析出的 idea 会被保存。

    参数里的 client/model 来自 ai_scientist.llm.create_client，因此本函数不关心
    底层是 OpenAI、Claude、Gemini 还是其他兼容后端。
    """
    research_profile = validate_research_profile(research_profile)
    system_prompt = build_system_prompt(research_profile)

    # archive 用字符串形式保存 idea，是为了能直接拼进 prompt，提醒 LLM 不要
    # 重复已有方向；最终写文件前再转回 dict。
    idea_str_archive = []

    # 如果已有 JSON 文件，先加载旧 idea。只接受 schema v2 envelope，避免旧
    # ML 默认数组格式悄悄进入 generalized pipeline。
    if reload_ideas and osp.exists(idea_fname):
        if osp.getsize(idea_fname) == 0:
            print(f"Idea file {idea_fname} is empty. Starting from scratch.")
        else:
            with open(idea_fname, "r") as f:
                idea_envelope = validate_idea_envelope(json.load(f))
                for idea in idea_envelope["ideas"]:
                    idea_str_archive.append(json.dumps(idea))
                print(f"Loaded {len(idea_str_archive)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    for gen_idx in range(max_num_generations):
        print()
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        try:
            # 每次准备生成新 proposal 时，把已有 idea 作为“不要重复”的上下文。
            prev_ideas_string = "\n\n".join(idea_str_archive)

            # last_tool_results 会把工具输出传给下一轮 reflection；msg_history 则让
            # 同一个 proposal 的多轮对话保持连续，但不同 proposal 之间互不污染。
            last_tool_results = ""
            idea_finalized = False
            msg_history = []

            for reflection_round in range(num_reflections):
                if reflection_round == 0:
                    # 第一轮要求 LLM 从 workshop 主题和历史 idea 出发，提出一个
                    # 新方向，通常也会先选择文献搜索动作。
                    prompt_text = idea_generation_prompt.format(
                        workshop_description=workshop_description,
                        prev_ideas_string=prev_ideas_string,
                    )
                else:
                    # 第二轮起进入“自我反思/工具反馈”模式：如果上一轮查了文献，
                    # 这里会把检索结果交回给 LLM，让它修正 novelty 和 related work。
                    prompt_text = idea_reflection_prompt.format(
                        current_round=reflection_round + 1,
                        num_reflections=num_reflections,
                        last_tool_results=last_tool_results or "No new results.",
                    )

                # get_response_from_llm 是与具体模型供应商的唯一耦合点。它返回
                # 本轮文本以及更新后的对话历史，后者会带入下一轮 reflection。
                response_text, msg_history = get_response_from_llm(
                    prompt=prompt_text,
                    client=client,
                    model=model,
                    system_message=system_prompt,
                    msg_history=msg_history,
                )

                # 下面开始执行“协议解析”：LLM 需要按 system_prompt 输出
                # ACTION 和 ARGUMENTS；如果连这两个字段都解析不出，本轮 proposal
                # 会被放弃。
                try:
                    # 用正则先切出动作名和参数文本，再用 json.loads 校验参数。
                    action_pattern = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
                    arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)"

                    action_match = re.search(
                        action_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                    arguments_match = re.search(
                        arguments_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )

                    if not all([action_match, arguments_match]):
                        raise ValueError("Failed to parse the LLM response.")

                    action = action_match.group(1).strip()
                    arguments_text = arguments_match.group(1).strip()
                    print(f"Action: {action}")
                    print(f"Arguments: {arguments_text}")

                    # LLM 有时会把 JSON 包在 ```json 代码块里；这里剥掉包裹层，
                    # 让后面的 json.loads 只处理纯 JSON。
                    if arguments_text.startswith("```json"):
                        arguments_text = re.search(
                            r"```json\s*(.*?)\s*```", arguments_text, re.DOTALL
                        ).group(1)

                    # 分支一：真实工具调用。目前主要是 SearchSemanticScholar。
                    # 工具结果不会立即保存，而是进入 last_tool_results，供下一轮
                    # reflection prompt 使用。
                    if action in tools_dict:
                        tool = tools_dict[action]
                        try:
                            arguments_json = json.loads(arguments_text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid arguments JSON for {action}.")

                        try:
                            # 参数名需要与工具的 use_tool 签名一致。例如
                            # SearchSemanticScholar 需要 {"query": "..."}。
                            result = tool.use_tool(**arguments_json)
                            last_tool_results = result
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeIdea":
                        # 分支二：LLM 认为 idea 已经足够成熟。这里要求它提交
                        # {"idea": {...}}，内部字段就是后续实验/写作流程依赖的契约。
                        try:
                            arguments_json = json.loads(arguments_text)
                            idea = arguments_json.get("idea")
                            if not idea:
                                raise ValueError("Missing 'idea' in arguments.")

                            # 保存到 archive 后，本轮 proposal 结束；下一轮外层循环
                            # 会把它加入 prev_ideas_string，帮助后续 idea 去重。
                            idea_str_archive.append(json.dumps(idea))
                            print(f"Proposal finalized: {idea}")
                            idea_finalized = True
                            break
                        except json.JSONDecodeError:
                            raise ValueError("Invalid arguments JSON for FinalizeIdea.")
                    else:
                        # 动作名不合法时不直接崩溃，而是打印可选动作。代码不会把
                        # 这个错误提示主动喂回模型，但对话历史里已保留模型原回复，
                        # 后续 reflection 仍可能回到合法动作。
                        print(
                            "Invalid action. Please specify one of the available tools."
                        )
                        print(f"Available actions are: {tool_names_str}")
                except Exception as e:
                    print(
                        f"Failed to parse LLM response. Response text:\n{response_text}"
                    )
                    traceback.print_exc()
                    break  # 当前 proposal 格式不可解析，退出它的反思循环。

            if idea_finalized:
                continue  # 成功得到一个 idea，继续尝试生成下一个 proposal。

        except Exception as e:
            # 外层兜底：单个 proposal 失败不影响整批生成，后面还能继续尝试。
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # 只把已 finalized 的 idea 写回 JSON。没完成 FinalizeIdea 的尝试会被丢弃。
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]
    if not ideas:
        raise RuntimeError(
            "No ideas were finalized. Check the earlier LLM/provider error or "
            "increase --num-reflections so the model has a chance to call FinalizeIdea."
        )

    with open(idea_fname, "w") as f:
        json.dump(make_idea_envelope(research_profile, ideas), f, indent=4)
    print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


if __name__ == "__main__":
    # 命令行入口：把 workshop Markdown、模型名和循环次数转成
    # generate_temp_free_idea 的参数。
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals - template free"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Model to use for AI Scientist. Supports listed models, qwen/<model>, "
            "and openai-compatible/<model>."
        ),
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        required=True,
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="auto",
        choices=["auto", *valid_domain_ids()],
        help="Research domain pack to use.",
    )
    parser.add_argument(
        "--execution-backend",
        type=str,
        default="auto",
        choices=["auto", *valid_execution_backend_ids()],
        help="Execution backend to use for downstream experiments.",
    )
    parser.add_argument(
        "--budget-profile",
        type=str,
        default="auto",
        choices=["auto", *valid_budget_profile_ids()],
        help="Static local resource budget profile.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    args = parser.parse_args()

    # 根据模型名创建具体客户端，并拿到 get_response_from_llm 实际使用的模型标识。
    client, client_model = create_client(args.model)

    # workshop 文件是 idea 生成的主题边界。README 中建议它包含 Title、
    # Keywords、TL;DR、Abstract 等信息，但代码本身只是把全文交给 LLM。
    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    research_profile = plan_research_profile(
        workshop_description,
        domain=args.domain,
        execution_backend=args.execution_backend,
        budget_profile=args.budget_profile,
    )
    print("Research Profile:")
    print(json.dumps(research_profile, indent=2))

    # 输出文件和输入 Markdown 同名，仅把 .md 替换成 .json。这个 JSON 会作为
    # 后续实验管线 launch_scientist_bfts.py --load_ideas 的输入。
    idea_fname = args.workshop_file.replace(".md", ".json")
    print("Starting idea generation for", idea_fname)
    ideas = generate_temp_free_idea(
        idea_fname=idea_fname,
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        research_profile=research_profile,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
    )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
