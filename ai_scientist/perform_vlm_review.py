"""VLM-based figure review utilities.

这个文件负责论文生成后的“看图审查”：从 PDF 中截出图像区域，收集 caption 和
正文里对该图的引用，然后把图片交给视觉语言模型（VLM）检查图本身、caption、
正文引用是否一致。它也提供重复图检测和“图是否值得放在正文”的选择性审查。

这里没有真正理解 PDF 版式的完整排版引擎；主要靠 PyMuPDF 的文本块坐标、
caption 正则和 VLM 的图像理解能力组成一个尽力而为的检查器。

关键耦合点：
- perform_llm_review.load_paper 负责把同一个 PDF 抽成文本，用于提取 abstract。
- ai_scientist.vlm.get_response_from_vlm / get_batch_responses_from_vlm 负责 VLM 调用。
- perform_icbinb_writeup.py 会在反思写作时调用这里的 caption/引用审查、
  重复图检测和正文图选择审查。
"""

import os
import hashlib
import pymupdf
import re
import base64
from ai_scientist.vlm import (
    get_response_from_vlm,
    get_batch_responses_from_vlm,
    extract_json_between_markers,
)

from ai_scientist.perform_llm_review import load_paper


def encode_image_to_base64(image_data):
    """把图片路径、bytes 或 PyMuPDF pixmap 数据转成 API 可用的 base64 字符串。"""
    if isinstance(image_data, str):
        with open(image_data, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_data, list):
        return base64.b64encode(image_data[0]).decode("utf-8")
    elif isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")


reviewer_system_prompt_base = (
    "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue."
    "Be critical and cautious in your decision."
)

# 下面三个 prompt 分别对应三种视觉审查粒度：
# - 图 + caption + 正文引用是否互相支持；
# - 在页数限制下，这张图是否值得留在正文；
# - 单独描述/评价一张图，供 writeup 阶段理解 figures/*.png。
img_cap_ref_review_prompt = """The abstract of the paper is:

{abstract}

You will be given an image via the vision API. As a careful scientist reviewer, your task is to:
  1. Examine the provided image closely.
  2. Describe in detail what the image shows in a scientific manner.
  3. Critically analyze whether the image content aligns with the given caption:

{caption}

  4. We also have references in the main text that mention the figure:

{main_text_figrefs}

You should:
  - Examine the figure in detail: conclude elements in figures (e.g. name of axis) and describe what information is shown (e.g. the line of loss decreases monotonically but plateaus after X epochs)
  - Suggest any potential improvements or issues in the figure itself (e.g., missing legend, unclear labeling, no meaningful conclusion, mismatch with what the caption claims).
  - Critique the caption: does it accurately describe the figure? Is it too long/short? Does it include a concise takeaway?
  - Review how well the main text references (figrefs) explain the figure: are they missing? Do they adequately describe the figure's content, context, or purpose?

Finally, respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```
In <JSON>, provide the review in JSON format with the following fields in the order:
- "Img_description": "<Describe the figure's contents here>"
- "Img_review": "<Your analysis of the figure itself, including any suggestions for improvement>"
- "Caption_review": "<Your assessment of how well the caption matches the figure and any suggestions>"
- "Figrefs_review": "<Your thoughts on whether the main text references adequately describe or integrate the figure>"

In <THOUGHT>, first, thoroughly reason through your observations, analysis of alignment, and any suggested improvements. It is okay to be very long.
Then provide your final structured output in <JSON>.
Make sure the JSON is valid and properly formatted, as it will be parsed automatically."""


img_cap_selection_prompt = """The abstract of the paper is:

{abstract}

You will be given an image via the vision API. As a careful scientist reviewer, your task is to:
  1. Examine the provided image closely.
  2. Describe in detail what the image shows in a scientific manner.
  3. Critically analyze whether the image content aligns with the given caption:

{caption}

  4. We also have references in the main text that mention the figure:

{main_text_figrefs}

  5. We have limited pages to present contents:

{reflection_page_info}

You should:
  - Examine the figure in detail: conclude elements in figures (e.g. name of axis) and describe what information is shown (e.g. the line of loss decreases monotonically but plateaus after X epochs)
  - Critique the caption: does it accurately describe the figure? Is it too long/short? Does it include a concise takeaway?
  - Review how well the main text references (figrefs) explain the figure: are they missing? Do they adequately describe the figure's content, context, or purpose?

After considering all of the above, you should carefully evaluate:
  - Given the current page limit, does this image and its relevant text add significant value to the paper's scientific argument?
  - Given the current page limit, is this image too sparse in information? Should it be combined with other figures in the main text?
  - Does this figure contain subfigures?
  - Is this figure not very informative? For example, some figures may show bars with very similar heights that are difficult to distinguish, or present data in a way that does not effectively communicate meaningful differences or patterns.

Finally, respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```
In <JSON>, provide the review in JSON format with the following fields in the order:
- "Img_description": "<Describe the figure's contents here>"
- "Img_review": "<Your analysis of the figure itself, including any suggestions for improvement>"
- "Caption_review": "<Your assessment of how well the caption matches the figure and any suggestions>"
- "Figrefs_review": "<Your thoughts on whether the main text references adequately describe or integrate the figure>"
- "Overall_comments": "<Your thoughts on whether this figure adds significant value to the paper. Should it be moved to the appendix or not?>"
- "Containing_sub_figures": "<Does this figure contain multiple sub-figures? Do you think the information in this figure is dense? If not, would you suggest combining it with other figures in the main text? If it contains subplots, are their sizes and positions nicely aligned? If not, describe the issues.>"
- "Informative_review": "<Is this figure informative? Does it effectively communicate meaningful differences or patterns? Or does it show data in a way that makes it difficult to distinguish differences (e.g. bars with very similar heights)?>"

In <THOUGHT>, first, thoroughly reason through your observations, analysis of alignment, and any suggested improvements. It is okay to be very long.
Then provide your final structured output in <JSON>.
Make sure the JSON is valid and properly formatted, as it will be parsed automatically."""

img_review_prompt = """

You will be given an image via the vision API. As a careful scientist reviewer, your task is to:
  1. Examine the provided image closely.
  2. Describe in detail what the image shows in a scientific manner.

You should:
  - Examine the figure in detail: conclude elements in figures (e.g. name of axis) and describe what information is shown (e.g. the line of loss decreases monotonically but plateaus after X epochs)
  - Suggest any potential improvements or issues in the figure itself (e.g., missing legend, unclear labeling, no meaningful conclusion, mismatch with what the caption claims).

Finally, respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```
In <JSON>, provide the review in JSON format with the following fields in the order:
- "Img_description": "<Describe the figure's contents here>"
- "Img_review": "<Your analysis of the figure itself, including any suggestions for improvement>"

In <THOUGHT>, first, thoroughly reason through your observations, analysis of alignment, and any suggested improvements. It is okay to be very long.
Then provide your final structured output in <JSON>.
Make sure the JSON is valid and properly formatted, as it will be parsed automatically."""


def extract_figure_screenshots(
    pdf_path,
    img_folder_path,
    num_pages=None,
    min_text_length=50,
    min_vertical_gap=30,
):
    """
    Extract screenshots for figure captions ("Figure X." or "Figure X:")
    and also gather text blocks (anywhere in the PDF) mentioning that
    exact figure with "Figure", "Fig.", or "Fig-ure" (including line breaks).
    Avoid partial matches, e.g. "Figure 11" doesn't match "Figure 1".

    重要：这里不是提取 PDF 内嵌图片对象，而是根据 caption 坐标截取其上方区域。
    caption 若写成 "Fig. 1"、跨文本块、在图侧边，或图跨页，都可能无法识别。
    """
    os.makedirs(img_folder_path, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    page_range = (
        range(len(doc)) if num_pages is None else range(min(num_pages, len(doc)))
    )

    # ---------- (A) EXTRACT ALL TEXT BLOCKS FROM THE DOCUMENT ----------
    # 先把整篇 PDF 的文本块和坐标读出来。后面既用坐标判断 caption 上方的图像
    # 截图范围，也用文本内容查找正文里提到该 Figure 的句段。
    text_blocks = []  # will hold dicts: { 'page': int, 'bbox': Rect, 'text': str }
    for page_num in page_range:
        page = doc[page_num]
        try:
            blocks = page.get_text("blocks")
            # blocks: [x0, y0, x1, y1, text, block_no, ...]
            for b in blocks:
                txt = b[4].strip()
                if txt:
                    bbox = pymupdf.Rect(b[0], b[1], b[2], b[3])
                    text_blocks.append({"page": page_num, "bbox": bbox, "text": txt})
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {e}")

    # ---------- (B) REGEX FOR FIGURE CAPTIONS  ----------
    # 这里只识别以 “Figure <编号>.” 或 “Figure <编号>:” 开头的 caption。
    # 如果模板使用 Fig.、图名跨块、或 caption 不在图下方，截图可能漏掉或截错。
    # Captures the figure label so we can reference it later (group name 'fig_label').
    # Example matches: "Figure 1:", "Figure (A).2.", "Figure A.1:"
    figure_caption_pattern = re.compile(
        r"^(?:Figure)\s+(?P<fig_label>"
        r"(?:\d+"  # "1", "11", ...
        r"|[A-Za-z]+\.\d+"  # "A.1", "S2.3"
        r"|\(\s*[A-Za-z]+\s*\)\.\d+"  # "(A).2"
        r")"
        r")(?:\.|:)",  # Must end with "." or ":"
        re.IGNORECASE,
    )

    # ---------- (C) DETECT SUB-FIGURE CAPTIONS (e.g. "(a)")  ----------
    subfigure_pattern = re.compile(r"\(\s*[a-zA-Z]\s*\)")

    def is_subfigure_caption(txt):
        return bool(subfigure_pattern.search(txt))

    # ---------- (D) MAIN ROUTINE: LOOP OVER PAGES AND CAPTIONS ----------
    result_pairs = []

    for page_num in page_range:
        page = doc[page_num]
        page_rect = page.rect

        # All text blocks for this page
        page_blocks = [b for b in text_blocks if b["page"] == page_num]
        # Sort top-to-bottom
        page_blocks.sort(key=lambda b: b["bbox"].y0)

        # ----- (D.1) Find figure captions -----
        for blk in page_blocks:
            caption_text = blk["text"]
            m = figure_caption_pattern.match(caption_text)
            if not m:
                continue  # not a figure caption

            fig_label = m.group("fig_label")  # e.g. "1", "A.1", "(A).2", etc.
            fig_x0, fig_y0, fig_x1, fig_y1 = blk["bbox"]

            # (a) Find a large text block above the caption (on the same page)
            # 简化假设：图通常就在 caption 上方，且与 caption 横向有明显重叠。
            # 这对常见论文有效，但对跨页图、caption 在侧边、复杂多栏排版并不稳。
            above_blocks = []
            for ab in page_blocks:
                if ab["bbox"].y1 < fig_y0:
                    # vertical gap
                    ab_height_gap = fig_y0 - ab["bbox"].y1
                    # horizontal overlap
                    overlap_x = min(fig_x1, ab["bbox"].x1) - max(fig_x0, ab["bbox"].x0)
                    width_min = min((fig_x1 - fig_x0), (ab["bbox"].x1 - ab["bbox"].x0))
                    horiz_overlap_ratio = (
                        overlap_x / float(width_min) if width_min > 0 else 0.0
                    )

                    if (
                        len(ab["text"]) >= min_text_length
                        and not is_subfigure_caption(ab["text"])
                        and ab_height_gap >= min_vertical_gap
                        and horiz_overlap_ratio > 0.3
                    ):
                        above_blocks.append(ab)

            # pick the block with the largest bottom edge
            if above_blocks:
                above_block = max(above_blocks, key=lambda b: b["bbox"].y1)
                clip_top = above_block["bbox"].y1
            else:
                clip_top = page_rect.y0

            clip_left = fig_x0
            clip_right = fig_x1
            clip_bottom = fig_y0

            # (b) Create figure screenshot
            if (clip_bottom > clip_top) and (clip_right > clip_left):
                clip_rect = pymupdf.Rect(clip_left, clip_top, clip_right, clip_bottom)
                pix = page.get_pixmap(clip=clip_rect, dpi=150)

                fig_label_escaped = re.escape(fig_label)
                # unique filename
                fig_hash = hashlib.md5(
                    f"figure_{fig_label_escaped}_{page_num}_{clip_rect}".encode()
                ).hexdigest()[:10]
                fig_filename = (
                    f"figure_{fig_label_escaped}_Page_{page_num+1}_{fig_hash}.png"
                )
                fig_filepath = os.path.join(img_folder_path, fig_filename)
                pix.save(fig_filepath)

                # (c) Now find references across the ENTIRE DOCUMENT
                # 正文引用查找不理解语义，只按 Figure/Fig. + 同一编号做正则匹配；
                # negative lookahead 用来避免 Figure 1 误匹配 Figure 11。
                #     We'll build a pattern that matches:
                #         Figure/Fig./Fig-ure + possible line break + fig_label
                #     We also ensure we do NOT match if there's a digit/letter
                #     immediately after fig_label (so "Figure 11" won't match "Figure 1").
                fig_label_escaped = re.escape(fig_label)
                # negative lookahead (?![0-9A-Za-z]) ensures no letter/digit follows
                main_text_figure_pattern = re.compile(
                    rf"(?:Fig(?:\.|-\s*ure)?|Figure)\s*{fig_label_escaped}(?![0-9A-Za-z])",
                    re.IGNORECASE,
                )

                references_in_doc = []
                for tb in text_blocks:
                    # exclude the caption block itself
                    if tb is blk:
                        continue
                    # see if it references this figure label
                    if main_text_figure_pattern.search(tb["text"]):
                        references_in_doc.append(tb["text"])

                # (d) Create the final result item
                result_pairs.append(
                    {
                        "img_name": f"figure_{fig_label_escaped}",
                        "caption": caption_text,
                        "images": [fig_filepath],
                        "main_text_figrefs": references_in_doc,
                    }
                )

    return result_pairs


def extract_abstract(text):
    """从 Markdown-ish 论文文本里抽取 Abstract 小节。"""
    # Split text into lines
    lines = text.split("\n")

    # Regex to identify a heading line: starts with # after optional spaces
    # e.g. "### Some Heading"
    heading_pattern = re.compile(r"^\s*#+\s*(.*)$")

    # Find the line containing "abstract" in a heading
    abstract_start = None
    for i, line in enumerate(lines):
        # Check if this line is a heading
        match = heading_pattern.match(line)
        if match:
            # Extract the heading text after '#'
            heading_text = match.group(1)
            if "abstract" in heading_text.lower():
                abstract_start = i
                break

    if abstract_start is None:
        # No abstract heading found
        return ""

    # From abstract_start, collect lines until the next heading
    abstract_lines = []
    for j in range(abstract_start + 1, len(lines)):
        # Check if this line is another heading
        if heading_pattern.match(lines[j]):
            # We've hit the next section heading, stop extraction
            break
        # Otherwise, accumulate the line as part of the abstract
        abstract_lines.append(lines[j])

    # Join the abstract lines into a single string
    abstract_text = "\n".join(abstract_lines).strip()
    return abstract_text


def generate_vlm_img_cap_ref_review(img, abstract, model, client):
    """让 VLM 检查单张图、caption 和正文引用的一致性。"""
    prompt = img_cap_ref_review_prompt.format(
        abstract=abstract,
        caption=img["caption"],
        main_text_figrefs=img["main_text_figrefs"],
    )
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_cap_ref_review_json = extract_json_between_markers(content)
    return img_cap_ref_review_json


def generate_vlm_img_review(img, model, client):
    """只让 VLM 描述并评价一张图，不使用 caption 或正文引用。"""
    prompt = img_review_prompt
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_review_json = extract_json_between_markers(content)
    return img_review_json


def perform_imgs_cap_ref_review(client, client_model, pdf_path):
    """对 PDF 中所有可截取的 Figure 做图像/caption/正文引用审查。"""
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)
    img_reviews = {}
    abstract = extract_abstract(paper_txt)
    # 每张图独立调用一次 VLM。返回值以 figure label 为 key，方便写作反思阶段定位。
    for img in img_pairs:
        review = generate_vlm_img_cap_ref_review(img, abstract, client_model, client)
        img_reviews[img["img_name"]] = review
    return img_reviews


def detect_duplicate_figures(client, client_model, pdf_path):
    """把 PDF 中截出的所有图一次性发给 VLM，请它判断是否有重复或高度相似图。"""
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at identifying duplicate or highly similar images. "
                "Please analyze these images and determine if they are duplicates or variations of the same visualization. "
                "Response format: reasoning, followed by `Duplicate figures: <list of duplicate figure names>`."
                "Make sure you use the exact figure names (e.g. Figure 1, Figure 2b, etc.) as they appear in the paper."
                "If you find no duplicates, respond with `No duplicates found`."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Are any of these images duplicates or highly similar? If so, please identify which ones are similar and explain why. Focus on content similarity, not just visual style.",
                }
            ],
        },
    ]

    # 这里直接走 OpenAI-compatible chat.completions 图片消息格式，而不是
    # get_response_from_vlm 包装器，因为任务需要一次发送多张图做相互比较。
    # 注意：prompt 没有逐张附上 img_name 到图片的显式映射，所以模型给出的
    # “exact figure names” 是弱约束，适合提示人工/LLM 反思，不适合当硬判据。
    for img_info in img_pairs:
        messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(img_info['images'][0])}"
                },
            }
        )

    try:
        response = client.chat.completions.create(
            model=client_model,
            messages=messages,
            max_tokens=1000,
        )

        analysis = response.choices[0].message.content

        return analysis

    except Exception as e:
        print(f"Error analyzing images: {e}")
        return {"error": str(e)}


def generate_vlm_img_selection_review(
    img, abstract, model, client, reflection_page_info
):
    """在已有页数压力信息下，让 VLM 评价某张图是否值得保留在正文。"""
    prompt = img_cap_selection_prompt.format(
        abstract=abstract,
        caption=img["caption"],
        main_text_figrefs=img["main_text_figrefs"],
        reflection_page_info=reflection_page_info,
    )
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_cap_ref_review_json = extract_json_between_markers(content)
    return img_cap_ref_review_json


def perform_imgs_cap_ref_review_selection(
    client, client_model, pdf_path, reflection_page_info
):
    """对 PDF 中所有图做“正文/附录/删除/合并”倾向的视觉审查。"""
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)
    img_reviews = {}
    abstract = extract_abstract(paper_txt)
    for img in img_pairs:
        review = generate_vlm_img_selection_review(
            img, abstract, client_model, client, reflection_page_info
        )
        img_reviews[img["img_name"]] = review
    return img_reviews
