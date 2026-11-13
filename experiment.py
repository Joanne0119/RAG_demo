"""
切段實驗：一次比較多種切段設定，看哪一種檢索效果最好。

每組設定都會「在記憶體裡」重新切段、向量化、檢索，不會動到 chroma_db。
用的測試題和 evaluate.py 一樣（檢索只看一般題，陷阱題只在 --llm 時評分）。

執行：python experiment.py                        只比檢索命中率（快，不呼叫 LLM）
      python experiment.py --llm                  連 LLM 回答一起評分（慢很多，含陷阱題）
      python experiment.py --llm --only baseline,parent-100
                                                  只跑指定的設定（用逗號分隔）
      python experiment.py --data data_plain      改用其他資料夾的文件（例如沒有章節結構的版本）
      python experiment.py --data data_car --tests tests_car
                                                  換一組測試題（tests_car.py / tests_rp2040.py，PDF 用 download_manuals.py 下載）

結果會印成表格，並存一份 CSV 到 results 資料夾。
"""
import argparse
import bisect
import csv
import hashlib
import importlib
import json
import re
import time
from pathlib import Path

import numpy as np

import rag_core as rag
from evaluate import contains_all

# ==================== 要比較的設定（可自行增減） ====================
# strategy：
#   heading   先依「## 」章節切，太長再用滑動視窗切小（= 目前 rag_core 的做法）
#   length    不看章節，整份文件直接用滑動視窗硬切
#   paragraph 依空行切成段落，再把相鄰段落合併到接近 max_chars
#   recursive 遞迴切分：依序嘗試用 空行 → 換行 → 句號 → 逗號 切，切不夠小才換下一種，
#             再把相鄰小塊合併到接近 max_chars（類似 LangChain 的 RecursiveCharacterTextSplitter）
#   table     和 heading 一樣，但 Markdown 表格不從中間硬切：按列切，每段都重複表格標題和表頭
#             （PDF 要用 --tables 讀取才會有表格；沒有表格的文件結果和 heading 相同）
#   token     和 heading 一樣先依章節切，但長度用 token 算：max_chars、overlap 在這裡代表「幾個 token」
#             token 用 embedding 模型 bge-m3 自己的 tokenizer 計算（中文約 1.5 字 = 1 token，英文約 3.4 字元）
# add_title：段首要不要加上「【文件標題｜章節標題】」
# context：Contextual Retrieval。用 LLM 看這段和它前面的內容，寫一句「這段在講什麼」放在段首。
#          檢索是否命中只看原本的段落內容（LLM 寫的句子可能剛好含有答案，不能算進去）
# hybrid：混合檢索。向量檢索和 BM25 關鍵字檢索各排一次名，再用 RRF 合併（型號、專有名詞比較找得到）
# query_en：先用 LLM 把問題翻成英文再檢索（給英文文件用；回答時仍用原本的中文問題）
#   "keyword" 只有 BM25 用英文問題，向量檢索仍用原本的問題
#   "all"     向量檢索和 BM25 都用英文問題
# parent：小段檢索、大段回答。用小段比對問題，但交給 LLM 的是小段所在的整個「## 」章節
#         （只支援 strategy="heading"；top_k 代表最後取幾個不重複的章節）
CONFIGS = [
    # baseline = 最初的設定（600/100）；heading-200 = 目前 rag_core.py 的設定
    {"name": "baseline",        "strategy": "heading",   "max_chars": 600, "overlap": 100, "add_title": True,  "top_k": 3},
    {"name": "heading-300",     "strategy": "heading",   "max_chars": 300, "overlap": 50,  "add_title": True,  "top_k": 3},
    {"name": "heading-200",     "strategy": "heading",   "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3},
    {"name": "heading-100",     "strategy": "heading",   "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 3},
    {"name": "heading-200-ov0", "strategy": "heading",   "max_chars": 200, "overlap": 0,   "add_title": True,  "top_k": 3},
    {"name": "no-title",        "strategy": "heading",   "max_chars": 600, "overlap": 100, "add_title": False, "top_k": 3},
    {"name": "length-400",      "strategy": "length",    "max_chars": 400, "overlap": 100, "add_title": True,  "top_k": 3},
    {"name": "length-200",      "strategy": "length",    "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3},
    {"name": "paragraph-300",   "strategy": "paragraph", "max_chars": 300, "overlap": 50,  "add_title": True,  "top_k": 3},
    {"name": "baseline-k1",     "strategy": "heading",   "max_chars": 600, "overlap": 100, "add_title": True,  "top_k": 1},
    {"name": "heading-200-k5",  "strategy": "heading",   "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 5},
    {"name": "recursive-300",   "strategy": "recursive", "max_chars": 300, "overlap": 50,  "add_title": True,  "top_k": 3},
    {"name": "recursive-200",   "strategy": "recursive", "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3},
    {"name": "recursive-100",   "strategy": "recursive", "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 3},
    {"name": "parent-100",      "strategy": "heading",   "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 3, "parent": True},
    {"name": "parent-200",      "strategy": "heading",   "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3, "parent": True},
    {"name": "parent-100-k2",   "strategy": "heading",   "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 2, "parent": True},
    # 用 token 算長度（重疊 25%）
    {"name": "token-32",        "strategy": "token",     "max_chars": 32,  "overlap": 8,   "add_title": True,  "top_k": 3},
    {"name": "token-64",        "strategy": "token",     "max_chars": 64,  "overlap": 16,  "add_title": True,  "top_k": 3},
    {"name": "token-128",       "strategy": "token",     "max_chars": 128, "overlap": 32,  "add_title": True,  "top_k": 3},
    {"name": "token-256",       "strategy": "token",     "max_chars": 256, "overlap": 64,  "add_title": True,  "top_k": 3},
    {"name": "token-512",       "strategy": "token",     "max_chars": 512, "overlap": 128, "add_title": True,  "top_k": 3},
    # 混合檢索：向量 + BM25 關鍵字，用 RRF 合併排名。切法分別和 heading-100 / 200 / 300 / baseline 相同
    {"name": "hybrid-100",      "strategy": "heading",   "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 3, "hybrid": True},
    {"name": "hybrid-200",      "strategy": "heading",   "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3, "hybrid": True},
    {"name": "hybrid-300",      "strategy": "heading",   "max_chars": 300, "overlap": 50,  "add_title": True,  "top_k": 3, "hybrid": True},
    {"name": "hybrid-600",      "strategy": "heading",   "max_chars": 600, "overlap": 100, "add_title": True,  "top_k": 3, "hybrid": True},
    # 問題先翻成英文（給英文文件用）：hybrid-en = 混合檢索但 BM25 用英文問題；vector-en = 只用向量、問題換成英文
    {"name": "hybrid-en-100",   "strategy": "heading",   "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 3, "hybrid": True, "query_en": "keyword"},
    {"name": "hybrid-en-200",   "strategy": "heading",   "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3, "hybrid": True, "query_en": "keyword"},
    {"name": "hybrid-en-300",   "strategy": "heading",   "max_chars": 300, "overlap": 50,  "add_title": True,  "top_k": 3, "hybrid": True, "query_en": "keyword"},
    {"name": "hybrid-en-600",   "strategy": "heading",   "max_chars": 600, "overlap": 100, "add_title": True,  "top_k": 3, "hybrid": True, "query_en": "keyword"},
    {"name": "vector-en-100",   "strategy": "heading",   "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 3, "query_en": "all"},
    {"name": "vector-en-200",   "strategy": "heading",   "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3, "query_en": "all"},
    {"name": "vector-en-300",   "strategy": "heading",   "max_chars": 300, "overlap": 50,  "add_title": True,  "top_k": 3, "query_en": "all"},
    {"name": "vector-en-600",   "strategy": "heading",   "max_chars": 600, "overlap": 100, "add_title": True,  "top_k": 3, "query_en": "all"},
    # 表格不從中間硬切：按列切，每段都附上表格標題和表頭（要搭配 --tables 才有表格）
    {"name": "table-100",       "strategy": "table",     "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 3},
    {"name": "table-200",       "strategy": "table",     "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3},
    {"name": "table-300",       "strategy": "table",     "max_chars": 300, "overlap": 50,  "add_title": True,  "top_k": 3},
    {"name": "table-600",       "strategy": "table",     "max_chars": 600, "overlap": 100, "add_title": True,  "top_k": 3},
    # Contextual Retrieval：用 LLM 幫每段寫一句上下文放在段首（很慢：每段都要呼叫一次 LLM）
    {"name": "context-100",     "strategy": "heading",   "max_chars": 100, "overlap": 30,  "add_title": True,  "top_k": 3, "context": True},
    {"name": "context-200",     "strategy": "heading",   "max_chars": 200, "overlap": 50,  "add_title": True,  "top_k": 3, "context": True},
]
# Contextual Retrieval 的設定
CONTEXT_WINDOW = 1500  # 給 LLM 看這段前面多少字（3B 模型讀不了整份文件，只給附近的內容）
CONTEXT_PROMPT = """<document_excerpt>
{before}
</document_excerpt>
Here is the chunk that comes right after the excerpt above:
<chunk>
{chunk}
</chunk>
Write one short sentence that situates this chunk within the document, naming the component, module, register or section it describes, to improve search retrieval. Answer only with the sentence."""
CONTEXT_CACHE_DIR = rag.BASE_DIR / "cache"  # LLM 產生的上下文存在這裡，重跑時不用重新產生（.gitignore 已排除）
TRANSLATE_PROMPT = """Translate the following question into English for searching a technical document.
Keep part numbers, register names, units and numbers unchanged. Output only the English question.

{question}"""
TOKENIZER_NAME = "BAAI/bge-m3"  # 第一次使用會從 Hugging Face 下載（約 17 MB），之後用快取
RESULTS_DIR = rag.BASE_DIR / "results"
EMBED_BATCH = 64  # 一次送幾段去向量化（長篇 PDF 會切出上千段，一次全送太吃記憶體）
# ===================================================================


# ---------- 切段 ----------
def split_by_paragraph(text, max_chars, overlap):
    """依空行切段落，相鄰段落合併到不超過 max_chars；單一段落太長再用滑動視窗切。"""
    merged, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            merged.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        merged.append(current)
    return [piece for m in merged for piece in rag.split_by_length(m, max_chars, overlap)]


RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "，"]


def split_recursive(text, max_chars, overlap, separators=RECURSIVE_SEPARATORS):
    """遞迴切分：用第一個切得開的分隔符號切，太長的小塊換下一個分隔符號繼續切，最後合併相鄰小塊。"""
    if len(text) <= max_chars:
        return [text]
    sep = next((s for s in separators if s in text), None)
    if sep is None:  # 所有分隔符號都用完了，只好硬切
        return rag.split_by_length(text, max_chars, overlap)

    # 切開時把分隔符號留在前一塊的結尾，合併回去才不會掉字
    parts = [p + sep for p in text.split(sep)]
    parts[-1] = parts[-1][:-len(sep)]
    rest = separators[separators.index(sep) + 1:]
    pieces = [
        q for p in parts if p.strip()
        for q in (split_recursive(p, max_chars, overlap, rest) if len(p) > max_chars else [p])
    ]

    # 把相鄰的小塊合併到不超過 max_chars；開新段時，保留上一段結尾幾塊（不超過 overlap 字）當重疊
    chunks, current = [], []
    size = lambda blocks: sum(len(b) for b in blocks)
    for piece in pieces:
        if current and size(current) + len(piece) > max_chars:
            chunks.append("".join(current))
            while current and (size(current) > overlap or size(current) + len(piece) > max_chars):
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append("".join(current))
    return [c.strip() for c in chunks if c.strip()]


# ---------- 讀 PDF：辨識表格（--tables）、標出每段所屬的章節（--sections） ----------
def page_text_with_tables(page):
    """讀一頁：pymupdf 找到的表格轉成 Markdown 表格，其他文字照舊。
    純文字版的表格會變成「一行一個欄位」，表頭和數值分開；轉成 Markdown 後每一列都是完整的一行。
    文字區塊維持 PDF 原本的閱讀順序（不能依座標重排，兩欄排版的頁面左右欄會交錯），
    表格放在它的第一個文字區塊原本的位置。"""
    import pymupdf

    tables = page.find_tables().tables
    boxes = [pymupdf.Rect(t.bbox) for t in tables]
    markdowns = [t.to_markdown().replace("<br>", " ").replace("**", "").strip() for t in tables]  # 儲存格內換行改成空白
    parts, placed = [], set()
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        inside = next((k for k, box in enumerate(boxes) if center in box), None)
        if inside is None:
            parts.append(text.strip())
        elif inside not in placed:  # 表格裡的文字改由 Markdown 表格提供，只在第一次遇到時放一次
            parts.append(f"\n{markdowns[inside]}\n")
            placed.add(inside)
    parts += [f"\n{md}\n" for k, md in enumerate(markdowns) if k not in placed]  # 沒有文字區塊落在裡面的表格
    return "\n".join(p for p in parts if p)


NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)+)\.\s+\S")  # 「2.10.4. Brown-out Detection」這種帶編號的標題
RUNNING_HEADER_REPEAT = 3  # 出現這麼多次以上的標題行是頁首，只有第一次算數


def breadcrumb(line):
    """車主手冊每頁頁首的麵包屑「uu 安全帶u 關於安全帶」→「安全帶 > 關於安全帶」（u 是箭頭符號轉成文字後的樣子）。"""
    parts = [x.strip(" *") for x in re.split(r"u(?=[^a-z]|$)", line.strip()[2:])]
    return " > ".join(x for x in parts if x)


def section_marks(pages):
    """找出每個章節從原文的哪個位置開始，回傳 [(位置, 章節路徑), ...]（依位置排序）。兩種規則：
    1. 帶編號的標題（RP2040）：依編號層級組成路徑，例如「2.10. Core Supply Regulator > 2.10.4. Brown-out Detection」
    2. 頁首的麵包屑（車主手冊）：整頁套用「章名｜麵包屑」。新主題的第一頁沒有麵包屑，改用下一頁的。
    這些規則是看過這兩份 PDF 後寫的，換一份文件可能要改。"""
    lines_per_page = [p.split("\n") for p in pages]
    counts = {}
    for lines in lines_per_page:
        for l in lines:
            counts[l.strip()] = counts.get(l.strip(), 0) + 1
    crumbs = [next((breadcrumb(l) for l in lines if l.strip().startswith("uu")), None) for lines in lines_per_page]

    marks, stack, seen, offset = [], [], set(), 0
    for n, lines in enumerate(lines_per_page):
        crumb = crumbs[n] or next((c for c in crumbs[n + 1:n + 2] if c), None)
        if crumb:
            chapter = next((l.strip() for l in lines if l.strip()), "")
            chapter = chapter if len(chapter) <= 8 and not chapter.isdigit() else ""
            marks.append((offset, "｜".join(x for x in (chapter, crumb) if x)))
        pos = offset
        for l in lines:
            m = NUMBERED_HEADING.match(l.strip())
            title = l.strip()
            if m and len(title) < 80 and not (counts[title] >= RUNNING_HEADER_REPEAT and title in seen):
                seen.add(title)
                depth = m.group(1).count(".") + 1
                stack = [h for h in stack if h[0] < depth] + [(depth, title)]
                marks.append((pos, " > ".join(t for _, t in stack)))
            pos += len(l) + 1
        offset += len(pages[n]) + 1  # 頁和頁之間用 "\n" 接起來
    return marks


def load_documents_for_experiment(data_dir, tables=False, sections=False):
    """和 rag.load_documents 一樣，但 PDF 可以辨識表格（tables），也可以標出每段所屬的章節（sections，存在 doc["sections"]）。"""
    import pymupdf

    docs = rag.load_documents(data_dir)
    for doc in docs:
        if not doc["source"].lower().endswith(".pdf") or not (tables or sections):
            continue
        with pymupdf.open(Path(data_dir) / doc["source"]) as pdf:
            pages = [page_text_with_tables(page) if tables else page.get_text() for page in pdf]
        doc["text"] = "\n".join(pages)
        if sections:
            doc["sections"] = section_marks(pages)
    return docs


def section_at(doc, pos):
    """原文位置 pos 所屬的章節路徑，找不到回傳 None。"""
    marks = doc.get("sections") or []
    i = bisect.bisect_right([m[0] for m in marks], pos) - 1
    return marks[i][1] if i >= 0 else None


CAPTION_LINES = 3       # 表格標題最多取表格前幾行文字
CAPTION_MAX_REPEAT = 5  # 在文件中出現這麼多次以上的行視為頁首頁尾


def split_keeping_tables(text, max_chars, overlap):
    """一般文字照 split_by_length 切；Markdown 表格按列切，每段 = 表格標題 + 表頭 + 幾列，
    就算超過 max_chars 也至少放一列，確保每一列都和表頭在同一段。"""
    lines = text.split("\n")
    # 頁首頁尾（例如「RP2040 Datasheet」）每頁都出現，不能拿來當表格標題
    counts = {}
    for l in lines:
        counts[l.strip()] = counts.get(l.strip(), 0) + 1
    is_caption = lambda l: l.strip() and not l.strip().isdigit() and counts[l.strip()] < CAPTION_MAX_REPEAT
    pieces, buffer, i = [], [], 0
    flush = lambda: pieces.extend(rag.split_by_length("\n".join(buffer).strip(), max_chars, overlap)) if "".join(buffer).strip() else None
    while i < len(lines):
        if not lines[i].startswith("|"):
            buffer.append(lines[i])
            i += 1
            continue
        # 表格開始：表格標題 = 表格前最多 CAPTION_LINES 行文字（例如「Table 190. BOD」或所在章節名稱），跳過頁碼和頁首頁尾
        caption = " ".join(reversed([l.strip() for l in reversed(buffer) if is_caption(l)][:CAPTION_LINES]))
        flush()
        buffer = []
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append(lines[i])
            i += 1
        header, body = rows[:2], rows[2:]  # 前兩行是表頭和 |---| 分隔線
        prefix = "\n".join(([caption] if caption else []) + header)
        current = []
        for row in body:
            if current and len(prefix) + sum(len(r) + 1 for r in current) + len(row) > max_chars:
                pieces.append(prefix + "\n" + "\n".join(current))
                current = []
            current.append(row)
        pieces.append(prefix + ("\n" + "\n".join(current) if current else ""))
    flush()
    return [p for p in pieces if p.strip()]


_tokenizer = None


def split_by_tokens(text, max_tokens, overlap):
    """滑動視窗，但長度用 token 算：每段 max_tokens 個 token，相鄰兩段重疊 overlap 個 token。
    用 tokenizer 回傳的位置（offsets）切原文，所以段落內容和原文一字不差。"""
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    offsets = _tokenizer.encode(text, add_special_tokens=False).offsets
    if len(offsets) <= max_tokens:
        return [text]
    step = max_tokens - overlap
    pieces = []
    for i in range(0, len(offsets) - overlap, step):
        window = offsets[i:i + max_tokens]
        pieces.append(text[window[0][0]:window[-1][1]].strip())
    return [p for p in pieces if p]


def heading_at(text, piece):
    """找出 piece 在原文中位於哪個「## 」章節底下，找不到回傳 None。"""
    pos = text.find(piece[:50])
    start = text.rfind("\n## ", 0, pos + 3)  # pos + 3：piece 本身以「## 」開頭時也找得到
    if pos == -1 or start == -1:
        return None
    end = text.find("\n", start + 1)
    return text[start + 4:end if end != -1 else None].strip()


def make_chunks(doc, cfg):
    """依設定把一份文件切段，回傳 [{"text", "source", "parent"}, ...]
    parent 是這段所在的整個章節（給「小段檢索、大段回答」用），其他切法則是自己。"""
    title = rag.doc_title(doc)
    text = doc["text"]
    max_chars, overlap = cfg["max_chars"], cfg["overlap"]
    with_label = lambda label, s: f"【{label}】\n{s}" if cfg["add_title"] else s

    triples = []  # (段首標籤, 段落內容, 所屬章節)
    if cfg["strategy"] == "heading":
        for section in rag.split_by_heading(text):
            label = rag.section_label(title, section)
            triples += [(label, p, with_label(label, section)) for p in rag.split_by_length(section, max_chars, overlap)]
    elif cfg["strategy"] == "table":
        for section in rag.split_by_heading(text):
            label = rag.section_label(title, section)
            triples += [(label, p, None) for p in split_keeping_tables(section, max_chars, overlap)]
    elif cfg["strategy"] == "token":
        for section in rag.split_by_heading(text):
            label = rag.section_label(title, section)
            triples += [(label, p, None) for p in split_by_tokens(section, max_chars, overlap)]
    elif cfg["strategy"] == "length":
        triples = [(title, p, None) for p in rag.split_by_length(text, max_chars, overlap)]
    elif cfg["strategy"] == "paragraph":
        triples = [(title, p, None) for p in split_by_paragraph(text, max_chars, overlap)]
    elif cfg["strategy"] == "recursive":
        for p in split_recursive(text, max_chars, overlap):
            heading = heading_at(text, p)
            triples.append((f"{title}｜{heading}" if heading else title, p, None))
    else:
        raise ValueError(f"不認識的 strategy：{cfg['strategy']}")

    if cfg.get("parent") and cfg["strategy"] != "heading":
        raise ValueError("parent 只支援 strategy='heading'")

    chunks, search_from = [], 0
    for n, (label, piece, section) in enumerate(triples):
        chunk_text = with_label(label, piece)
        if cfg.get("context") or doc.get("sections"):
            # 找出這段在原文的位置（段落依序排列，從上一段之後找起）。table 切法的段落開頭有另外加上的
            # 表格標題和表頭，原文找不到整段，改找最後一行；都找不到就沿用上一段的位置
            pos = text.find(piece, search_from)
            if pos == -1:
                pos = text.find(piece.split("\n")[-1], search_from)
            pos = pos if pos != -1 else max(search_from - 1, 0)
            search_from = pos + 1
        if doc.get("sections"):  # --sections：段首標籤加上這段所屬的章節路徑
            where = section_at(doc, pos)
            label = f"{title}｜{where}" if where else label
            chunk_text = with_label(label, piece)
        chunk = {"text": chunk_text, "source": doc["source"], "parent": section or chunk_text}
        if cfg.get("context"):  # 取這段前面 CONTEXT_WINDOW 字給 LLM 看
            sentence = situate_chunk(doc["source"], text[max(0, pos - CONTEXT_WINDOW):pos], piece, n + 1, len(triples))
            chunk["text"] = with_label(label, f"{sentence}\n{piece}")
            chunk["parent"] = section or chunk["text"]
            chunk["eval_text"] = chunk_text  # 評分只看原本的內容
        chunks.append(chunk)
    save_context_cache()
    return chunks


# ---------- Contextual Retrieval：用 LLM 幫每段寫一句上下文 ----------
_context_cache, _context_dirty = {}, 0


def _context_cache_path(source):
    return CONTEXT_CACHE_DIR / f"context_{Path(source).stem}.json"


def save_context_cache():
    global _context_dirty
    if not _context_dirty:
        return
    CONTEXT_CACHE_DIR.mkdir(exist_ok=True)
    for source, entries in _context_cache.items():
        _context_cache_path(source).write_text(json.dumps(entries, ensure_ascii=False, indent=0), encoding="utf-8")
    _context_dirty = 0


def situate_chunk(source, before, piece, n, total):
    """回傳 LLM 為這段寫的一句上下文。結果存到 cache/，同樣的內容不會再問一次；每 20 段存檔一次，中途停掉也能接著跑。"""
    global _context_dirty
    if source not in _context_cache:
        path = _context_cache_path(source)
        _context_cache[source] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    entries = _context_cache[source]
    key = hashlib.sha1(f"{CONTEXT_PROMPT}\0{before}\0{piece}".encode("utf-8")).hexdigest()
    if key not in entries:
        resp = rag.ollama.chat(
            model=rag.LLM_MODEL,
            messages=[{"role": "user", "content": CONTEXT_PROMPT.format(before=before, chunk=piece)}],
            options={"temperature": 0, "seed": rag.SEED, "num_predict": 80},
        )
        entries[key] = " ".join(resp["message"]["content"].split())  # 壓成一行
        _context_dirty += 1
        if _context_dirty >= 20:
            save_context_cache()
            print(f"\n  上下文 {n}/{total}：{entries[key][:80]}", end="", flush=True)
    return entries[key]


# ---------- 向量化（加快取：同樣的文字只算一次） ----------
_embed_cache = {}


def embed_cached(texts):
    missing = [t for t in dict.fromkeys(texts) if t not in _embed_cache]
    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start:start + EMBED_BATCH]
        _embed_cache.update(zip(batch, rag.embed(batch)))
    vectors = np.array([_embed_cache[t] for t in texts], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # 正規化後內積 = cosine 相似度


# ---------- 建索引、檢索（app.py 的比較模式也會用到） ----------
def build_config_index(cfg, docs):
    """依設定切段並向量化，回傳 (段落清單, 正規化後的向量矩陣)。"""
    chunks = [c for doc in docs for c in make_chunks(doc, cfg)]
    return chunks, embed_cached([c["text"] for c in chunks])


# ---------- 關鍵字檢索（BM25），給混合檢索用 ----------
BM25_K1, BM25_B = 1.2, 0.75  # BM25 的標準參數：k1 控制詞頻的影響上限，b 控制段落長度的懲罰
RRF_K = 60                   # 合併兩種排名時的常數（Reciprocal Rank Fusion 論文的預設值）


def keyword_terms(text):
    """把文字切成關鍵字。用 bge-m3 的 tokenizer，中英文都能切；去掉純標點，英文轉小寫。"""
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    terms = (t.lstrip("▁") for t in _tokenizer.encode(text.lower(), add_special_tokens=False).tokens)
    return [t for t in terms if t and any(ch.isalnum() for ch in t)]


_bm25_cache = {}  # id(段落清單) → 索引，同一份段落清單只建一次


def bm25_scores(chunks, question):
    """回傳問題對每一段的 BM25 分數（numpy 陣列）。"""
    key = id(chunks)
    if key not in _bm25_cache:
        postings, lengths = {}, []  # postings：關鍵字 → [(段落編號, 出現次數), ...]
        for j, c in enumerate(chunks):
            terms = keyword_terms(c["text"])
            lengths.append(len(terms))
            counts = {}
            for t in terms:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                postings.setdefault(t, []).append((j, tf))
        _bm25_cache[key] = (chunks, postings, np.array(lengths, dtype=np.float32))
    _, postings, lengths = _bm25_cache[key]

    n, avg_len = len(lengths), lengths.mean()
    scores = np.zeros(n, dtype=np.float32)
    for t in set(keyword_terms(question)):
        if t not in postings:
            continue
        idf = np.log(1 + (n - len(postings[t]) + 0.5) / (len(postings[t]) + 0.5))
        for j, tf in postings[t]:
            scores[j] += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * lengths[j] / avg_len))
    return scores


def rrf(*score_lists):
    """Reciprocal Rank Fusion：每種檢索各自排名，分數 = Σ 1 / (RRF_K + 名次)。只看名次，不用管兩種分數的尺度不同。"""
    fused = np.zeros(len(score_lists[0]), dtype=np.float32)
    for scores in score_lists:
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
        fused += 1 / (RRF_K + ranks)
    return fused


def retrieve(cfg, chunks, similarity, question=None):
    """similarity 是問題和每一段的相似度。依相似度由高到低取 top_k 段；
    hybrid 模式改用「向量 + BM25 關鍵字」兩種排名合併後的順序（需要傳入 question）；
    parent 模式改成取段落所屬的章節，重複的章節跳過。"""
    order_by = similarity
    if cfg.get("hybrid"):
        if question is None:
            raise ValueError("hybrid 模式需要傳入 question")
        order_by = rrf(similarity, bm25_scores(chunks, question))
    hits, seen = [], set()
    for j in np.argsort(-order_by, kind="stable"):
        text = chunks[j]["parent"] if cfg.get("parent") else chunks[j]["text"]
        if text in seen:
            continue
        seen.add(text)
        hit = {"text": text, "source": chunks[j]["source"], "distance": 1 - float(similarity[j])}
        if "eval_text" in chunks[j] and not cfg.get("parent"):
            hit["eval_text"] = chunks[j]["eval_text"]
        hits.append(hit)
        if len(hits) == cfg["top_k"]:
            break
    return hits


# ---------- 把問題翻成英文（query rewriting） ----------
_translation_cache = {}


def translate_to_english(question):
    """用回答問題的同一個 LLM 把問題翻成英文。temperature 0 + 固定 seed，每次翻譯結果相同。"""
    if question not in _translation_cache:
        resp = rag.ollama.chat(
            model=rag.LLM_MODEL,
            messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(question=question)}],
            options={"temperature": 0, "seed": rag.SEED},
        )
        _translation_cache[question] = resp["message"]["content"].strip()
    return _translation_cache[question]


# ---------- 跑一組設定 ----------
def run_config(cfg, docs, tests, trap_start, question_vecs, use_llm):
    chunks, chunk_vecs = build_config_index(cfg, docs)
    search_questions = [q for q, _ in tests]  # 檢索用的問題（可能翻成英文）；回答時一律用原本的問題
    if cfg.get("query_en"):
        search_questions = [translate_to_english(q) for q in search_questions]
        if cfg["query_en"] == "all":
            question_vecs = embed_cached(search_questions)
    similarity = question_vecs @ chunk_vecs.T  # [題數, 段數]

    retrieval_ok = answer_ok = context_chars = 0
    missed, wrong = [], []  # 檢索沒命中、回答錯誤的題號
    for i, (question, keywords) in enumerate(tests):
        is_trap = i >= trap_start
        if is_trap and not use_llm:
            continue

        hits = retrieve(cfg, chunks, similarity[i], search_questions[i])

        if not is_trap:
            context_chars += sum(len(h["text"]) for h in hits)
            if contains_all("".join(h.get("eval_text", h["text"]) for h in hits), keywords):  # context 設定只看原本的內容
                retrieval_ok += 1
            else:
                missed.append(i + 1)

        if use_llm:
            if contains_all(rag.generate_answer(question, hits), keywords):
                answer_ok += 1
            else:
                wrong.append(i + 1)

    lengths = [len(c["text"]) for c in chunks]
    return {
        "name": cfg["name"],
        "chunks": len(chunks),
        "avg_len": round(sum(lengths) / len(lengths)),
        "max_len": max(lengths),
        "retrieval": f"{retrieval_ok}/{trap_start}",
        "answer": f"{answer_ok}/{len(tests)}" if use_llm else "-",
        "avg_context": round(context_chars / trap_start),
        "missed": " ".join(map(str, missed)) or "-",
        "wrong": (" ".join(map(str, wrong)) or "-") if use_llm else "",
    }


def print_table(rows):
    headers = {
        "name": "設定", "chunks": "段數", "avg_len": "平均字數", "max_len": "最長",
        "retrieval": "檢索命中", "answer": "回答正確", "avg_context": "每題上下文字數", "missed": "沒命中的題號", "wrong": "答錯的題號",
    }
    # 中文字在終端機佔兩格，算寬度時要考慮
    width = lambda s: sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))
    pad = lambda s, w: str(s) + " " * (w - width(s))
    col_w = {k: max(width(h), *(width(r[k]) for r in rows)) for k, h in headers.items()}

    print("  ".join(pad(h, col_w[k]) for k, h in headers.items()))
    print("  ".join("-" * col_w[k] for k in headers))
    for r in rows:
        print("  ".join(pad(r[k], col_w[k]) for k in headers))


def main():
    parser = argparse.ArgumentParser(description="比較不同切段設定的 RAG 效果")
    parser.add_argument("--llm", action="store_true", help="也讓 LLM 回答並評分（很慢）")
    parser.add_argument("--only", help="只跑指定的設定名稱，用逗號分隔，例如 baseline,parent-100")
    parser.add_argument("--data", default=str(rag.DATA_DIR), help="知識文件的資料夾（預設 data）")
    parser.add_argument("--tests", default="evaluate", help="測試題所在的模組，要有 TESTS 和 TRAP_START（預設 evaluate）")
    parser.add_argument("--tables", action="store_true", help="讀 PDF 時辨識表格，轉成 Markdown 表格（較慢）")
    parser.add_argument("--sections", action="store_true", help="PDF 的段首標籤加上段落所屬的章節路徑")
    args = parser.parse_args()

    configs = CONFIGS
    if args.only:
        names = args.only.split(",")
        unknown = set(names) - {c["name"] for c in CONFIGS}
        if unknown:
            parser.error(f"找不到這些設定：{', '.join(sorted(unknown))}")
        configs = [c for c in CONFIGS if c["name"] in names]

    tests_module = importlib.import_module(args.tests)
    tests, trap_start = tests_module.TESTS, tests_module.TRAP_START

    docs = load_documents_for_experiment(args.data, tables=args.tables, sections=args.sections)
    print(f"文件：{args.data}（{len(docs)} 份）　測試題：{args.tests}（{len(tests)} 題）")
    question_vecs = embed_cached([q for q, _ in tests])

    rows = []
    for cfg in configs:
        start = time.time()
        print(f"▶ {cfg['name']} …", end="", flush=True)
        rows.append(run_config(cfg, docs, tests, trap_start, question_vecs, args.llm))
        print(f" {time.time() - start:.1f}s")

    print()
    print_table(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    data_name = Path(args.data).resolve().name
    suffix = "" if data_name == rag.DATA_DIR.name else f"_{data_name}"  # 用其他資料夾時，檔名加上資料夾名稱
    if args.tables:
        suffix += "_tables"
    if args.sections:
        suffix += "_sections"
    out = RESULTS_DIR / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}{suffix}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 打開中文不會亂碼
        # 有些設定多了 parent 欄位，所以欄位名稱要取所有設定的聯集
        fieldnames = list(dict.fromkeys(k for cfg, row in zip(configs, rows) for k in {**cfg, **row}))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg, row in zip(configs, rows):
            writer.writerow({**cfg, **row})
    print(f"\n結果已存到 {out.relative_to(rag.BASE_DIR)}")

    if _translation_cache:  # 有設定把問題翻成英文時，另存一份翻譯結果，方便檢查翻得對不對
        out_tr = out.with_name(out.stem + "_translations.csv")
        with open(out_tr, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["題號", "原本的問題", "英文翻譯"])
            for i, (question, _) in enumerate(tests):
                writer.writerow([i + 1, question, _translation_cache.get(question, "")])
        print(f"翻譯結果已存到 {out_tr.relative_to(rag.BASE_DIR)}")


if __name__ == "__main__":
    main()
