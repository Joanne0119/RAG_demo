"""
切段實驗：一次比較多種切段設定，看哪一種檢索效果最好。

每組設定都會「在記憶體裡」重新切段、向量化、檢索，不會動到 chroma_db。
用的測試題和 evaluate.py 一樣（檢索只看一般題，陷阱題只在 --llm 時評分）。

執行：python experiment.py                        只比檢索命中率（快，不呼叫 LLM）
      python experiment.py --llm                  連 LLM 回答一起評分（慢很多，含陷阱題）
      python experiment.py --llm --only baseline,parent-100
                                                  只跑指定的設定（用逗號分隔）
      python experiment.py --data data_plain      改用其他資料夾的文件（例如沒有章節結構的版本）

結果會印成表格，並存一份 CSV 到 results 資料夾。
"""
import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np

import rag_core as rag
from evaluate import TESTS, TRAP_START, contains_all

# ==================== 要比較的設定（可自行增減） ====================
# strategy：
#   heading   先依「## 」章節切，太長再用滑動視窗切小（= 目前 rag_core 的做法）
#   length    不看章節，整份文件直接用滑動視窗硬切
#   paragraph 依空行切成段落，再把相鄰段落合併到接近 max_chars
#   recursive 遞迴切分：依序嘗試用 空行 → 換行 → 句號 → 逗號 切，切不夠小才換下一種，
#             再把相鄰小塊合併到接近 max_chars（類似 LangChain 的 RecursiveCharacterTextSplitter）
# add_title：段首要不要加上「【文件標題｜章節標題】」
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
]
RESULTS_DIR = rag.BASE_DIR / "results"
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

    chunks = []
    for label, piece, section in triples:
        chunk_text = with_label(label, piece)
        chunks.append({"text": chunk_text, "source": doc["source"], "parent": section or chunk_text})
    return chunks


# ---------- 向量化（加快取：同樣的文字只算一次） ----------
_embed_cache = {}


def embed_cached(texts):
    missing = [t for t in dict.fromkeys(texts) if t not in _embed_cache]
    if missing:
        _embed_cache.update(zip(missing, rag.embed(missing)))
    vectors = np.array([_embed_cache[t] for t in texts], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # 正規化後內積 = cosine 相似度


# ---------- 建索引、檢索（app.py 的比較模式也會用到） ----------
def build_config_index(cfg, docs):
    """依設定切段並向量化，回傳 (段落清單, 正規化後的向量矩陣)。"""
    chunks = [c for doc in docs for c in make_chunks(doc, cfg)]
    return chunks, embed_cached([c["text"] for c in chunks])


def retrieve(cfg, chunks, similarity):
    """similarity 是問題和每一段的相似度。依相似度由高到低取 top_k 段；
    parent 模式改成取段落所屬的章節，重複的章節跳過。"""
    hits, seen = [], set()
    for j in np.argsort(-similarity):
        text = chunks[j]["parent"] if cfg.get("parent") else chunks[j]["text"]
        if text in seen:
            continue
        seen.add(text)
        hits.append({"text": text, "source": chunks[j]["source"], "distance": 1 - float(similarity[j])})
        if len(hits) == cfg["top_k"]:
            break
    return hits


# ---------- 跑一組設定 ----------
def run_config(cfg, docs, question_vecs, use_llm):
    chunks, chunk_vecs = build_config_index(cfg, docs)
    similarity = question_vecs @ chunk_vecs.T  # [題數, 段數]

    retrieval_ok = answer_ok = context_chars = 0
    missed = []
    for i, (question, keywords) in enumerate(TESTS):
        is_trap = i >= TRAP_START
        if is_trap and not use_llm:
            continue

        hits = retrieve(cfg, chunks, similarity[i])

        if not is_trap:
            context_chars += sum(len(h["text"]) for h in hits)
            if contains_all("".join(h["text"] for h in hits), keywords):
                retrieval_ok += 1
            else:
                missed.append(i + 1)

        if use_llm:
            answer_ok += contains_all(rag.generate_answer(question, hits), keywords)

    lengths = [len(c["text"]) for c in chunks]
    return {
        "name": cfg["name"],
        "chunks": len(chunks),
        "avg_len": round(sum(lengths) / len(lengths)),
        "max_len": max(lengths),
        "retrieval": f"{retrieval_ok}/{TRAP_START}",
        "answer": f"{answer_ok}/{len(TESTS)}" if use_llm else "-",
        "avg_context": round(context_chars / TRAP_START),
        "missed": " ".join(map(str, missed)) or "-",
    }


def print_table(rows):
    headers = {
        "name": "設定", "chunks": "段數", "avg_len": "平均字數", "max_len": "最長",
        "retrieval": "檢索命中", "answer": "回答正確", "avg_context": "每題上下文字數", "missed": "沒命中的題號",
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
    args = parser.parse_args()

    configs = CONFIGS
    if args.only:
        names = args.only.split(",")
        unknown = set(names) - {c["name"] for c in CONFIGS}
        if unknown:
            parser.error(f"找不到這些設定：{', '.join(sorted(unknown))}")
        configs = [c for c in CONFIGS if c["name"] in names]

    docs = rag.load_documents(args.data)
    print(f"文件：{args.data}（{len(docs)} 份）")
    question_vecs = embed_cached([q for q, _ in TESTS])

    rows = []
    for cfg in configs:
        start = time.time()
        print(f"▶ {cfg['name']} …", end="", flush=True)
        rows.append(run_config(cfg, docs, question_vecs, args.llm))
        print(f" {time.time() - start:.1f}s")

    print()
    print_table(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    data_name = Path(args.data).resolve().name
    suffix = "" if data_name == rag.DATA_DIR.name else f"_{data_name}"  # 用其他資料夾時，檔名加上資料夾名稱
    out = RESULTS_DIR / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}{suffix}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 打開中文不會亂碼
        # 有些設定多了 parent 欄位，所以欄位名稱要取所有設定的聯集
        fieldnames = list(dict.fromkeys(k for cfg, row in zip(configs, rows) for k in {**cfg, **row}))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg, row in zip(configs, rows):
            writer.writerow({**cfg, **row})
    print(f"\n結果已存到 {out.relative_to(rag.BASE_DIR)}")


if __name__ == "__main__":
    main()
