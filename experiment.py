"""
切段實驗：一次比較多種切段設定，看哪一種檢索效果最好。

每組設定都會「在記憶體裡」重新切段、向量化、檢索，不會動到 chroma_db。
用的測試題和 evaluate.py 一樣（前 15 題，不含陷阱題）。

執行：python experiment.py          只比檢索命中率（快，不呼叫 LLM）
      python experiment.py --llm    連 LLM 回答一起評分（慢很多，含陷阱題）

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
# add_title：段首要不要加上「【文件標題｜章節標題】」
CONFIGS = [
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


def make_chunks(doc, cfg):
    """依設定把一份文件切段，回傳 [{"text", "source"}, ...]"""
    title = rag.doc_title(doc)
    text = doc["text"]
    max_chars, overlap = cfg["max_chars"], cfg["overlap"]

    labeled = []  # (段首標籤, 段落內容)
    if cfg["strategy"] == "heading":
        for section in rag.split_by_heading(text):
            heading = section.splitlines()[0].lstrip("# ").strip()
            labeled += [(f"{title}｜{heading}", p) for p in rag.split_by_length(section, max_chars, overlap)]
    elif cfg["strategy"] == "length":
        labeled = [(title, p) for p in rag.split_by_length(text, max_chars, overlap)]
    elif cfg["strategy"] == "paragraph":
        labeled = [(title, p) for p in split_by_paragraph(text, max_chars, overlap)]
    else:
        raise ValueError(f"不認識的 strategy：{cfg['strategy']}")

    return [
        {"text": f"【{label}】\n{piece}" if cfg["add_title"] else piece, "source": doc["source"]}
        for label, piece in labeled
    ]


# ---------- 向量化（加快取：同樣的文字只算一次） ----------
_embed_cache = {}


def embed_cached(texts):
    missing = [t for t in dict.fromkeys(texts) if t not in _embed_cache]
    if missing:
        _embed_cache.update(zip(missing, rag.embed(missing)))
    vectors = np.array([_embed_cache[t] for t in texts], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # 正規化後內積 = cosine 相似度


# ---------- 跑一組設定 ----------
def run_config(cfg, docs, question_vecs, use_llm):
    chunks = [c for doc in docs for c in make_chunks(doc, cfg)]
    chunk_vecs = embed_cached([c["text"] for c in chunks])
    similarity = question_vecs @ chunk_vecs.T  # [題數, 段數]

    retrieval_ok = answer_ok = context_chars = 0
    missed = []
    for i, (question, keywords) in enumerate(TESTS):
        is_trap = i >= TRAP_START
        if is_trap and not use_llm:
            continue

        top = np.argsort(-similarity[i])[:cfg["top_k"]]
        hits = [
            {"text": chunks[j]["text"], "source": chunks[j]["source"], "distance": 1 - float(similarity[i, j])}
            for j in top
        ]

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
    args = parser.parse_args()

    docs = rag.load_documents()
    question_vecs = embed_cached([q for q, _ in TESTS])

    rows = []
    for cfg in CONFIGS:
        start = time.time()
        print(f"▶ {cfg['name']} …", end="", flush=True)
        rows.append(run_config(cfg, docs, question_vecs, args.llm))
        print(f" {time.time() - start:.1f}s")

    print()
    print_table(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 打開中文不會亂碼
        writer = csv.DictWriter(f, fieldnames=list({**CONFIGS[0], **rows[0]}))
        writer.writeheader()
        for cfg, row in zip(CONFIGS, rows):
            writer.writerow({**cfg, **row})
    print(f"\n結果已存到 {out.relative_to(rag.BASE_DIR)}")


if __name__ == "__main__":
    main()
