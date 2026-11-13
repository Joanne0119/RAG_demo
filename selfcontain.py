"""
自足性指標：一段文字「單獨拿出來看，看不看得出在講什麼」。

想驗證的假設：段落越自足的文件，最佳段落長度越小。
如果成立，就能在切段前先量自足性，自動決定要切大還是切小。

把每份文件用 token 切成不同長度（不加段首標題，只看段落本身），每種長度隨機抽 SAMPLE 段，量兩個指標：

  主題可找回率（topic）：請 LLM 只看這段，寫出它的主題（例如「後輪胎壓」），再用這個主題去檢索同一份文件的
      所有段落，看這段本身有沒有排進前 TOPIC_TOP_K 名。
      段落自足 → LLM 寫得出具體主題 → 找得回來；只剩一串數值 → 只能寫出「電壓值」之類的籠統主題 → 找不回來
  上下文相似度（context）：這段的向量，和「這段加上前後各 CONTEXT_TOKENS 個 token」的向量的 cosine 相似度。
      不用 LLM。段落自足 → 前後文沒有帶來太多新的意思 → 相似度高

（原本想直接問 LLM「看得出在講什麼嗎？是/否」，但 qwen2.5:3b 幾乎一律回答「否」，連很好懂的段落也是，
  判斷不可靠，所以改成上面兩個不靠 LLM 下判斷的指標。）

執行：python selfcontain.py
      python selfcontain.py --data data_car,data_rp2040 --sizes 32,64,128,256
      python selfcontain.py --answers --data data_car     只量「含有答案的段落」（見 main_answers）
結果印成表格，逐段的主題和排名存到 results/selfcontain_*.csv（方便人工抽查）。
"""
import argparse
import csv
import hashlib
import json
import random
import time

import numpy as np
import ollama
from tokenizers import Tokenizer

import experiment as exp
import rag_core as rag
from evaluate import contains_all

SAMPLE = 40           # 每種文件 × 每種長度抽幾段
SEED = 42
TOPIC_TOP_K = 5       # 主題檢索時，這段排進前幾名算「找得回來」（前後段有 25% 重疊，所以不只看第 1 名）
CONTEXT_TOKENS = 128  # 上下文相似度：前後各多看幾個 token
CACHE_PATH = rag.BASE_DIR / "cache" / "selfcontain_topics.json"

TOPIC_PROMPT = """以下是從一份說明文件中隨機抽出的一段文字，看不到它前後的內容。
請只根據這段文字，用不超過 12 個字寫出它在講的主題，例如「後輪胎壓」「UART 鮑率設定」「BOD 電壓門檻」。
只寫主題名稱，不要寫數值，不要解釋。

段落：
{chunk}

主題："""


def windows(text, offsets, size):
    """和 experiment.split_by_tokens 一樣的滑動視窗（重疊 25%），但同時回傳每段在 token 序列中的起訖位置。"""
    overlap = size // 4
    if len(offsets) <= size:
        return [(0, len(offsets))]
    return [(i, min(i + size, len(offsets))) for i in range(0, len(offsets) - overlap, size - overlap)]


def span(text, offsets, i, j):
    return text[offsets[i][0]:offsets[j - 1][1]].strip()


def topic_of(chunk, cache):
    key = hashlib.sha1((TOPIC_PROMPT + chunk).encode("utf-8")).hexdigest()
    if key not in cache:
        resp = ollama.chat(
            model=rag.LLM_MODEL,
            messages=[{"role": "user", "content": TOPIC_PROMPT.format(chunk=chunk)}],
            options={"temperature": 0, "seed": rag.SEED, "num_predict": 24},
        )
        cache[key] = resp["message"]["content"].strip().split("\n")[0].strip("「」\"' ")
    return cache[key]


def main():
    parser = argparse.ArgumentParser(description="量測不同文件、不同段落長度的自足性")
    parser.add_argument("--data", default="data_car,data_rp2040,data,data_plain", help="文件資料夾，用逗號分隔")
    parser.add_argument("--sizes", default="32,64,128,256", help="段落長度（token），用逗號分隔")
    parser.add_argument("--answers", action="store_true", help="只量含有測試題答案的段落")
    args = parser.parse_args()
    if args.answers:
        return main_answers(args)
    sizes = [int(s) for s in args.sizes.split(",")]
    tokenizer = Tokenizer.from_pretrained(exp.TOKENIZER_NAME)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}

    records, table = [], {}
    for data in args.data.split(","):
        docs = rag.load_documents(rag.BASE_DIR / data)
        encoded = [(d["text"], tokenizer.encode(d["text"], add_special_tokens=False).offsets) for d in docs]
        for size in sizes:
            start = time.time()
            # 所有段落（檢索的對象）和它們的上下文
            pieces, contexts = [], []
            for text, offsets in encoded:
                for i, j in windows(text, offsets, size):
                    pieces.append(span(text, offsets, i, j))
                    contexts.append(span(text, offsets, max(0, i - CONTEXT_TOKENS), min(len(offsets), j + CONTEXT_TOKENS)))
            sample = random.Random(SEED).sample(range(len(pieces)), min(SAMPLE, len(pieces)))

            piece_vecs = exp.embed_cached(pieces)
            context_vecs = exp.embed_cached([contexts[n] for n in sample])
            topics = [topic_of(pieces[n], cache) for n in sample]
            topic_vecs = exp.embed_cached(topics)

            found, sims = [], []
            for k, n in enumerate(sample):
                scores = piece_vecs @ topic_vecs[k]
                rank = int((scores > scores[n]).sum()) + 1
                sim = float(piece_vecs[n] @ context_vecs[k])
                found.append(rank <= TOPIC_TOP_K)
                sims.append(sim)
                records.append({"data": data, "size": size, "topic": topics[k], "rank": rank, "pool": len(pieces),
                                "context_sim": round(sim, 3), "chunk": pieces[n]})
            CACHE_PATH.parent.mkdir(exist_ok=True)
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
            table[data, size] = (np.mean(found), np.mean(sims), len(pieces))
            print(f"{data:12s} token-{size:<4d} 主題可找回 {np.mean(found):4.0%}　上下文相似度 {np.mean(sims):.3f}"
                  f"　（共 {len(pieces)} 段，{time.time() - start:.0f}s）", flush=True)

    for name, idx, fmt in [("主題可找回率", 0, "{:.0%}"), ("上下文相似度", 1, "{:.3f}")]:
        print(f"\n{name}")
        print("文件".ljust(14) + "".join(f"token-{s}".rjust(11) for s in sizes))
        for data in args.data.split(","):
            print(data.ljust(14) + "".join(fmt.format(table[data, s][idx]).rjust(11) for s in sizes))

    out = exp.RESULTS_DIR / f"selfcontain_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n逐段結果已存到 {out.relative_to(rag.BASE_DIR)}")


# ---------- 只量「含有答案的段落」 ----------
TESTS_OF = {"data": "evaluate", "data_plain": "evaluate", "data_car": "tests_car", "data_rp2040": "tests_rp2040"}
MAX_ANSWER_CHUNKS = 3  # 關鍵字出現在超過這麼多段的題目（例如答案只是「80」）找不出哪段才是答案，略過


def main_answers(args):
    """隨機抽樣量不出差別（結果十一），可能是因為出問題的只有少數「含有答案的段落」。
    這裡改成只量這些段落：用測試題的關鍵字找出含有答案的段落，量三個指標：
      主題對得上問題：LLM 只看這段寫出的主題，和問題的 cosine 相似度（越高 = 這段自己就看得出是在回答這個問題）
      主題排名：用主題檢索所有段落，這段排在前百分之幾（越小越好；用百分比，段落總數不同的文件才能比）
      上下文增益：問題和「這段 + 前後文」的相似度，減掉問題和「這段」的相似度（越大 = 越需要前後文才看得出關聯）"""
    import importlib
    sizes = [int(s) for s in args.sizes.split(",")]
    tokenizer = Tokenizer.from_pretrained(exp.TOKENIZER_NAME)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}

    records = []
    for data in args.data.split(","):
        tests_module = importlib.import_module(TESTS_OF[data])
        tests = tests_module.TESTS[:tests_module.TRAP_START]
        docs = rag.load_documents(rag.BASE_DIR / data)
        encoded = [(d["text"], tokenizer.encode(d["text"], add_special_tokens=False).offsets) for d in docs]
        for size in sizes:
            start = time.time()
            pieces, contexts = [], []
            for text, offsets in encoded:
                for i, j in windows(text, offsets, size):
                    pieces.append(span(text, offsets, i, j))
                    contexts.append(span(text, offsets, max(0, i - CONTEXT_TOKENS), min(len(offsets), j + CONTEXT_TOKENS)))
            targets = []  # (題號, 問題, 段落編號)
            for q, (question, keywords) in enumerate(tests):
                matches = [n for n, p in enumerate(pieces) if contains_all(p, keywords)]
                if 0 < len(matches) <= MAX_ANSWER_CHUNKS:
                    targets += [(q + 1, question, n) for n in matches]

            piece_vecs = exp.embed_cached(pieces)
            question_vecs = exp.embed_cached([t[1] for t in targets])
            context_vecs = exp.embed_cached([contexts[t[2]] for t in targets])
            topics = [topic_of(pieces[t[2]], cache) for t in targets]
            topic_vecs = exp.embed_cached(topics)
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")

            rows = []
            for k, (q, question, n) in enumerate(targets):
                scores = piece_vecs @ topic_vecs[k]
                rank = int((scores > scores[n]).sum()) + 1
                rows.append({"data": data, "size": size, "q": q, "question": question, "topic": topics[k],
                             "topic_question_sim": round(float(topic_vecs[k] @ question_vecs[k]), 3),
                             "topic_rank_pct": round(rank / len(pieces), 4), "rank": rank, "pool": len(pieces),
                             "context_gain": round(float(context_vecs[k] @ question_vecs[k] - piece_vecs[n] @ question_vecs[k]), 3),
                             "chunk": pieces[n]})
            records += rows
            mean = lambda key: np.mean([r[key] for r in rows])
            median = lambda key: np.median([r[key] for r in rows])
            print(f"{data:12s} token-{size:<4d} {len({r['q'] for r in rows}):2d} 題 {len(rows):2d} 段　"
                  f"主題對得上問題 {mean('topic_question_sim'):.3f}　主題排名中位數 前 {median('topic_rank_pct'):.1%}　"
                  f"上下文增益 {mean('context_gain'):+.3f}（{time.time() - start:.0f}s）", flush=True)

    data_name = "_".join(args.data.split(","))
    out = exp.RESULTS_DIR / f"selfcontain_answers_{time.strftime('%Y%m%d_%H%M%S')}_{data_name}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"逐段結果已存到 {out.relative_to(rag.BASE_DIR)}")


if __name__ == "__main__":
    main()
