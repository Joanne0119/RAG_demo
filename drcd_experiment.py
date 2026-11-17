"""
用 DRCD（別人人工標註的題庫）比較切段設定。

執行：python drcd_experiment.py --only token-32,token-64,token-128
  知識庫：DRCD test 的 378 篇維基文章全部放進同一個索引（約 43 萬字），題目 3,493 題。
  評分：取回的前 top_k 段之中，有沒有一段「涵蓋答案在原文中的位置」（同一篇文章、段落範圍包含整個答案）。
        另外也算舊的「答案字串有沒有出現」，看兩種評分差多少。
切段設定直接沿用 experiment.py 的 CONFIGS。
"""
import argparse
import csv
import time

import numpy as np

import drcd
import experiment as exp
import rag_core as rag

DEFAULT_CONFIGS = ["token-32", "token-64", "token-128", "token-256", "token-512",
                   "heading-200", "baseline", "recursive-200", "hybrid-200", "hybrid-600"]


def chunk_spans(chunks, docs, cfg):
    """每段在原文中的 (文章編號, 起點, 終點)。段落依序排列，從上一段之後找起；找不到的記成 None。"""
    doc_of = {d["source"]: n for n, d in enumerate(docs)}
    spans, search_from, current = [], 0, None
    for c in chunks:
        d = doc_of[c["source"]]
        if d != current:
            current, search_from = d, 0
        piece = c["text"].split("\n", 1)[1] if cfg["add_title"] else c["text"]  # 去掉段首標籤
        pos = docs[d]["text"].find(piece, search_from)
        spans.append((d, pos, pos + len(piece)) if pos != -1 else None)
        if pos != -1:
            search_from = pos + 1
    return spans


class FastBM25:
    """和 experiment.bm25_scores 同一個公式（k1、b、idf 都一樣），只是用 numpy 一次算完一個關鍵字，題目多時才跑得動。"""

    def __init__(self, chunks):
        postings, lengths = {}, []
        for j, c in enumerate(chunks):
            terms = exp.keyword_terms(c["text"])
            lengths.append(len(terms))
            counts = {}
            for t in terms:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                postings.setdefault(t, []).append((j, tf))
        self.postings = {t: (np.array([j for j, _ in p]), np.array([tf for _, tf in p], dtype=np.float32))
                         for t, p in postings.items()}
        self.lengths = np.array(lengths, dtype=np.float32)

    def scores(self, question):
        n, avg_len = len(self.lengths), self.lengths.mean()
        scores = np.zeros(n, dtype=np.float32)
        for t in set(exp.keyword_terms(question)):
            if t not in self.postings:
                continue
            idx, tf = self.postings[t]
            idf = np.log(1 + (n - len(idx) + 0.5) / (len(idx) + 0.5))
            scores[idx] += idf * tf * (exp.BM25_K1 + 1) / (tf + exp.BM25_K1 * (1 - exp.BM25_B + exp.BM25_B * self.lengths[idx] / avg_len))
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(DEFAULT_CONFIGS), help="要跑的設定名稱，用逗號分隔")
    parser.add_argument("--split", default="test", help="DRCD 的哪一組（test / dev / training）")
    parser.add_argument("--limit", type=int, help="只用前 N 篇文章（試跑用）")
    args = parser.parse_args()

    docs, qas = drcd.load(args.split)
    if args.limit:
        docs, qas = docs[:args.limit], [q for q in qas if q[0] < args.limit]
    print(f"DRCD {args.split}：{len(docs)} 篇文章、{len(qas)} 題、共 {sum(len(d['text']) for d in docs):,} 字", flush=True)
    start = time.time()
    question_vecs = exp.embed_cached([q[3] for q in qas])
    print(f"題目向量化 {time.time() - start:.0f}s", flush=True)

    names = args.only.split(",")
    unknown = set(names) - {c["name"] for c in exp.CONFIGS}
    if unknown:
        raise SystemExit(f"不認識的設定：{', '.join(sorted(unknown))}")

    rows, per_question = [], {}
    for name in names:
        cfg = next(c for c in exp.CONFIGS if c["name"] == name)
        start = time.time()
        print(f"▶ {name} …", end="", flush=True)
        chunks, vecs = exp.build_config_index(cfg, docs)
        spans = chunk_spans(chunks, docs, cfg)
        bm25 = FastBM25(chunks) if cfg.get("hybrid") else None

        hits, string_hits = [], []
        for k, (d, ans_start, ans_end, question, answer) in enumerate(qas):
            similarity = vecs @ question_vecs[k]  # 一題一題算，不建（題數 × 段數）的大矩陣，省記憶體
            order_by = exp.rrf(similarity, bm25.scores(question)) if bm25 else similarity
            top = np.argsort(-order_by, kind="stable")[:cfg["top_k"]]
            hits.append(any(spans[j] and spans[j][0] == d and spans[j][1] <= ans_start and ans_end <= spans[j][2] for j in top))
            string_hits.append(any(answer in chunks[j]["text"] for j in top))
        hits, string_hits = np.array(hits), np.array(string_hits)
        exp._embed_cache.clear()  # 每個設定的段落都不一樣，清掉向量快取省記憶體（題目向量已經另外存著）
        per_question[name] = hits
        rate = hits.mean()
        rows.append({"name": name, "chunks": len(chunks), "avg_len": round(float(np.mean([len(c["text"]) for c in chunks]))),
                     "unlocated": sum(s is None for s in spans), "hits": int(hits.sum()), "total": len(qas),
                     "hit_rate": round(float(rate), 4), "stderr": round(float(np.sqrt(rate * (1 - rate) / len(qas))), 4),
                     "string_hits": int(string_hits.sum())})
        print(f" {time.time() - start:.0f}s　命中 {hits.sum()}/{len(qas)}（{rate:.1%}）", flush=True)

    print(f"\n{'設定':14s}{'段數':>7s}{'平均字數':>8s}{'找不到位置':>8s}{'命中（看位置）':>14s}{'標準誤':>7s}{'命中（看字串）':>12s}")
    for r in rows:
        print(f"{r['name']:14s}{r['chunks']:>7d}{r['avg_len']:>10d}{r['unlocated']:>10d}"
              f"{r['hits']:>9d}（{r['hit_rate']:.1%}）{r['stderr']:>8.1%}{r['string_hits']:>11d}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = exp.RESULTS_DIR / f"drcd_{stamp}_{args.split}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # 逐題結果（只存題號和 0/1，不存文章內容）
    out_q = exp.RESULTS_DIR / f"drcd_{stamp}_{args.split}_per_question.csv"
    with open(out_q, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["question_index"] + names)
        for k in range(len(qas)):
            writer.writerow([k] + [int(per_question[n][k]) for n in names])
    print(f"\n結果已存到 {out.relative_to(rag.BASE_DIR)}、{out_q.relative_to(rag.BASE_DIR)}")


if __name__ == "__main__":
    main()
