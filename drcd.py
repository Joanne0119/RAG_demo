"""
DRCD（台達閱讀理解資料集）：別人人工標註的繁體中文題庫，用來取代自己出的測試題。

來源：https://github.com/DRCKnowledgeTeam/DRCD（論文 https://arxiv.org/abs/1806.00920）
授權：改編自維基百科，CC BY-SA 3.0。原始 JSON 下載到 cache/drcd/（不放進 git，用 download() 重新下載）。

每篇文章的段落用空行接回一整篇，答案位置換算成「在整篇文章中的位置」，
評分時看取回的段落有沒有涵蓋這個位置（不是只看答案字串有沒有出現，因為短答案常在別處重複出現）。
"""
import json
import urllib.request

import rag_core as rag

CACHE_DIR = rag.BASE_DIR / "cache" / "drcd"
URL = "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/master/DRCD_{split}.json"
SEPARATOR = "\n\n"  # 段落之間用空行隔開


def download(split="test"):
    path = CACHE_DIR / f"DRCD_{split}.json"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split=split), path)
    return path


def load(split="test"):
    """回傳 (docs, qas)。
    docs：[{"source": 文章標題, "text": 全文}, ...]，和 rag.load_documents 的格式相同（沒有「# 」標題行，所以段首標籤就是文章標題）。
    qas：[(文章編號, 答案起點, 答案終點, 問題, 答案), ...]，起點、終點是在全文中的位置。"""
    data = json.loads(download(split).read_text(encoding="utf-8"))["data"]
    docs, qas = [], []
    for article in data:
        offset, parts = 0, []
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                answer = qa["answers"][0]  # id "1" 是人工標註的答案
                start = offset + answer["answer_start"]
                qas.append((len(docs), start, start + len(answer["text"]), qa["question"], answer["text"]))
            parts.append(paragraph["context"])
            offset += len(paragraph["context"]) + len(SEPARATOR)
        docs.append({"source": article["title"], "text": SEPARATOR.join(parts)})
    for d, start, end, _, answer in qas:  # 確認換算後的位置正確
        assert docs[d]["text"][start:end] == answer
    return docs, qas


if __name__ == "__main__":
    docs, qas = load()
    print(f"{len(docs)} 篇文章、{len(qas)} 題、共 {sum(len(d['text']) for d in docs):,} 字")
