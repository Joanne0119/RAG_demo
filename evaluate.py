"""
自動跑一輪測試題（與「測試問題與標準答案.md」相同），統計：
  - 檢索命中：找到的段落裡有沒有包含答案
  - 回答正確：LLM 的回答裡有沒有出現關鍵字
每次調整參數後跑一次，就能比較改得好不好。

執行：python evaluate.py
"""
import rag_core as rag

# (問題, 關鍵字)：關鍵字「全部」都要出現才算對；用 | 分隔表示「任一個出現即可」
TESTS = [
    ("咖啡機出現 E03 是什麼意思？", ["渣盒"]),
    ("咖啡機多久要除鈣一次？", ["3個月|3个月|三個月|三个月"]),
    ("咖啡機除鈣要多久？", ["25"]),
    ("App 一直找不到咖啡機怎麼辦？", ["2.4"]),
    ("咖啡喝起來太淡怎麼調整？", ["細|细"]),
    ("咖啡機保固多久？", ["2年|兩年|两年"]),
    ("咖啡機可以直接放咖啡粉嗎？放多少？", ["7"]),
    ("打奶泡時牛奶要加熱到幾度？", ["60"]),
    ("黑客松報名截止日期是哪天？", ["11月20|11/20"]),
    ("黑客松一隊可以幾個人？", ["3", "5"]),
    ("我是外校學生可以當黑客松隊長嗎？", ["南星"]),
    ("黑客松報名費多少？", ["500", "250"]),
    ("黑客松第一名獎金多少？要扣稅嗎？", ["50,000|50000|5萬|5万", "10%"]),
    ("黑客松程式碼最晚什麼時候要交？", ["15:00|下午3|下午三"]),
    ("黑客松決賽評分中技術完成度佔多少？", ["30%"]),
    # 陷阱題：資料中沒有答案，期望模型老實說沒有
    ("咖啡機可以做冰咖啡嗎？", ["沒有提到|没有提到"]),
    ("黑客松有提供停車位嗎？", ["沒有提到|没有提到"]),
]
TRAP_START = 15  # 從第 16 題開始是陷阱題，不檢查檢索


def contains_all(text, keywords):
    text = text.replace(" ", "")
    return all(any(alt in text for alt in kw.split("|")) for kw in keywords)


def main():
    if rag.index_size() == 0:
        print("資料庫是空的，請先執行 python ingest.py")
        raise SystemExit

    retrieval_ok = answer_ok = 0
    for i, (question, keywords) in enumerate(TESTS):
        hits = rag.search(question)
        answer = rag.generate_answer(question, hits)

        is_trap = i >= TRAP_START
        hit_ok = is_trap or contains_all("".join(h["text"] for h in hits), keywords)
        ans_ok = contains_all(answer, keywords)
        if not is_trap:
            retrieval_ok += hit_ok
        answer_ok += ans_ok

        mark = lambda ok: "✅" if ok else "❌"
        print(f"\n{i + 1:>2}. {question}")
        print(f"    檢索 {'—' if is_trap else mark(hit_ok)}   回答 {mark(ans_ok)}   關鍵字：{keywords}")
        print(f"    {answer.strip().replace(chr(10), ' ')[:150]}")

    print("\n" + "=" * 50)
    print(f"檢索命中：{retrieval_ok}/{TRAP_START}")
    print(f"回答正確：{answer_ok}/{len(TESTS)}")


# experiment.py 會 import 這個檔案的測試題，所以直接執行時才跑評分
if __name__ == "__main__":
    main()
