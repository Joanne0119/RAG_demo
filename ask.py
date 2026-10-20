"""
在終端機問答，會顯示找到哪些段落，方便觀察檢索結果。

執行：python ask.py      （輸入 q 離開）
"""
import rag_core as rag

if rag.index_size() == 0:
    print("資料庫是空的，請先執行 python ingest.py")
    raise SystemExit

while True:
    question = input("\n❓ 問題：").strip()
    if question.lower() == "q":
        break
    if not question:
        continue

    hits = rag.search(question)
    print("\n🔍 找到的段落（距離越小越相關）：")
    for hit in hits:
        first_line = hit["text"].splitlines()[0]
        print(f"   {hit['distance']:.3f}  {first_line}")

    print("\n💬 回答：")
    for token in rag.stream_answer(question, hits):
        print(token, end="", flush=True)
    print()
