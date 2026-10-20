"""
建立知識庫：讀取 data 資料夾 → 切段 → 向量化 → 存進 chroma_db。
data 裡的文件有新增或修改時，重新執行一次即可。

執行：python ingest.py
"""
import rag_core as rag

# 先印出切段結果，方便檢查切得好不好
for doc in rag.load_documents():
    chunks = rag.chunk_document(doc)
    print(f"\n📄 {doc['source']}：{len(doc['text'])} 字，切成 {len(chunks)} 段")
    for c in chunks:
        first_line = c["text"].splitlines()[0]
        print(f"   - {c['id']}（{len(c['text'])} 字）{first_line}")

print("\n正在向量化並存入資料庫…")
n = rag.build_index()
print(f"✅ 完成！資料庫共 {n} 段")
