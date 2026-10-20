"""
RAG 核心功能：讀檔 → 切段 → 向量化 → 存入資料庫 → 檢索 → 生成回答。
其他程式（ingest.py / ask.py / evaluate.py / app.py）都會 import 這個檔案。
"""
from pathlib import Path

import chromadb
import ollama
import pymupdf

# ==================== 可調整的參數 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"          # 知識文件放這裡（支援 .md / .txt / .pdf）
DB_DIR = BASE_DIR / "chroma_db"       # 向量資料庫存檔位置
COLLECTION_NAME = "docs"

EMBED_MODEL = "bge-m3"                # 把文字轉成向量的模型
LLM_MODEL = "qwen2.5:3b"              # 負責回答的語言模型

MAX_CHARS = 600                       # 每段最多幾個字，超過就再切小
OVERLAP = 100                         # 再切小時，相鄰兩段重疊幾個字
TOP_K = 3                             # 每次檢索取回幾段
TEMPERATURE = 0.1                     # 越低回答越穩定保守，越高越有變化

SYSTEM_PROMPT = "你是一個客服助理，只根據提供的參考資料回答問題，一律使用繁體中文回答。"

PROMPT_TEMPLATE = """請只根據以下參考資料回答問題。
如果參考資料中沒有答案，請直接回答「資料中沒有提到」，不要自行推測。

【參考資料】
{context}

【問題】
{question}"""
# =====================================================


# ---------- 1. 讀取文件 ----------
def load_documents():
    """讀取 data 資料夾內所有文件，回傳 [{"source": 檔名, "text": 全文}, ...]"""
    docs = []
    for path in sorted(DATA_DIR.iterdir()):
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            with pymupdf.open(path) as pdf:
                text = "\n".join(page.get_text() for page in pdf)
        else:
            continue  # 其他格式先略過
        docs.append({"source": path.name, "text": text})
    return docs


# ---------- 2. 切段 ----------
def split_by_heading(text):
    """依照 Markdown 的「## 」章節標題切開，每個章節一段。"""
    sections, current = [], []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    sections.append("\n".join(current).strip())
    return [s for s in sections if s]


def split_by_length(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """太長的段落用「滑動視窗」切小：每段 max_chars 字，相鄰兩段重疊 overlap 字。"""
    if len(text) <= max_chars:
        return [text]
    step = max_chars - overlap
    return [text[i:i + max_chars] for i in range(0, len(text) - overlap, step)]


def doc_title(doc):
    """文件標題：取第一行「# 」開頭的標題，沒有的話用檔名。"""
    for line in doc["text"].splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return doc["source"]


def chunk_document(doc):
    """把一份文件切成多個小段，每段開頭加上「文件標題｜章節標題」。"""
    title = doc_title(doc)

    chunks = []
    for section in split_by_heading(doc["text"]):
        heading = section.splitlines()[0].lstrip("# ").strip()
        for piece in split_by_length(section):
            chunks.append({
                "id": f"{doc['source']}_{len(chunks)}",
                # 加上標題，讓「保固多久」這類問題能分辨是咖啡機還是黑客松
                "text": f"【{title}｜{heading}】\n{piece}",
                "source": doc["source"],
            })
    return chunks


# ---------- 3. 向量化 ----------
def embed(texts):
    """把一串文字轉成向量（每個向量 1024 個數字）。"""
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


# ---------- 4. 建立向量資料庫 ----------
def get_client():
    return chromadb.PersistentClient(path=str(DB_DIR))


def build_index():
    """讀檔、切段、向量化後存入資料庫。每次執行都會整個重建。回傳段落數。"""
    chunks = []
    for doc in load_documents():
        chunks += chunk_document(doc)

    client = get_client()
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    # space="cosine"：用 cosine 距離比較向量，0 = 意思完全一樣，越大越不像
    collection = client.create_collection(
        COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return len(chunks)


def index_size():
    """資料庫裡有幾段，還沒建立的話回傳 0。"""
    try:
        return get_client().get_collection(COLLECTION_NAME).count()
    except chromadb.errors.NotFoundError:
        return 0


# ---------- 5. 檢索 ----------
def search(question, top_k=TOP_K):
    """找出和問題最相關的 top_k 段，回傳 [{"text", "source", "distance"}, ...]"""
    collection = get_client().get_collection(COLLECTION_NAME)
    results = collection.query(query_embeddings=embed([question]), n_results=top_k)
    return [
        {"text": text, "source": meta["source"], "distance": dist}
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


# ---------- 6. 生成回答 ----------
def build_messages(question, hits):
    """把檢索到的段落和問題組成要給 LLM 的訊息。"""
    context = "\n---\n".join(hit["text"] for hit in hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=question)},
    ]


def generate_answer(question, hits):
    """一次拿到完整回答。"""
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=build_messages(question, hits),
        options={"temperature": TEMPERATURE},
    )
    return resp["message"]["content"]


def stream_answer(question, hits):
    """邊生成邊輸出（像 ChatGPT 一個字一個字跑出來），給網頁介面用。"""
    for part in ollama.chat(
        model=LLM_MODEL,
        messages=build_messages(question, hits),
        options={"temperature": TEMPERATURE},
        stream=True,
    ):
        yield part["message"]["content"]
