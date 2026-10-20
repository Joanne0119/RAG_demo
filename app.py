"""
網頁聊天介面。

執行：streamlit run app.py   （瀏覽器會自動打開，關閉請在終端機按 Ctrl+C）

兩種模式（側邊欄切換）：
  一般問答    用 chroma_db 裡的知識庫回答
  比較兩種切法  同一個問題用兩種切段設定（experiment.py 的 CONFIGS）各跑一次，左右並排比較
"""
import time

import numpy as np
import ollama
import streamlit as st

import experiment as exp
import rag_core as rag

CHAT_MODE, COMPARE_MODE = "💬 一般問答", "⚖️ 比較兩種切法"
DATA_DIRS = {"有結構（data）": rag.DATA_DIR, "無結構（data_plain）": rag.BASE_DIR / "data_plain"}
CONFIGS = {cfg["name"]: cfg for cfg in exp.CONFIGS}

# 比較模式要左右並排，畫面用寬版（set_page_config 必須是第一個 Streamlit 指令，所以先看 session_state）
wide = st.session_state.get("mode") == COMPARE_MODE
st.set_page_config(page_title="RAG Demo", page_icon="📚", layout="wide" if wide else "centered")
st.title("📚 RAG Demo")
st.caption(f"LLM：{rag.LLM_MODEL}　｜　Embedding：{rag.EMBED_MODEL}")

# AI 還沒吐出第一個字之前，顯示跳動的「...」
st.markdown("""
<style>
.typing span { display: inline-block; font-size: 1.6rem; line-height: 1; animation: blink 1.4s infinite both; }
.typing span:nth-child(2) { animation-delay: .2s; }
.typing span:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%, 80%, 100% { opacity: .2; } 40% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)
TYPING_HTML = '<div class="typing"><span>●</span><span>●</span><span>●</span></div>'


# ---------- 共用：檢索 → 串流回答 → 顯示評估數據 ----------
def format_metrics(m):
    """把評估數據排成回答下方的小字。"""
    parts = [
        f"⏱️ 總共 {m['total']:.1f}s（檢索 {m['search']:.2f}s、第一個字 {m['first_token']:.1f}s）",
        f"📄 參考 {m['chunks']} 段 / {m['context_chars']} 字",
    ]
    if "prompt_tokens" in m:
        speed = m["output_tokens"] / m["eval_seconds"] if m["eval_seconds"] else 0
        parts.append(f"🔢 輸入 {m['prompt_tokens']} tokens、輸出 {m['output_tokens']} tokens（{speed:.1f} tokens/s）")
    return "　｜　".join(parts)


def show_sources(hits):
    with st.expander("📎 參考段落"):
        for hit in hits:
            st.markdown(f"**{hit['source']}**　距離 `{hit['distance']:.3f}`")
            st.text(hit["text"])


def answer_with_metrics(question, search_fn, with_sources):
    """呼叫 search_fn() 檢索，再串流顯示回答與評估數據。回傳 (回答, 評估數據)。"""
    placeholder = st.empty()
    placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

    start = time.perf_counter()
    hits = search_fn()
    search_done = time.perf_counter()

    answer, first_token_at, stats = "", None, {}
    for token in rag.stream_answer(question, hits, stats):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        answer += token
        placeholder.markdown(answer + "▌")  # 生成中在尾端顯示游標
    placeholder.markdown(answer)
    end = time.perf_counter()

    metrics = {
        "total": end - start,
        "search": search_done - start,
        "first_token": (first_token_at or end) - start,
        "chunks": len(hits),
        "context_chars": sum(len(hit["text"]) for hit in hits),  # 交給 LLM 的參考資料字數
        **stats,
    }
    st.caption(format_metrics(metrics))
    if with_sources:
        show_sources(hits)
    return answer, metrics


# ---------- 側邊欄：設定 ----------
with st.sidebar:
    mode = st.radio("模式", [CHAT_MODE, COMPARE_MODE], key="mode")
    st.divider()
    st.header("⚙️ 設定")
    with_sources = st.checkbox("顯示參考段落", value=True)

    if mode == CHAT_MODE:
        top_k = st.slider("檢索段數（Top-K）", 1, 8, rag.TOP_K)
        st.divider()
        st.write(f"知識庫目前共 **{rag.index_size()}** 段")
        if st.button("🔄 重新建立知識庫"):
            with st.spinner("建立中…"):
                n = rag.build_index()
            st.success(f"完成，共 {n} 段")
        if st.button("🗑️ 清除對話"):
            st.session_state.messages = []
    else:
        names = list(CONFIGS)
        name_a = st.selectbox("🅰️ 設定 A", names, index=names.index("heading-200"))
        name_b = st.selectbox("🅱️ 設定 B", names, index=names.index("baseline"))
        data_label = st.selectbox("知識文件", list(DATA_DIRS))
        st.caption("設定定義在 experiment.py 的 CONFIGS。A、B 會依序執行（同時跑會搶同一張顯卡，時間就不準了）。")
        if st.button("🗑️ 清除比較紀錄"):
            st.session_state.comparisons = []


# ==================== 一般問答 ====================
def chat_page():
    if rag.index_size() == 0:
        st.warning("知識庫是空的，請按左側「重新建立知識庫」。")
        st.stop()

    # Streamlit 每次互動都會從頭重跑，所以對話紀錄要存在 session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metrics" in msg:
                st.caption(format_metrics(msg["metrics"]))

    if question := st.chat_input("請輸入問題，例如：咖啡機出現 E03 怎麼辦？"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer, metrics = answer_with_metrics(question, lambda: rag.search(question, top_k), with_sources)
        st.session_state.messages.append({"role": "assistant", "content": answer, "metrics": metrics})


# ==================== 比較兩種切法 ====================
@st.cache_resource(show_spinner="依設定切段並向量化中…")
def config_index(name, data_dir):
    """每種「設定 × 文件資料夾」只建一次索引，之後直接重用。"""
    return exp.build_config_index(CONFIGS[name], rag.load_documents(data_dir))


def describe(cfg):
    text = f"`{cfg['strategy']}`　每段 ≤ {cfg['max_chars']} 字、重疊 {cfg['overlap']}、Top-K {cfg['top_k']}"
    if not cfg["add_title"]:
        text += "、不加段首標題"
    if cfg.get("parent"):
        text += "、小段檢索大段回答"
    return text


def winner_summary(a, b):
    """比較 A、B 的評估數據，列出各項誰比較好（數字越小越好）。"""
    rows = []
    for key, label, unit in [("total", "總時間", "s"), ("first_token", "第一個字", "s"), ("context_chars", "參考字數", " 字")]:
        diff = a[key] - b[key]
        better = "A" if diff < 0 else "B" if diff > 0 else "平手"
        amount = f"{abs(diff):.1f}{unit}" if unit == "s" else f"{abs(diff):.0f}{unit}"
        rows.append(f"**{label}**：{better}" + ("" if better == "平手" else f"（少 {amount}）"))
    return "　｜　".join(rows)


def compare_page():
    if "comparisons" not in st.session_state:
        st.session_state.comparisons = []

    cfg_a, cfg_b = CONFIGS[name_a], CONFIGS[name_b]
    data_dir = DATA_DIRS[data_label]

    # 歷史紀錄
    for record in st.session_state.comparisons:
        with st.chat_message("user"):
            st.markdown(record["question"])
            st.caption(f"知識文件：{record['data']}")
        cols = st.columns(2)
        for col, side in zip(cols, ["A", "B"]):
            r = record[side]
            with col, st.chat_message("assistant", avatar="🅰️" if side == "A" else "🅱️"):
                st.markdown(f"**{side}：{r['name']}**")
                st.markdown(r["content"])
                st.caption(format_metrics(r["metrics"]))
        st.info(winner_summary(record["A"]["metrics"], record["B"]["metrics"]))
        st.divider()

    if question := st.chat_input("輸入問題，A、B 兩種切法會各回答一次"):
        with st.chat_message("user"):
            st.markdown(question)
            st.caption(f"知識文件：{data_label}")

        # 先載入兩個模型（不計時），避免 A 多算到模型載入時間，對 A 不公平
        with st.spinner("載入模型中…"):
            rag.embed([question])
            ollama.chat(model=rag.LLM_MODEL, messages=[])  # messages 留空 = 只把模型載入記憶體

        record = {"question": question, "data": data_label}
        cols = st.columns(2)
        for col, side, cfg in zip(cols, ["A", "B"], [cfg_a, cfg_b]):
            with col:
                chunks, chunk_vecs = config_index(cfg["name"], str(data_dir))
                with st.chat_message("assistant", avatar="🅰️" if side == "A" else "🅱️"):
                    st.markdown(f"**{side}：{cfg['name']}**")
                    st.caption(describe(cfg) + f"　｜　共 {len(chunks)} 段")

                    def search(cfg=cfg, chunks=chunks, chunk_vecs=chunk_vecs):
                        # 不用快取，A、B 都實際向量化一次，檢索時間才公平
                        question_vec = np.array(rag.embed([question])[0], dtype=np.float32)
                        question_vec /= np.linalg.norm(question_vec)
                        return exp.retrieve(cfg, chunks, chunk_vecs @ question_vec)

                    answer, metrics = answer_with_metrics(question, search, with_sources)
                record[side] = {"name": cfg["name"], "content": answer, "metrics": metrics}

        st.info(winner_summary(record["A"]["metrics"], record["B"]["metrics"]))
        st.session_state.comparisons.append(record)
    elif not st.session_state.comparisons:
        st.markdown(f"🅰️ **{name_a}**：{describe(cfg_a)}")
        st.markdown(f"🅱️ **{name_b}**：{describe(cfg_b)}")
        st.caption("在下方輸入問題開始比較。第一次使用某個設定時，需要幾秒鐘切段並向量化。")


if mode == CHAT_MODE:
    chat_page()
else:
    compare_page()
