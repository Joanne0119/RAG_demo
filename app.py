"""
網頁聊天介面。

執行：streamlit run app.py   （瀏覽器會自動打開，關閉請在終端機按 Ctrl+C）
"""
import streamlit as st

import rag_core as rag

st.set_page_config(page_title="RAG Demo", page_icon="📚")
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

# ---------- 側邊欄：設定 ----------
with st.sidebar:
    st.header("⚙️ 設定")
    top_k = st.slider("檢索段數（Top-K）", 1, 8, rag.TOP_K)
    show_sources = st.checkbox("顯示參考段落", value=True)

    st.divider()
    st.write(f"知識庫目前共 **{rag.index_size()}** 段")
    if st.button("🔄 重新建立知識庫"):
        with st.spinner("建立中…"):
            n = rag.build_index()
        st.success(f"完成，共 {n} 段")
    if st.button("🗑️ 清除對話"):
        st.session_state.messages = []

if rag.index_size() == 0:
    st.warning("知識庫是空的，請按左側「重新建立知識庫」。")
    st.stop()

# ---------- 對話紀錄（Streamlit 每次互動都會從頭重跑，所以要存在 session_state） ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 輸入問題 → 檢索 → 回答 ----------
if question := st.chat_input("請輸入問題，例如：咖啡機出現 E03 怎麼辦？"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

        hits = rag.search(question, top_k)
        answer = ""
        for token in rag.stream_answer(question, hits):
            answer += token
            placeholder.markdown(answer + "▌")  # 生成中在尾端顯示游標
        placeholder.markdown(answer)
        if show_sources:
            with st.expander("📎 參考段落"):
                for hit in hits:
                    st.markdown(f"**{hit['source']}**　距離 `{hit['distance']:.3f}`")
                    st.text(hit["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
