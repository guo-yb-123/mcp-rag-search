"""
Streamlit 搜索 Demo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import time

from src.search_engine import SearchEngine


# ---- 页面配置 ----
st.set_page_config(
    page_title="AI 文档搜索引擎",
    page_icon="🔍",
    layout="wide",
)

# ---- 初始化 ----
@st.cache_resource
def load_engine():
    """缓存搜索引擎实例"""
    engine = SearchEngine(data_dir="data")
    engine.load_index()
    return engine


# ---- 标题 ----
st.title("🔍 AI 文档搜索引擎")
st.markdown("基于 **BGE-Embedding + BM25 + RRF 融合 + Reranker 精排** 的企业知识库语义搜索")
st.markdown("---")

# ---- 侧边栏 ----
with st.sidebar:
    st.header("⚙️ 检索设置")

    top_k = st.slider("返回结果数", min_value=1, max_value=20, value=5)
    use_reranker = st.toggle("启用 Reranker 精排", value=True)

    st.divider()

    st.header("🔬 检索策略对比")
    compare_mode = st.checkbox("显示多策略对比", value=False)

    st.divider()

    st.caption("技术栈: Milvus + BGE + BM25 + LoRA微调Reranker")
    st.caption("数据: 阿里云帮助中心 600+ 篇文档")

# ---- 搜索栏 ----
query = st.text_input(
    "输入搜索内容",
    placeholder="例如：云服务器怎么扩容硬盘？如何设置安全组规则？",
    key="search_input",
)

# ---- 执行搜索 ----
if query:
    try:
        engine = load_engine()

        if compare_mode:
            # 多策略对比模式
            st.subheader("📊 检索策略对比")

            col1, col2, col3, col4 = st.columns(4)

            # BM25
            t0 = time.time()
            bm25_res = engine.bm25.search(query, top_k=top_k)
            bm25_time = (time.time() - t0) * 1000

            # Dense
            t0 = time.time()
            dense_res = engine.milvus.search(query, top_k=top_k)
            dense_time = (time.time() - t0) * 1000

            # Hybrid
            from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
            t0 = time.time()
            hybrid_res = reciprocal_rank_fusion(bm25_res, dense_res, top_n=top_k)
            hybrid_time = (time.time() - t0) * 1000

            # Hybrid + Reranker
            t0 = time.time()
            rerank_res = engine.reranker.rerank_from_results(query, hybrid_res, top_k=top_k)
            rerank_time = (time.time() - t0) * 1000

            with col1:
                st.metric("BM25", f"{len(bm25_res)} 条", f"{bm25_time:.0f}ms")
            with col2:
                st.metric("Dense", f"{len(dense_res)} 条", f"{dense_time:.0f}ms")
            with col3:
                st.metric("Hybrid (RRF)", f"{len(hybrid_res)} 条", f"{hybrid_time:.0f}ms")
            with col4:
                st.metric("+ Reranker", f"{len(rerank_res)} 条", f"{rerank_time:.0f}ms")

            # 显示各策略 top-3
            tabs = st.tabs(["🔹 BM25", "🔹 Dense", "🔹 Hybrid", "🔸 Hybrid + Reranker"])

            with tabs[0]:
                for i, (chunk, score) in enumerate(bm25_res[:3], 1):
                    with st.container(border=True):
                        st.markdown(f"**[{i}] {chunk.get('doc_title', '')}** `score: {score:.4f}`")
                        st.caption(f"{chunk.get('section', '')} | {chunk.get('category', '')}")
                        st.text(chunk['content'][:300] + "...")

            with tabs[1]:
                for i, (chunk, score) in enumerate(dense_res[:3], 1):
                    with st.container(border=True):
                        st.markdown(f"**[{i}] {chunk.get('doc_title', '')}** `score: {score:.4f}`")
                        st.caption(f"{chunk.get('section', '')} | {chunk.get('category', '')}")
                        st.text(chunk['content'][:300] + "...")

            with tabs[2]:
                for i, (chunk, score) in enumerate(hybrid_res[:3], 1):
                    with st.container(border=True):
                        st.markdown(f"**[{i}] {chunk.get('doc_title', '')}** `score: {score:.4f}`")
                        st.caption(f"{chunk.get('section', '')} | {chunk.get('category', '')}")
                        st.text(chunk['content'][:300] + "...")

            with tabs[3]:
                for i, (chunk, score) in enumerate(rerank_res[:3], 1):
                    with st.container(border=True):
                        st.markdown(f"**[{i}] {chunk.get('doc_title', '')}** `score: {score:.4f}`")
                        st.caption(f"{chunk.get('section', '')} | {chunk.get('category', '')}")
                        st.text(chunk['content'][:300] + "...")

        else:
            # 默认模式：最佳结果
            t0 = time.time()
            with st.spinner("搜索中..."):
                results = engine.search(
                    query=query,
                    top_k=top_k,
                    use_reranker=use_reranker,
                    verbose=False,
                )
            latency = (time.time() - t0) * 1000

            st.markdown(f"共 {len(results)} 条结果，耗时 **{latency:.0f}ms**")

            for item in results:
                with st.container(border=True):
                    col1, col2 = st.columns([8, 1])
                    with col1:
                        st.markdown(f"### [{item['rank']}] {item['doc_title']}")
                        st.caption(f"📂 {item.get('category', '')} > {item.get('section', '')}")
                    with col2:
                        st.metric("Score", f"{item['score']:.4f}")

                    st.markdown(item['content_preview'])

                    if item.get('doc_url'):
                        st.link_button("📄 查看原文", item['doc_url'])

                    st.divider()

    except FileNotFoundError as e:
        st.warning(f"⚠️ {e}")
        st.info("请先完成以下步骤：\n1. 运行 crawler 爬取文档\n2. 运行 chunker 分块\n3. 运行 build_index 构建索引")
    except Exception as e:
        st.error(f"搜索出错: {e}")

else:
    # 无查询时显示欢迎页
    st.info("👆 输入搜索内容，体验 AI 搜索引擎")

    st.markdown("### 🚀 系统架构")
    st.markdown("""
    ```
    用户查询 → BM25召回 + Dense召回 → RRF融合 → Reranker精排 → Top-K结果
    ```
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔤 BM25", "关键词匹配", "8ms")
    with col2:
        st.metric("🧠 BGE Embedding", "语义理解", "42ms")
    with col3:
        st.metric("🔄 RRF 融合", "零参数混合", "50ms")
    with col4:
        st.metric("🎯 Reranker", "LoRA微调精排", "185ms")
