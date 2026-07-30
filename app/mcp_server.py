"""
MCP (Model Context Protocol) RAG 知识服务

面向 Claude Code / GitHub Copilot / OpenClaw 等 Agent 工具，
通过 MCP 协议暴露知识检索能力，让 Agent 可以：
  - search_knowledge: 语义搜索阿里云文档知识库
  - get_document: 获取单篇文档完整内容
  - get_index_info: 查看索引状态和统计

用法：
  python app/mcp_server.py                        # stdio 模式（默认）
  python app/mcp_server.py --transport sse --port 8080  # SSE 模式

Claude Code 配置 (.claude.json 或 claude_desktop_config.json)：
  {
    "mcpServers": {
      "aliyun-docs": {
        "command": "python",
        "args": ["app/mcp_server.py"],
        "cwd": "D:/python/Pythonfastapi/PythonProject6",
        "env": { "HF_ENDPOINT": "https://hf-mirror.com" }
      }
    }
  }
"""

import sys
import os
import json
from pathlib import Path
from collections import Counter

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from mcp.server import MCPServer

# ============================================================
# 初始化
# ============================================================

mcp = MCPServer(
    name="阿里云文档知识检索",
    version="1.0.0",
    description="基于 BGE + BM25 + RRF + Reranker 的阿里云帮助文档语义搜索引擎，"
    "600+ 篇文档覆盖 ECS/VPC/RDS/OSS 等产品线",
)

# 懒加载引擎（首次调用时才加载模型，避免 stdio 启动超时）
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        import time
        from src.search_engine import SearchEngine

        sys.stderr.write("[MCP] 正在加载搜索引擎...\n")
        sys.stderr.flush()
        t0 = time.time()
        _engine = SearchEngine(data_dir="data")
        _engine.load_index()
        sys.stderr.write(f"[MCP] 加载完成 ({time.time() - t0:.1f}s)\n")
        sys.stderr.flush()
    return _engine


# ============================================================
# MCP 工具
# ============================================================

@mcp.tool()
def search_knowledge(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
) -> str:
    """
    搜索阿里云帮助文档知识库。

    适用场景：
      - 查询阿里云产品使用方法、API文档、最佳实践
      - 查找配置指南、故障排查步骤
      - 了解 ECS、VPC、RDS、OSS、RAM 等云产品的功能说明

    Args:
        query: 搜索查询，支持自然语言。例如 "如何给ECS实例扩容系统盘"
        top_k: 返回结果数量，默认 5，最大 20
        use_reranker: 是否启用 Reranker 精排。True=更准但较慢，False=更快

    Returns:
        格式化的搜索结果，包含文档标题、URL、相关内容和相似度分数
    """
    engine = _get_engine()
    results = engine.search(
        query=query, top_k=min(top_k, 20), use_reranker=use_reranker
    )

    if not results:
        return f"未找到与「{query}」相关的文档。请尝试更换关键词或缩小搜索范围。"

    lines = [f"## 搜索「{query}」- Top {len(results)} 结果\n"]
    for r in results:
        lines.append(f"### [{r['rank']}] {r['doc_title']}")
        lines.append(f"- **分类**: {r.get('category', '')} > {r.get('section', '')}")
        lines.append(f"- **相关度**: {r['score']:.4f}")
        lines.append(f"- **链接**: {r.get('doc_url', '')}")
        lines.append("")
        lines.append(f"> {r['content_preview']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """
    根据文档ID获取完整文档内容。

    适用场景：
      - 查看某篇文档的全文（不限于分块预览）
      - 深入了解某个搜索结果对应的完整文章

    Args:
        doc_id: 文档ID（chunk_id 前12位，即文档URL的MD5哈希前缀）

    Returns:
        文档的完整 Markdown 内容
    """
    raw_dir = Path("data/raw")
    prefix = doc_id[:12] if len(doc_id) > 12 else doc_id

    for f in raw_dir.glob(f"{prefix}*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        return (
            f"# {doc['title']}\n\n"
            f"- **分类**: {doc.get('category', '')}\n"
            f"- **URL**: {doc['url']}\n"
            f"- **爬取时间**: {doc.get('crawl_time', '')}\n\n"
            f"---\n\n"
            f"{doc['content']}"
        )

    return f"错误：未找到文档 ID `{doc_id}`"


@mcp.tool()
def get_index_info() -> str:
    """
    查看知识库索引的统计信息。

    Returns:
        索引状态、文档数、chunk数、模型信息、产品线分布
    """
    engine = _get_engine()
    chunks = engine.bm25.chunks
    if not chunks:
        return "索引未加载或为空"

    raw_dir = Path("data/raw")
    doc_count = len(list(raw_dir.glob("*.json")))

    cat_counter = Counter()
    for ch in chunks:
        cat_counter[ch.get("category", "?")] += 1

    lines = [
        "## 知识库索引状态",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 状态 | ready |",
        f"| 原始文档数 | {doc_count} |",
        f"| Chunk 总数 | {len(chunks)} |",
        f"| Embedding 模型 | {engine.milvus.embedding_model_name} |",
        f"| Reranker 模型 | {engine.reranker.model_name} |",
        f"| 向量维度 | {engine.milvus.dimension} |",
        "",
        "### 产品线分布",
        "",
    ]
    for cat, cnt in cat_counter.most_common(20):
        lines.append(f"- **{cat}**: {cnt} chunks")

    return "\n".join(lines)


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP RAG 知识服务")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输协议: stdio (默认) 或 sse",
    )
    parser.add_argument("--port", type=int, default=8080, help="SSE 模式端口 (默认 8080)")
    args = parser.parse_args()

    if args.transport == "sse":
        import asyncio
        asyncio.run(mcp.run_sse_async(host="0.0.0.0", port=args.port))
    else:
        import asyncio
        asyncio.run(mcp.run_stdio_async())
