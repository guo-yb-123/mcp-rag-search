# MCP RAG 知识检索服务系统

面向 Claude Code、GitHub Copilot 等 Agent 工具的知识检索基础设施。基于 MCP 协议提供可插拔、可评测的知识服务。

## 快速开始

```bash
pip install -r requirements.txt

# 爬取文档
python main.py crawl

# 分块 + 构建索引
python main.py chunk
python main.py build

# 启动搜索 Demo
python main.py demo

# 启动 MCP 服务（供外部 Agent 调用）
python app/mcp_server.py
```

## 架构

```
用户查询 → BM25召回 + Dense召回 → RRF融合 → Reranker精排 → Top-K结果
```

## 技术栈

- **检索**: BM25 + BGE-Small Embedding + Milvus Lite
- **融合**: Reciprocal Rank Fusion (RRF)
- **精排**: BGE-Reranker (LoRA 微调)
- **协议**: MCP 2.0 (stdio / SSE)
- **后端**: FastAPI + Streamlit

## 消融实验结果

| 策略 | MRR@10 | 延迟 |
|------|--------|------|
| BM25 only | 0.91 | 64ms |
| Dense only | 0.93 | 398ms |
| Hybrid (BM25+Dense+RRF) | 0.91 ★ | 57ms |
| Hybrid + Reranker | 0.73 | 4948ms |

## 项目结构

```
├── src/
│   ├── crawler/       # 文档爬虫
│   ├── indexing/      # 分块 + BM25 + Milvus
│   ├── retrieval/     # RRF融合 + Reranker
│   ├── eval/          # 消融实验 + 评测指标
│   └── finetune/      # LoRA微调
├── app/
│   ├── api.py         # FastAPI 接口
│   ├── demo.py        # Streamlit Demo
│   └── mcp_server.py  # MCP 知识服务
└── data/
    └── eval/          # 评测集
```
