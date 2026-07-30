"""
AI 文档搜索引擎 — 主入口
用法：
    python main.py crawl          # 爬取文档
    python main.py chunk          # 文档分块
    python main.py build          # 构建索引
    python main.py search         # 交互式搜索
    python main.py eval           # 运行消融实验
    python main.py api            # 启动 FastAPI
    python main.py demo           # 启动 Streamlit Demo
    python main.py finetune       # 微调 Reranker
    python main.py pipeline       # 一键运行全流程 (crawl→chunk→build→eval)
"""
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_crawl():
    """爬取文档"""
    from src.crawler.crawl_docs import DocCrawler, ALIYUN_START_URLS

    crawler = DocCrawler(
        base_url="https://help.aliyun.com",
        output_dir="data/raw",
        max_docs=600,
    )
    crawler.crawl(ALIYUN_START_URLS)


def cmd_chunk():
    """文档分块"""
    from src.indexing.chunker import DocumentChunker, load_documents, save_chunks

    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    docs = load_documents("data")
    print(f"加载了 {len(docs)} 篇文档")

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"共生成 {len(all_chunks)} 个 chunks")
    save_chunks(all_chunks, "data")


def cmd_build():
    """构建索引"""
    from src.search_engine import SearchEngine

    engine = SearchEngine(data_dir="data")
    engine.build_index()


def cmd_search():
    """交互式搜索"""
    from src.search_engine import SearchEngine

    engine = SearchEngine(data_dir="data")
    engine.load_index()

    print("\n" + "=" * 60)
    print("AI 文档搜索引擎 — 交互式搜索")
    print("输入 'quit' 退出, 'reranker off' 关闭 Reranker")
    print("=" * 60 + "\n")

    use_reranker = True
    while True:
        query = input("搜索 > ").strip()

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "reranker off":
            use_reranker = False
            print("Reranker 已关闭\n")
            continue
        if query.lower() == "reranker on":
            use_reranker = True
            print("Reranker 已开启\n")
            continue

        results = engine.search(query, top_k=5, use_reranker=use_reranker, verbose=True)

        for item in results:
            print(f"  [{item['rank']}] {item['doc_title']}")
            print(f"      Section: {item['section']} | Score: {item['score']:.4f}")
            print(f"      {item['content_preview']}")
            print()


def cmd_eval():
    """运行消融实验"""
    import json
    from src.search_engine import SearchEngine
    from src.eval.run_experiments import ExperimentRunner

    engine = SearchEngine(data_dir="data")
    engine.load_index()

    runner = ExperimentRunner(
        bm25_index=engine.bm25,
        milvus_index=engine.milvus,
        reranker=engine.reranker,
        test_queries_path="data/eval/test_queries.json",
    )

    all_results = runner.run_all()
    runner.print_summary(all_results)

    # 保存结果
    output_path = PROJECT_ROOT / "data" / "eval" / "experiment_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"实验结果已保存到 {output_path}")


def cmd_api():
    """启动 FastAPI"""
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)


def cmd_demo():
    """启动 Streamlit Demo"""
    import subprocess
    subprocess.run([
        "streamlit", "run",
        str(PROJECT_ROOT / "app" / "demo.py"),
    ])


def cmd_finetune():
    """微调 Reranker"""
    from src.finetune.train_reranker import RerankerTrainer

    trainer = RerankerTrainer(
        base_model="BAAI/bge-reranker-base",
        output_dir="./models/reranker-finetuned",
    )

    trainer.train(
        train_data_path="data/eval/reranker_train_data.json",
        eval_data_path="data/eval/reranker_eval_data.json",
        epochs=2,
        batch_size=8,
        learning_rate=2e-5,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
    )


def cmd_pipeline():
    """一键运行全流程"""
    print("=" * 60)
    print("全流程自动化: crawl → chunk → build → eval")
    print("=" * 60)

    cmd_crawl()
    cmd_chunk()
    cmd_build()

    # 问用户是否有评测集
    eval_path = PROJECT_ROOT / "data" / "eval" / "test_queries.json"
    if eval_path.exists():
        print("\n检测到评测集，是否运行消融实验？(y/n)")
        if input().strip().lower() == "y":
            cmd_eval()


if __name__ == "__main__":
    commands = {
        "crawl": cmd_crawl,
        "chunk": cmd_chunk,
        "build": cmd_build,
        "search": cmd_search,
        "eval": cmd_eval,
        "api": cmd_api,
        "demo": cmd_demo,
        "finetune": cmd_finetune,
        "pipeline": cmd_pipeline,
    }

    if len(sys.argv) < 2:
        print("用法: python main.py <command>")
        print(f"可用命令: {', '.join(commands.keys())}")
        print("\n推荐顺序:")
        print("  1. python main.py pipeline   # 一键全流程")
        print("  2. python main.py search     # 交互式搜索")
        print("  3. python main.py demo       # Streamlit Demo")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd not in commands:
        print(f"未知命令: {cmd}")
        print(f"可用命令: {', '.join(commands.keys())}")
        sys.exit(1)

    commands[cmd]()
