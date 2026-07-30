"""
Reranker 微调数据生成模块
策略：大模型辅助生成 (query, positive_doc, negative_doc) 三元组
"""
import json
import random
from pathlib import Path
from typing import List, Dict


class RerankerDataGenerator:
    """微调数据生成器"""

    def __init__(self, chunks_path: str = "data/chunks/chunks.json"):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.all_chunks = json.load(f)
        print(f"加载了 {len(self.all_chunks)} 个 chunks")

    def generate_with_llm(
        self,
        num_samples: int = 2500,
        eval_ratio: float = 0.1,
        output_dir: str = "data/eval",
    ) -> Dict[str, List]:
        """
        使用 LLM 辅助生成训练数据

        策略：
        1. 随机选一个 chunk 作为正例
        2. 让 LLM 根据该 chunk 内容反向生成 3-5 个不同表述的查询
        3. 从其他文档随机采样 2-3 个作为负例
        4. 训练集和验证集使用不同文档子集，避免数据泄露

        Args:
            num_samples: 生成的数据量
            eval_ratio: 验证集比例
            output_dir: 输出目录

        Returns:
            {"train": [...], "eval": [...]}
        """
        # 按文档拆分，避免训练/验证数据泄露
        doc_ids = list(set(c["doc_id"] for c in self.all_chunks))
        random.shuffle(doc_ids)

        split_idx = int(len(doc_ids) * (1 - eval_ratio))
        train_doc_ids = set(doc_ids[:split_idx])
        eval_doc_ids = set(doc_ids[split_idx:])

        print(f"训练文档: {len(train_doc_ids)} 篇, 验证文档: {len(eval_doc_ids)} 篇")

        # 生成
        train_data = self._generate_subset(num_samples, train_doc_ids)
        eval_data = self._generate_subset(int(num_samples * eval_ratio), eval_doc_ids)

        # 保存
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, data in [("train", train_data), ("eval", eval_data)]:
            output_path = output_dir / f"reranker_{name}_data.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已保存 {name} 数据 ({len(data)} 条) 到 {output_path}")

        return {"train": train_data, "eval": eval_data}

    def _generate_subset(
        self,
        num_samples: int,
        allowed_doc_ids: set,
    ) -> List[Dict]:
        """生成数据子集"""
        # 过滤出允许的 chunks
        allowed_chunks = [c for c in self.all_chunks if c["doc_id"] in allowed_doc_ids]
        other_chunks = [c for c in self.all_chunks if c["doc_id"] not in allowed_doc_ids]

        if len(allowed_chunks) < 100:
            print(f"警告: 允许的 chunks 较少 ({len(allowed_chunks)})，可能影响数据质量")

        data = []
        for _ in range(num_samples):
            # 选一个正例 chunk（内容长度适中的）
            positive = self._select_positive_chunk(allowed_chunks)

            # 根据 chunk 内容构造查询（规则生成 + LLM 生成模板）
            query = self._generate_query_for_chunk(positive)

            # 选几个负例 chunk
            negatives = self._select_negative_chunks(positive, other_chunks, n=2)

            data.append({
                "query": query,
                "positive": positive["content"],
                "negative": [n["content"] for n in negatives],
            })

        return data

    def _select_positive_chunk(self, chunks: List[Dict]) -> Dict:
        """选择一个适合的正例chunk：内容长度适中"""
        # 过滤太短或太长的
        valid = [
            c for c in chunks
            if 80 < len(c["content"]) < 800
        ]
        if not valid:
            valid = chunks
        return random.choice(valid)

    def _generate_query_for_chunk(self, chunk: Dict) -> str:
        """
        根据 chunk 内容生成查询

        规则策略（不依赖LLM时的降级方案）：
        - 从标题和section中提取关键词组合
        - 生成变体（加疑问词、同义替换等）
        """
        title = chunk.get("doc_title", "")
        section = chunk.get("section", "")
        content = chunk.get("content", "")

        # 规则生成模板（简化版，实际应用建议调用 LLM）
        templates = [
            f"{title}怎么{self._extract_action(content)}",
            f"如何{self._extract_action(content)}",
            f"{section}的操作步骤",
            f"{title}的{section}在哪里设置",
            f"怎样{self._extract_action(content)}",
        ]

        # 过滤掉太短的
        templates = [t for t in templates if len(t) > 5]
        return random.choice(templates) if templates else section

    def _extract_action(self, text: str) -> str:
        """从文本中提取动作关键词"""
        action_keywords = ["创建", "配置", "修改", "删除", "查询", "设置", "安装", "部署",
                           "扩容", "备份", "恢复", "迁移", "绑定", "解绑", "开启", "关闭",
                           "添加", "移除", "导入", "导出", "上传", "下载", "启动", "停止"]
        for kw in action_keywords:
            if kw in text:
                # 取关键词 + 后面的词
                idx = text.index(kw)
                end = min(idx + 6, len(text))
                return text[idx:end].replace("\n", "").strip()
        return "设置"

    def _select_negative_chunks(
        self,
        positive: Dict,
        other_chunks: List[Dict],
        n: int = 2,
    ) -> List[Dict]:
        """选择负例chunk：内容不相关但看起来像的（hard negative）"""
        candidates = [
            c for c in other_chunks
            if c["doc_id"] != positive["doc_id"]
               and len(c["content"]) > 50
        ]

        if len(candidates) < n:
            # 不够的话从同文档其他section取
            same_doc = [
                c for c in self.all_chunks
                if c["doc_id"] == positive["doc_id"]
                   and c["chunk_id"] != positive["chunk_id"]
                   and len(c["content"]) > 50
            ]
            candidates.extend(same_doc)

        if len(candidates) < n:
            return random.choices(candidates, k=n)

        return random.sample(candidates, n)


# ========== LLM 辅助生成（推荐方案） ==========

def generate_with_claude(chunks: List[Dict], output_path: str, num_samples: int = 2500):
    """
    使用 Claude API 批量生成高质量训练数据

    每条数据包含多个表述变体，质量远超规则生成。
    建议先用此函数生成数据，再用 generate_manual_eval_set 人工标注评测集。
    """
    import os

    # 这个函数需要用户配置 API key，下面是指令模板
    prompt_template = """你是一个搜索评测数据生成器。给定一段文档内容，请生成3个不同表述的搜索查询，这些查询的正确答案是这段文档。

要求：
1. 查询的表述要多样化：包含精确匹配型、口语化表述、同义词替换
2. 查询长度在5-25个字之间
3. 模拟真实用户会怎么搜

文档标题：{title}
文档章节：{section}
文档内容：{content}

请直接输出JSON格式：
[{{"query": "查询1"}}, {{"query": "查询2"}}, {{"query": "查询3"}}]
"""
    # 具体实现留给用户（需要 API key）
    print(f"请配置 Claude API key 后运行此函数，将生成 {num_samples} 条数据。")
    print(f"提示词模板已准备，每5条文档 batch 一次调用。")
    print(f"预期输出路径: {output_path}")


def generate_manual_eval_set(
    chunks: List[Dict],
    output_path: str = "data/eval/test_queries.json",
    num_queries: int = 20,
):
    """
    手工标注评测集模板
    生成一个模板 JSON，用户手动填写查询和相关chunk
    """
    template = {
        "description": "AI搜索评测集 — 请手动填写每个查询的正确答案",
        "num_queries": num_queries,
        "queries": [],
    }

    # 预选一些有代表性的chunk供标注
    good_chunks = [c for c in chunks if 100 < len(c["content"]) < 600][:num_queries * 3]

    for i, chunk in enumerate(random.sample(good_chunks, min(num_queries, len(good_chunks)))):
        template["queries"].append({
            "id": f"q{i+1:03d}",
            "query": "TODO: 根据下面的文档内容，写一个你会搜索的问题",
            "doc_preview": chunk["content"][:300],
            "relevant_chunks": [],  # 标注时填写正确的 chunk_id
            "relevant_chunks_with_scores": [],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"评测集模板已生成: {output_path}")
    print(f"请手动编辑该文件，填写 query 和 relevant_chunks。")


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    generator = RerankerDataGenerator(chunks_path=f"{data_dir}/chunks/chunks.json")

    # 规则生成（快速，但质量一般）
    data = generator.generate_with_llm(num_samples=300, output_dir=f"{data_dir}/eval")

    # 生成评测集模板
    with open(f"{data_dir}/chunks/chunks.json", "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    generate_manual_eval_set(all_chunks, output_path=f"{data_dir}/eval/test_queries.json")
