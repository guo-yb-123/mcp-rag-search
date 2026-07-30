"""
文档分块模块
策略：按 Markdown 标题层级切分 + 滑动窗口重叠
"""
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional


class DocumentChunker:
    """文档分块器"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        """
        Args:
            chunk_size: 每个 chunk 的最大字符数（中文约等于token数）
            chunk_overlap: 相邻 chunk 重叠的字符数
            min_chunk_size: 最小 chunk 大小，小于此值的 chunk 会被合并到前一个
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: Dict) -> List[Dict]:
        """
        对单篇文档进行分块

        Args:
            doc: {"doc_id": str, "title": str, "url": str, "content": str, "category": str}

        Returns:
            List of chunk dicts
        """
        content = doc["content"]
        title = doc.get("title", "")
        doc_id = doc.get("doc_id", "")

        # 1. 按二级标题切分段落
        sections = self._split_by_headings(content)

        # 2. 对每个段落按 chunk_size 切分（滑动窗口）
        chunks = []
        for section_title, section_text in sections:
            section_chunks = self._sliding_window_split(section_text)

            for i, chunk_text in enumerate(section_chunks):
                chunk_id = self._generate_chunk_id(doc_id, section_title, i)
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_title": title,
                    "doc_url": doc.get("url", ""),
                    "category": doc.get("category", ""),
                    "section": section_title,
                    "content": chunk_text,
                    "chunk_index": len(chunks),
                })

        # 3. 合并过短的 chunk
        chunks = self._merge_short_chunks(chunks)

        return chunks

    def _split_by_headings(self, content: str) -> List[tuple]:
        """
        按 ## 二级标题切分，返回 [(section_title, section_text), ...]
        """
        # 匹配 ## 标题（排除 ### 三级标题等）
        pattern = r'^##\s+(.+)$'
        lines = content.split('\n')
        sections = []
        current_title = "正文"
        current_lines = []

        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                # 保存上一个段落
                if current_lines:
                    sections.append((current_title, '\n'.join(current_lines)))
                current_title = match.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # 最后一个段落
        if current_lines:
            sections.append((current_title, '\n'.join(current_lines)))

        return sections

    def _sliding_window_split(self, text: str) -> List[str]:
        """
        滑动窗口切分
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # 尽量在句号、换行等自然边界处断开
            if end < len(text):
                # 在 end 往前找最近的断句点
                for sep in ['\n\n', '\n', '。', '；', '. ', ' ']:
                    search_start = start + self.chunk_size // 2
                    if search_start >= end:
                        break
                    last_sep = text.rfind(sep, search_start, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            new_start = end - self.chunk_overlap
            if new_start <= start:
                # 防止死循环: 确保 start 至少前进 1
                new_start = end
            start = new_start

        return chunks

    def _generate_chunk_id(self, doc_id: str, section: str, index: int) -> str:
        """生成唯一的 chunk ID"""
        raw = f"{doc_id}#{section}#{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _merge_short_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """合并过短的 chunk 到前一个"""
        if len(chunks) <= 1:
            return chunks

        merged = []
        for chunk in chunks:
            if merged and len(chunk["content"]) < self.min_chunk_size:
                # 合并到前一个
                merged[-1]["content"] += "\n" + chunk["content"]
            else:
                merged.append(chunk)

        # 重新分配 index
        for i, chunk in enumerate(merged):
            chunk["chunk_index"] = i

        return merged


def load_documents(data_dir: str) -> List[Dict]:
    """加载所有原始文档"""
    docs = []
    raw_dir = Path(data_dir) / "raw"
    for json_file in raw_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
            docs.append(doc)
    return docs


def save_chunks(chunks: List[Dict], output_dir: str):
    """保存分块结果"""
    output_path = Path(output_dir) / "chunks" / "chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(chunks)} 个 chunks 到 {output_path}")


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)

    docs = load_documents(data_dir)
    print(f"加载了 {len(docs)} 篇文档")

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"共生成 {len(all_chunks)} 个 chunks")
    save_chunks(all_chunks, data_dir)
