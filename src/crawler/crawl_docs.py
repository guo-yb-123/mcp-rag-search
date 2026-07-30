"""
文档爬取模块 — 爬取云产品帮助文档
"""
import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


class DocCrawler:
    """帮助文档爬虫"""

    def __init__(
        self,
        base_url: str,
        output_dir: str = "data/raw",
        delay: float = 0.5,
        max_docs: int = 600,
    ):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.max_docs = max_docs
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        self.visited = set()
        self.doc_count = 0

    def crawl(self, start_urls: List[str]):
        """开始爬取"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        url_queue = list(start_urls)

        while url_queue and self.doc_count < self.max_docs:
            url = url_queue.pop(0)
            if url in self.visited:
                continue

            print(f"[{self.doc_count + 1}/{self.max_docs}] 爬取: {url}")
            self.visited.add(url)

            try:
                html = self._fetch(url)
                if not html:
                    continue

                doc = self._parse_doc(html, url)
                if doc and doc["content"].strip():
                    self._save_doc(doc)
                    self.doc_count += 1

                # 提取页面中的链接作为候选
                new_urls = self._extract_links(html, url)
                url_queue.extend(new_urls)

            except Exception as e:
                print(f"  错误: {e}")

            time.sleep(self.delay)

        print(f"爬取完成，共获取 {self.doc_count} 篇文档")

    def _fetch(self, url: str) -> Optional[str]:
        """获取页面 HTML"""
        try:
            resp = self.session.get(url, timeout=30)
            resp.encoding = resp.apparent_encoding or "utf-8"
            if resp.status_code == 200:
                return resp.text
            else:
                print(f"  HTTP {resp.status_code}")
                return None
        except requests.RequestException as e:
            print(f"  请求失败: {e}")
            return None

    def _parse_doc(self, html: str, url: str) -> Optional[Dict]:
        """解析文档页面"""
        soup = BeautifulSoup(html, "lxml")

        # 提取标题
        title = ""
        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # 清理标题中的站点名后缀
            title = re.sub(r'\s*[-–|]\s*.*$', '', title)

        # 提取正文（尝试常见的内容容器）
        # 优先级: 具体类名 > 通用标签; 用 min_length 过滤掉 footer/侧栏等短内容
        content_selectors = [
            (".markdown-body", 100),
            ("main", 100),
            (".doc-content", 100),
            (".article-content", 100),
            ("article", 200),      # article 常匹配到 footer，提高门槛
            (".content", 100),
            ("#content", 100),
        ]

        content_html = None
        for selector, min_len in content_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                if len(text) >= min_len:
                    content_html = str(elem)
                    break

        if not content_html:
            # 兜底：取 body
            body = soup.find("body")
            if body:
                # 移除 script, style, nav, footer
                for tag in body(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                content_html = str(body)

        if not content_html:
            return None

        # HTML → Markdown（保留 <a> 链接文字，只去掉 <img>）
        content_md = md(content_html, heading_style="ATX", strip=["img"])

        # 清理
        content_md = self._clean_content(content_md)

        # 提取分类
        category = self._extract_category(url, soup)

        doc_id = hashlib.md5(url.encode()).hexdigest()[:12]

        return {
            "doc_id": doc_id,
            "title": title,
            "url": url,
            "content": content_md,
            "category": category,
            "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _clean_content(self, text: str) -> str:
        """清理文本"""
        # 去掉多余空行
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        # 去掉只有空白的行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 去掉阿里云页面 footer 关注/联系方式
        text = re.sub(r'### 关注阿里云\n\n.*$', '', text, flags=re.DOTALL)
        text = re.sub(r'该文章对您有帮助吗\?.*$', '', text, flags=re.DOTALL)
        # 去掉首尾空白
        return text.strip()

    def _extract_category(self, url: str, soup: BeautifulSoup) -> str:
        """从 URL 或面包屑提取分类"""
        # 尝试从面包屑提取
        breadcrumb = soup.select_one(".breadcrumb, .breadcrumbs, [aria-label='breadcrumb']")
        if breadcrumb:
            parts = [a.get_text(strip=True) for a in breadcrumb.find_all("a")]
            if parts:
                return " > ".join(parts)

        # 从 URL 路径推断
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p and p not in ("zh", "doc", "document", "help")]
        if len(path_parts) >= 1:
            return path_parts[0]

        return "未分类"

    def _extract_links(self, html: str, current_url: str) -> List[str]:
        """提取页面中的文档链接"""
        soup = BeautifulSoup(html, "lxml")
        links = []

        # 只提取同域名下的链接
        base_domain = urlparse(self.base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(current_url, href)

            parsed = urlparse(full_url)
            if parsed.netloc != base_domain:
                continue

            # 去 fragment
            full_url = full_url.split("#")[0]

            # 过滤掉非文档链接（登录、API、下载等）
            skip_patterns = ["login", "signup", "download", "api.", "console.", "/tag/"]
            if any(p in full_url.lower() for p in skip_patterns):
                continue

            if full_url not in self.visited:
                links.append(full_url)

        return links

    def _save_doc(self, doc: Dict):
        """保存文档"""
        filename = f"{doc['doc_id']}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)


# ========== 预置的爬取入口 ==========

ALIYUN_START_URLS = [
    "https://help.aliyun.com/zh/ecs/",
    "https://help.aliyun.com/zh/oss/",
    "https://help.aliyun.com/zh/vpc/",
    "https://help.aliyun.com/zh/rds/",
]

FEISHU_START_URLS = [
    "https://www.feishu.cn/hc/zh-CN/categories/6947731315800522754",
    "https://www.feishu.cn/hc/zh-CN/categories/6947731315800522756",
]


if __name__ == "__main__":
    import sys

    # 选择爬取目标
    target = sys.argv[1] if len(sys.argv) > 1 else "aliyun"

    if target == "aliyun":
        crawler = DocCrawler(
            base_url="https://help.aliyun.com",
            output_dir="data/raw",
            max_docs=600,
        )
        crawler.crawl(ALIYUN_START_URLS)
    elif target == "feishu":
        crawler = DocCrawler(
            base_url="https://www.feishu.cn",
            output_dir="data/raw",
            max_docs=600,
        )
        crawler.crawl(FEISHU_START_URLS)
    else:
        print(f"未知目标: {target}，可选: aliyun, feishu")
