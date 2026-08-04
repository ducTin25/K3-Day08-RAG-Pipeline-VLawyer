"""Task 2 - Crawl 5 bai huong dan phap luat lao dong Viet Nam moi nhat.

Nguon duoc chon la Bao Dien tu Chinh phu. Script chi dung Python standard
library, vi vay van chay duoc khi may chua cai Crawl4AI/Chromium.

Chay:
    python -X utf8 src/task2_crawl_news.py
"""

import asyncio
import gzip
import html
import json
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Sap xep theo ngay dang giam dan, da doi chieu tai ngay 04/08/2026.
ARTICLE_URLS = [
    "https://baochinhphu.vn/hop-dong-lam-nhiem-vu-cua-cong-chuc-co-duoc-che-do-vung-dbkk-102260707095336567.htm",
    "https://baochinhphu.vn/quy-dinh-moi-ve-hop-dong-thuc-hien-cong-viec-trong-don-vi-su-nghiep-cong-lap-102260630062226594.htm",
    "https://baochinhphu.vn/hop-dong-theo-nghi-dinh-111-huong-luong-theo-thoa-thuan-102260529141445147.htm",
    "https://baochinhphu.vn/truong-hop-nguoi-nuoc-ngoai-khong-thuoc-dien-cap-giay-phep-lao-dong-102260520143311769.htm",
    "https://baochinhphu.vn/lam-hop-dong-co-duoc-huong-chinh-sach-vung-kho-khan-102260512145214742.htm",
]


def setup_directory() -> None:
    """Tao thu muc output neu chua co."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


class GovernmentNewsParser(HTMLParser):
    """Trich title, sapo va noi dung chinh tu trang baochinhphu.vn."""

    BLOCK_TAGS = {"p", "h2", "h3", "h4", "li", "blockquote"}
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.sapo_parts: list[str] = []
        self.content_parts: list[str] = []
        self.published_at = ""
        self._section: str | None = None
        self._section_depth = 0
        self._skip_until_depth: int | None = None

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key: value or "" for key, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = self._attrs(attrs)
        if tag == "meta" and values.get("property") == "article:published_time":
            self.published_at = html.unescape(values.get("content", ""))

        role = values.get("data-role")
        if self._section is None and role in {"title", "sapo", "content"}:
            self._section = role
            self._section_depth = 1
            return

        if self._section is not None:
            if tag not in self.VOID_TAGS:
                self._section_depth += 1
            is_related_box = (
                values.get("type") == "RelatedNewsBox"
                or "RelatedNewsBox" in values.get("class", "")
            )
            if self._skip_until_depth is None and (
                tag in {"script", "style", "figure"} or is_related_box
            ):
                self._skip_until_depth = self._section_depth
            elif self._skip_until_depth is None and tag in self.BLOCK_TAGS:
                prefix = "\n- " if tag == "li" else "\n"
                if tag in {"h2", "h3", "h4"}:
                    prefix += "#" * int(tag[1]) + " "
                self._append(prefix)
            elif self._skip_until_depth is None and tag == "br":
                self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._section is None:
            return
        was_skipping = self._skip_until_depth is not None
        if not was_skipping and tag in self.BLOCK_TAGS:
            self._append("\n")
        if self._skip_until_depth == self._section_depth:
            self._skip_until_depth = None
        self._section_depth -= 1
        if self._section_depth == 0:
            self._section = None

    def handle_data(self, data: str) -> None:
        if self._section is not None and self._skip_until_depth is None:
            self._append(data)

    def _append(self, value: str) -> None:
        if self._section == "title":
            self.title_parts.append(value)
        elif self._section == "sapo":
            self.sapo_parts.append(value)
        elif self._section == "content":
            self.content_parts.append(value)


def _clean_text(parts: list[str]) -> str:
    text = html.unescape("".join(parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _download_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; RAG-Lab-News-Crawler/1.0)",
            "Accept-Language": "vi-VN,vi;q=0.9",
        },
    )
    with urlopen(request, timeout=30) as response:
        body = response.read()
        if response.headers.get("Content-Encoding", "").lower() == "gzip":
            body = gzip.decompress(body)
        charset = response.headers.get_content_charset() or "utf-8"
        return body.decode(charset, errors="replace")


def _crawl_article_sync(url: str) -> dict:
    parser = GovernmentNewsParser()
    parser.feed(_download_html(url))

    title = _clean_text(parser.title_parts)
    sapo = _clean_text(parser.sapo_parts)
    body = _clean_text(parser.content_parts)
    content_markdown = "\n\n".join(part for part in (sapo, body) if part)
    if not title or len(content_markdown) < 100:
        raise ValueError(f"Khong trich duoc noi dung bai viet: {url}")

    return {
        "url": url,
        "title": title,
        "published_at": parser.published_at,
        "date_crawled": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "Bao Dien tu Chinh phu",
        "topic": "Phap luat lao dong Viet Nam dang co hieu luc",
        "content_markdown": content_markdown,
    }


async def crawl_article(url: str) -> dict:
    """Crawl mot bai viet ma khong chan event loop."""
    return await asyncio.to_thread(_crawl_article_sync, url)


async def crawl_all() -> None:
    """Crawl va luu moi bai thanh mot file JSON UTF-8."""
    setup_directory()
    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        filepath = DATA_DIR / f"article_{index:02d}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  Saved: {filepath} ({len(article['content_markdown'])} chars)")


if __name__ == "__main__":
    asyncio.run(crawl_all())
