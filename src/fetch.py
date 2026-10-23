"""
采集器（临床版）。

源分两类：
  rss  —— 英文顶刊/媒体 + 药时代
  html —— 生物谷专科频道、摩熵、医药魔方

沿用了 BD 雷达踩出来的几个关键修复：
  1. 列表页条目【不填抓取时刻】。早期版本给列表页条目一律填 now，
     导致几个月前的旧文看起来像刚发布（把 2026-08-18 的稿子写成 10-11 头条）。
  2. 日期做二次探测：从文章页的 JSON-LD / <time> / 中文日期 / URL 路径取，
     取不到就是"未知"（age_hours 返回 -1 哨兵值），不当成"很旧"。
  3. 列表页只负责发现链接，正文与日期在第二个 pass 一起取（一次请求两用）。
"""
from __future__ import annotations

import asyncio
import html as html_lib
import json
import os
import re
import sys
from datetime import datetime, timezone

import feedparser
import httpx

try:
    from selectolax.lexbor import LexborHTMLParser as HTMLParser
except ImportError:  # selectolax 1.0 起 parser 模块被弃用
    from selectolax.parser import HTMLParser

from common import (RAW_DIR, RawItem, load_yaml, make_id, now_iso,
                    today_cst, write_json)
from factguard import split_entity_tags
from noise import is_clinical

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config", "sources.yaml")

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# ============ 列表页选择器 ============
# 只放已验证过的。新增源必须先跑 probe_sources.py 确认能取到链接。
LIST_PATTERNS = {
    # 生物谷专科频道：实测频道页 19 篇文章链接，
    # 链接形如 http://news.bioon.com/article/<hex>.html
    "bioon_trials": 'a[href*="/article/"]',
    "bioon_tumor_immunotherapy": 'a[href*="/article/"]',
    "bioon_celltherapy": 'a[href*="/article/"]',
    "bioon_genetherapy": 'a[href*="/article/"]',
    "bioon_antibodies": 'a[href*="/article/"]',
    "bioon_neuroscience": 'a[href*="/article/"]',
    "pharnexcloud": 'a[href*="/zixun/"]',
    "pharmcube": 'a[href*="/news/"]',
}

# ============ 正文容器（按站点） ============
# 实测：生物谷正文在 div.content（2600-3500 字符）
# 实测：摩熵医药正文在 div.left（1686 中文字）——注意 div.content 只有 365 字符，
#       是个诱饵；div.left 才是真正文。选错容器会抓到空壳。
ARTICLE_CONTENT = {
    "bioon": ["div.content", "div.article-content", "article"],
    "pharnexcloud": ["div.left", "div.report-content", "div.article-content"],
    "pharmcube": ["div.article-content", "div.content", "article"],
    "default": ["article", "div.content", "main"],
}

# ESMO 是 Nuxt3 SPA（serverRendered:false），正文【不在 HTML 里】，
# 而在 <script id="__NUXT_DATA__"> 的 devalue 扁平数组里。
# 实测：article 对象含 introduction / content 两个 rich_text，value 字段是正文 HTML，
#       合起来 3.6–4.3K 字符，带研究名、摘要号、NCT 号。纯 httpx 可取，无需浏览器。
RE_NUXT = re.compile(
    r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', re.S)


def _devalue(arr, i, depth=0, seen=None):
    """解 Nuxt/devalue 的整数引用。返回 Python 值。"""
    if seen is None:
        seen = set()
    if depth > 14:
        return None
    if isinstance(i, (str, float, bool)) or i is None:
        return i
    if not isinstance(i, int):
        return i
    if i < 0 or i >= len(arr) or i in seen:
        return None
    seen = seen | {i}
    v = arr[i]
    if isinstance(v, (str, float, bool)) or v is None:
        return v
    if isinstance(v, int):
        return _devalue(arr, v, depth + 1, seen)
    if isinstance(v, list):
        # Nuxt 的包装类型：[["ShallowReactive", n]] / ["Reactive", n] / ["Set", n]
        if v and isinstance(v[0], str) and v[0] in (
                "ShallowReactive", "Reactive", "Ref", "ShallowRef", "Set", "Map"):
            return _devalue(arr, v[1], depth + 1, seen) if len(v) > 1 else None
        return [_devalue(arr, x, depth + 1, seen) for x in v]
    if isinstance(v, dict):
        return {k: _devalue(arr, x, depth + 1, seen) for k, x in v.items()}
    return None


def extract_esmo_body(html_text: str) -> tuple[str, str]:
    """从 ESMO 页面取正文。返回 (正文, 发布日期)。取不到返回 ("","")。"""
    m = RE_NUXT.search(html_text or "")
    if not m:
        return "", ""
    try:
        arr = json.loads(m.group(1))
    except Exception:
        return "", ""
    for idx, v in enumerate(arr):
        # 文章对象同时带 introduction 和 content
        if not (isinstance(v, dict) and "introduction" in v and "content" in v):
            continue
        obj = _devalue(arr, idx)
        if not isinstance(obj, dict):
            continue
        parts = []
        for key in ("introduction", "content"):
            node = obj.get(key)
            val = node.get("value") if isinstance(node, dict) else None
            if isinstance(val, str) and val.strip():
                parts.append(val)
        if not parts:
            continue
        raw = "\n".join(parts)
        raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"<br\s*/?>|</p>|</li>|</h[1-6]>", "\n", raw, flags=re.I)
        txt = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw))
        txt = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in txt.split("\n"))
        txt = re.sub(r"\n{2,}", "\n", txt).strip()
        pub = obj.get("publish_date")
        pub = (pub.get("value") if isinstance(pub, dict) else "") or ""
        return _strip_nav(txt), pub
    return "", ""

# ⚠️ JSON-LD 的 type 属性不一定紧跟 <script>，早先写死
# `<script[^>]+type="application/ld+json"` 会漏掉属性顺序不同的页面，
# 导致大量条目取不到日期。改为任意属性顺序都能匹配。
RE_JSONLD = re.compile(
    r"<script[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.I | re.S)
RE_TIME = re.compile(r"<time[^>]+datetime=\"([^\"]+)\"", re.I)
RE_PUBLISHED_EN = re.compile(
    r"(?:Published|Posted|Updated)[:\s]*([A-Z][a-z]{2,8}\.?\s+\d{1,2},\s+20\d\d)",
    re.I)
RE_PUBLISHED_CN = re.compile(
    r"(20\d\d)[-年/\.](\d{1,2})[-月/\.](\d{1,2})日?", re.I)
RE_URL_DATE = re.compile(r"/(20\d\d)/(\d{1,2})/(\d{1,2})/")

# ============ WAF / 验证码检测 ============
#
# 实测（生物谷）：连续抓 6 个专科频道后触发腾讯 WAF，
# 之后**所有路径**（首页、频道、文章、RSS）都返回同一个 1681 字节的
# 验证码页，HTTP 状态码仍是 200。
#
# 这类拦截最危险的地方是【假装成功】：
#   - HTTP 200，不会报错
#   - 正文长度很小，但短页面本身也可能是正常的
# 如果不显式识别，表现就是"这个源今天突然没内容了"，
# 会被误诊成选择器坏了或网站改版——BD 雷达就在这个坑里绕过一圈。
WAF_MARKERS = re.compile(
    r"TCaptcha|captcha\.qq\.com|waf-captcha|x-waf-captcha|"
    r"请完成安全验证|人机验证|滑动验证|"
    r"cf-browser-verification|__cf_chl|Just a moment",
    re.I)


def looks_blocked(html: str) -> bool:
    """识别 WAF/验证码拦截页。注意 HTTP 状态码是 200，不能靠状态码判断。"""
    h = html or ""
    if len(h) > 60000:          # 正常页面远大于此，先短路
        return False
    return bool(WAF_MARKERS.search(h))


# 标题前导日期/栏目名，需要剥掉
LEADING_DATE = re.compile(
    r"^\s*(?:20\d\d[-年/\.]\d{1,2}[-月/\.]\d{1,2}日?|\d{1,2}[-月/]\d{1,2}日)"
    r"\s*[:：\-—]?\s*")
LEADING_TAG = re.compile(r"^\s*【[^】]{1,10}】\s*")


def clean_title(t: str) -> str:
    """剥掉标题前导的日期与栏目标签"""
    t = re.sub(r"\s+", " ", t or "").strip()
    for _ in range(2):
        t = LEADING_DATE.sub("", t)
        t = LEADING_TAG.sub("", t)
    return t.strip()


def _norm_date(y, m, d) -> str:
    try:
        return datetime(int(y), int(m), int(d),
                        tzinfo=timezone.utc).isoformat()
    except Exception:
        return ""


def extract_pub_date(html: str, url: str = "") -> str:
    """
    从文章页抽发布日期。取不到返回 ""（未知），**绝不返回当前时刻**。

    顺序：JSON-LD datePublished → <time datetime> → 英文 Published →
          中文日期 → URL 路径日期
    """
    # 1. JSON-LD（可能嵌套在 @graph 里）
    for block in RE_JSONLD.findall(html or "")[:6]:
        try:
            data = json.loads(block.strip())
        except Exception:
            # 有的页面 JSON-LD 里有未转义字符，退化成正则直接取
            m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', block)
            if m:
                try:
                    return datetime.fromisoformat(
                        m.group(1).replace("Z", "+00:00")).isoformat()
                except Exception:
                    pass
            continue
        nodes = data if isinstance(data, list) else [data]
        # 展开 @graph
        flat = []
        for nd in nodes:
            if isinstance(nd, dict) and isinstance(nd.get("@graph"), list):
                flat.extend(nd["@graph"])
            else:
                flat.append(nd)
        for nd in flat:
            if not isinstance(nd, dict):
                continue
            for k in ("datePublished", "dateCreated", "dateModified"):
                v = nd.get(k)
                if v and isinstance(v, str):
                    try:
                        return datetime.fromisoformat(
                            v.replace("Z", "+00:00")).isoformat()
                    except Exception:
                        m = RE_PUBLISHED_CN.search(v)
                        if m:
                            return _norm_date(*m.groups())
    # 2. <time datetime>
    m = RE_TIME.search(html or "")
    if m:
        try:
            return datetime.fromisoformat(
                m.group(1).replace("Z", "+00:00")).isoformat()
        except Exception:
            pass
    # 3. 英文 "Published Aug. 18, 2026"
    m = RE_PUBLISHED_EN.search(html or "")
    if m:
        try:
            return datetime.strptime(
                m.group(1).replace(".", ""),
                "%B %d, %Y").replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            try:
                return datetime.strptime(
                    m.group(1).replace(".", ""),
                    "%b %d, %Y").replace(tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    # 4. 中文日期（取正文里第一个像发布日期的）
    m = RE_PUBLISHED_CN.search((html or "")[:6000])
    if m:
        return _norm_date(*m.groups())
    # 5. URL 路径
    m = RE_URL_DATE.search(url or "")
    if m:
        return _norm_date(*m.groups())
    return ""


def _strip_nav(text: str) -> str:
    """
    去掉正文里混进来的导航/版权/推广/作者块尾巴。

    实测（摩熵医药）：正文容器 div.left 里前面贴着作者简介
      "张鹏 · 新药研发分析师 张 张鹏 新药研发分析师 审核：王正晴 · 药学博士
       关注全球创新药研发管线…累计发布研发分析 102+ 篇。"
    直接当正文用会污染成稿，必须清掉。
    """
    noise = re.compile(
        r"版权所有|免责声明|转载请注明|扫码关注|点击阅读原文|"
        r"热门推荐|相关阅读|猜你喜欢|投稿|广告|友情链接|"
        r"新药研发分析师|分析师|审核：|药学博士|"
        r"累计发布|关注全球|专注于|"
        r"©\s*20\d\d|All rights reserved|Cookie|ICP备|公网安备|"
        r"上一篇|下一篇|返回列表|分享到|阅读\s*\d+", re.I)
    out = []
    for line in re.split(r"[\n\r]+", text):
        s = line.strip()
        if not s or len(s) < 2:
            continue
        if noise.search(s):
            continue
        out.append(s)
    return "\n".join(out)


def _text_ratio(html: str) -> float:
    """正文密度，用于挑最佳容器"""
    if not html:
        return 0.0
    txt = re.sub(r"<[^>]+>", "", html)
    return len(txt) / max(len(html), 1)


async def fetch_article(client: httpx.AsyncClient, url: str,
                        source_id: str) -> tuple[str, str]:
    """
    抓文章页，返回 (正文, 发布日期)。一次请求两用。
    """
    try:
        r = await client.get(url, timeout=25)
    except Exception:
        return "", ""
    if r.status_code != 200 or len(r.text) < 500:
        return "", ""
    if looks_blocked(r.text):
        return "", ""
    pub = extract_pub_date(r.text, url)

    site = source_id.split("_")[0]
    # ESMO：正文在 __NUXT_DATA__ 里，常规模板选择器一个也命中不了（实测 0/16）
    if site == "esmo":
        body, npub = extract_esmo_body(r.text)
        return body, (npub or pub)

    selectors = ARTICLE_CONTENT.get(site, ARTICLE_CONTENT["default"])
    html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>",
                  " ", r.text)
    try:
        tree = HTMLParser(html)
    except Exception:
        return "", pub
    best, blen = "", 0
    for sel in selectors:
        for nd in tree.css(sel)[:2]:
            t = re.sub(r"[ \t]+", " ", nd.text()).strip()
            if len(t) > blen:
                best, blen = t, len(t)
        if blen > 800:
            break
    return _strip_nav(best), pub


# ============ 列表页解析 ============
async def fetch_rss(client: httpx.AsyncClient, src: dict) -> list[RawItem]:
    try:
        r = await client.get(src["url"], timeout=25)
    except Exception as e:
        print(f"  [{src['id']}] RSS 失败: {type(e).__name__}")
        return []
    if r.status_code != 200:
        print(f"  [{src['id']}] RSS HTTP {r.status_code}")
        return []
    feed = feedparser.parse(r.content)
    out: list[RawItem] = []
    for e in feed.entries[:40]:
        title = clean_title(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        if not title or not link:
            continue
        summary = re.sub(
            r"<[^>]+>", " ",
            getattr(e, "summary", "") or getattr(e, "description", "") or "")
        summary = re.sub(r"\s+", " ", summary).strip()[:800]
        # RSS 的 published 可信，直接用；没有就是未知（不要填 now）
        pub = ""
        for k in ("published", "updated"):
            v = getattr(e, k, "")
            if v:
                try:
                    pub = datetime(*getattr(e, f"{k}_parsed")[:6],
                                   tzinfo=timezone.utc).isoformat()
                    break
                except Exception:
                    pass
        # RSS 分类/栏目。ESMO 用它在噪声过滤里区分
        # Scientific News / Drug Approval（要）与 Interview / Editorial（不要）。
        tags = [t.get("term", "").strip() for t in (getattr(e, "tags", None) or [])
                if getattr(t, "term", "")]
        out.append(RawItem(
            id=make_id(link, title), source=src["id"], lang=src.get("lang", "en"),
            title=title, url=link, published=pub,
            access=src.get("access", "rss"), mode="rss",
            summary=summary, weight=src.get("weight", 1.0),
            topics=tags or list(src.get("topics") or []),
            fetch_time=now_iso()))
    return out


async def fetch_html_list(client: httpx.AsyncClient, src: dict) -> list[RawItem]:
    sel = LIST_PATTERNS.get(src["id"])
    try:
        r = await client.get(src["url"], timeout=25)
    except Exception as e:
        print(f"  [{src['id']}] 列表失败: {type(e).__name__}")
        return []
    if r.status_code != 200:
        print(f"  [{src['id']}] 列表 HTTP {r.status_code}")
        return []
    if looks_blocked(r.text):
        print(f"  [{src['id']}] ⚠️ 被 WAF/验证码拦截（HTTP 200 但是拦截页）")
        return []
    try:
        tree = HTMLParser(r.text)
    except Exception:
        return []
    seen, out = set(), []
    for a in (tree.css(sel) if sel else tree.css("a")):
        href = (a.attributes.get("href") or "").strip()
        title = clean_title(a.text() or "")
        if not href or not title:
            continue
        if not href.startswith("http"):
            if href.startswith("/"):
                href = src["url"].split("/")[0] + "//" + \
                       src["url"].split("/")[2] + href
            else:
                continue
        if len(title) < 12 or len(title) > 120:
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append(RawItem(
            id=make_id(href, title), source=src["id"], lang=src.get("lang", "cn"),
            title=title, url=href,
            # ⚠️ 列表页日期一律留空，交给探测环节。填 now 会让旧文变"新文"
            published="",
            access=src.get("access", "full"), mode="html",
            weight=src.get("weight", 1.0), fetch_time=now_iso()))
    return out


def _drop_stale(items: list[RawItem], max_age_days: int):
    """丢弃发布日期超过阈值者。日期未知的不丢（未知≠很旧）"""
    kept, dropped = [], 0
    for it in items:
        h = it.age_hours
        if h >= 0 and h > max_age_days * 24:
            dropped += 1
            continue
        kept.append(it)
    return kept, dropped


async def run(date: str | None = None) -> list[RawItem]:
    date = date or today_cst()
    cfg = yaml_load(CONFIG)
    sources = [s for s in cfg.get("sources", [])
               if s.get("mode") in ("rss", "html")
               and s.get("access") not in ("blocked", "disabled")]

    print(f"=== 采集 {date}：{len(sources)} 个启用源 ===")
    items: list[RawItem] = []
    async with httpx.AsyncClient(headers=UA, follow_redirects=True,
                                 timeout=25) as client:
        # --- 1. 列表/RSS ---
        # ⚠️ 同域请求之间加延迟。
        # 实测：连续请求 bioon.com 的 6 个频道（无间隔）触发腾讯 WAF，
        # 之后全站返回验证码页。降频是唯一可靠的规避手段。
        last_host = ""
        for src in sources:
            host = src["url"].split("/")[2] if "//" in src["url"] else ""
            if host and host == last_host:
                await asyncio.sleep(2.0)
            last_host = host
            fn = fetch_rss if src["mode"] == "rss" else fetch_html_list
            got = await fn(client, src)
            print(f"  [{src['id']:28}] {len(got):>3} 条")
            items.extend(got)

        print(f"\n=== 初步 {len(items)} 条 ===")

        # --- 2. 噪声过滤（临床）---
        kept, dropped = [], {}
        for it in items:
            ok, why = is_clinical(it.title, it.source, it.topics)
            if ok:
                kept.append(it)
            else:
                dropped[why] = dropped.get(why, 0) + 1
        print(f"=== 临床过滤：{len(items)} -> {len(kept)} 条（丢弃 {len(items)-len(kept)}）===")
        for why, n in sorted(dropped.items(), key=lambda x: -x[1]):
            print(f"     {why:28} {n}")
        items = kept

        # --- 3. 正文 + 日期（一次请求两用）---
        quota = (cfg.get("fetch", {}) or {}).get("article_quota", {}) or {}
        # fetch_body: true 的源即使走 RSS 也要抓正文
        # （ESMO 的 RSS summary 大面积为空，不抓正文就只有标题）
        body_srcs = {s["id"] for s in sources if s.get("fetch_body")}
        need = [i for i in items
                if i.access == "full" and (i.mode == "html" or i.source in body_srcs)]
        # 每个源限量，避免单源刷满
        by_src: dict[str, int] = {}
        picked = []
        for it in need:
            cap = quota.get(it.source, 8)
            if by_src.get(it.source, 0) >= cap:
                continue
            by_src[it.source] = by_src.get(it.source, 0) + 1
            picked.append(it)
        print(f"\n=== 抓正文+日期 {len(picked)} 条 ===")
        sem = asyncio.Semaphore(8)

        async def _one(it: RawItem):
            async with sem:
                body, pub = await fetch_article(client, it.url, it.source)
            if pub:
                it.published = pub
            if body and len(body) > 300:
                # ⚠️ 新闻站常把自动实体标注塞在正文最前面。留着它，成稿模型会
                #    把标签里的靶点/公司当成事实依据。这里剥离，标签单独存
                #    （见 factguard.split_entity_tags 的说明与事故记录）。
                body2, etags = split_entity_tags(body)
                it.entity_tags = etags
                it.content = body2
                it.access = "full"
            elif not it.summary:
                it.access = "headline"

        await asyncio.gather(*[_one(i) for i in picked])
        got_body = sum(1 for i in picked if i.content)
        got_date = sum(1 for i in picked if i.published)
        print(f"  取得正文 {got_body}/{len(picked)}，取得日期 {got_date}/{len(picked)}")

        # --- 4. 陈旧过滤 ---
        max_age = (cfg.get("fetch", {}) or {}).get("max_age_days", 21)
        before = len(items)
        items, stale = _drop_stale(items, max_age)
        print(f"\n=== 陈旧过滤：丢 {stale} 条（>{max_age}天），{before} -> {len(items)} ===")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RAW_DIR / f"{date}.json", [i.to_dict() for i in items])
    print(f"\n共 {len(items)} 条 -> data/raw/{date}.json")
    ac: dict[str, int] = {}
    for i in items:
        ac[i.access] = ac.get(i.access, 0) + 1
    print("  access 分布:", ac)
    return items


def yaml_load(p):
    import yaml
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run(d))
