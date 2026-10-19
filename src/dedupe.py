"""
去重模块。

两个层级：
1. 完全重复 —— 同 URL 或同 hash
2. 语义重复 —— 不同媒体报同一件事（标题相似度）

用 rapidfuzz 做标题近似，因为同一条交易在不同媒体的标题
往往只差几个词（"Novo to pay Hengrui up to $2.6B for a
once-weekly obesity pill" vs "Novo Nordisk licenses Hengrui
oral GLP-1 for $2.6 billion"）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz import process

from common import (RawItem, RAW_DIR, SCORED_DIR, SEEN_FILE, load_yaml,
                    read_json, write_json, today_cst)


def normalize_title(t: str) -> str:
    """去掉标点、公司后缀等干扰，让相似度更准"""
    t = t.lower()
    t = re.sub(r"\s+[—\-|·].*$", "", t)          # 去掉破折号后的副标题
    t = re.sub(r"\(.*?\)", "", t)                  # 去括号
    t = re.sub(r"[^\w\s一-鿿]", " ", t)       # 去标点（保留中文）
    t = re.sub(r"\b(inc|corp|ltd|plc|co|company|holdings|therapeutics|"
               r"biosciences|biopharma|pharmaceuticals?)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def entity_hint(t: str) -> str:
    """抽取公司名片段，作为辅助判据"""
    caps = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", t)
    stop = {"The", "This", "That", "With", "From", "After", "What", "Why",
            "How", "Its", "His", "Her", "They", "Their", "Both", "More",
            "Here", "Now", "New", "First", "Two", "Three", "Four", "Five"}
    caps = [c for c in caps if c not in stop]
    return " ".join(caps[:4]).lower()


def dedupe(items: list[RawItem], threshold: int = 82) -> list[RawItem]:
    """
    返回去重后的列表，被合并的来源会记到 keep.sources 里（用 extra_sources 字段）。
    """
    # 按发布时间倒序，保留最新的
    items = sorted(items, key=lambda x: x.published, reverse=True)

    kept: list[RawItem] = []
    norms: list[str] = []
    url_seen: set[str] = set()
    hash_seen: set[str] = set()
    merged: dict[str, list[str]] = {}

    for it in items:
        base = it.url.split("?")[0]
        if it.id in hash_seen or base in url_seen:
            # 完全重复，只记录来源
            for k in kept:
                if k.id == it.id or k.url.split("?")[0] == base:
                    merged.setdefault(k.id, []).append(it.source)
                    break
            continue

        n = normalize_title(it.title)
        dup_of = None
        if n:
            m = process.extractOne(n, norms, scorer=fuzz.token_set_ratio,
                                   score_cutoff=threshold)
            if m:
                dup_of = m[0]

        if dup_of is not None:
            for k in kept:
                if normalize_title(k.title) == dup_of:
                    merged.setdefault(k.id, []).append(it.source)
                    break
        else:
            kept.append(it)
            norms.append(n)
            url_seen.add(base)
            hash_seen.add(it.id)

    # 把合并的来源写回去
    for it in kept:
        extra = merged.get(it.id, [])
        if extra:
            it.topics = list(set(it.topics + [f"also:{s}" for s in extra]))

    return kept


def filter_history(items: list[RawItem], today: str | None = None) -> list[RawItem]:
    """
    跨天去重：只丢弃"已经写成稿"的条目。

    ⚠️ 两个踩过的坑，都在这里：

    【坑一】seen.json 早期不带日期，导致同一天的重跑把当天所有新闻
    都当成"已报道"。工作流每天跑两次，第二次永远产出 0 篇。
    实测：139 条被丢 110 条，只剩 2 条，最终 0 篇成稿。
    修法：条目带日期。

    【坑二】更隐蔽——去重基准用的是"抓过的"而不是"写过的"。
    一天抓 100+ 条但只写 2-3 篇，剩下 97 条明明没写过，
    却被永久标记为"已报道"。次日它们仍在源站首页，直接被过滤，
    于是天天产出 0 篇。
    实测：次日采集 143 条 → 丢弃 105 条 → 只剩 8 条 → 选题 0 篇。
    修法：seen.json 只记录【已发表】的选题（见 record_published），
    其余条目每次照常进入评分，让没入选的高分条目有机会第二天补上。
    """
    today = today or today_cst()
    data = read_json(SEEN_FILE, {})
    entries = _load_entries(data)

    # 已发表选题：不过滤同日的（同日重跑靠 compose 的文件幂等兜住）
    old = [e for e in entries if e.get("date", "") < today]
    # ⚠️ 按 8 位前缀比对，不按全量 id。
    # seen.json 里的 id 有两种来源：
    #   - compose 写入的是完整 hash（16 位）
    #   - 从 posts 文件名反查重建时只能拿到 8 位前缀
    # 早先按全量比对，导致 8 位的历史条目永远匹配不上，
    # 同一笔交易在 10-10 和 10-11 各发了一遍（诺诚健华×礼来）。
    seen_ids = {str(e["id"])[:8] for e in old if e.get("id")}
    seen_titles = [normalize_title(e["title"]) for e in old if e.get("title")]

    out, dropped = [], 0
    for it in items:
        if it.id[:8] in seen_ids:
            dropped += 1
            continue
        if seen_titles:
            m = process.extractOne(normalize_title(it.title), seen_titles,
                                   scorer=fuzz.token_set_ratio, score_cutoff=90)
            if m:
                dropped += 1
                continue
        out.append(it)

    if dropped:
        print(f"  历史去重：丢弃 {dropped} 条（往日已发表 {len(old)} 篇为基准）")
    else:
        print(f"  历史去重：基准 {len(old)} 篇往日已发表，本轮无丢弃")
    return out


def _load_entries(data: dict) -> list[dict]:
    """
    读 seen 条目。只接受 v3。

    版本历史（每次都是因为语义或格式错了才升版）：
      v1  {"ids":[], "titles":[]}
          无日期。同日重跑会把当天所有新闻当成"已报道"。
      v2  {"entries":[{id,title,date}]}
          有日期了，但记录的是【抓取到的全部条目】，
          导致没写过的高分条目被永久埋掉。
      v3  {"entries":[{id,title,date}]}
          记录【已发表选题】。格式与 v2 相同，语义不同，
          所以按 version 字段严格区分。

    旧版一律丢弃：宁可让最近一轮多评几条，也不能让错误的基准
    继续把好条目挡在外面。
    """
    if not isinstance(data, dict):
        return []
    ver = data.get("version")
    if ver == 3:
        return [e for e in data.get("entries", []) if isinstance(e, dict)]
    if data.get("entries") or data.get("ids") or data.get("titles"):
        print(f"  [seen] 检测到旧版格式 v{ver}，已丢弃"
              f"（语义已变更：只记录已发表选题）")
    return []


def record_published(topics: list[dict], today: str | None = None,
                     cap: int = 2000) -> None:
    """
    记录【已发表】的选题，作为跨天去重的基准。

    ⚠️ 只记录真正写成稿的，不要记录抓取到的全部条目。
    记录全部会把"没写过的高分条目"永久埋掉——见 filter_history 的坑二。

    ⚠️ 必须一并记录 cluster_key。同一件临床事件常被多个源报道，
    标题措辞差异大（模糊匹配拦不住），但 cluster_key 相同。
    不记 cluster_key 的话，同一件事会被不同源各发一遍。

    topics: [{"lead_id":..., "title":..., "cluster_key":...}, ...]
    """
    today = today or today_cst()
    data = read_json(SEEN_FILE, {})
    entries = _load_entries(data)

    known = {e.get("id") for e in entries}
    added = 0
    for t in topics:
        tid = t.get("lead_id") or t.get("id")
        title = t.get("title") or ""
        if not tid or tid in known:
            continue
        # ⚠️ 必须连【事件类型】一起记（2026-10-19 补）。
        #    跨天去重要回答的是"这是不是同一件新闻"，
        #    而 cluster_key（drug:xxx）只说明"同一个药"——
        #    同一个药会接连出不同事件，不能一律当重复：
        #      orforglipron: 10-17 readout → 10-19 milestone  ← 两条都要写
        #      BNT113:       10-18 halt    → 10-19 halt       ← 同一条，要排除
        #    判据是 (cluster_key, type) 一致，不是 cluster_key 一致。
        entries.append({"id": tid, "title": title, "date": today,
                        "cluster_key": t.get("cluster_key") or "",
                        "type": t.get("type") or "",
                        "verdict": t.get("endpoint_verdict") or ""})
        known.add(tid)
        added += 1

    entries = entries[-cap:]
    write_json(SEEN_FILE, {"version": 3, "entries": entries})
    print(f"  [seen] 记录已发表 {added} 篇（库中共 {len(entries)} 篇）")


def published_cluster_keys(today: str) -> set[str]:
    """
    今天已发表过的 cluster_key 集合。

    用途：防止"同一件临床事件、不同源的不同标题"被重复写成两篇。
    实测踩坑：众生药业 RAY1225 的 III 期结果，摩熵医药和药时代
    各报一次，标题措辞差很远（模糊去重拦不住），于是发了两篇。

    只限今天——跨天用 `covered_events()`（按事件类型判，见下）。
    为什么跨天不能只按 cluster_key：同一个药会接连出不同事件
    （II 期今天、III 期下个月），那些都是不同的新闻。
    """
    data = read_json(SEEN_FILE, {})
    out = set()
    for e in _load_entries(data):
        if e.get("date") == today and e.get("cluster_key"):
            out.add(str(e["cluster_key"]))
    return out


def covered_events(today: str, days: int = 14) -> set[tuple[str, str]]:
    """
    近 N 天内已发表过的 (cluster_key, 事件类型) 组合——**跨天**去重。

    ⚠️ 为什么需要它（2026-10-19 实测事故）：
       BNT113 的"PFS 终点未达"连续 3 天进入选题：
         10-17 cluster_key=drug:BNT113, lead=dd92cea6（进选题未成稿）
         10-18 cluster_key=drug:bnt113, lead=dd92cea6（写成稿）
         10-19 cluster_key=drug:bnt113, lead=abf57879（又进选题）
       10-19 换了 lead_id，所以 `published`（按 id[:8]）拦不住；
       `filter_history` 的标题模糊匹配也拦不住——因为 seen.json 里
       存的是**改写后的中文标题**，而新条目是**英文原标题**，跨语种
       的 token_set_ratio 接近 0。而 cluster_key 明明存了却没用于跨天。

    关键是带上【事件类型】：只按 cluster_key 会把"同药的新进展"误杀
      · ("drug:orforglipron", "readout")   vs ("drug:orforglipron", "milestone")
        → 不同事件，都该写
      · ("drug:bnt113", "halt")            vs ("drug:bnt113", "halt")
        → 同一事件重复报道，排除
    """
    from datetime import date as _date, timedelta as _td
    try:
        cut = (_date.fromisoformat(today) - _td(days=days)).isoformat()
    except Exception:
        return set()
    out: set[tuple[str, str]] = set()
    for e in _load_entries(read_json(SEEN_FILE, {})):
        d = str(e.get("date") or "")
        ck = str(e.get("cluster_key") or "")
        if not ck or not d:
            continue
        if cut <= d < today:                      # 只算"之前 N 天内"
            out.add((ck, str(e.get("type") or "")))
    return out


def run(date: str | None = None) -> list[RawItem]:
    date = date or today_cst()
    cfg = load_yaml("sources.yaml")
    thr = cfg.get("dedupe", {}).get("threshold", 82)

    raw = read_json(RAW_DIR / f"{date}.json", [])
    if not raw:
        print(f"[dedupe] 没有原始数据: {RAW_DIR}/{date}.json")
        return []

    items = [RawItem.from_dict(d) for d in raw]
    print(f"[dedupe] 输入 {len(items)} 条")

    items = filter_history(items, today=date)
    before = len(items)
    items = dedupe(items, threshold=thr)
    print(f"[dedupe] 去重后 {len(items)} 条（合并 {before - len(items)} 条）")

    # 注意：这里不写 seen.json。
    # seen 记录的是【已发表】选题，由 compose 成功后调用 record_published。
    write_json(RAW_DIR / f"{date}.deduped.json", [i.to_dict() for i in items])
    return items


if __name__ == "__main__":
    # Windows 控制台默认 GBK，输出特殊符号会崩
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    run(sys.argv[1] if len(sys.argv) > 1 else None)