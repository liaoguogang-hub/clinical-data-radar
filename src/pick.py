"""
选题模块（临床数据雷达）。

与 BD 雷达的关键差异：**按栏目分别选题**。
用户要求中国和全球都要，所以两个栏目各自独立跑一遍选题逻辑，
各自有每日上限——不能出现"全球大新闻把中国名额挤光"的情况。

规则：
1. >= keep 进池，>= must_publish 必选
2. 同 cluster_key（同药/同靶点）>= 2 条 → 合并成一篇 landscape 综述
3. 分数接近时优先有正文素材、有多源交叉印证的
4. 每栏目各自 daily_max 上限，宁缺毋滥
5. 已发表过的 lead_id 不重复选
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

from common import (ROOT, SCORED_DIR, load_yaml, read_json, write_json,
                    today_cst)

POSTS_DIR = ROOT / "posts"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _has_body(s: dict) -> bool:
    return len(s.get("content") or "") > 500


def _published_leads(date: str) -> set[str]:
    """任意日期已成稿的 lead_id 前 8 位。

    ⚠️ 必须跨全部日期查，不能只看今天——BD 雷达踩过这个坑：
    同一笔交易在 10-10 和 10-11 各发了一遍。
    """
    if not POSTS_DIR.exists():
        return set()
    out = set()
    for p in POSTS_DIR.rglob("*.html"):
        m = re.search(r"-([0-9a-f]{8})\.html$", p.name)
        if m:
            out.add(m.group(1))
    return out


def _evidence(s: dict) -> int:
    """素材完整度，用于同分 tie-break"""
    n = 0
    if _has_body(s):
        n += 3
    if len(s.get("sources") or []) > 1:
        n += 2              # 多家媒体交叉印证
    if s.get("access") == "full":
        n += 1
    # 有一手材料（注册库）的优先，因为能给出准确分期/样本量/NCT号
    if s.get("registry_nct"):
        n += 2
    if s.get("verdict_basis") in ("statistical", "registry_status"):
        n += 2              # 有可信统计量或状态记录
    return n


def _select_for_channel(pool: list[dict], dmax: int, published: set[str],
                        cluster_max: int = 3,
                        published_ck: set[str] | None = None,
                        covered_ev: set[tuple[str, str]] | None = None) -> list[dict]:
    """单个栏目内的选题逻辑"""
    published_ck = published_ck or set()
    covered_ev = covered_ev or set()

    # 排除已发表与低置信度
    #
    # ⚠️ 按 cluster_key 排除是必需的，不能只按 lead_id。
    # 实测踩坑：众生药业 RAY1225 的 III 期结果，摩熵医药和药时代
    # 各报一次，标题措辞差很远（模糊去重拦不住 lead_id 也不同），
    # 结果同一件事发了两篇稿。cluster_key 是"同一件临床事件"的标识，
    # 用它去重才拦得住跨源重复。
    #
    # ⚠️ covered_ev 是【跨天】的近 7 天已写过的 cluster_key：同一件事连着几天
    #    被不同源报出来时，只有它能拦住（2026-10-19 BNT113 连续 3 天进选题）。
    #
    # ⚠️ 但 `milestone` 例外：监管审批/获批是**新的进展**，不是重复报道。
    #    实测两条对照：
    #      BNT113  10-18 写过 halt   → 10-19 又是 halt   → 该排除
    #      orforglipron 10-17 readout → 10-19 milestone   → 该保留
    #    用 type 做完全匹配不行（type 会因"是否合并成综述"变化：
    #    10-18 那条 BNT113 记的是 landscape），所以只按 cluster_key，
    #    再单独豁免 milestone。
    def _atype(s: dict) -> str:
        return (s.get("article_type") or "readout").strip()

    before = len(pool)
    pool = [s for s in pool
            if s["id"][:8] not in published
            and "not_clinical" not in (s.get("flags") or [])
            and (s.get("cluster_key") or "") not in published_ck
            and not ((s.get("cluster_key") or "") in covered_ev
                     and _atype(s) != "milestone")]
    dropped = before - len(pool)
    if dropped:
        print(f"    （排除已发表/已覆盖 {dropped} 条）")

    clusters: dict[str, list[dict]] = defaultdict(list)
    singles: list[dict] = []
    for s in pool:
        ck = (s.get("cluster_key") or "").strip()
        if ck:
            clusters[ck].append(s)
        else:
            singles.append(s)

    plan: list[dict] = []

    # 1) 多条同簇 → 合并成综述
    for ck, group in clusters.items():
        if len(group) >= 2:
            group.sort(key=lambda x: (x.get("score", 0), _evidence(x)),
                       reverse=True)
            lead = group[0]
            plan.append({
                "type": "landscape",
                "cluster_key": ck,
                "lead_id": lead["id"],
                "title": lead.get("title", ""),
                "score": lead.get("score", 0),
                "channel": lead.get("channel", ""),
                "member_ids": [g["id"] for g in group[:cluster_max]],
                "members": [
                    {"title": g.get("title"), "url": g.get("url"),
                     "score": g.get("score"), "reasoning": g.get("reasoning")}
                    for g in group[:cluster_max]
                ],
                "reasoning": lead.get("reasoning", ""),
            })
        else:
            singles.append(group[0])

    # 2) 单篇按分数排
    singles.sort(key=lambda x: (x.get("score", 0), _evidence(x)),
                 reverse=True)
    for s in singles:
        if len(plan) >= dmax:
            break
        plan.append({
            "type": s.get("article_type") or "readout",
            # 保留 cluster_key：成稿后要写进 seen.json，
            # 供"同一事件、不同源"的同日去重使用（见 _select_for_channel）
            "cluster_key": s.get("cluster_key") or None,
            "lead_id": s["id"],
            "title": s.get("title", ""),
            "score": s.get("score", 0),
            "channel": s.get("channel", ""),
            "member_ids": [s["id"]],
            "members": [{"title": s.get("title"), "url": s.get("url"),
                         "score": s.get("score"),
                         "reasoning": s.get("reasoning")}],
            "reasoning": s.get("reasoning", ""),
            "verify_todo": s.get("verify_todo", []),
            "flags": s.get("flags", []),
            "endpoint_verdict": s.get("endpoint_verdict", ""),
        })

    # 3) 综述优先（landscape 放前面），且保证至少一篇有实质数据
    plan.sort(key=lambda p: (p["type"] != "landscape", -p.get("score", 0)))
    return plan[:dmax]


def run(date: str | None = None) -> dict:
    date = date or today_cst()
    cfg_all = load_yaml("scoring.yaml")
    cfg = cfg_all["score"]
    channels = cfg_all.get("channels", {}) or {}

    scored = read_json(SCORED_DIR / f"{date}.json", [])
    if not scored:
        print(f"[select] 没有评分数据: {SCORED_DIR}/{date}.json")
        return {}

    keep_thr = cfg.get("keep", 6.0)
    min_conf = cfg.get("min_confidence", 0.5)
    cluster_max = cfg.get("same_target_cluster_max", 3)
    published = _published_leads(date)
    if published:
        print(f"[select] 排除已发表 {len(published)} 篇")
    # 今天已覆盖的临床事件（cluster_key），防止跨源重复
    from dedupe import covered_events, published_cluster_keys
    published_ck = published_cluster_keys(date)
    if published_ck:
        print(f"[select] 今日已覆盖事件 {len(published_ck)} 个："
              f"{'、'.join(sorted(published_ck)[:5])}")
    # ⚠️ 跨天：近 7 天内已写过的 cluster_key（milestone 在筛选中例外）。
    covered_ev = covered_events(date, days=7)
    if covered_ev:
        print(f"[select] 近 7 天已写过的事件 {len(covered_ev)} 个（milestone 除外）")

    pool = [s for s in scored
            if s.get("score", 0) >= keep_thr
            and s.get("confidence", 0) >= min_conf]
    print(f"[select] 候选池 {len(pool)} 条（>= {keep_thr} 分）")

    # ---- 按栏目分流 ----
    # 未标栏目的按关键词兜底判断，避免因 LLM 漏标而丢失
    china_kw = channels.get("china", {}).get("keywords", []) or []

    def _chan(s: dict) -> str:
        c = (s.get("channel") or "").strip()
        if c in ("china", "global"):
            return c
        t = s.get("title", "")
        return "china" if any(k in t for k in china_kw) else "global"

    by_chan: dict[str, list[dict]] = {"china": [], "global": []}
    for s in pool:
        by_chan[_chan(s)].append(s)
    print(f"  栏目分布：中国 {len(by_chan['china'])} / "
          f"全球 {len(by_chan['global'])}")

    plan: list[dict] = []
    for ch in ("china", "global"):
        dmax = channels.get(ch, {}).get("daily_max", cfg.get("daily_max", 3))
        got = _select_for_channel(by_chan[ch], dmax, published, cluster_max,
                                  published_ck, covered_ev)
        label = channels.get(ch, {}).get("label", ch)
        print(f"\n  [{label}] 选定 {len(got)} 篇")
        for p in got:
            print(f"    · [{p['type']}] {p['score']}分  {p['title'][:50]}")
        plan.extend(got)

    # 兜底：若两个栏目都没选出（阈值过高或数据太差），
    # 放宽到全池前三，避免整天零产出
    #
    # ⚠️ 兜底也要过 covered_ev：否则"只剩重复事件"时会把已写过的
    #    同一件事再发一遍——比零产出更糟。
    if not plan and pool:
        print("\n[select] 两栏目均为空，放宽到全池前三")
        plan = _select_for_channel(
            sorted(pool, key=lambda x: -x.get("score", 0)),
            cfg.get("daily_max", 3), published, cluster_max, published_ck,
            covered_ev)

    for i, p in enumerate(plan, 1):
        p["_idx"] = i

    out = {"date": date, "count": len(plan), "plan": plan}
    write_json(SCORED_DIR / f"{date}.plan.json", out)
    print(f"\n[select] 合计选定 {len(plan)} 篇")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
