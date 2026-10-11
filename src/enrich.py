"""
一手材料补全（临床版）。

在评分之后、成稿之前运行：
  1. 取评分达标的条目（默认 >= keep 阈值）
  2. 从标题/正文抽检索词 → ClinicalTrials.gov 找对应试验
  3. 把注册库数值与新闻措辞合起来做终点判定
  4. 存 data/enriched/<date>.json，供成稿阶段使用

为什么不放在评分之前（BD 雷达的 EDGAR 是放在采集后的）：
注册库查询每个条目要 2-6 次 API 请求，83 条全做要 300+ 次，
既慢又浪费——大部分条目根本不会入选。
先按标题评分（便宜），只给可能入选的做一手材料（贵），
这与 BD 雷达"补全名额有限必须按重要性排序"是同一条教训。

⚠️ 判定结果必须带来源层级（statistical / textual / numeric），
成稿阶段要据此标注"据注册库统计分析"还是"据新闻措辞"。
"""
from __future__ import annotations

import asyncio
import sys

from common import (DATA_DIR, RawItem, SCORED_DIR, load_yaml, read_json,
                    today_cst, write_json)
from registry import enrich as registry_enrich
from verdict import reconcile, to_prompt_block

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENRICHED_DIR = DATA_DIR / "enriched"


async def run(date: str | None = None, max_items: int = 20,
              concurrency: int = 4) -> dict:
    date = date or today_cst()
    cfg = load_yaml("scoring.yaml")
    keep = cfg["score"].get("keep", 6.0)

    scored = read_json(SCORED_DIR / f"{date}.json", [])
    if not scored:
        print(f"[enrich] 没有评分数据: {SCORED_DIR / f'{date}.json'}")
        return {}

    # 只给"可能入选"的做一手材料
    cands = [s for s in scored
             if s.get("score", 0) >= keep
             and s.get("confidence", 0) >= cfg["score"].get("min_confidence", 0.5)
             and "not_clinical" not in (s.get("flags") or [])]
    cands.sort(key=lambda x: -x.get("score", 0))
    cands = cands[:max_items]

    print(f"\n=== 一手材料补全：从 {len(scored)} 条中取 {len(cands)} 条"
          f"（>= {keep} 分）===")

    out: dict[str, dict] = {}
    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0}

    async def _one(s: dict):
        async with sem:
            title = s.get("title", "")
            body = s.get("content") or s.get("summary") or ""
            try:
                reg = await registry_enrich(title, body)
            except Exception as e:
                reg = {"trials": [], "best": None,
                       "note": f"检索异常 {type(e).__name__}"}
            v = reconcile(title, body, reg.get("best"))
        done["n"] += 1
        best = reg.get("best") or {}
        nct = best.get("nct") or ""
        tag = f"{nct} {best.get('phase','')}" if nct else "未命中"
        print(f"  [{done['n']:>2}/{len(cands)}] {s.get('score'):>4} "
              f"{v.get('verdict','?'):20} {tag:18} {title[:40]}")
        out[s.get("id", "")] = {
            "id": s.get("id", ""),
            "title": title,
            "score": s.get("score"),
            "channel": s.get("channel", ""),
            "verdict": v,
            "verdict_block": to_prompt_block(v),
            "registry": {
                "best": best,
                "trials": reg.get("trials") or [],
                "terms": reg.get("terms") or {},
                "note": reg.get("note") or "",
            },
        }

    await asyncio.gather(*[_one(s) for s in cands])

    # 统计
    hits = [v for v in out.values() if (v["registry"]["best"] or {}).get("nct")]
    stat = [v for v in out.values()
            if v["verdict"].get("basis") == "statistical"]
    print(f"\n=== 补全结果 ===")
    print(f"  注册库命中 {len(hits)}/{len(out)} 条")
    print(f"  其中拿到统计检验（可信） {len(stat)} 条")
    from collections import Counter
    vc = Counter(v["verdict"].get("verdict", "?") for v in out.values())
    print(f"  判定分布: {dict(vc)}")

    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ENRICHED_DIR / f"{date}.json", out)
    print(f"  -> data/enriched/{date}.json")
    return out


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else None))
