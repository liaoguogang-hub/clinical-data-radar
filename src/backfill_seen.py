"""
一次性迁移：给 seen.json 的历史条目回填 type / verdict。

# 为什么需要（2026-10-19）

`dedupe.covered_events()` 跨天去重靠 (cluster_key, 事件类型) 匹配。
但 2026-10-19 之前写入 seen.json 的条目**只有 cluster_key、没有 type**，
于是算出来是 ("drug:bnt113", "")，和 ("drug:bnt113", "halt") 永远不相等
—— 过滤条件写了却不起作用，BNT113 这类重复仍会进选题。

这个脚本从 `data/scored/*.plan.json`（里面有 type / endpoint_verdict）
按 lead_id 回填到 seen.json 的历史条目上。

安全说明：
  · 只补**缺失**字段，已有 type 的条目不动
  · 匹配用 8 位短 id 和 16 位全 id 两种（seen 里两种都出现过）
  · 找不到对应 plan 条目的（多半是早期已清理的日期）保持原样，只是回填不到
  · 幂等：重复跑结果一样

用法：
    cd src
    python backfill_seen.py            # 只报告，不写
    python backfill_seen.py --apply    # 真的写回
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SCORED_DIR, SEEN_FILE, read_json, write_json  # noqa: E402
from dedupe import _load_entries  # noqa: E402


def _plan_index() -> dict[str, dict]:
    """lead_id → plan 条目（同时登记 8 位短 id 和 16 位全 id）。"""
    idx: dict[str, dict] = {}
    for p in sorted(SCORED_DIR.glob("*.plan.json")):
        data = read_json(p, {}) or {}
        for it in (data.get("plan") or []):
            lid = str(it.get("lead_id") or "")
            if not lid:
                continue
            idx[lid] = it
            idx[lid[:8]] = it
    return idx


def main() -> int:
    apply = "--apply" in sys.argv
    idx = _plan_index()
    if not idx:
        print("✗ data/scored/ 下没有任何 plan.json，无法回填")
        return 1

    data = read_json(SEEN_FILE, {}) or {}
    entries = _load_entries(data)
    print(f"seen.json 现有 {len(entries)} 条；plan 索引 {len(set(id(x) for x in idx.values()))} 条")

    filled_t = filled_v = missed = 0
    for e in entries:
        eid = str(e.get("id") or "")
        it = idx.get(eid) or idx.get(eid[:8])
        if not it:
            if not e.get("type"):
                missed += 1
            continue
        if not e.get("type"):
            e["type"] = str(it.get("type") or "")
            filled_t += 1
        if not e.get("verdict"):
            e["verdict"] = str(it.get("endpoint_verdict") or "")
            filled_v += 1

    print(f"  回填 type {filled_t} 条、verdict {filled_v} 条；"
          f"另有 {missed} 条找不到对应 plan（保持原样）")

    # 回填后能看到跨天去重真正生效的证据
    from dedupe import covered_events
    from common import today_cst
    ce = covered_events(today_cst(), days=14)
    if ce:
        print(f"\n近 14 天已覆盖 (cluster_key, type) {len(ce)} 个：")
        for k in sorted(ce):
            print(f"   {k}")

    if not apply:
        print("\n（只读模式。加 --apply 才会写回）")
        return 0

    data["entries"] = entries
    data["version"] = data.get("version", 3)
    write_json(SEEN_FILE, data)
    print(f"\n✓ 已写回 {SEEN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
