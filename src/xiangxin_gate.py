"""
象信 AI 前置闸门（Xiangxin pre-filter）——临床数据雷达。

作用：在昂贵的 LLM 打分之前，用象信先判断「这条到底是不是一条临床数据读出」，
把明显不是临床内容的条目拦掉（纯 BD 交易 / 人事 / 融资 / 综述行业新闻 /
会议预告 / 财经股价），省 LLM 调用。

⚠️ 与 BD 雷达的闸门【判据完全不同】，不要照搬：
  · BD 雷达问的是「有没有交易对手方」
  · 本雷达问的是「有没有具体的临床试验结果或关键进展节点」
  用错判据会大量误杀——BD 判据会把临床读出全拦掉，反之亦然。

为什么需要这一层：
  本雷达的 _prefilter 只在【标题】上匹配排除词，且必须「命中排除词 且
  无临床信号」才丢。实测大量非临床内容（行业综述、财经股价、会议预告、
  学术会议整体报道）根本不命中任何排除词，会全部涌进 LLM。
  象信能读摘要，判断力比关键词表强。

边界（重要）：
  - 象信【不】替代 LLM 打分。终点是否达到、试验规模、适应症未满足需求
    这些仍然只有 LLM 能评。
  - 闸门只拦「高置信度的非临床内容」，低置信度一律放行给 LLM。
  - ⚠️ 临床读出的类型很多（终点读出/终止/暂停/入组/注册/审批/安全性），
    判据里【必须】把它们全部列为「是」，否则会误杀本雷达的正常产出。
    实测最危险的是把「获批」当成非临床——本雷达明确做监管审批。
  - 象信服务不可用时，闸门整体旁路，绝不阻断生产。
"""
from __future__ import annotations

import os
import sys
from typing import Any

# 置信度门限：只有象信非常确定「不是临床内容」时才拦。
# 宁可多花钱给 LLM，也不要漏杀一条值得成稿的临床读出。
BLOCK_CONFIDENCE = 0.80

# 送给象信的摘要最大字符数。象信对长 state 效果下降（SKILL.md §6），
# 且只放判断需要的内容——无关的长文本会降低准确率。
MAX_SUMMARY_CHARS = 800

# 题目定义（口径对齐 score.py 的 SYSTEM_PROMPT + exclude_keywords）
#
# ⚠️ 「是临床」这一栏列得越全，漏杀越少。多列几类不花钱，
# 少列一类可能整条产线归零。
Q_CLINICAL = {
    "type": "choice",
    "instructions": (
        "`标题` 和 `摘要` 描述的是否是一条具体的临床研究进展或结果，"
        "即涉及某个药物、疗法或疾病领域的临床试验数据或关键节点？"
    ),
    "criteria": {
        "是临床": (
            "以下任一均算「是临床」："
            "① 临床试验结果读出（达到或未达到主要终点/次要终点、疗效数据、"
            "安全性数据、统计显著性结论）；"
            "② 试验终止、暂停、失败、被撤回；"
            "③ 关键进展节点（入组完成/启动、期中分析、注册试验启动、"
            "关键数据披露）；"
            "④ 监管审批与上市申请（FDA/EMA/NMPA 等获批、受理、优先审评、"
            "突破性疗法认定）——本雷达明确把监管审批作为临床进展的结果；"
            "⑤ 学术会议上的逐研究读出（含摘要号的研究报告）；"
            "⑥ 药物警戒/安全性事件（不良事件、召回、警示）；"
            "⑦ 与某个具体临床项目直接相关的研发进展（如新药获批临床、"
            "适应症扩展获批）；"
            "⑧ 临床前研究（动物实验、体外实验、分子设计论文）——"
            "本雷达会评临床前数据，只是降权"
        ),
        "非临床": (
            "以下均算「非临床」："
            "纯 BD 交易（授权/许可、收购并购、合作开发、期权）；"
            "融资（IPO/定增/Series A/B）、股价波动、市值变化、财经分析；"
            "人事变动、组织架构调整；"
            "行业综述、趋势解读、市场分析、观点文章；"
            "会议预告、会议日程、征稿通知、奖项公告、学会活动通知；"
            "公司战略、财报、营收、产能、供应链、组织新闻；"
            "与临床试验无关的一般性医学科普或常识；"
            "【复盘/盘点类综述】——只汇总项目数量与失败率、"
            "不含任何单项试验结论的内容，不算临床读出"
        ),
    },
}


def _state(item: Any) -> dict[str, str]:
    """构造象信 state。只放判断需要的内容。"""
    if isinstance(item, dict):
        title = item.get("title") or ""
        body = item.get("summary") or item.get("content") or ""
    else:  # RawItem
        title = getattr(item, "title", "") or ""
        body = (getattr(item, "summary", "") or
                getattr(item, "content", "") or "")
    return {
        "标题": title,
        "摘要": (body or "")[:MAX_SUMMARY_CHARS],
    }


def _has_text(item: Any) -> bool:
    """条目是否至少有标题或摘要可判。

    ⚠️ 这条不能省。实测踩过的坑：历史 scored 数据里存在
    title / summary / content 【全空】的条目——正文抓取失败时
    只剩下 id。这类条目送进象信时 state 是空的，象信只能凭
    "标题"和"摘要"两个空字符串猜，结果一律猜成「非临床」，
    于是一条条真实临床读出被误杀。

    实测（2026-11-25，43 天历史数据回归）：
      拦掉 1176 条（28.8%），其中 10 条是 score>=6.0 的真实读出，
      全部是空 state 的条目。
      典型：id=6bed09b0cc4ff6a4（score=6.7, flags=['met']）、
            id=ae983f23fdb58df（score=6.9, flags=['met']）、
            id=c1159daa1e770f05（score=6.9, flags=['halted']）。

    规则：**没有文本就不判**。空条目直接放行给 LLM——
    反正 LLM 也拿不到素材（compose 阶段的 material_is_thin 会拦），
    但至少不会在闸门这层被误杀成"非临床"。
    """
    st = _state(item)
    return bool((st["标题"] or "").strip() or (st["摘要"] or "").strip())


def _has_key() -> bool:
    return bool(os.getenv("XIANGXIN_API_KEY"))


def filter_items(items: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    """
    象信前置闸门。

    返回 (放行的条目, 统计信息)。
    任何异常都会退化为「全部放行」，绝不阻断生产。

    统计信息:
      blocked_ids   被拦掉的条目 id
      blocked       拦掉条数
      passed        放行条数
      error         旁路原因（None 表示正常启用）
    """
    stats: dict[str, Any] = {
        "enabled": False,
        "total": len(items),
        "blocked": 0,
        "passed": len(items),
        "error": None,
        "blocked_ids": [],
    }

    if not _has_key():
        stats["error"] = "XIANGXIN_API_KEY 未设置，闸门旁路"
        return items, stats

    try:
        from xiangxin import Choice, XiangxinClient
    except ImportError:
        stats["error"] = "xiangxin-sdk 未安装，闸门旁路"
        return items, stats

    question = Choice(
        instructions=Q_CLINICAL["instructions"],
        criteria=Q_CLINICAL["criteria"],
    )

    def _item_id(it: Any) -> str:
        if isinstance(it, dict):
            return it.get("id") or ""
        return getattr(it, "id", "") or ""

    try:
        with XiangxinClient() as client:
            passed = []
            for it in items:
                iid = _item_id(it)
                # 空 state 不判：象信只能凭空字符串猜，一定猜成「非临床」，
                # 会把真实临床读出误杀。详见 _has_text 的注释。
                if not _has_text(it):
                    stats["no_text"] = stats.get("no_text", 0) + 1
                    passed.append(it)
                    continue
                try:
                    resp = client.system_one(
                        state=_state(it),
                        questions={"clinical": question},
                    )
                    ans = resp.answers["clinical"]
                    if ans.choice == "非临床" and ans.confidence >= BLOCK_CONFIDENCE:
                        stats["blocked"] += 1
                        stats["blocked_ids"].append(iid)
                        continue
                except Exception as e:  # 单条失败不阻断整批
                    # 鉴权失败是全局性的，继续打只会白费 401 —— 整体旁路
                    if type(e).__name__ == "AuthenticationError":
                        stats["error"] = f"AuthenticationError: {e}"
                        print(f"[xiangxin-gate] 鉴权失败，全部放行: {stats['error']}")
                        return items, stats
                    print(f"[xiangxin-gate] 单条失败，放行: {type(e).__name__}: {e}")
                passed.append(it)
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        print(f"[xiangxin-gate] 初始化失败，全部放行: {stats['error']}")
        return items, stats

    stats["enabled"] = True
    stats["passed"] = len(passed)
    return passed, stats


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    demo = [
        {"id": "a", "title": "默沙东 TROP2 ADC 胃食管腺癌三期总生存达标",
         "summary": "III 期研究达到主要终点总生存，HR 0.62，473 例三线患者。"},
        {"id": "b", "title": "和黄医药「索乐匹尼布」获批上市",
         "summary": "国家药监局批准其 BTK 抑制剂新适应症上市。"},
        {"id": "c", "title": "礼来以 2 亿美元首付款获得诺诚健华 BTK 降解剂全球权益",
         "summary": "双方达成授权许可交易，礼来支付首付款。"},
        {"id": "d", "title": "英伟达被调查，连续 7 个交易日下跌",
         "summary": "股价接近腰斩，单周市值缩水近 9000 亿美元。"},
        {"id": "e", "title": "2026 ESMO 肿瘤药物研发盘点",
         "summary": "盘点本届 ESMO 会议重点研究，生存证据与临床定位。"},
    ]
    passed, st = filter_items(demo)
    print(f"总 {st['total']} → 放行 {st['passed']}，拦掉 {st['blocked']}")
    print(f"error: {st['error']}")
    for it in passed:
        iid = it.get("id")
        print("  放行:", iid, (it.get("title") or "")[:60])
    for iid in st["blocked_ids"]:
        t = next((x.get("title") for x in demo if x.get("id") == iid), "")
        print("  拦掉:", iid, (t or "")[:60])
