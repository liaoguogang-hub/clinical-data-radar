"""
噪声过滤器（临床版）。

与 BD 雷达的过滤逻辑【方向相反】：
  BD 雷达：留"交易"，剔"临床数据"
  本雷达：留"临床数据"，剔"交易"

实测依据：药时代 RSS 10 条里混着三类内容：
  · 临床进展   "士泽生物通用细胞治疗帕金森病亮相MDS国际大会丨
                临床I期突破进展+盲法II期全部入组"        ← 要
  · BD 交易    "三迭纪与强生达成合作"                    ← 不要（另一个雷达）
  · 人事变动   "百济神州高管履新吉利德！本土BD人才反向赋能MNC"  ← 不要

所以必须有明确的"这是临床内容"判据，而不是靠关键词白名单默认通过。
"""
from __future__ import annotations

import re

# ============ 第一层：临床强信号（命中即保留）============
#
# ⚠️ 第一版写得太窄，把最好的一条误杀了：
#   "NEJM：CAR-T新星 anito-cel 治疗复发/难治多发性骨髓瘤——38 例全部应答，
#    近 80% 完全缓解，3 年生存 65%，无迟发神经毒性"
# 当时只有"生存期"没有"生存"，只有"应答率"没有"应答"，
# 于是这条被判"无临床信号"。
# 教训：列举式的关键词表总有遗漏，必须补一个"上下文+数值"的兜底层。
CLINICAL_STRONG = re.compile(
    # 中文
    r"临床|"
    r"III期|II期|I期|3期|2期|1期|三期|二期|一期|"
    r"主要终点|次要终点|终点|达到|未达到|"
    r"疗效|安全性|不良事件|耐受性|"
    r"缓解|应答|客观缓解|疾病控制|"
    r"生存|无进展|疾病进展|死亡率|"
    r"顶线|数据读出|中期分析|入组(?:完成|结束)|"
    r"获批|批准|上市申请|受理|突破性疗法|优先审评|"
    r"ORR|PFS|OS\b|DOR|CR\b|PR\b|HR\s*=|"
    # 英文
    r"\bphase\s*(?:1|2|3|i{1,3})\b|\bclinical\s+(?:trial|study|data|results?)\b|"
    r"\bprimary\s+endpoint\b|\bmet\s+(?:the\s+)?(?:primary\s+)?endpoint\b|"
    r"\bmissed\b|\bfailed\s+to\s+meet\b|\btopline\b|\binterim\s+analysis\b|"
    r"\befficacy\b|\bsafety\b|\btolerability\b|\bremission\b|"
    r"\bresponse\s+rate\b|\bcomplete\s+response\b|\bpartial\s+response\b|"
    r"\boverall\s+survival\b|\bprogression-free\b|\bsurvival\b|\bORR\b|\bPFS\b|"
    r"\bapproval\b|\bapproved\b|\bfiling\b|\bNDA\b|\bBLA\b|"
    r"\bdiscontinu(?:ed|ation)\b|\bterminated\b|\bhalted\b|"
    r"\bdata\s+readout\b|\bcohort\b|\benroll(?:ed|ment)\b",
    re.I,
)

# ============ 第二层：临床上下文（需与数值同时出现才保留）============
#
# 兜底层。期刊名 + 疾病名 + 患者/受试者，单独出现不算临床内容
# （"患者满意度调查"不是临床数据），但配上具体数值就很可能是。
CLINICAL_CONTEXT = re.compile(
    r"NEJM|N\s*Engl\s*J\s*Med|Lancet|JAMA|BMJ|Nat\s*Med|Nature\s+Medicine|"
    r"Ann\s+Oncol|JCO|Blood|Cancer\s+Cell|Cell\s+Rep|Sci\s+Transl\s+Med|"
    r"肿瘤|癌|白血病|淋巴瘤|骨髓瘤|实体瘤|"
    r"糖尿病|肥胖|减重|心血管|卒中|阿尔茨海默|帕金森|"
    r"免疫|炎症|感染|病毒|肝炎|肾病|肝病|罕见病|"
    r"患者|受试者|病例|队列|"
    r"\bpatients?\b|\bsubjects?\b|\bcarcinoma\b|\btumou?r\b|\bleukemia\b|"
    r"\blymphoma\b|\bmyeloma\b|\bdiabetes\b|\bobesity\b|\bcovid\b",
    re.I,
)

# ============ 其他领域（命中即排除）============
# 本雷达只做临床数据，交易归 BD 雷达，人事/融资都不做
OTHER_DOMAIN = re.compile(
    # 纯 BD 交易
    r"达成合作|签署(?:协议|合作)|授权|引进|收购|并购|"
    r"license[- ]?(?:out|in)|licensing\s+deal|\bacqui(?:re|sition)\b|"
    r"\bmerger\b|\bbuyout\b|\bpartnership\b|\bcollaboration\b|"
    r"首付款|里程碑付款|对外授权|独家权益|"
    # 人事
    r"履新|任命|加盟|离职|辞任|上任|出任|晋升|"
    r"\bappoint(?:ed|ment)\b|\bhire[sd]?\b|\bCEO\b|\bCTO\b|\bCFO\b|"
    r"高管|核心团队|人才|招聘|"
    # 融资
    r"\bIPO\b|融资|定增|轮融资|Series\s+[A-D]\b|上市募资|"
    r"\braises?\b|\bfunding\b|\bventure\b",
    re.I,
)

# ============ 临床数据的具体证据（数字型）============
# 有具体数值的临床内容价值最高，用于"交易+数据"混合场景的取舍
CLINICAL_NUMERIC = re.compile(
    r"缓解率|应答率|生存期|PFS|OS\b|ORR|HR\s*=|"
    r"\d+(?:\.\d+)?\s*%|\d+\s*例|n\s*=\s*\d+|"
    r"\d+(?:\.\d+)?\s*(?:个月|周|天)|\b\d+(?:\.\d+)?\s*months?\b",
    re.I,
)

# ============ 通用噪声 ============
NOISE = re.compile(
    r"诺贝尔|招聘|会议(?:通知|邀请)|论坛|峰会|报名|直播预告|"
    r"诚邀|诚聘|招商|代理|促销|"
    r"\bnobel\b|\bwebinar\b|\bsponsored\b|\bpodcast\b|\bawards?\b|\bwatch\b",
    re.I,
)


def is_clinical(title: str, source: str = "") -> tuple[bool, str]:
    """
    判断是否属于"临床研究数据与进展"。
    返回 (是否保留, 原因)
    """
    t = (title or "").strip()
    if len(t) < 12:
        return False, "过短"

    if NOISE.search(t):
        return False, "通用噪声"

    has_clinical = bool(CLINICAL_STRONG.search(t))
    has_other = bool(OTHER_DOMAIN.search(t))
    has_numeric = bool(CLINICAL_NUMERIC.search(t))
    has_context = bool(CLINICAL_CONTEXT.search(t))

    # 第二层兜底：期刊名/疾病名 + 具体数值 → 认定为临床内容
    if not has_clinical and has_context and has_numeric:
        has_clinical = True

    if not has_clinical:
        return False, "无临床信号"

    # 混合场景：既像交易/人事，又像临床。
    # 只有带具体临床数值才留——否则那是 BD 雷达的内容。
    if has_other:
        if has_numeric:
            return True, "临床数据（含具体数值）"
        return False, "疑似交易/人事（无临床数值支撑）"

    return True, "临床内容"


def filter_items(items: list) -> tuple[list, list]:
    """返回 (保留, 丢弃)"""
    kept, dropped = [], []

    def get(x, k):
        return x.get(k) if isinstance(x, dict) else getattr(x, k, "")

    for it in items:
        ok, reason = is_clinical(get(it, "title"), get(it, "source"))
        if ok:
            kept.append(it)
        else:
            dropped.append((it, reason))
    return kept, dropped


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    cases = [
        "士泽生物通用细胞治疗帕金森病亮相MDS国际大会丨临床I期突破进展+盲法II期全部入组",
        "百济神州高管履新吉利德！本土BD人才反向赋能MNC",
        "三迭纪与强生达成合作",
        "迪拓生物与罗氏达成合作",
        "NEJM：CAR-T新星 anito-cel 治疗复发/难治多发性骨髓瘤——38 例全部应答，近 80% 完全缓解，3 年生存 65%，无迟发神经毒性",
        "恒瑞医药SHR-1819 III期达到主要终点，IL-4Rα单抗治疗特应性皮炎",
        "辉诺医药新药HN0037再获突破性疗法认定",
        "2026年诺贝尔化学奖公布，不对称有机合成领域又一次摘得桂冠！",
        "某公司宣布终止II期临床研究，因安全性问题",
        "AstraZeneca's lung cancer drug met the primary endpoint in Phase 3 trial",
        "Lilly to acquire a biotech for $2B",
        "Phase 2 study of XYZ missed its primary endpoint",
    ]
    print("=== 临床噪声过滤器测试 ===\n")
    for c in cases:
        ok, why = is_clinical(c)
        print(f"  {'保留' if ok else '剔除'}  {why:28} {c[:52]}")
