"""
ClinicalTrials.gov v2 一手材料查询。

定位：**不做发现源**，只在"搜一手材料"环节被调用——
从新闻话题里抽检索词，去注册库找对应的试验，拿回结构化结果。

实测（2026-10-11）：
  · 全量字段可一次请求取回，字段名见 FIELDS
  · query.term 按药名检索：anito-cel → 1 条
  · query.spons 按申办方检索：Hengrui → 916 条
  · resultsSection.outcomeMeasures[].analyses[] 含 pValue / ciLowerLimit /
    ciUpperLimit / nonInferiorityType（但仅约 1/3 的试验有）

⚠️ 最重要的约束：注册库**不给"是否达到终点"的结论**，只给数值和统计量。
   判定逻辑全部放在 verdict.py，且必须区分"机械判定"与"数值推断"。
"""
from __future__ import annotations

import asyncio
import re
import sys
from typing import Any

import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API = "https://clinicaltrials.gov/api/v2/studies"
UA = {"User-Agent": "clinical-data-radar/0.1 (research; contact: user@example.com)"}

# 实测通过的全量字段名
FIELDS = ",".join([
    "NCTId", "BriefTitle", "OfficialTitle", "Acronym", "OverallStatus",
    "Phase", "WhyStopped", "HasResults", "StartDate", "PrimaryCompletionDate",
    "CompletionDate", "LastUpdatePostDate", "ResultsFirstPostDate",
    "EnrollmentCount", "EnrollmentType", "LeadSponsorName", "CollaboratorName",
    "Condition", "InterventionName", "InterventionType",
    "PrimaryOutcomeMeasure", "LocationCountry",
])

# ============ 检索词抽取 ============
# ⚠️ 不要用 \b 划边界。实测踩坑：标题 "恒瑞医药SHR-1819 III期…" 里
#   "药" 和 "S" 之间【没有】\b——因为 Python 的 \w 默认包含中日韩字符，
#   两个都是 word char，于是 \b 不成立，药物代号 SHR-1819 整个抽不出来，
#   该条只能退化到按申办方搜，最终被判不匹配。
#   改用显式的"前一个字符不是字母数字"断言，中文不在 [A-Za-z0-9] 里，
#   所以 药SHR-1819 能正确切开。
_L = r"(?<![A-Za-z0-9])"      # 左边界
_R = r"(?![A-Za-z0-9])"       # 右边界

RE_NCT = re.compile(_L + r"(NCT\d{8})" + _R, re.I)

# 药物代号。实测标题里的形态：
#   SPR2015  HRS-1596  XS411  KR25102  HN0037  SHR-1819  BGB-16673
#   anito-cel（全小写的 INN 形态）
#
# ⚠️ 小写分支必须限定"药名后缀"，否则会把英文短语当药名。
# 实测踩坑：`[a-z]{4,12}-[a-z]{2,6}` 抓到了
#   double-blind / parallel-group / open-label / treat-to
# 这些是试验设计术语，拿去 CT.gov 搜会拉回完全无关的试验，
# 而且下面的锚机制会让它"自我确认"通过闸门。
# INN 后缀（长后缀放前面，避免 tide 抢先匹配掉 glutide）
#
# ⚠️ 这份表永远不够全。实测漏过：
#   Iberdomide（-mide 没收录）→ 只抽到对照药 Daratumumab
#   safiglipron（-glipron 没收录）→ 只抽到对照药 dapagliflozin
# 所以不能只靠后缀表：enrich 里还有一道"最终闸门"兜底。
INN_SUFFIX = (r"(?:glutide|zumab|ximab|omab|umab|leucel|tinib|ciclib|parib|"
              r"sortib|siran|rsen|glipron|domide|metinib|lisib|degib|prubart|"
              r"fusp|cel|mab|nib|tide|vir|stat|sen|vec|gene|ted|mide|azine|"
              r"azepam|prazole|sartan|gliptin|gliflozin)")
# ⚠️ 首字母大写也要能匹配。标题里的 INN 常写成 "Retatrutide" 而非全小写，
# 早先只写 [a-z]，导致 Retatrutide / Semaglutide 这类标准药名全抽不出来。
_INN = r"[A-Za-z][a-z]{2,13}"
RE_DRUG = re.compile(
    _L + r"(?:"
    r"[A-Z]{2,6}-?\d{3,6}"                      # SHR-1819 / KR25102 / SPR2015
    r"|" + _INN + r"-" + INN_SUFFIX + r"|"      # anito-cel / tabirafusp-ted
    r"" + _INN + INN_SUFFIX +                   # Retatrutide / Semaglutide
    r")" + _R)

# 明确不是药名的连字符英文词（双保险）
DRUG_STOP = {
    "double-blind", "open-label", "parallel-group", "treat-to", "first-in",
    "placebo-controlled", "randomized-controlled", "real-world", "long-term",
    "head-to", "proof-of", "best-in", "end-to", "dose-ranging", "multi-center",
    "single-arm", "cross-over", "follow-up", "high-dose", "low-dose",
    "phase-3", "phase-2", "phase-1", "intent-to",
}

# 常见靶点，仅用于日志与人工排查（实际过滤靠上面的位数判据）
TARGET_HINT = re.compile(
    r"^(?:GLP|GIP|GCG|IL|PD|PDL|CTLA|HER|EGFR|VEGF|TNF|JAK|BTK|CD|BCMA|"
    r"TIGIT|LAG|CLDN|KRAS|BRAF|ALK|ROS|NTRK|FGFR|PI3K|mTOR|PARP|SGLT|"
    r"DPP|TSLP|TSHR|CCR|CXCR|STING|TLR|RANK)[-‑]?\d{1,2}[A-Za-z]?$", re.I)

# 中文公司名。实测：末字必须落在"企业类型后缀"上。
# ⚠️ 早先的 {2,6} + 后缀表（含"细胞""基因"）会贪婪匹配出
#   "士泽生物通用细胞"（把"通用细胞"当后缀），必须收紧：
#   前缀 2-4 字，后缀只留真正的企业类型词。
RE_CN_CO = re.compile(
    r"[\u4e00-\u9fff]{2,4}(?:药业|生物|制药|医药|医疗|科技|集团|控股|股份|"
    r"生物医药|生物科技|制药集团)")

# 英文公司名（首字母大写的连续词）
RE_EN_CO = re.compile(
    r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)\s+"
    r"(?:Pharma|Biosciences|Biotech|Therapeutics|Oncology|Inc|Corp|Ltd)\b")

# 常见无关词，抽公司名时排除
CO_STOP = {"临床研究", "临床试验", "临床数据", "研究", "治疗", "药物", "新药",
           "全球首个", "国内首个", "国产首个", "细胞治疗", "基因治疗"}

# ============ 中英文公司名映射 ============
#
# ⚠️ 实测（2026-10-11）：ClinicalTrials.gov **完全不收录中文公司名**：
#     query.spons=恒瑞医药 → 0    query.spons=恒瑞 → 0
#     query.spons=百济神州 → 0    query.spons=众生药业 → 0
#   但英文名都有效：Hengrui → 916  BeiGene → 189  Innovent → 268
#
# 所以中文话题必须先映射成英文名，否则一手材料检索永远为空。
# 这张表只收"确认在 CT.gov 上有注册"的主流公司；
# 小公司查不到是正常的，返回"未找到"比返回错材料好。
CN_EN_COMPANY = {
    "恒瑞医药": "Hengrui", "恒瑞": "Hengrui",
    "百济神州": "BeiGene", "百济": "BeiGene",
    "信达生物": "Innovent",
    "君实生物": "Junshi",
    "石药集团": "CSPC",
    "中国生物制药": "Sino Biopharm",
    "复星医药": "Fosun",
    "再鼎医药": "Zai Lab",
    "康方生物": "Akeso",
    "诺诚健华": "InnoCare",
    "翰森制药": "Hansoh", "豪森药业": "Hansoh",
    "齐鲁制药": "Qilu",
    "正大天晴": "Chia Tai Tianqing",
    "科伦药业": "Kelun",
    "先声药业": "Simcere",
    "和黄医药": "Hutchmed",
    "传奇生物": "Legend Biotech",
    "驯鹿生物": "IASO",
    "亘喜生物": "Gracell",
    "荣昌生物": "RemeGen",
    "贝达药业": "Betta",
    "三生制药": "3SBio",
    "天境生物": "I-Mab",
    "基石药业": "CStone",
    "亚盛医药": "Ascentage",
    "德琪医药": "Antengene",
    "加科思": "Jacobio",
    "康宁杰瑞": "Alphamab",
    "泽璟制药": "Zelgen",
    "微芯生物": "Chipscreen",
    "艾力斯": "Allist",
    "迪哲医药": "Dizal",
    "益方生物": "InventisBio",
    "海思科": "Haisco",
    "恩华药业": "Nhwa",
    "人福医药": "Humanwell",
    "丽珠集团": "Livzon",
    "华海药业": "Huahai",
    "东阳光药": "HEC",
    "金赛药业": "GeneScience",
    "药明生物": "WuXi Biologics",
    "药明康德": "WuXi AppTec",
    "凯莱英": "Asymchem",
}


def to_english_co(name: str) -> str:
    """中文公司名 → CT.gov 可检索的英文名。找不到返回原值。"""
    n = (name or "").strip()
    if n in CN_EN_COMPANY:
        return CN_EN_COMPANY[n]
    for cn, en in CN_EN_COMPANY.items():
        if cn in n or n in cn:
            return en
    return n


def extract_terms(*texts: str) -> dict[str, list[str]]:
    """
    从标题/正文里抽检索词。

    返回 {"nct": [...], "drug": [...], "sponsor": [...]}
    """
    blob = " ".join(t for t in texts if t)
    nct = list(dict.fromkeys(RE_NCT.findall(blob)))

    drug = []
    for m in RE_DRUG.findall(blob):
        if m.lower() in DRUG_STOP:
            continue
        if re.fullmatch(r"(?:PHASE|NCT|FDA|NMPA|CDE|III|II|IV)\w*", m, re.I):
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", m):
            continue
        drug.append(m)
    drug = list(dict.fromkeys(drug))[:4]

    sponsor = []
    for m in RE_CN_CO.findall(blob):
        if m in CO_STOP:
            continue
        sponsor.append(m)
    for m in RE_EN_CO.findall(blob):
        sponsor.append(m)
    sponsor = list(dict.fromkeys(sponsor))[:4]
    # 中文名映射成英文（CT.gov 不收录中文名），并保留原文便于溯源
    sponsor_en = [to_english_co(s) for s in sponsor]

    return {"nct": nct, "drug": drug, "sponsor": sponsor,
            "sponsor_en": sponsor_en}


# ============ 查询 ============
async def _get(client: httpx.AsyncClient, params: dict) -> dict:
    try:
        r = await client.get(API, params=params, timeout=30)
    except Exception:
        return {}
    if r.status_code != 200:
        return {}
    try:
        return r.json()
    except Exception:
        return {}


async def search_by_term(client: httpx.AsyncClient, term: str,
                         page_size: int = 8) -> list[dict]:
    """按药名/关键词全文检索"""
    d = await _get(client, {"query.term": f'"{term}"', "pageSize": page_size,
                            "fields": FIELDS, "countTotal": "true"})
    return d.get("studies", []) or []


async def search_by_sponsor(client: httpx.AsyncClient, name: str,
                            page_size: int = 8) -> list[dict]:
    """
    按申办方检索。

    ⚠️ 不要在这里加 AREA[Phase]PHASE3 预过滤。
    实测（Arcellx）：加 PHASE3 过滤只返回 1 条，把 PHASE2 的
    iMMagine-1（就是新闻里那个 38 例的试验）整个丢掉了。
    正确做法是全取回来，交给 rank_trials 排序。
    """
    d = await _get(client, {"query.spons": name, "pageSize": page_size,
                            "fields": FIELDS})
    return d.get("studies", []) or []


async def get_trial(client: httpx.AsyncClient, nct: str) -> dict:
    """按 NCT 号取单个试验（含 resultsSection）"""
    try:
        r = await client.get(f"{API}/{nct}", params={"fields": FIELDS + ",ResultsSection"},
                             timeout=30)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}


# ============ 相关度校验 ============
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# 非药物性干预，不参与"同一药物"的判定
IV_STOP = {"placebo", "standardlymphodepletionregimen", "standardofcare",
           "best supportive care", "chemotherapy", "salinetreatment"}


def _iv_names(study: dict) -> set[str]:
    """试验的干预名称集合（归一化），用作"同一药物"的锚"""
    ps = (study or {}).get("protocolSection", {}) or {}
    arms = (ps.get("armsInterventionsModule", {}) or {}).get("interventions", []) or []
    out = set()
    for a in arms:
        n = _norm((a or {}).get("name", ""))
        if len(n) > 3 and n not in IV_STOP:
            out.add(n)
    return out


def _cond_names(study: dict) -> set[str]:
    ps = (study or {}).get("protocolSection", {}) or {}
    conds = (ps.get("conditionsModule", {}) or {}).get("conditions", []) or []
    return {_norm(c) for c in conds if c}


def _iv_match(a: str, b: str) -> bool:
    """
    两个干预名称是否指同一个药。

    ⚠️ 实测必需：CT.gov 上同一个药在不同试验里的写法不一致——
      NCT06626919 写 "anito-cel"                （缩写）
      NCT06413498 写 "Anitocabtagene Autoleucel"（全称 INN）
    精确比对必然漏掉 pivotal 试验。

    判据（任一成立即算同药）：
      · 完全相同
      · 一个是另一个的子串
      · 共同前缀 >= 6 字符（anitoc… 命中 anitocel / anitocabtagene…）
    """
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    if len(a) >= 6 and len(b) >= 6:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        if n >= 6:
            return True
    return False


def score_relevance(study: dict, terms: dict,
                    anchors_iv: set[str] | None = None) -> int:
    """
    试验与话题的相关度 = 命中的独立证据数。

    ⚠️ EDGAR 那次的教训：只检查"搜到的名字出现在结果里"是确认偏误。
    所以要求 **>= 2 个独立证据**。

    ⚠️ 但纯字面匹配又太死（实测 anito-cel vs Anitocabtagene）。
    所以加**数据驱动的双锚**：
      · 干预名称模糊匹配（+3）—— 最强证据，基本可确定是同一个药
      · 适应症匹配（+1）—— 弱证据，单独不达标；
        但配上申办方命中（+1）就到 2，足以通过
        这样既捞回 pivotal 试验，又不会只因为"同一个病"就乱收
    """
    ps = (study or {}).get("protocolSection", {}) or {}
    ident = ps.get("identificationModule", {}) or {}
    sponsor = (ps.get("sponsorCollaboratorsModule", {}) or {}).get("leadSponsor", {}) or {}
    cond = (ps.get("conditionsModule", {}) or {}).get("conditions", []) or []
    arms = (ps.get("armsInterventionsModule", {}) or {}).get("interventions", []) or []

    hay = " ".join([
        ident.get("briefTitle", "") or "",
        ident.get("officialTitle", "") or "",
        ident.get("acronym", "") or "",
        sponsor.get("name", "") or "",
        " ".join(cond),
        " ".join((a or {}).get("name", "") or "" for a in arms),
    ]).lower()

    score = 0
    for d in terms.get("drug", []):
        if d.lower() in hay:
            score += 2
    for s in terms.get("sponsor", []) + (terms.get("sponsor_en") or []):
        if s.lower() in hay:
            score += 1
    # 药物锚（唯一的强证据）
    if anchors_iv:
        for iv in _iv_names(study):
            if any(_iv_match(iv, a) for a in anchors_iv):
                score += 3
                break
    # ⚠️ 曾经这里还有"适应症锚 +1"：同申办方 + 同适应症 = 2 分即通过。
    # 那是错的——同公司同疾病完全不等于同一个药。实测因此放进了：
    #   Retatrutide（肥胖）→ 匹配到 Tirzepatide 试验（同为礼来）
    #   Iberdomide + Daratumumab → 匹配到 Daratumumab 单药试验
    # 适应症太宽，不能当"同一件事"的证据。已移除。
    return score


# ============ 结构化提取 ============
def summarize(study: dict) -> dict:
    """把试验压成写作要用的字段（不返回原始大 JSON）"""
    ps = (study or {}).get("protocolSection", {}) or {}
    ident = ps.get("identificationModule", {}) or {}
    status = ps.get("statusModule", {}) or {}
    design = ps.get("designModule", {}) or {}
    sponsor = (ps.get("sponsorCollaboratorsModule", {}) or {}).get("leadSponsor", {}) or {}
    cond = (ps.get("conditionsModule", {}) or {}).get("conditions", []) or []
    arms = (ps.get("armsInterventionsModule", {}) or {}).get("interventions", []) or []
    outcomes = (ps.get("outcomesModule", {}) or {}).get("primaryOutcomes", []) or []
    locs = (ps.get("contactsLocationsModule", {}) or {}).get("locations", []) or []
    enroll = design.get("enrollmentInfo", {}) or {}

    countries = []
    for l in locs:
        c = (l or {}).get("country")
        if c and c not in countries:
            countries.append(c)

    return {
        "nct": ident.get("nctId", ""),
        "title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "acronym": ident.get("acronym", ""),
        "status": status.get("overallStatus", ""),
        "why_stopped": status.get("whyStopped", ""),
        "phase": ", ".join(design.get("phases", []) or []),
        "sponsor": sponsor.get("name", ""),
        "conditions": cond,
        "interventions": [a.get("name", "") for a in arms if a.get("name")],
        "primary_outcomes": [o.get("measure", "") for o in outcomes
                             if o.get("measure")],
        "enrollment": enroll.get("count"),
        "enrollment_type": enroll.get("type", ""),
        "start_date": (status.get("startDateStruct", {}) or {}).get("date", ""),
        "primary_completion": (status.get("primaryCompletionDateStruct", {})
                               or {}).get("date", ""),
        "completion": (status.get("completionDateStruct", {}) or {}).get("date", ""),
        "last_update": (status.get("lastUpdatePostDateStruct", {})
                        or {}).get("date", ""),
        "results_first_post": (status.get("resultsFirstPostDateStruct", {})
                               or {}).get("date", ""),
        "has_results": bool(study.get("hasResults")),
        "countries": countries[:12],
        "results": extract_results(study),
    }


def extract_results(study: dict) -> dict:
    """
    抽结果模块：主要终点 + 分组数值 + 统计检验。

    ⚠️ 实测结构（2026-10-11）：
      outcomeMeasures[]
        ├─ type: PRIMARY / SECONDARY / OTHER
        ├─ title / description / timeFrame
        ├─ groups[]                     分组定义
        ├─ classes[].categories[].measurements[]
        │     {"groupId","value","spread"}         ← 数值在这
        └─ analyses[]                   ← 仅部分试验有
              {"pValue","ciLowerLimit","ciUpperLimit",
               "nonInferiorityType","statisticalMethod"}
    注意 pValue 实测形态是 ".01"（前导点，无 0），解析要兼容。
    """
    rs = (study or {}).get("resultsSection", {}) or {}
    oms = (rs.get("outcomeMeasuresModule", {}) or {}).get("outcomeMeasures", []) or []

    primaries, secondaries = [], []
    for o in oms:
        item = {
            "title": o.get("title", ""),
            "description": (o.get("description") or "")[:300],
            "time_frame": o.get("timeFrame", ""),
            "type": o.get("type", ""),
            "param_type": o.get("paramType", ""),
            "unit": o.get("unitOfMeasure", ""),
            "reporting_status": o.get("reportingStatus", ""),
            "groups": [{"id": g.get("id", ""),
                        "title": (g.get("title") or "")[:120]}
                       for g in (o.get("groups") or [])],
            "values": [],
            "analyses": [],
        }
        # 数值：classes[].categories[].measurements[]
        for cl in (o.get("classes") or []):
            for cat in (cl.get("categories") or []):
                for m in (cat.get("measurements") or []):
                    v = {"group_id": m.get("groupId", ""),
                         "value": m.get("value", ""),
                         "spread": m.get("spread", "")}
                    if v["value"] not in (None, "") and v not in item["values"]:
                        item["values"].append(v)
        # 直接挂在 outcome 上的 measureValues（部分试验用这个形态）
        for m in (o.get("measureValues") or []):
            v = {"group_id": m.get("groupId", ""),
                 "value": m.get("value", ""),
                 "spread": m.get("spread", "")}
            if v not in item["values"]:
                item["values"].append(v)
        # 统计检验
        for a in (o.get("analyses") or []):
            item["analyses"].append({
                "p_value": a.get("pValue"),
                "ci_lower": a.get("ciLowerLimit"),
                "ci_upper": a.get("ciUpperLimit"),
                "ci_pct": a.get("ciPctValue"),
                "test_type": a.get("nonInferiorityType"),
                "method": (a.get("statisticalMethod") or "")[:120],
                "param_type": a.get("paramType"),
                "param_value": a.get("paramValue"),
                "group_ids": a.get("groupIds") or [],
            })
        if item["type"] == "PRIMARY":
            primaries.append(item)
        else:
            secondaries.append(item)

    return {
        "primary": primaries[:4],
        "secondary": secondaries[:6],
        "n_primary": len(primaries),
        "n_secondary": len(secondaries),
        "has_analyses": any(p["analyses"] for p in primaries),
    }


# ============ 试验优先级 ============
PHASE_RANK = {"PHASE3": 5, "PHASE2": 4, "PHASE2, PHASE3": 4, "PHASE1": 2,
              "PHASE4": 3, "EARLY_PHASE1": 1, "NA": 0}
# 有结果或已结束的，比还在招募的更可能有数据
STATUS_RANK = {"COMPLETED": 3, "TERMINATED": 3, "ACTIVE_NOT_RECRUITING": 2,
               "SUSPENDED": 2, "WITHDRAWN": 2, "RECRUITING": 1,
               "NOT_YET_RECRUITING": 1, "ENROLLING_BY_INVITATION": 1,
               "UNKNOWN": 0}


def rank_trials(studies: list[dict]) -> list[dict]:
    """
    排序：III期 > II期 > I期；有结果优先；入组人数多的优先。

    实测（anito-cel）：同一药物在 CT.gov 上有 5 个试验——
      NCT06413498 PHASE3 n=452   ← 该优先取这个
      NCT05396885 PHASE2 n=136
      NCT06626919 PHASE1 n=30
      NCT04155749 PHASE1 n=40
      NCT07045909 PHASE2 n=30
    不做排序的话，取到的是 I 期小样本，写出来的稿子会严重低估这个资产。
    """
    def key(st: dict) -> tuple:
        ps = (st or {}).get("protocolSection", {}) or {}
        design = ps.get("designModule", {}) or {}
        status = ps.get("statusModule", {}) or {}
        phases = design.get("phases", []) or []
        pr = max((PHASE_RANK.get(p, 0) for p in phases), default=0)
        sr = STATUS_RANK.get(status.get("overallStatus", ""), 0)
        has_res = 1 if st.get("hasResults") else 0
        n = (design.get("enrollmentInfo", {}) or {}).get("count") or 0
        return (has_res, pr, sr, n)

    return sorted(studies, key=key, reverse=True)


# ============ 主入口 ============
def _term_matches(term: str, study: dict) -> bool:
    """
    该试验是否**真的**对应用这个检索词。

    ⚠️ 这是防"自我确认"的关键。实测踩坑：
    用垃圾词（如 IT101）搜 CT.gov，接口会松匹配返回一个无关试验
    （CRLX101），而如果把这个结果当成"锚"，它就会匹配自己的锚
    → 必然通过相关度闸门 → 错误材料进了文章。
    这是 EDGAR 那次确认偏误的新变种：**不是"搜到什么就信什么"，
    而是"搜到什么就以什么为标准"**。

    判据：检索词出现在试验标题/干预名称里（归一化后互为子串）。
    """
    t = _norm(term)
    if len(t) < 3:
        return False
    ps = (study or {}).get("protocolSection", {}) or {}
    ident = ps.get("identificationModule", {}) or {}
    hay = _norm(" ".join([
        ident.get("briefTitle", "") or "",
        ident.get("officialTitle", "") or "",
        ident.get("acronym", "") or "",
        " ".join((a or {}).get("name", "") or ""
                 for a in ((ps.get("armsInterventionsModule", {}) or {})
                           .get("interventions", []) or [])),
    ]))
    return t in hay


async def enrich(title: str, content: str = "",
                 min_relevance: int = 2, max_trials: int = 3) -> dict:
    """
    给一个新闻话题找注册库一手材料。

    检索策略（两跳）：
      ① NCT 号最直接
      ② 药名全文检索
      ③ **从命中的试验里取申办方，再按申办方搜一次**
         —— 实测必需：新闻标题常只写药名（"anito-cel"），
            按药名只命中 1 个 I 期试验；拿到申办方 Arcellx 后再搜，
            才能找到 III 期 pivotal 试验。
      ④ 药名无果时才退到申办方（中文名先映射成英文）
    """
    terms = extract_terms(title, content)
    out: dict[str, Any] = {"terms": terms, "trials": [], "best": None, "note": ""}

    # 能否定位：要么有 NCT 号/药代号，要么有能映射成英文的申办方。
    # ⚠️ 只有未映射的中文公司名时不搜——CT.gov 查不到中文名，
    #   强行搜只会拿到 0 结果或靠靶点拉回别人的试验。
    #   宁可诚实返回"未找到"，也不要塞错材料（EDGAR 的教训）。
    mapped_sponsor = [s for s in (terms.get("sponsor_en") or [])
                      if not re.search(r"[\u4e00-\u9fff]", s)]
    if not (terms["nct"] or terms["drug"] or mapped_sponsor):
        named = terms.get("sponsor") or []
        out["note"] = (
            f"注册库无法定位：抽到的检索词只有未映射的中文主体"
            f"（{'、'.join(named)}）。CT.gov 不收录中文公司名，"
            f"需要补映射表或改用中国注册库。" if named
            else "未能从标题抽出可检索的药名/公司名/NCT号")
        return out

    async with httpx.AsyncClient(headers=UA, follow_redirects=True) as client:
        studies: list[dict] = []
        drug_hits: list[dict] = []
        # 药物锚：由"匹配检索词的那个干预"构成（见下面 ② 的说明）。
        # 必须在使用前初始化。
        anchors_iv: set[str] = set()
        # ① NCT 号
        for nct in terms["nct"][:2]:
            s = await get_trial(client, nct)
            if s:
                studies.append(s)
                drug_hits.append(s)
        # ② 药名
        # ⚠️ 只有**真正匹配**该药名的结果才记入 drug_hits（用于建锚）。
        # 接口会松匹配，把垃圾词的结果也返回来；如果那些也当锚，
        # 就会自我确认通过闸门。实测因此放进了 4/8 条错误材料。
        #
        # ⚠️ 锚**只能取匹配检索词的那个干预**，不能收整条试验的干预列表。
        # 实测踩坑：搜 "Retatrutide" 命中正确试验 NCT06662383，
        # 但它的干预是 ['tirzepatide', 'retatrutide']——对照药也在里面。
        # 全收进锚集合后，任何 Tirzepatide 试验都能匹配上锚，
        # 于是"Retatrutide 肥胖"被匹配到"Tirzepatide 试验"。
        # ⚠️ **只搜主药名**（标题里第一个，通常是试验药），不要搜对照药。
        #
        # 实测踩坑：标题 "Iberdomide Plus Daratumumab/Dexamethasone…"
        # 抽出 ['Iberdomide', 'Daratumumab']。两个都搜之后，
        # Daratumumab 的命中被建成锚，于是一条**不含 iberdomide** 的
        # Daratumumab 单药试验（NCT03158688，干预只有 carfilzomib/
        # daratumumab/dexamethasone）通过了最终闸门——答非所问却自洽。
        # safiglipron vs dapagliflozin 同理。
        #
        # 对照药是标题里的常客，把它当检索词必然引入误匹配。
        # 只搜主药名后：iberdomide 能命中真正的 iberdomide 试验，
        # dapagliflozin 那条则因为主药名查不到而干净地返回"未找到"。
        for d in terms["drug"][:1]:
            hits = await search_by_term(client, d)
            studies.extend(hits)
            dn = _norm(d)
            for h in hits:
                if not _term_matches(d, h):
                    continue
                drug_hits.append(h)
                for iv in _iv_names(h):
                    if _iv_match(iv, dn):
                        anchors_iv.add(iv)
            await asyncio.sleep(0.3)

        # ③ 两跳：从已命中的试验里挖申办方，再搜一次
        hop_sponsors: list[str] = []
        for st in drug_hits or studies:
            sp = ((st.get("protocolSection", {}) or {})
                  .get("sponsorCollaboratorsModule", {}) or {})
            nm = (sp.get("leadSponsor", {}) or {}).get("name", "")
            nm = re.sub(r",?\s*(Inc\.?|Corp\.?|Ltd\.?|LLC|Co\.?|GmbH|S\.A\.|"
                        r"B\.V\.|AG|plc|Limited)\.?\s*$", "", nm).strip()
            if nm and len(nm) > 3 and nm not in hop_sponsors:
                hop_sponsors.append(nm)
        for sp in hop_sponsors[:2]:
            studies.extend(await search_by_sponsor(client, sp, page_size=12))
            await asyncio.sleep(0.3)

        # ④ 药名无果时才用申办方
        if not studies:
            for sp in mapped_sponsor[:2]:
                studies.extend(await search_by_sponsor(client, sp))
                await asyncio.sleep(0.3)

    # 去重 + 相关度打分（带药物锚）
    seen, ranked = set(), []
    for st in studies:
        nct = ((st.get("protocolSection", {}) or {})
               .get("identificationModule", {}) or {}).get("nctId", "")
        if not nct or nct in seen:
            continue
        seen.add(nct)
        ranked.append((score_relevance(st, terms, anchors_iv), st))

    # 相关度达标的按优先级排序
    good = rank_trials([st for sc, st in ranked if sc >= min_relevance])

    # ===== 最终闸门 =====
    #
    # ⚠️ 这是防"错材料"的最后一道，必须保留。
    # 理由：上面的相关度是"命中证据数"，容易被两种方式绕过——
    #   1. 标题里的**对照药**被抽成检索词（Iberdomide + Daratumumab，
    #      只抽到后者，于是搜到 daratumumab 的试验，自洽但答非所问）
    #   2. 接口的**松匹配**返回同公司同适应症的别的药
    #      （Retatrutide → Tirzepatide）
    # 所以最后再确认一次：**选中的试验必须真的能对上药名**。
    # 对不上就宁可返回空——错材料比没有材料更糟（EDGAR 的教训）。
    #
    # ⚠️ 判据要"主药名优先"，不能把所有药名一视同仁。
    # 实测踩坑：标题 "Iberdomide Plus Daratumumab/Dexamethasone…"
    # 抽出 ['Iberdomide', 'Daratumumab']，其中后者是**对照药**。
    # 如果允许任意一个药名命中就放行，那么一条 Daratumumab 单药试验
    # 会被 Daratumumab 这个词"验证"通过——答非所问却自洽。
    #
    # 规则：
    #   只认可**主药名**（标题里第一个药名，通常是试验药而不是对照药）
    #   对不上就返回空。
    #
    # ⚠️ 曾经允许"主药名找不到时用其余药名兜底"，实测反而制造误匹配：
    #   标题 "Iberdomide Plus Daratumumab/Dexamethasone…" 抽出
    #   ['Iberdomide', 'Daratumumab']，Iberdomide 在注册库查不到，
    #   于是用对照药 Daratumumab 兜底 → 命中一条 Daratumumab 单药试验。
    #   答非所问，但形式上"自洽"。
    #
    #   兜底唯一的收益是同一药物在新闻与注册库用不同名字的场景
    #   （Kodiak 的 tabirafusp-ted ↔ KSI-501），但代价是引入错误材料。
    #   错材料比没有材料严重得多，所以宁可误杀。
    #   名字映射问题留给 CN_EN_COMPANY 那样的映射表解决，不靠兜底猜。
    drugs = [d for d in (terms.get("drug") or []) if d]
    primary = drugs[0] if drugs else ""

    final: list[dict] = []
    for st in good:
        if not drugs:
            final.append(st)          # 只能靠申办方定位，不做这层
            continue
        if primary and _term_matches(primary, st):
            final.append(st)
            continue
        # 或对上主药名对应的干预锚（解决 INN 全称/缩写不一致）
        if anchors_iv and any(_iv_match(iv, a)
                              for iv in _iv_names(st) for a in anchors_iv):
            final.append(st)

    if good and not final:
        out["note"] = (
            f"注册库检索到 {len(good)} 个试验，但没有一个能对上主药名"
            f"「{primary}」（其余抽到的药名多为对照药，不作依据），"
            f"判定为不匹配，不采用")
        return out

    out["trials"] = [summarize(s) for s in final[:max_trials]]
    if not out["trials"]:
        # good 为空（相关度全不达标）或 final 为空，都到此为止。
        # ⚠️ 早先这里直接取 [0]，空列表会 IndexError——
        # 而"没找到"是正常情况，不该让整个补全流程崩掉。
        out["note"] = out["note"] or "注册库未找到相关度足够的试验"
        return out
    with_res = [t for t in out["trials"] if t.get("has_results")]
    out["best"] = (with_res or out["trials"])[0]
    return out


if __name__ == "__main__":
    import json
    tests = [
        "脑癌疫苗Ⅱ期未达终点，公司仍有信心",
        "NEJM：CAR-T新星 anito-cel 治疗复发/难治多发性骨髓瘤——38 例全部应答，近 80% 完全缓解，3 年生存 65%",
        "众生药业GLP-1R/GIPR激动剂III期研究成功",
        "恒瑞医药SHR-1819 III期达到主要终点，IL-4Rα单抗治疗特应性皮炎",
        "士泽生物通用细胞治疗帕金森病亮相MDS国际大会：临床I期突破进展+盲法II期全部入组",
    ]

    async def main():
        for t in tests:
            print(f"\n{'='*74}\n标题: {t[:66]}")
            r = await enrich(t)
            print("  检索词:", {k: v for k, v in r["terms"].items() if v})
            if r.get("note"):
                print("  说明:", r["note"])
            for tr in r["trials"]:
                print(f"  · {tr['nct']}  [{tr['phase']}] {tr['status']}  "
                      f"入组{tr['enrollment']}")
                print(f"      {tr['title'][:64]}")
                print(f"      申办方: {tr['sponsor'][:40]} | 有结果: {tr['has_results']}")
                if tr["why_stopped"]:
                    print(f"      终止原因: {tr['why_stopped'][:60]}")
    asyncio.run(main())
