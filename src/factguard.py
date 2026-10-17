"""
事实防线的公共件：实体标签剥离 + 事实闸门。

⚠️ 本文件与 pharma-bd-radar/src 下同名逻辑是【各一份拷贝】。
   两个雷达是互相独立的项目（不能互相 import，否则破坏各自可独立部署），
   所以这里刻意重复而不是抽公共库。
   ⚠️ 改动时【两个项目要同步改】，否则两个雷达的防线会漂移。
   对应的写法与理由见 pharma-bd-radar/src/common.py 的 split_entity_tags
   和 compose.py 的 check_facts 注释。
"""
from __future__ import annotations

import re

# ============================================================
# 1. 新闻站自动实体标签剥离
# ============================================================
# 摩熵/pharnexcloud 等新闻站会把自动实体标注放在正文最前面：
#   "…摩熵医药 2026/10/08 2846  Eli Lilly & Co  TYK2
#        北京诺诚健华医药科技有限公司  …  近期，诺诚健华公告称…"
# 那串只是新闻站的实体标注，不是正文内容。留着它，成稿模型会把标签里的
# 靶点/公司当成事实依据（实测事故：postmortem_b4985179，"TYK2 平台"）。
_BYLINE = re.compile(r"20\d\d[/-]\d{1,2}[/-]\d{1,2}\s+\d{3,7}(?=\s)")

_ENT_CJK_TAIL = ("公司", "药业", "生物", "科技", "医药", "制药", "股份",
                 "集团", "有限", "控股", "研究院", "大学")
_ENT_UPPER_CODE = re.compile(r"^[A-Z][A-Z0-9/-]{1,11}$")
_ENT_CJK_WORD = re.compile(
    r"^[一-鿿A-Za-z0-9（）()·]{2,40}(" + "|".join(_ENT_CJK_TAIL) + r")[\s]*$")
_LATIN_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.&'-]*$")
_LATIN_CONNECT = {"&", "and", "of", "de", "the", "inc", "corp", "ltd", "gmbh"}


def _consume_latin_run(raw: str) -> tuple[str, int]:
    """吃掉可能跨多个 token 的拉丁公司名（Eli Lilly & Co）。"""
    pos, words = 0, []
    while True:
        mm = re.match(r"\s*(\S+)", raw[pos:])
        if not mm:
            break
        t = mm.group(1)
        if _LATIN_TOKEN.match(t) or t.lower() in _LATIN_CONNECT:
            words.append(t)
            pos += mm.end()
        else:
            break
    while words and words[-1].lower() in _LATIN_CONNECT:
        words.pop()
    if len(words) < 2:
        return "", 0
    return " ".join(words), pos


def _consume_entities(rest: str, limit: int = 8) -> tuple[list[str], int]:
    tags: list[str] = []
    pos = 0
    while len(tags) < limit:
        while pos < len(rest) and rest[pos].isspace():
            pos += 1
        mm = re.match(r"\S+\s*", rest[pos:])
        if not mm:
            break
        tok = mm.group(0).rstrip()
        if _ENT_CJK_WORD.match(tok) or _ENT_UPPER_CODE.match(tok):
            tags.append(tok)
            pos += mm.end()
            continue
        run, used = _consume_latin_run(rest[pos:])
        if run:
            tags.append(run)
            pos += used
            continue
        break
    return tags, pos


def split_entity_tags(content: str) -> tuple[str, list[str]]:
    """
    把正文开头的新闻站实体标签行剥离出来，返回 (干净的正文, 标签列表)。

    三个信号必须同时成立才动手（CMS byline / ≥2 个实体形态 token /
    标签后是足够长的新闻导语），任一不满足就原样返回——
    误伤正文的代价远大于漏一行标签。
    """
    if not content:
        return content, []
    m = _BYLINE.search(content)
    if not m:
        return content, []
    rest = content[m.end():].lstrip()
    tags, used = _consume_entities(rest)
    if len(tags) < 2:
        return content, []
    body = rest[used:].lstrip()
    if len(body) < 200 or body.count("。") < 2:
        return content, []
    if not re.match(r"(?:近期|据悉|日前|近日|昨日|本周|宣布|公告)", body):
        return content, []
    return body, tags


# ============================================================
# 2. 事实闸门：成稿里有、素材里没有的高风险实体
# ============================================================
# ⚠️ 边界【不能用 \b】——Python 的 \b 基于 \w，中文字符在 Unicode 里属于 \w，
#    所以 "的ICP-488抑" 在 ICP 前【没有】词边界，正则会整段匹配不到。
#    实测踩过：闸门写了 \b，结果 ICP-488 和 TYK2 一个都没抓到，闸门形同虚设。
#    必须用否定环视，只把 ASCII 字母数字算作"词内字符"。
_DRUG_CODE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z][A-Z]{1,6}[-]?\d{2,5})(?![A-Za-z0-9])")
_TARGET = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{2,7})(?![A-Za-z0-9])")

# 细胞因子/通路族名：IL-17、IL-23、TNF-α、IFN-γ… 是生物学分类，
# 不是专利分子代号，正文写"靶向 IL-23 通路"是正常的，不该拦。
_CYTOKINE = re.compile(r"^(IL|TNF|IFN|IL23|IL17|IL6|TGF|CHEMOKINE)")

# 突变位点命名法：G12C / G12D / G12V / T790M / L858R …
# ⚠️ 这些是标准科学记法，不是某家的专利代号，任何临床稿都可能正当引用
#    （包括"无法确认是 G12D 还是 G12C"这种如实说明未知的情形）。
#    实测踩过：daraxonrasib 那篇老实写"无法确认其具体突变亚型（如 G12D、G12V、
#    G12C 等）"，结果被闸门当成捏造的靶点拦了——正是 skill §6 说的
#    "规则不能比问题更伤人"。与细胞因子同理，放行。
_MUTATION = re.compile(r"^[A-Z][0-9]{2,4}[A-Z]$|^G[0-9]{2}[A-Z]$")

_TARGET_ALLOW = {
    "TYK2", "JAK", "GLP", "ALK", "BTK", "SHP2", "SOS1", "KRAS", "EGFR",
    "PD", "PD1", "HER2", "CD19", "CD20", "BCMA", "IL", "TNF", "MCL1",
    "BRAF", "MEK", "FGFR", "MET", "ROS1", "NTRK", "CDK", "MDM2",
    "A", "I", "II", "III", "IV", "FDA", "NMPA", "EMA", "US", "EU", "MNC",
    "BD", "CEO", "IPO", "IND", "NDA", "PCC", "GLP1", "GLP2", "SIK2",
    "RNAI", "SIRNA", "ASO", "ADC", "CAR", "T", "Q", "R", "H1", "H2",
    "PCR", "ELISA", "DNA", "RNA", "P", "L", "D",
}


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/]", "", s).upper()


def check_facts(body_text: str, material: str,
                extra_allow: set[str] | None = None) -> list[str]:
    """
    事实闸门：抓出成稿里"素材中没有"的高风险实体。

    body_text: 成稿正文（已拼好的纯文本）
    material:  可用素材（拼好的纯文本）
    extra_allow: 额外放行词（如注册库补全里确认存在的药名）
    """
    material_n = _norm(material or "")
    if not material_n:
        return []                 # 没有素材可比对，放行

    allow = set(_TARGET_ALLOW)
    if extra_allow:
        allow |= {_norm(x) for x in extra_allow}

    errs: list[str] = []
    seen: set[str] = set()

    for m in _DRUG_CODE.finditer(body_text):
        code = m.group(1)
        key = _norm(code)
        if key in seen or _CYTOKINE.match(code):
            continue
        seen.add(key)
        if key in material_n:
            continue
        errs.append(f"分子代号「{code}」在素材中找不到——疑似凭空捏造。"
                    f"公告/原文里没有出现的药物代号一律不能写。")

    for m in _TARGET.finditer(body_text):
        tok = m.group(1)
        key = _norm(tok)
        if key in seen or key in allow:
            continue
        seen.add(key)
        if key in material_n:
            continue
        # 突变位点命名法（G12C/T790M…）是标准科学记法，放行
        if _MUTATION.match(tok):
            continue
        if len(key) >= 3 and any(c.isdigit() for c in key):
            errs.append(f"靶点/代号「{tok}」在素材中找不到——请核实是否原始"
                        f"披露。新闻站自动标签不等于事实披露。")
    return errs


# ============================================================
# 3. 素材"太薄"判定：薄到不足以成稿
# ============================================================
def material_is_thin(material: str) -> tuple[bool, str]:
    """
    素材是否薄到不足以成稿。返回 (是否太薄, 原因)。

    ⚠️ 为什么必须有这道闸（2026-10-17 事故）：
      实测当天 82 条候选里 **52 条（63%）正文为空**（源站付费墙/反爬）。
      这些条目只有标题 + 一句评分理由，成稿阶段却照样写满 5000 字——
      内容全部来自模型的世界知识。结果 retlirafusp alfa 那篇：
      把恒瑞的药写成君实的、把 TGF-β 写成 VEGF、把已发表数据写成"未披露"。

      更糟的是：**素材为空时原有事实闸门会静默放行**
      （check_facts 见 material 为空直接 return []），
      于是最需要闸门的场景反而完全没有闸门。所以这里单独判"太薄"。

    判据（要正向证据）：素材里至少要有一段真正的原文/摘要，
    或一条带 NCT 号且含关键字段的注册库记录。两者皆无＝只有标题。
    """
    m = material or ""
    if not m.strip():
        return True, "素材为空"

    # 注册库记录：有 NCT 号且带关键字段
    if re.search(r"NCT\d{8}", m) and re.search(
            r"入组|enrollment|主要终点|phase", m, re.I):
        return False, ""

    # 真正的正文/摘要：剥掉我们自己的模板行后还剩多少
    body = re.sub(r"^.*?(?:正文摘录|摘要)[:：]\s*", "", m, flags=re.S)
    body = re.sub(r"【[^】]*】", " ", body)
    body = re.sub(r"^(?:---|标题|来源|链接|评分|评分理由|标记|待核查|"
                  r"本条新闻的发布日期|检索词|说明)[:：].*$", " ", body, flags=re.M)
    body = re.sub(r"\s+", "", body)
    if len(body) < 200:
        return True, (f"剥离模板行后正文仅 {len(body)} 字（<200），"
                      f"相当于只有标题")
    return False, ""
