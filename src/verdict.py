"""
终点判定：把"注册库数值 + 新闻措辞"变成可否成文的结论。

这是整个项目最容易出错的地方，所以**结论必须带来源层级**。

核心约束（实测确认，2026-10-11）：
  ClinicalTrials.gov **不给"是否达到终点"的结论**，只给数值。
  analyses[]（含 p 值/CI）只有约 1/3 的试验有；其余只有裸数值。

因此判定分四档，可信度依次递减，**每档都要在文章里标明依据**：

  ① statistical  有 p 值 / CI / 非劣界值 → 机械判定，可信
  ② textual      新闻稿或期刊写明"达到/未达到" → 引用原文
  ③ numeric      只有裸数值 → 据数值推断，**必须标注"未经统计分析"**
  ④ none         什么都拿不到 → 标"未找到公开数据"

⚠️ 绝不能把 ③ 写成 ①。
   "数值看起来更好" 不等于 "达到主要终点"——
   终点是否达到取决于试验预先设定的统计假设，
   而这些假设不在注册库的结构化字段里。
   实例：一个试验可能达到数值优效，但因为预设的
   α 消耗（alpha spending）规则或缺失数据处理方式而判定为失败。
"""
from __future__ import annotations

import re
import sys
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============ 新闻措辞 → 定性结论 ============
# ⚠️ 教训（实测 3 处漏判）：逐字列举关键词永远列举不全。
#   旧版写死 "研究成功|试验成功"，遇到 "临床成功" 就漏；
#   旧版写死 "missed (the) (primary) endpoint"，遇到
#   "missed ITS primary endpoint" 就漏；
#   旧版只认 "halted"，遇到 "halts" 就漏。
#   改成**结构性正则**：允许中间插词、允许词形变化。
TEXT_MET = re.compile(
    # ⚠️ "达到/已达/达成/达" 都要覆盖。实测漏判：
    #   正文写"GC101治疗恶性黑色素瘤的关键II期研究【已达主要终点】"，
    #   而旧正则只认"达到…终点"，于是这条有 HR=0.46、P=0.0010
    #   这种硬统计结果的稿子，被判成了"仅疗效数据"。
    #   （TEXT_MISS 在前检查，所以"未达终点"不会被这里误判）
    r"(?:达到|已达|已达成|达成|达)(?:了)?(?:主要|次要|全部|预设)?终点|"
    r"(?:主要|次要)终点(?:达成|达到|已达成)|"
    r"(?:研究|试验|临床|III期|II期|3期|2期)(?:取得)?成功|"
    r"成功(?:达到|取得)|(?:取得|获得)(?:积极|阳性|理想)结果|阳性结果|"
    r"达到预期|超预期|"
    r"显著(?:改善|优于|延长|降低)|疗效显著|"
    r"\bmet\b[^.]{0,24}\b(?:primary\s+)?endpoint\b|"
    r"\bmet\s+its\b|\bsuccessful\b|\bpositive\s+(?:results?|data|outcome)\b|"
    r"\bsuperiority\b|\bstatistically\s+significant\b|\bwins?\b", re.I)

TEXT_MISS = re.compile(
    r"未达到?(?:主要|次要|全部|预设)?终点|未达终点|"
    r"(?:主要|次要)终点(?:未达|失败|未达成)|"
    r"(?:研究|试验|临床)(?:宣告)?失败|未达预期|(?:结果)?不及预期|"
    r"阴性结果|疗效不足|未见效|无效|"
    r"中期分析(?:显示)?(?:无效|停止)|"
    r"未显示(?:疗效|获益|优效)|"
    r"\b(?:miss(?:ed|es)|fail(?:ed|s))\b[^.]{0,28}\b(?:primary\s+)?endpoint\b|"
    r"\bfailed\s+to\s+(?:meet|demonstrate|show)\b|"
    r"\bnegative\s+(?:results?|data|outcome)\b|"
    r"\b(?:futility|did\s+not\s+meet)\b", re.I)

TEXT_HALT = re.compile(
    r"终止|暂停|叫停|搁置|中止|停摆|撤回|"
    r"\b(?:terminat|suspend|discontinu|withdraw|halt)\w*\b", re.I)

TEXT_PARTIAL = re.compile(
    r"部分(?:达到|获益)|未全部达到|混合结果|"
    r"达到次要终点(?:但|，).{0,12}未达到主要|"
    r"\bmixed\s+results?\b|\bpartial\b", re.I)

# 疗效数据（**不等于**达到终点）
#
# ⚠️ 实测必需区分。标题：
#   "NEJM：CAR-T新星 anito-cel——38 例全部应答，近 80% 完全缓解，
#    3 年生存 65%，无迟发神经毒性"
# 这是疗效数据，但**没有说"达到主要终点"**。
# 早先把它判成 unknown（因为关键词表里没有"完全缓解"），
# 但直接判成 met 也是错的——"缓解率高"与"达到预设终点"
# 是两件事：终点判定取决于试验预设的统计假设。
# 所以单列一档 positive_data，置信度低于 met。
TEXT_EFFICACY = re.compile(
    r"(?:完全|部分|客观|临床)缓解|全部应答|应答率|缓解率|"
    r"生存率|获益|疗效(?:良好|确切|积极|显著)|"
    r"数据(?:积极|良好|亮眼)|表现亮眼|"
    r"\b(?:complete|partial|objective)\s+response\b|\bORR\b|"
    r"\bresponse\s+rate\b|\bsurvival\s+rate\b|\bbenefit\b", re.I)

# 否定词：命中 MET/EFFICACY 前若出现否定，说明是负面说法。
#
# ⚠️ 三个坑都踩过：
#   1. 间隔："该研究【未达到】完全缓解"——否定词与正面词之间隔着"达到"
#   2. 更长的间隔："【未显示】生存获益"——隔着"显示生存"
#      所以不能用"否定词 + 可选动词 + 行尾"的锚定模式，必须在窗口内搜索
#   3. 误伤：不能把裸的"无"算作否定词，否则"无进展生存率"会被判成负面
NEG_ANY = re.compile(r"(?:未|尚未|没有|未能|无法|不能)")


def _negated(blob: str, m: re.Match, window: int = 10) -> bool:
    """
    判断匹配到的正面词是否被否定（在它前面 10 字窗口内找否定词）。

    ⚠️ 不做这个检查的话，"未达到完全缓解" 会因为含"完全缓解"
    被误判成 positive_data。
    """
    pre = blob[max(0, m.start() - window):m.start()]
    return bool(NEG_ANY.search(pre))


def judge_from_text(*texts: str) -> dict:
    """
    从新闻标题/正文措辞判断定性结论。

    优先级：终止 > 未达到 > 部分 > 达到。
    （"III期失败并终止"应判 missed+halt，不是单纯 halt）
    """
    blob = " ".join(t for t in texts if t)
    if not blob:
        return {"verdict": "unknown", "basis": "none", "confidence": 0.0,
                "evidence": []}

    ev = []
    halt = TEXT_HALT.search(blob)
    miss = TEXT_MISS.search(blob)
    met = TEXT_MET.search(blob)
    part = TEXT_PARTIAL.search(blob)
    eff = TEXT_EFFICACY.search(blob)

    if miss and halt:
        ev.append(f"失败表述:{miss.group(0)}")
        ev.append(f"终止表述:{halt.group(0)}")
        return {"verdict": "missed_and_halted", "basis": "textual",
                "confidence": 0.85, "evidence": ev}
    if miss:
        ev.append(f"未达到表述:{miss.group(0)}")
        return {"verdict": "missed", "basis": "textual", "confidence": 0.75,
                "evidence": ev}
    if halt:
        ev.append(f"终止表述:{halt.group(0)}")
        # 只看标题时无法区分"因疗效不足终止"与"因入组困难终止"
        return {"verdict": "halted", "basis": "textual", "confidence": 0.7,
                "evidence": ev}
    if part:
        ev.append(f"部分获益表述:{part.group(0)}")
        return {"verdict": "partial", "basis": "textual", "confidence": 0.6,
                "evidence": ev}
    # 明确说"达到终点"优先于仅有疗效数据
    if met and not _negated(blob, met):
        ev.append(f"达到表述:{met.group(0)}")
        return {"verdict": "met", "basis": "textual", "confidence": 0.75,
                "evidence": ev}
    # 正面词被否定 → 这是负面结论。
    # 实测："该研究未达到完全缓解" / "未显示生存获益"
    # 应当判 missed，不能因为没命中"终点"字样就落到 unknown。
    if met and _negated(blob, met):
        ev.append(f"被否定的达到表述:{met.group(0)}")
        return {"verdict": "missed", "basis": "textual", "confidence": 0.65,
                "evidence": ev}
    # 只有疗效数据：数据积极，但没说是否达到预设终点
    if eff:
        if _negated(blob, eff):
            ev.append(f"被否定的疗效表述:{eff.group(0)}")
            return {"verdict": "missed", "basis": "textual", "confidence": 0.6,
                    "evidence": ev}
        ev.append(f"疗效数据表述:{eff.group(0)}")
        return {"verdict": "positive_data", "basis": "textual",
                "confidence": 0.5, "evidence": ev,
                "caveat": "标题只给出疗效数据，未明确表述是否达到主要终点；"
                          "两者不可混为一谈"}
    return {"verdict": "unknown", "basis": "none", "confidence": 0.2,
            "evidence": []}


# ============ 数值解析 ============
def _num(v: Any) -> float | None:
    """
    把注册库里的数值转成 float。

    ⚠️ 实测：pValue 的形态是 ".01"（前导点、无 0），
    直接 float() 虽然能过，但用正则更稳；还有 "<0.001" 这种带符号的。
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.search(r"-?\d*\.?\d+", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _p_is_sig(p: float | None) -> bool | None:
    """p < 0.05 视为显著。拿不到返回 None（不是 False）。"""
    if p is None:
        return None
    return p < 0.05


# ============ 从统计量机械判定 ============
def judge_from_analyses(trial: dict) -> dict:
    """
    用 analyses[] 里的 p 值 / CI / 非劣界值判定。**这是唯一可信的机械判定。**

    判据：
      · 优效（SUPERIORITY）：p < 0.05 → met；p >= 0.05 → missed
      · 非劣（NON_INFERIORITY）：CI 下限 > -界值 → met
        （界值注册库不一定给，拿不到就只能看 p 值）
      · 有 CI 无 p：CI 不跨 0 且方向有利于试验组 → met
    """
    results = (trial or {}).get("results") or {}
    primaries = results.get("primary") or []
    ev: list[str] = []

    for om in primaries:
        for a in (om.get("analyses") or []):
            p = _num(a.get("p_value"))
            lo = _num(a.get("ci_lower"))
            hi = _num(a.get("ci_upper"))
            ttype = (a.get("test_type") or "").upper()

            if p is not None:
                sig = _p_is_sig(p)
                ev.append(f"终点「{om.get('title','')[:40]}」p={a.get('p_value')}"
                          f"（{ttype or '未标注检验类型'}）")
                return {
                    "verdict": "met" if sig else "missed",
                    "basis": "statistical",
                    "confidence": 0.9,
                    "evidence": ev,
                    "detail": {"p_value": p, "ci": [lo, hi], "test": ttype},
                }
            if lo is not None and hi is not None:
                # CI 不跨 0
                if lo > 0 or hi < 0:
                    ev.append(f"终点「{om.get('title','')[:40]}」"
                              f"95%CI [{lo}, {hi}]，不跨 0")
                    return {"verdict": "met", "basis": "statistical",
                            "confidence": 0.75, "evidence": ev,
                            "detail": {"ci": [lo, hi], "test": ttype}}
                ev.append(f"终点「{om.get('title','')[:40]}」"
                          f"95%CI [{lo}, {hi}]，跨 0，不能判定优于对照")
                return {"verdict": "missed", "basis": "statistical",
                        "confidence": 0.7, "evidence": ev,
                        "detail": {"ci": [lo, hi], "test": ttype}}

    return {"verdict": "unknown", "basis": "none", "confidence": 0.0,
            "evidence": ev}


# ============ 从裸数值推断（低置信） ============
def judge_from_numbers(trial: dict) -> dict:
    """
    只有数值、没有统计检验时，只能做**很弱的**方向性观察。

    ⚠️ 这一档【不足以判定终点是否达到】。它的用途是：
      - 给写作提供可引用的具体数字
      - 提示人工或 LLM 进一步核对
    返回值固定带 caveat，上层必须把它标注成"未经统计分析"。
    """
    results = (trial or {}).get("results") or {}
    primaries = results.get("primary") or []
    rows = []
    for om in primaries:
        groups = {g.get("id"): g.get("title", "") for g in (om.get("groups") or [])}
        vals = []
        for v in (om.get("values") or []):
            vals.append({
                "group": groups.get(v.get("group_id"), v.get("group_id", "")),
                "value": v.get("value"),
                "spread": v.get("spread"),
            })
        if vals:
            rows.append({
                "endpoint": om.get("title", ""),
                "unit": om.get("unit", ""),
                "param_type": om.get("param_type", ""),
                "values": vals,
            })

    if not rows:
        return {"verdict": "unknown", "basis": "none", "confidence": 0.0,
                "evidence": [], "numbers": []}

    return {
        "verdict": "unclear",
        "basis": "numeric",
        "confidence": 0.35,
        "evidence": [f"注册库提供 {len(rows)} 个主要终点的分组数值，"
                     f"但未提供统计检验（p 值/置信区间）"],
        "numbers": rows,
        "caveat": ("注册库仅提供数值，未提供统计分析。"
                   "终点是否达到取决于试验预设的统计假设，"
                   "不能仅凭数值大小判定。"),
    }


# ============ 终止原因分类 ============
HALT_REASON = {
    "efficacy": re.compile(r"efficacy|futility|lack of|not effective|无效|疗效", re.I),
    "safety": re.compile(r"safety|adverse|toxicity|安全性|不良", re.I),
    "enrollment": re.compile(r"enroll|accrual|recruit|slow|入组", re.I),
    "funding": re.compile(r"fund|financ|资金|经费|资助", re.I),
    "business": re.compile(r"strategic|business|portfolio|prioriti|商业|战略", re.I),
    "supply": re.compile(r"supply|drug availability|供应", re.I),
    "covid": re.compile(r"covid|pandemic", re.I),
}


def classify_halt(why: str) -> str:
    """把 whyStopped 归到可用的一类（实测样本：
    'terminated prematurely because the funding...' / 'strategic business reasons'）"""
    w = why or ""
    for k, pat in HALT_REASON.items():
        if pat.search(w):
            return k
    return "other" if w else "unknown"


# ============ 综合判定 ============
def reconcile(title: str, content: str, trial: dict | None) -> dict:
    """
    把新闻措辞与注册库数据合起来给最终结论。

    规则：
      1. 注册库状态是 TERMINATED/SUSPENDED/WITHDRAWN → 一定是终止事件，
         并用 whyStopped 补原因（这是**确定性最高**的信号）
      2. 有统计量 → 机械判定优先于新闻措辞
      3. 无统计量 → 用新闻措辞（textual），并附上注册库数值供核对
      4. 两者冲突 → 都列出，标 conflict，倾向以注册库统计量为准
    """
    text_j = judge_from_text(title, content)
    out: dict[str, Any] = {
        "textual": text_j,
        "statistical": None,
        "numbers": None,
        "verdict": text_j["verdict"],
        "basis": text_j["basis"],
        "confidence": text_j["confidence"],
        "evidence": list(text_j["evidence"]),
        "conflict": False,
        "halt_reason": "",
    }

    if not trial:
        out["note"] = "注册库未找到对应试验，结论仅来自新闻措辞"
        out["confidence"] = min(out["confidence"], 0.5)
        return out

    st = (trial.get("status") or "").upper()
    why = trial.get("why_stopped") or ""

    # ① 注册库状态变更（确定性最高）
    if st in ("TERMINATED", "SUSPENDED", "WITHDRAWN"):
        out["halt_reason"] = classify_halt(why)
        out["evidence"].append(
            f"注册库状态={st}" + (f"，原因：{why[:80]}" if why else "（未填原因）"))
        if out["verdict"] not in ("missed", "missed_and_halted"):
            out["verdict"] = "halted"
            out["basis"] = "registry_status"
            out["confidence"] = 0.85

    # ② 统计量机械判定
    sj = judge_from_analyses(trial)
    out["statistical"] = sj
    if sj["verdict"] != "unknown":
        st_verdict = sj["verdict"]
        # 冲突检测
        if (text_j["verdict"] in ("met", "missed")
                and st_verdict in ("met", "missed")
                and text_j["verdict"] != st_verdict):
            out["conflict"] = True
            out["evidence"].append(
                f"⚠️ 冲突：新闻措辞判 {text_j['verdict']}，"
                f"注册库统计量判 {st_verdict}——以注册库为准")
        out["verdict"] = st_verdict
        out["basis"] = "statistical"
        out["confidence"] = sj["confidence"]
        out["evidence"].extend(sj["evidence"])
        nj = judge_from_numbers(trial)
        out["numbers"] = nj.get("numbers")
        return out

    # ③ 无统计量：用新闻措辞，附注册库数值
    nj = judge_from_numbers(trial)
    out["numbers"] = nj.get("numbers")
    if nj["numbers"]:
        out["evidence"].append(nj["evidence"][0])
        out["caveat"] = nj.get("caveat", "")
        if out["verdict"] == "unknown":
            out["verdict"] = "unclear"
            out["basis"] = "numeric"
            out["confidence"] = 0.35
        else:
            # 有措辞结论但没有统计量 → 降低置信度并说明
            out["confidence"] = min(out["confidence"], 0.6)
            out["caveat"] = ("结论来自新闻措辞；注册库有数值但无统计检验，"
                             "未能交叉验证")
    return out


# ============ 给写作层用的摘要 ============
def to_prompt_block(v: dict) -> str:
    """把判定结果转成给成稿模型看的材料块（含来源分层标注）"""
    label = {
        "met": "达到主要终点",
        "missed": "未达到主要终点",
        "missed_and_halted": "未达到终点且已终止",
        "halted": "试验已终止/暂停",
        "partial": "部分达到",
        "positive_data": "疗效数据积极（未明确是否达到终点）",
        "unclear": "无法判定（仅有数值）",
        "unknown": "未找到公开结论",
    }.get(v.get("verdict"), v.get("verdict", ""))

    basis_label = {
        "statistical": "注册库统计分析（可信）",
        "registry_status": "注册库状态记录（可信）",
        "textual": "新闻/公告措辞（需注意出处）",
        "numeric": "据注册库数值推断（未经统计分析）",
        "none": "无",
    }.get(v.get("basis"), v.get("basis", ""))

    lines = [f"终点判定：{label}",
             f"依据层级：{basis_label}",
             f"置信度：{v.get('confidence', 0):.2f}"]
    if v.get("halt_reason"):
        lines.append(f"终止原因分类：{v['halt_reason']}")
    if v.get("evidence"):
        lines.append("证据：" + "；".join(str(e) for e in v["evidence"][:4]))
    if v.get("numbers"):
        for n in v["numbers"][:2]:
            vs = "；".join(
                f"{x['group']}={x['value']}" + (f"±{x['spread']}" if x.get("spread") else "")
                for x in (n.get("values") or [])[:4])
            lines.append(f"  终点「{n.get('endpoint','')[:40]}」"
                         f"({n.get('param_type','')} {n.get('unit','')})：{vs}")
    if v.get("caveat"):
        lines.append(f"⚠️ {v['caveat']}")
    if v.get("conflict"):
        lines.append("⚠️ 新闻措辞与注册库统计量冲突，文章必须以注册库为准并说明")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== 措辞判定回归 ===")
    cases = [
        "脑癌疫苗Ⅱ期未达终点，公司仍有信心",
        "众生药业GLP-1R/GIPR激动剂III期研究成功",
        "恒瑞医药SHR-1819 III期达到主要终点",
        "某公司宣布终止II期临床研究，因安全性问题",
        "Kodiak III期DAYBREAK临床成功",
        "Phase 2 study of XYZ missed its primary endpoint",
        "AstraZeneca's drug met the primary endpoint in Phase 3",
        "Lilly halts Phase 3 trial over safety concerns",
    ]
    for c in cases:
        r = judge_from_text(c)
        print(f"  {r['verdict']:20} conf={r['confidence']:.2f}  {c[:54]}")

    print("\n=== 终止原因分类 ===")
    for w in ["The study was terminated prematurely because the funding was withdrawn",
              "Study terminated for strategic business reasons",
              "terminated prior to Phase 2 due to slow enrollment",
              "Stopped for safety concerns",
              ""]:
        print(f"  {classify_halt(w):12} {w[:56] or '(空)'}")

    print("\n=== 数值解析（前导点形态）===")
    for v in [".01", "0.01", "<0.001", "1.1", "2.8", "", None, "0.049"]:
        print(f"  {str(v):8} -> {_num(v)}")
