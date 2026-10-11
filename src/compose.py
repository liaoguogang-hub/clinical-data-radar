"""
文章生成模块（内容层）。

架构（实测得出的分工）：
  LLM  → 只写内容和轻量标记，不写任何 HTML 标签
  render → 负责所有标签、CSS、转义

为什么这样分：
让 LLM 一次写"内容 + HTML + CSS"，成功率只有 ~1/3，
失败全在格式层（标签不配平、CSS 丢失、输出截断、返回空）。
拆开后格式错误从架构上不可能发生，模型只需专注内容。
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

from common import (ROOT, SCORED_DIR, RAW_DIR, DATA_DIR, load_yaml,
                    read_json, write_json, today_cst)
from render import (parse_content, render_html, body_chars,
                    structure_report, content_text)
from score import _client, _call_llm, _strip_think

POSTS_DIR = ROOT / "posts"


# ============================================================
# 素材组装
# ============================================================
ENRICHED_DIR = DATA_DIR / "enriched"


# 新闻标题里的分期表述 → 归一化
_PHASE_CN = {
    "I期": "PHASE1", "1期": "PHASE1", "一期": "PHASE1", "I 期": "PHASE1",
    "II期": "PHASE2", "2期": "PHASE2", "二期": "PHASE2", "II 期": "PHASE2",
    "III期": "PHASE3", "3期": "PHASE3", "三期": "PHASE3", "III 期": "PHASE3",
    "IV期": "PHASE4", "4期": "PHASE4",
    "phase 1": "PHASE1", "phase 2": "PHASE2", "phase 3": "PHASE3",
}


def _news_phase(title: str) -> set[str]:
    t = (title or "").lower()
    out = set()
    for k, v in _PHASE_CN.items():
        if k.lower() in t:
            out.add(v)
    return out


def _trial_matches_news(trial: dict, news_title: str) -> bool:
    """
    判断注册库命中的试验是否很可能就是新闻讲的那个。

    只做粗判：分期对得上就算。中国试验大量注册在 CDE 而不是 CT.gov，
    所以对不上是常态，这时必须显式提示，避免把两个试验的数据混写。
    """
    nph = _news_phase(news_title)
    if not nph:
        return True                 # 新闻没提分期，无法判断，不做提示
    tph = {p.strip().upper() for p in (trial.get("phase") or "").split(",")
           if p.strip()}
    return bool(nph & tph)


def build_material(date: str, plan_item: dict) -> str:
    """
    组装给成稿模型的素材。

    与 BD 雷达的关键差异：这里会把**一手材料与终点判定**放在最前面，
    并要求模型按"来源层级"表述——
    有统计检验的可以说"达到主要终点"，
    只有疗效数字的只能说"数据积极"，绝不能等号连接。
    """
    scored = read_json(SCORED_DIR / f"{date}.json", [])
    raw = read_json(RAW_DIR / f"{date}.json", [])
    enriched = read_json(ENRICHED_DIR / f"{date}.json", {}) or {}
    by_id = {s["id"]: s for s in scored}
    raw_by_id = {r["id"]: r for r in raw}

    ids = plan_item.get("member_ids") or [plan_item["lead_id"]]
    blocks: list[str] = []

    for i, sid in enumerate(ids, 1):
        s = by_id.get(sid, {})
        r = raw_by_id.get(sid, {})
        ex = enriched.get(sid) or {}

        blocks.append(f"--- 素材 {i} ---")
        if s:
            blocks.append(f"标题：{s.get('title','')}")
            blocks.append(f"来源：{', '.join(s.get('sources',[]) or [])}")
            blocks.append(f"链接：{s.get('url','')}")
            # ⚠️ 发布日期必须显式给出，且要标成"这就是新闻的日期"。
            # 实测踩坑：素材里没有日期时，模型会自己编年份
            # （2026-10-09 的新闻被写成"2025年10月8日"）。
            # 也不能只给一个裸的 ISO 串——要说明它是发布日，
            # 否则模型可能把它当成别的日期。
            pub = s.get("published") or r.get("published") or ""
            if pub:
                blocks.append(f"**本条新闻的发布日期：{pub[:10]}**"
                              f"（HOOK 里的日期必须与此一致）")
            else:
                blocks.append("**发布日期：未找到**"
                              "（HOOK 里写\"未找到公开数据\"，"
                              "绝对不要自己编一个年份）")
            blocks.append(f"评分：{s.get('score','')}/10")
            if s.get("reasoning"):
                blocks.append(f"评分理由：{s['reasoning']}")
            if s.get("flags"):
                blocks.append(f"标记：{', '.join(s['flags'])}")
            if s.get("verify_todo"):
                blocks.append(f"待核查：{'; '.join(s['verify_todo'])}")

        # ---- 一手材料与终点判定（最重要，放前面）----
        v = ex.get("verdict") or {}
        reg = ex.get("registry") or {}
        best = reg.get("best") or {}
        if v or best:
            blocks.append("")
            blocks.append("【一手材料核查结果】")
            if ex.get("verdict_block"):
                blocks.append(ex["verdict_block"])
            if best.get("nct"):
                blocks.append(f"注册库试验：{best['nct']} "
                              f"（{best.get('phase','')}，"
                              f"{best.get('status','')}）")
                blocks.append(f"  试验名称：{best.get('title','')}")
                # ⚠️ 注册库命中的试验**未必就是本条新闻讲的那个**。
                # 实测：君赛生物那条新闻讲的是 MIZAR-003（II 期黑色素瘤），
                # 但中国试验多在 CDE 注册、CT.gov 上只有同药的其他试验
                # （命中的 NCT07294872 是 I 期实体瘤）。必须显式提醒，
                # 否则成稿会把两个试验的数据混在一起。
                if not _trial_matches_news(best, s.get("title", "")):
                    blocks.append("  ⚠️ **这可能不是本条新闻讲的那个试验**："
                                  "分期或适应症与新闻对不上。"
                                  "只能当作同药背景资料引用，"
                                  "不要把它的样本量/终点/状态写成"
                                  "新闻那个试验的。")
                if best.get("official_title"):
                    blocks.append(f"  正式名称：{best['official_title'][:220]}")
                if best.get("sponsor"):
                    blocks.append(f"  申办方：{best['sponsor']}")
                if best.get("enrollment"):
                    blocks.append(f"  入组人数：{best['enrollment']}"
                                  f"（{best.get('enrollment_type','')}）")
                if best.get("conditions"):
                    blocks.append(f"  适应症：{'、'.join(best['conditions'][:5])}")
                if best.get("interventions"):
                    blocks.append(f"  干预：{'、'.join(best['interventions'][:6])}")
                if best.get("primary_outcomes"):
                    blocks.append(f"  主要终点："
                                  f"{'、'.join(best['primary_outcomes'][:3])}")
                if best.get("start_date"):
                    blocks.append(f"  启动：{best['start_date']}"
                                  f"  主要完成：{best.get('primary_completion','')}")
                if best.get("why_stopped"):
                    blocks.append(f"  ⚠️ 终止原因：{best['why_stopped']}")
            else:
                note = reg.get("note") or ""
                if note:
                    blocks.append(f"注册库：{note}")
            # 其他相关试验（用于横向对比）
            others = [t for t in (reg.get("trials") or [])
                      if t.get("nct") and t["nct"] != best.get("nct")]
            if others:
                blocks.append("  同药其他试验（可作横向对比）：")
                for t in others[:3]:
                    blocks.append(f"    · {t['nct']} [{t.get('phase','')}] "
                                  f"n={t.get('enrollment','?')} "
                                  f"{t.get('title','')[:60]}")
            blocks.append("")

        content = (s.get("content") or r.get("content") or "")
        if len(content) > 200:
            blocks.append(f"正文摘录：\n{content[:7000]}")
        else:
            summ = (s.get("summary") or r.get("summary") or "")
            if summ:
                blocks.append(f"摘要：{summ[:1200]}")
            else:
                blocks.append("⚠️ 本条无正文，仅有标题。"
                              "不要在文中编造任何未在标题出现的事实。")
        blocks.append("")

    return "\n".join(blocks)


# ============================================================
# Prompt —— 只要求轻量标记，不要求 HTML
# ============================================================
SYSTEM_TPL = """你是临床研究（clinical development）领域的资深撰稿人，
为中文读者写临床数据解读。**写内容，不写 HTML。**

# 输出的标记格式

KICKER: 栏目标签（如「临床数据」「试验终止」）
TITLE: 文章标题
SUB: 副标题（一句话，可选）
HOOK: 开篇第一段

## 01 小节标题
正文段落……
## 02 小节标题
正文段落……

TABLE: 表头1 | 表头2
单元格 | 单元格
ENDTABLE

BOX w 小标题 | 内容      （w=口径提示 g=正面 r=风险）
ENDCARD 结语标题 | 结语内容
FN 1. 来源：xxx（机构｜日期）

规则：
- `## ` 小节必须有编号，如 `## 01 终点结果`
- 表格第一行是表头，`|` 分隔，每行列数要一致
- `BOX` 三个变体：`w` 琥珀（口径/依据说明）/ `g` 绿（正面）/ `r` 红（风险）
- `ENDCARD` 和 `FN` 各只能出现一次/在末尾，FN 可以多行
- 正文里可用 `**加粗**` 强调关键数字

# 内容规范

- **正文 2800-3200 字**（不含表格和脚注）。硬指标，宁可写满
- **HOOK 第一句必须交代时间**（年-月-日，或"2026年9月24日"格式）。
  读者第一眼就要知道这是什么时候的数据。查不到就写"未找到公开数据"，
  但不要略过不提。同时给出反常识数字或矛盾对比。

{structure}

# 终点结论必须标来源层级（这是本雷达最重要的一条）

素材里会给出【一手材料核查结果】，其中「依据层级」分四档，
**表述方式必须与之匹配，不得越级**：

| 依据层级 | 允许的表述 | 禁止的表述 |
|---|---|---|
| 注册库统计分析（p值/CI） | 「达到主要终点」「统计学显著」 | —— |
| 注册库状态记录 | 「试验已终止/暂停」 | —— |
| 新闻/公告措辞 | 「据公司公告，称达到主要终点」 | 不能说成已确证 |
| **仅数值推断** | 「注册库显示 X 组为 Y，未提供统计检验」 | **绝不能写「达到终点」** |

**最关键的一条：疗效数据 ≠ 达到终点。**
"38 例全部应答、80% 完全缓解" 是疗效数字；
"达到主要终点"是统计结论。终点是否达到取决于试验**预设的统计假设**
（α 消耗、缺失数据处理、非劣界值），这些通常不在新闻里。
所以：
- 素材只说疗效数字、没说终点结论 → 写"数据积极"，**不要写"试验成功"**
- 只有拿到统计检验或权威来源明确表述时，才可以写"达到主要终点"
- 必须注明依据来源（注册库 / 公司公告 / 期刊 / 会议）

# 其他规则

- **失败和终止是好选题**。不要因为消息是坏的就把语气写软。
  III 期失败是行业级事件，要写透影响。
- 有注册库试验号（NCT 号）的，正文中要写出——这是读者可核查的锚点
- 有准确的分期、样本量、适应症、主要终点的，要写出来
- 查不到的数据写"未找到公开数据"，**绝对禁止编造**
- **已知信息要写足**：不要把"未披露"当成段落主体
- 每个小节至少 2-3 段，表格后要有解读，不要甩个表就走
- 结尾 ENDCARD 给判断，不要复述前文

# 禁止脑补（实测踩过的坑）

素材里没有的事实、数字、试验、公司，一律不得出现：

- ❌ 素材只说"II 期数据亮眼"，却写"已获 FDA 批准" —— 素材里没有
- ❌ 素材只有一行标题，却编出样本量、终点名称、统计结果
- ❌ 把"疗效数据积极"写成"达到主要终点"（最常犯，也最严重）
- ❌ 给素材里没有的公司安角色（"某某或是潜在买家"）

规则：
1. **所有数字必须来自素材**（正文摘录或注册库字段）。没有就写"未找到公开数据"
2. **不要把推断写成事实**。推断要加"可能""或"这类限定词，并说明依据
3. **素材只有标题时**：坦率说明信息有限，聚焦标题支撑得起的分析，
   不要用推测填满 2800 字
4. **注册库没有命中时**：写"注册库未检索到对应试验"，不要假装有 NCT 号

# 来源标注

文末 FN 脚注要列出来源，并注明类型：
- 注册库：写 NCT 号 + ClinicalTrials.gov
- 期刊：写期刊名 + 发表日期
- 会议：写会议名 + 场次
- 媒体：写媒体名 + 链接
- 公司公告：写公司 + 公告日期

只输出上面的标记文本，第一个字符应该是 KICKER 或 TITLE。"""


def build_prompt(plan_item: dict, material: str, style_cfg: dict) -> tuple[str, str]:
    atype = plan_item.get("type", "event_report")
    struct = style_cfg.get("article_types", {}).get(atype, {})
    system = SYSTEM_TPL.replace("{structure}", struct.get("structure", ""))
    target = style_cfg.get("generation", {}).get("target_words", 3000)

    user = f"""# 今日选题
类型：{atype}（{struct.get('name','')}）
标题参考：{plan_item.get('title','')}
评分：{plan_item.get('score','')}/10
选题理由：{plan_item.get('reasoning','')}

# 可用素材
{material}

# 要求
标题要吸引人点击，但不要标题党。
正文 {target} 字左右（不含表格和脚注）。
素材里没有的数字不要写。素材里标了"待核查"的点，要么核实，
要么在文中写明"未找到公开数据"。

现在按标记格式输出。"""
    return system, user


# ============================================================
# 内容层校验
# ============================================================
# 中文金额："5亿美元" "50亿" "1.25亿美元"
MONEY_CN = re.compile(r"(\d+(?:\.\d+)?)\s*亿(?:美元|美金|元)?")
# 金额限定词
MONEY_QUAL = ("首付款", "首付", "总对价", "总额", "里程碑")


def _title_money_pairs(text: str) -> list[tuple[str, str]]:
    """提取「金额 + 限定词」组合。限定词取金额后 8 字内出现的那个。"""
    out = []
    for m in MONEY_CN.finditer(text):
        tail = text[m.end():m.end() + 8]
        qual = next((q for q in MONEY_QUAL if q in tail), "")
        out.append((m.group(1), qual))
    return out


def _check_title_numbers(doc: dict, body: str) -> list[str]:
    """
    标题里的金额与正文是否自洽。
    只检查带限定词的金额——不带限定词的（如"60亿押注"）噪声太大，
    误报成本高于收益，不做强制。
    """
    errs: list[str] = []
    title = doc.get("title", "")
    if not title:
        return errs

    body_pairs = _title_money_pairs(body)
    body_nums = {n for n, _ in body_pairs}
    body_qual_nums = {(q, n) for n, q in body_pairs if q}

    for num, qual in _title_money_pairs(title):
        if not qual:
            continue
        # 正文里必须有同限定词同数字的组合（限定的就近匹配）
        if (qual, num) in body_qual_nums:
            continue
        # 该限定词在正文中出现的所有数字
        same_qual = sorted({n for q, n in body_qual_nums
                            if q == qual or q.startswith(qual[:2])})
        if same_qual:
            errs.append(
                f"标题金额与正文不符：标题称 {num}亿{qual}，"
                f"正文中{qual}为 {'/'.join(same_qual)}亿")
        elif num not in body_nums:
            errs.append(f"标题金额 {num}亿{qual} 未在正文出现")
    return errs


def validate_content(doc: dict, text: str, style_cfg: dict,
                     src_title: str = "", material: str = "",
                     src_published: str = "") -> list[str]:
    """
    校验结构化内容（不涉及 HTML，因为 HTML 由代码保证）。

    src_title: 源新闻标题。判断"这是不是并购类"必须用它，不能用生成的文章标题
    ——文章标题可能写成"18亿美元买一个止痛贴"，不含"收购"二字；
    而正文的对比表格里又可能出现别的公司的"收购"，用正文判断会误报。

    material:  可用素材。用来判断"该事实是否适用且可得"。
    """
    errs: list[str] = []
    gen = style_cfg.get("generation", {})
    target = gen.get("target_words", 3000)
    tol = gen.get("word_tolerance", 0.75)
    wmax = gen.get("max_words", 3900)

    # 1. 必需元素
    if not doc.get("title"):
        errs.append("缺少 TITLE")
    if not doc.get("hook"):
        errs.append("缺少 HOOK")
    if len(doc.get("sections") or []) < 3:
        errs.append(f"小节太少（{len(doc.get('sections') or [])} 个，至少 3 个）")
    if not doc.get("footnotes"):
        errs.append("缺少 FN 来源脚注")

    # 2. 字数
    n = body_chars(doc)
    lo, hi = int(target * tol), int(wmax)
    if not (lo <= n <= hi):
        errs.append(f"正文字数 {n} 超出范围（目标 {lo}-{hi}）")

    # 2.5 空壳表格检测（仅在【整张表全空】时才判失败）
    #
    # 实测案例：一篇讲"临床三期成功"的稿子（素材里没有任何交易），
    # 却套了「交易要素」表，各行全是"未找到公开数据"——根本没有这笔交易。
    #
    # ⚠️ 第一版用"≥50% 且 ≥3 行未找到"就判失败，结果【误杀了诚实稿件】：
    #   · 综述里的多笔交易对比表，部分交易金额未披露是如实报告
    #   · 条款表里确实没披露的项目，写"未找到公开数据"是对的
    #   实测一轮里 3 篇有 1 篇因这条规则连续 4 次重写全部失败。
    #
    # 而且根因已由别处解决：评分层第 ⑧ 条会把"非交易"标 not_bd，
    # 选题层再拦一道，非交易条目根本到不了成稿。所以这里只保留
    # 最保守的兜底——整张表没有任何一行有实质信息，才算空壳。
    for sec in doc.get("sections") or []:
        for b in sec.get("blocks") or []:
            if b.get("type") != "table":
                continue
            rows = b.get("rows") or []
            if len(rows) < 3:
                continue
            blank = sum(1 for r in rows
                        if any("未找到公开数据" in c for c in r))
            if blank == len(rows):
                head = " / ".join(b.get("head") or [])[:40]
                errs.append(
                    f"表格「{head}」{len(rows)} 行全部是「未找到公开数据」，"
                    f"是一张没有任何信息的空壳表。请删掉它——"
                    f"没有内容支撑的表格不如不写。")

    # 2.7 并购类稿件的关键事实
    #
    # 实测案例：一篇"Viatris 18亿美元收购 Pacira"的稿子里
    # 既没有每股收购价（实际 $36.50/股），也没有交易日期。
    # 只写总额的并购报道，读者看不出这是新闻，也判断不了贵不贵
    # ——每股价格才是估值锚点（对比市价能看出多少溢价）。
    #
    # 只对并购类做要求；授权类交易没有"每股价格"这个概念。
    # 判据用源标题（见函数说明），不用正文——正文对比表里常有
    # 别家公司的"收购"字样，会误报。
    subj = src_title or doc.get("title", "")
    facts = doc.get("title", "") + " " + content_text(doc)
    if re.search(r"收购|并购|买下|acqui|buyout|takeover|merger", subj, re.I):
        # 每股收购价只对【上市公司整体收购】适用。
        # 判据：素材或标题里出现交易所代码，如 (Nasdaq: PCRX)。
        # 反例："Merck Acquires SPR2015" 是资产收购（SPR2015 是药物代号），
        # 根本没有"每股价格"这个概念，强制要求会把好稿误拦。
        is_listed = bool(re.search(
            r"\((?:NASDAQ|NYSE|Nasdaq|NYSE|HKEX|SEHK)[:：]\s*[A-Z0-9]{2,6}",
            (material or "") + " " + subj))
        if is_listed and not re.search(
                r"(?:每股|per\s+share)[^。；]{0,24}[\d$]|"
                r"\$\s?\d[\d.,]*\s*(?:per\s+share|/\s*股)|"
                r"[\d.]+\s*美元\s*/\s*股", facts, re.I):
            errs.append(
                "上市公司整体收购，稿件缺少【每股收购价】——"
                "这是估值锚点，能看出溢价多少。"
                "可从 SEC 8-K / 公司公告取。")
        # 交易日期对所有并购类都适用
        if not re.search(
                r"20\d\d\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
                r"\d{1,2},?\s+20\d\d|"
                r"\d{1,2}\s+(?:January|February|March|April|May|June|July|"
                r"August|September|October|November|December)\s+20\d\d",
                facts, re.I):
            errs.append(
                "并购类稿件缺少【交易/公告日期】——读者据此判断是不是新闻")

    # 2.6 HOOK 必须交代交易时间
    #
    # 实测被用户指出：那篇「礼来1亿美元首付"敲门"」的开头写
    #   "礼来2026年单笔首付均值已达2.17亿美元，而这一次它只掏出
    #    1亿美元'零头'，就绑定了诺诚健华最多5个创新靶点的研发管线。"
    # 通篇没有交易日期，读者无从判断时效性——不像新闻。
    #
    # ⚠️ 只看 HOOK 段落，不查全文：日期写在文末或表格里，
    # 读者第一眼看不到，等于没写。
    hook = doc.get("hook", "") or ""
    if hook:
        DATE_RE = (
            r"20\d\d\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
            r"[a-z]*\.?\s+\d{1,2},?\s+20\d\d|"
            r"\d{1,2}\s+(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+20\d\d|"
            r"20\d\d-\d{1,2}-\d{1,2}"
        )
        if not re.search(DATE_RE, hook, re.I):
            errs.append(
                "HOOK 里没有交易时间。读者第一眼就要知道这是什么时候的事"
                "——请在开头写明具体年月日（如「2026年9月24日」或"
                "「Oct. 8, 2026」）。查不到就写「未找到公开数据」，"
                "但不要略过不提。")
        # 年份必须与源发布日期一致
        #
        # ⚠️ 实测踩坑：素材里没给发布日期时，模型会**自己编年份**——
        #   2026-10-09 的新闻被写成"2025年10月8日"，2026-10-07 的
        #   被写成"2024年10月7日"。日期错一年，整篇稿子的时效性就假了。
        #   所以这里硬校验：HOOK 里出现的年份不能和素材发布日期差超过 1 年。
        elif src_published:
            try:
                pub_year = int(str(src_published)[:4])
            except Exception:
                pub_year = 0
            if pub_year:
                years = [int(y) for y in re.findall(r"(20\d\d)\s*年", hook)]
                years += [int(y) for y in
                          re.findall(r"\b(20\d\d)\b", hook)]
                wrong = [y for y in years if abs(y - pub_year) >= 1]
                if wrong:
                    errs.append(
                        f"HOOK 里的年份 {wrong} 与素材发布日期 "
                        f"{src_published[:10]} 不符。"
                        f"这是编造日期——必须用素材给出的发布日，"
                        f"查不到就写「未找到公开数据」。")

    # 3. AI 味（分档）
    body = content_text(doc)
    tolerate = gen.get("ai_word_tolerate", 2)
    hard = gen.get("ai_word_hard_limit", 5)
    counts = {w: body.count(w) for w in style_cfg.get("banned_words", [])}
    total = sum(counts.values())
    if total > hard:
        top = sorted(counts.items(), key=lambda x: -x[1])[:5]
        errs.append(f"AI味词过多({total}处): "
                    + ", ".join(f"{w}x{c}" for w, c in top))
    elif total > tolerate:
        # ⚠️ 软档只提示，不判失败。
        # 实测教训（BD 雷达）：为几个 AI 味词重写整篇，代价远大于收益——
        # 本轮就有一篇第 3 次尝试已写出 3797 字（在字数范围内），
        # 仅因 3 个 AI 味词被打回，第 4 次掉到 1621 字，最终成稿失败。
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        print("      [提示] AI味词偏多: "
              + ", ".join(f"{w}x{c}" for w, c in top) + "（不判失败）")

    for p in style_cfg.get("banned_patterns", []):
        if re.search(p, body):
            errs.append(f"AI味句式: {p}")

    # 4. 标题金额 vs 正文金额（实测抓到的真错误）    #
    # 案例：CI 生成的稿件标题写「GSK豪掷50亿美元首付」，
    # 正文却写「GSK向恒瑞支付了5亿美元首付」——差 10 倍。
    # 正文是对的，标题错了。而标题是读者第一眼看到的数字，必须拦。
    #
    # 规则：标题里「数字+限定词」（首付/总额/里程碑）的组合，
    #       必须在正文中找到同一组合。限定词不去管的话，
    #       「50亿」会误配到表格里康方的「50亿美元总额」而放行。
    errs.extend(_check_title_numbers(doc, body))

    # 4. 表格列数一致
    for sec in doc.get("sections") or []:
        for b in sec.get("blocks") or []:
            if b.get("type") == "table":
                head = len(b.get("head") or [])
                for r in b.get("rows") or []:
                    if head and len(r) != head:
                        errs.append(
                            f"表格列数不一致：表头 {head} 列，"
                            f"有一行 {len(r)} 列")
                        break

    # 5. 标记残留（模型误把标记写进正文）
    if re.search(r"^\s*(TABLE|BOX|FN|KICKER|TITLE|SUB|HOOK)\s*[:：]",
                 body, re.M):
        pass  # 已被解析器消费，不算错
    if "<p>" in text or "<div" in text:
        errs.append("正文里混入了 HTML 标签（应使用标记格式）")

    return errs


# ============================================================
# 生成
# ============================================================
LENGTH_HINT = """

# 篇幅不足，必须加长

上一版正文只有 {n} 字，目标是 {target} 字。请按下面的方式扩写：

1. 每个 `## ` 小节至少 3 段（当前普遍只有 1-2 段）
2. 交易拆解加一张"条款逐项对照"表格（首付/里程碑/总额/权益范围/开发责任）
3. 增加"同类交易历史对照"段落，用素材里的数据库数据做纵向比较
4. 增加"买方动机"和"卖方视角"两段独立分析
5. 行业背景数据（授权支出占比、领域分布、阶段分布）展开成表格并解读
6. 结尾加"未来 12-18 个月时间表"

**注意：已知信息要展开写透，不要把多个"未找到公开数据"堆成段落。**
"""

STYLE_HINT = """

# 文风有问题

上一版出现了 AI 味表达，必须替换：

- "值得注意的是" → 直接给结论
- "综上所述"/"总而言之" → 直接给结论
- "不言而喻"/"至关重要" → 删掉或换成具体描述
- "赋能" → "支撑""帮上"
- "底层逻辑" → "基本逻辑""实际原因"
- "本文将"/"让我们" → 直接开始讲

篇幅保持不变，只替换词。
"""

STRUCT_HINT = """

# 结构有问题

请检查并修正：

1. 必须有 `TITLE:`、`HOOK:`、至少 3 个 `## ` 小节、以及 `FN ` 脚注
2. `TABLE:` 之后每行的 `|` 数量必须与表头一致
3. 不要输出任何 HTML 标签，只用标记
4. 结尾用 `ENDCARD 标题 | 正文`
"""


async def generate_one(client, eff: dict, plan_item: dict, material: str,
                       style_cfg: dict, out_dir: Path,
                       src_published: str = "") -> dict:
    gen_cfg = style_cfg.get("generation", {})
    retries = gen_cfg.get("max_retries", 3)
    target = gen_cfg.get("target_words", 3000)

    system, user = build_prompt(plan_item, material, style_cfg)
    feedback = ""
    last_doc = None

    for attempt in range(retries + 1):
        msg = user
        if feedback:
            msg += f"\n\n# 上次的问题（必须修正）\n{feedback}\n"
            if "字数" in feedback:
                n = body_chars(last_doc) if last_doc else 0
                msg += LENGTH_HINT.replace("{n}", str(n)).replace(
                    "{target}", str(target))
            if "AI味" in feedback:
                msg += STYLE_HINT
            if any(k in feedback for k in ("缺少", "小节太少", "列数", "HTML")):
                msg += STRUCT_HINT

        try:
            txt = _strip_think(_call_llm(client, eff, system, msg, retries=1))
        except Exception as e:
            print(f"    尝试{attempt+1}: LLM 异常 {type(e).__name__}")
            continue

        if not txt or len(txt) < 200:
            print(f"    尝试{attempt+1}: 返回为空/过短（{len(txt or '')} 字符）")
            # 存下原始输出，便于诊断"为什么模型输出骤降"
            dbg = DATA_DIR / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            (dbg / f"short_{plan_item.get('lead_id','')[:8]}_{attempt+1}.txt"
             ).write_text(txt or "", encoding="utf-8")
            feedback = ("上次返回为空或过短。请输出完整内容："
                        "TITLE、HOOK、至少 3 个 ## 小节、表格、FN 脚注，"
                        "正文 2800-3200 字。")
            continue

        doc = parse_content(txt)
        last_doc = doc
        errs = validate_content(
                doc, txt, style_cfg,
                src_title=plan_item.get('title', ''),
                material=material,
                src_published=src_published)
        st = structure_report(doc)

        print(f"    尝试{attempt+1}: 正文{st['body_chars']}字 "
              f"小节{st['sections']} 表{st['tables']} "
              f"框{st['boxes']} 注{st['footnotes']} "
              f"{'✅通过' if not errs else '❌' + str(len(errs)) + '个问题'}")

        if not errs:
            html = render_html(doc)
            idx = plan_item.get("_idx", 1)
            atype = plan_item.get("type", "event_report")
            fname = f"{idx:02d}-{atype}-{plan_item.get('lead_id','')[:8]}.html"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / fname).write_text(html, encoding="utf-8")
            # 同时存一份标记原文，便于人工比对
            (out_dir / (fname.replace(".html", ".txt"))).write_text(
                txt, encoding="utf-8")
            return {"ok": True, "file": fname, "stats": st,
                    "html_len": len(html)}

        feedback = "\n".join(f"- {e}" for e in errs)
        print(f"      问题: {feedback[:180]}")

    return {"ok": False, "error": feedback[:400],
            "stats": structure_report(last_doc) if last_doc else {}}


# ============================================================
# 主流程
# ============================================================
def run(date: str | None = None) -> list[dict]:
    date = date or today_cst()
    plan = read_json(SCORED_DIR / f"{date}.plan.json", {})
    if not plan or not plan.get("plan"):
        print("[compose] 没有选题计划")
        return []

    style_cfg = load_yaml("style_guide.yaml")
    cfg = load_yaml("scoring.yaml")
    out_dir = POSTS_DIR / date[:4] / date[5:7] / date[8:10]

    print(f"\n=== 生成 {len(plan['plan'])} 篇 -> {out_dir} ===")
    results: list[dict] = []

    async def _all():
        gcfg = cfg.get("gen_llm") or cfg["llm"]
        client, eff = _client(gcfg, cfg.get("providers", {}))
        print(f"       model={eff['model']} max_tokens={eff['max_tokens']}")
        for i, item in enumerate(plan["plan"], 1):
            item["_idx"] = i
            # 幂等：同一 lead_id 只要【任何日期】成过稿就不再生成。
            # ⚠️ 早先只查今天的目录，导致同一笔交易跨天重复发布
            # （诺诚健华×礼来在 10-10 和 10-11 各发了一遍）。
            # 根因是 seen.json 里存了 8 位短 id、比对时按 16 位全量比，
            # 历史去重没拦住；这里再加一道跨日期检查兜底。
            lead = (item.get("lead_id") or "")[:8]
            if lead and POSTS_DIR.exists() and any(
                    lead in p.name
                    for p in POSTS_DIR.rglob("*.html")):
                print(f"\n  [{i}] 已存在成稿（任意日期），跳过 "
                      f"{item.get('title','')[:50]}")
                results.append({"ok": True, "skipped": True,
                                "title": item.get("title", ""),
                                "score": item.get("score", 0),
                                "type": item.get("type", ""),
                                "stats": {}})
                continue
            material = build_material(date, item)
            print(f"\n  [{i}] {item.get('type')} | {item.get('score')}分")
            print(f"      {item.get('title','')[:60]}")
            # 取源发布日期，用于校验 HOOK 里的年份（防模型编日期）
            _lead = item.get("lead_id", "")
            _scored = read_json(SCORED_DIR / f"{date}.json", [])
            _pub = next((x.get("published", "") for x in _scored
                         if x.get("id") == _lead), "")
            if not _pub:
                _raw = read_json(RAW_DIR / f"{date}.json", [])
                _pub = next((x.get("published", "") for x in _raw
                             if x.get("id") == _lead), "")
            r = await generate_one(client, eff, item, material,
                                    style_cfg, out_dir,
                                    src_published=_pub)
            r["title"] = item.get("title", "")
            r["score"] = item.get("score", 0)
            r["type"] = item.get("type", "")
            results.append(r)

    asyncio.run(_all())

    ok = sum(1 for r in results if r.get("ok"))
    new = sum(1 for r in results if r.get("ok") and not r.get("skipped"))
    print(f"\n=== 完成 {ok}/{len(results)} 篇"
          f"（新生成 {new}，跳过已存在 {ok - new}）===")
    for r in results:
        mark = "✅" if r.get("ok") else "❌"
        st = r.get("stats") or {}
        print(f"  {mark} {r.get('type','')[:16]:16} "
              f"{st.get('body_chars',0):>5}字  {r.get('title','')[:36]}")
        if not r.get("ok"):
            print(f"      {r.get('error','')[:140]}")

    # 把【已发表】的选题写进 seen.json，作为跨天去重基准。
    # 只记写成的，不记抓到的——否则没入选的高分条目会被永久埋掉。
    published = [
        {"lead_id": it.get("lead_id"), "title": it.get("title", "")}
        for it, r in zip(plan["plan"], results)
        if r.get("ok")
    ]
    if published:
        try:
            from dedupe import record_published
            record_published(published, today=date)
        except Exception as e:
            print(f"  [seen] 记录失败（不影响成稿）: {type(e).__name__}")

    write_json(DATA_DIR / f"compose_{date}.json", results)
    return results


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    run(sys.argv[1] if len(sys.argv) > 1 else None)