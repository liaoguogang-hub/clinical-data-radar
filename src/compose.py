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
from factguard import check_facts, split_entity_tags, material_is_thin  # noqa: F401
import claimcheck  # noqa: E402

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
            # ⚠️ 新闻站自动实体标签【不是原始披露内容】——显式隔离，
            #    否则成稿会把标签里的靶点/公司当成事实（实测事故：
            #    标签里的一个靶点名被写成了"该药的机制"，还顺藤摸瓜编出
            #    一个原文没有的分子代号）。标签已在抓取阶段从正文剥离。
            etags = (s.get("entity_tags") or []) + (r.get("entity_tags") or [])
            if etags:
                blocks.append(
                    f"⚠️ 新闻站自动标注的实体标签：{'、'.join(etags)}\n"
                    "   这是新闻网站自己的实体识别结果，【不是原文披露的事实】。"
                    "除非正文里明确写了某个靶点/药物，否则不要把它当作"
                    "本次事件的机制、适应症或结论来写。")
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

    # ── 公司背景（可引用）───────────────────────────────────────────
    #
    # 用户 2026-10-17 定的规则：素材不足时，首选扩写方向是"介绍相关公司"。
    # 但扩写**必须有出处**，否则就是下一个胡编入口。所以这里主动去把
    # 可引用的公司公开信息取回来（美国批准产品 / CT.gov 在研管线 / 官网自述），
    # 让成稿写得出来、断言核验也验得过。
    #
    # ⚠️ 取不到就不写（companyinfo 会如实说明"未取到"），绝不编。
    try:
        import companyinfo
        import registry as _R
        # 从标题 + 各素材的标题/摘要/正文开头里找公司。
        # ⚠️ 必须带上正文：实测新闻**标题常常不含公司名**
        #    （如某药 III 期读出那条，标题只有药名，正文才提公司），
        #    只看标题会让公司背景对一大半条目失效。
        _parts = [plan_item.get("title", "")]
        for i in ids:
            _s = by_id.get(i, {}) or {}
            _r = raw_by_id.get(i, {}) or {}
            _parts += [_s.get("title", ""), _s.get("summary", "")[:600],
                       (_s.get("content") or "")[:1500],
                       _r.get("title", ""), (_r.get("content") or "")[:800]]
        _blob = " ".join(p for p in _parts if p)
        _found = _R.find_companies(_blob, limit=1)
        if _found:
            _zh, _en = _found[0]
            brief = companyinfo.company_brief(_zh, _en)
            if brief:
                blocks.append(brief)
                blocks.append("")
    except Exception as _e:
        blocks.append(f"（公司背景补全失败：{type(_e).__name__}，"
                      f"请勿编造公司信息）")
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
- `ENDCARD` 和 `FN` 各只能出现一次、且在末尾，FN 可以多行

  ⚠️ **ENDCARD 是硬格式：`ENDCARD 标题 | 正文`，横线两边都要有内容。**
  实测事故：模型把结尾写成单独一行 `ENDCARD`，解析器不匹配，
  被当普通段落输出成 `<p>ENDCARD</p>`，稿子发出去读者看到两个大写的 ENDCARD。

  ❌ `ENDCARD`                    ← 只有词，两个部分都缺
  ❌ `ENDCARD 这笔交易说明了什么`    ← 缺 `|` 和正文
  ✅ `ENDCARD 真正的看点 | 这不是一次普通的读出，而是赛道的分水岭。`

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

## ⚠️ 素材很短怎么办：写公司与药物背景，别编临床数据（2026-10-16 用户定）

素材常常只有一句标题（一则 CHMP 批准意见、一个会议摘要），
但正文要求 2800-3200 字。**扩写的正确方向是介绍背景**，写它不产生事实风险：

- ✅ 药企背景：主营方向、研发管线规模、已上市产品与商业化能力、出海历史
- ✅ 靶点/疾病背景：该靶点的生物学、已上市同靶点药物的机制与获批情况、
  该适应症的流行病学与现有治疗格局
- ✅ 监管流程背景：CHMP/FDA 审评机制、positive opinion 与正式批准的区别、
  分阶段审评（phased review）是什么
- ✅ 临床研究背景：分期、终点设计、常见统计口径怎么读

**但临床事实本身仍然只能来自素材**，素材没给就写"未披露"，绝不脑补：

- ❌ 具体的分子代号 / 代号（原文没出现的，一个字都不能编）
- ❌ 具体样本量、期中分析时间点、效应量（ORR/PFS/OS 的具体数值）
- ❌ 具体的突变亚型、作用机制断言（素材只说靶点 X，就只写靶点 X）
- ❌ 申办方名称、试验编号（NCT 号）

⚠️ **实测事故（2026-10-16 daraxonrasib 那条）就是这么来的**：素材是一句
"CHMP 建议批准 daraxonrasib 用于经治转移性胰腺癌"。成稿为了凑字数，
编了一个原文没有的分子代号，还断言了具体突变亚型和机制——而这些信息
其实公开可查，正确的做法是**去查**（注册库、公告、学术会议），
查不到就写未披露，而不是自己编。

⚠️⚠️ **"注册库里没有结果" ≠ "数据未公开"。** 这是最贵的一个教训：
   实测某三期试验在 CT.gov 上 `has_results: false`，但它的结果早已在
   ASCO 全体大会发表并同步上了 NEJM。**写"数据未披露"之前，
   必须同时查注册库、公司公告和学术会议**；只有全都查不到才能这么写。

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


def word_bounds(gen: dict) -> tuple[int, int]:
    """
    正文字数的目标区间 (lo, hi)。**唯一口径。**

    ⚠️ 为什么要抽出来（2026-10-21）：
      以前"校验"和"放弃成稿"两处各算各的下限（一个用 config 里的
      word_tolerance=0.70 → 2100，另一个用默认值 0.75 → 2250），
      两处不一致 = 合格稿子被静默扔掉，而且日志里看不出为什么。
      现在两处都走这里，配置改了只有一个地方生效。
    """
    target = gen.get("target_words", 3000)
    tol = gen.get("word_tolerance", 0.75)
    wmax = gen.get("max_words", 3900)
    return int(target * tol), int(wmax)


def word_floor(gen: dict) -> int:
    """
    实际生效的**下限** = 目标下限 - 近下限宽限。

    ⚠️ 近下限宽限（word_floor_slack，默认 0）：
      实测 2026-10-21 有一篇正文 2099 字（下限 2100，差 1 个字）
      的稿子被字数判死——8 个小节、5 条脚注、结语卡齐全、事实全部可溯源，
      却因为 1 个字被扔掉；重试后模型掉到 1588 字，整篇彻底失败。
      硬门槛管的是"明显过短"，不该管"差几个字"。

      口径唯一：validate_content 和 giveup 分支必须都用这个函数。
    """
    lo, _ = word_bounds(gen)
    return max(0, lo - int(gen.get("word_floor_slack", 0) or 0))


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

    # 1. 必需元素
    if not doc.get("title"):
        errs.append("缺少 TITLE")
    if not doc.get("hook"):
        errs.append("缺少 HOOK")
    if len(doc.get("sections") or []) < 3:
        errs.append(f"小节太少（{len(doc.get('sections') or [])} 个，至少 3 个）")
    if not doc.get("footnotes"):
        errs.append("缺少 FN 来源脚注")
    # ⚠️ 结语卡必须检查。
    # 实测事故：模型把结尾写成单独一行 `ENDCARD`（漏了「标题 | 内容」），
    # 解析器不匹配 → 被当普通段落 → 输出 <p>ENDCARD</p>，
    # 稿子发到公众号，读者看到两个大写的 ENDCARD。
    # 当时校验只查 TITLE/HOOK/小节数/脚注，正好漏了这条。
    if not doc.get("end"):
        errs.append(
            "缺少 ENDCARD 结语卡。它必须是【全文最后一块】，写在所有 `## ` "
            "小节之后、FN 脚注之前，格式是 `ENDCARD 标题 | 内容`"
            "（竖线两边都要有内容，例如："
            "`ENDCARD 真正的看点 | 这不是一次普通的读出，而是赛道的分水岭。`）。"
            "不能只写一个孤零零的 ENDCARD，也不要把它甩在正文中段。")
    # 解析层记下的"标记格式不全"一并上报
    for me in (doc.get("marker_errors") or []):
        errs.append(me)

    # 2. 字数
    #
    # 三档（2026-10-21）：
    #   < floor   → 判死（明显过短）
    #   floor~lo  → 【近下限】放行（只差几个字，重写的代价远大于那几字）
    #   > hi      → 判死（超上限）
    n = body_chars(doc)
    lo, hi = word_bounds(gen)
    floor = word_floor(gen)
    if n < floor:
        errs.append(f"正文字数 {n} 低于下限 {floor}（目标 {lo}-{hi}）")
    elif n < lo:
        print(f"      · 正文字数 {n}：接近下限 {lo}（宽限 {lo - floor} 字），"
              f"按近下限放行")
    elif n > hi:
        errs.append(f"正文字数 {n} 超出上限（目标 {lo}-{hi}）")

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
                "HOOK 里没有时间。读者第一眼就要知道这是什么时候的数据"
                "——请在开头写明具体年月日（如「2026年10月8日」或"
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

    # 6. 事实闸门 —— 成稿里有、素材里没有的高风险实体。
    #
    # 实测事故（daraxonrasib 那条，2026-10-16）：素材是一句 CHMP 批准的标题，
    # 成稿为了凑字数和"显得有料"，编了一个原文没有的分子代号，还顺着新闻站
    # 实体标签里的靶点写了一套机制。结构校验（字数/表格/日期）全过——
    # 因为它"写得很完整"，只有拿实体和素材比对才能发现。
    #
    # 放在最后：先确认这是个体完整的稿子，再去质疑它的真伪，
    # 避免对着半成品反复报错烧掉重试次数。
    errs.extend(check_facts(content_text(doc) + " " + (doc.get("title") or ""),
                            material))

    # 7. 素材太薄 —— 直接拒绝，不给"用世界知识填满字数"的机会。
    #
    # 实测事故（retlirafusp alfa，2026-10-17）：当天 63% 的条目正文为空，
    # 成稿阶段凭标题写满 5000 字，把恒瑞的药写成君实的、
    # 把 TGF-β 写成 VEGF、把已发表数据写成"未披露"。
    # 这条必须在最前面就拦住——事后的事实闸门对空素材是静默放行的。
    thin, why = material_is_thin(material)
    if thin:
        errs.append(
            f"素材太薄，禁止成稿：{why}。本条只有标题/少量信息，"
            f"凭世界知识扩写成 2800 字必然产生假事实。"
            f"正确做法是跳过本条，或先补一手材料"
            f"（公司公告 / 期刊 / 学术会议 / 注册库）。")

    return errs


# ============================================================
# 生成
# ============================================================
LENGTH_HINT = """

# 篇幅不足，必须加长

上一版正文只有 {n} 字，目标是 {target} 字。请按下面的方式扩写：

1. 每个 `## ` 小节至少 3 段（当前普遍只有 1-2 段）
2. **介绍公司与药物背景**——这是最推荐的扩写方向，写它不产生事实风险：
   - 申办方的主营方向、研发管线规模、已上市产品与商业化能力、出海历史
   - 靶点/疾病的生物学背景、已上市同靶点药物的机制与获批情况
   - 该适应症的流行病学、现有治疗格局与未满足需求
   - 监管流程背景：positive opinion 与正式批准的区别、分阶段审评是什么、
     常见统计口径（PFS/OS/ORR）怎么读
3. 素材里**确实有**的终点结果展开成表格：主要终点名称 / 试验组数值 /
   对照组数值 / 统计学指标 / 依据层级（注册库统计量 or 公告措辞）
4. 增加"未满足需求"段落：这个适应症现在的治疗现状、还有哪些没被解决
5. 增加"安全性数据"段落（仅当素材里有安全性数字时才写）
6. 结尾加"下一步看什么"的时间表（读出节点、申报计划、随访成熟时间）

**注意：素材里已有的数字要展开写透，不要只罗列。**
**已知信息要写足，不要把多个"未找到公开数据"堆成段落。**

⚠️ **扩写背景 ≠ 可以编临床数据。** 公司/靶点/疾病/监管流程背景可以写，
但分子代号、具体样本量、具体效应量、突变亚型、申办方名称、NCT 号
仍然只能来自素材；素材没披露就写"未披露"，绝不脑补。
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
4. **结尾必须是 `ENDCARD 标题 | 正文` 这一整行**——
   横线两边都要有内容，缺一不可。

   ❌ 错误（只写一个词，读者会看到两个大写的 ENDCARD）：
      ENDCARD
   ❌ 错误（只有标题没有正文）：
      ENDCARD 这笔交易说明了什么
   ✅ 正确：
      ENDCARD 真正的看点 | 这不是一次普通的授权，而是平台级合作的起点。

   **ENDCARD 后面必须跟一个竖线 `|` 和一段正文，这是硬格式。**
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
            # 结构过了，再跑【逐条断言核验】——要调 LLM，所以放在结构校验之后：
            # 别对着格式都不对的半成品烧核查调用。
            #
            # ⚠️ 判据用 claimcheck.should_block()，**不是"有错就拦"**。
            #    2026-10-19 事故：按"有错就拦"接线后，跨语种素材（英文公告 +
            #    中文成稿）的误报把 3 篇稿子全拦了，重试 12 次产出归零，
            #    下游还把它报成"当天没选题"。详见 should_block 的说明。
            print("      → 断言核验…")
            try:
                cok, cerrs, cstat = claimcheck.check_article(
                    client, eff, txt, material)
            except Exception as e:
                # 核查本身异常不能当成"通过"（那会让闸门静默失效）
                cok, cerrs, cstat = False, [f"断言核验异常 {type(e).__name__}"], {}
            block, why = claimcheck.should_block(cstat, cerrs)
            print(f"        断言 {cstat.get('ok', 0)}/{cstat.get('total', 0)} 有出处"
                  f"（无出处 {cstat.get('no_evidence', 0)}，"
                  f"编造出处 {cstat.get('fake_evidence', 0)}，"
                  f"跨语种无法判定 {cstat.get('cross_lang', 0)}）"
                  + (f"  ❌拦：{why}" if block else "  ✅放行"))
            if not block:
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
                        "claims": cstat,
                        "claim_warnings": (cerrs if cerrs else []),
                        "html_len": len(html)}
            # ── 被拦：先把草稿存下来 ────────────────────────────────
            # ⚠️ 原来这里直接丢弃，于是"为什么被拦"事后完全无据可查。
            #
            # ⚠️⚠️ 第一版存进了 `data/debug/`——那个目录**在 .gitignore 里**
            #     （第 21 行），云端跑完随 runner 一起销毁，等于没存。
            #     2026-10-19 就因此又白查了一轮。
            #     现在改存 `data/rejected/`（未忽略），会随"提交产出"进仓库。
            rej = ROOT / "data" / "rejected"
            rej.mkdir(parents=True, exist_ok=True)
            (rej / f"{today_cst()}_{plan_item.get('lead_id','')[:8]}"
                   f"_try{attempt+1}.txt").write_text(
                f"# 拦因：{why}\n# 统计：{cstat}\n# 问题：\n"
                + "\n".join(f"- {e}" for e in cerrs[:40])
                + f"\n\n{txt}", encoding="utf-8")
            errs = cerrs

        # ── 字数不达标 / 断言无依据，重试都救不回来时：放弃，不写 ──
        #    用户 2026-10-17 定的规则：扩写只允许写【可核验的公司背景】；
        #    连那样都到不了字数，就**不写**，不要用世界知识凑。
        if attempt >= retries:
            short = body_chars(doc)
            # ⚠️ 下限必须和 validate_content 用**同一个**口径：
            #    统一走 word_floor()（2026-10-21）。历史上这里出过两次事故：
            #    ① 这里写 target*0.9、validate_content 用 target*0.75，
            #       2250–2700 字的稿子"结构校验通过、却被判成字数不达标"
            #       直接丢弃（2026-10-19）。
            #    ② 两处各读各的 word_tolerance 默认值（0.75 vs config 0.70），
            #       下限对不上，日志里看不出为什么被扔（2026-10-21）。
            #    两个阈值不一致 = 合格稿子被静默扔掉。
            floor = word_floor(gen_cfg)
            if short < floor:
                reason = (f"字数不达标（{short} 字 < 下限 {floor}），"
                          f"扩写可核验内容后仍不足，按规则放弃成稿"
                          f"（不用世界知识凑字数）")
            elif errs:
                reason = f"断言核验未通过：{errs[0][:160]}"
            else:
                reason = "重试用尽"
            # ⚠️ 放弃时也要留证据（2026-10-19 教训）：
            #    这条分支原来只返回一个 reason 字符串，正文直接丢弃，
            #    于是"为什么 3 篇全没成"事后完全查不出来。
            #    存到 data/rejected/（未 gitignore），随提交进仓库。
            try:
                rej = ROOT / "data" / "rejected"
                rej.mkdir(parents=True, exist_ok=True)
                (rej / f"{today_cst()}_{plan_item.get('lead_id','')[:8]}"
                       f"_giveup.txt").write_text(
                    f"# 放弃原因：{reason}\n# 正文 {short} 字（目标 {target}，"
                    f"下限 {int(target * tol)}）\n# 问题：\n"
                    + "\n".join(f"- {e}" for e in (errs or [])[:40])
                    + f"\n\n{txt}", encoding="utf-8")
            except Exception:
                pass
            return {"ok": False, "skipped": True, "reason": reason,
                    "stats": st}

        fact_errs = [e for e in errs if "找不到" in e or "捏造" in e]
        feedback = "\n".join(f"- {e}" for e in errs)
        if fact_errs:
            feedback += (
                "\n\n# ⚠️ 事实错误（必须删掉，不能换个说法保留）\n"
                "上面这些断言在本次素材里**找不到出处**。正确做法：\n"
                "  · 删掉没有出处的断言，或改成素材支持的表述\n"
                "  · 素材只说靶点 X，就只写靶点 X，不要自行补亚型或机制\n"
                "  · 篇幅不够就写【公司与药物背景】——但**背景里的每个数字和\n"
                "    产品名也要能在素材里找到**；素材没有就写不了\n"
                "  · 素材里确实没有、但公开可查的，写进脚注标明需核实渠道，\n"
                "    不要直接当事实写\n"
                "不要解释这些说法是否真实——素材里没有出处就是没有。"
            )
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
            # ⚠️ 素材太薄就别生成（2026-10-17）。
            #    当天 63% 的条目正文为空，成稿却能凭标题写满 5000 字——
            #    retlirafusp alfa 那篇就是这么把恒瑞的药写成君实的。
            #    在这里拦住，比让它生成完再被 validate 拒绝更省（不烧 LLM 调用），
            #    而且日志里能一眼看出"今天有多少条因素材不足被跳过"。
            thin, why = material_is_thin(material)
            if thin:
                print(f"\n  [{i}] ⏭ 素材太薄，跳过：{why}")
                print(f"      {item.get('title','')[:60]}")
                results.append({"ok": False, "skipped": True,
                                "reason": f"素材太薄：{why}",
                                "title": item.get("title", ""),
                                "score": item.get("score", 0),
                                "type": item.get("type", ""),
                                "stats": {}})
                continue
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
    #
    # ⚠️ cluster_key 必须一并记录。同一件临床事件常被多个源报道，
    # 标题措辞差异大（模糊匹配拦不住），只有 cluster_key 相同。
    # 不记的话，同一件事会被不同源各发一遍（实测踩过）。
    published = [
        {"lead_id": it.get("lead_id"), "title": it.get("title", ""),
         "cluster_key": it.get("cluster_key") or ""}
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