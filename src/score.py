"""
LLM 评分模块（临床数据雷达）。

核心价值：把人工分析的判断规则编码进 prompt。
这些规则来自实战踩过的坑：

1. 【疗效数据 ≠ 达到终点】"38 例全部应答、80% 完全缓解"是疗效数字；
    "达到主要终点"是统计结论。终点是否达到取决于试验预设的统计假设
    （α 消耗、缺失数据处理、非劣界值），这些不在新闻里。
    早期把两者混为一谈，会把"数据好看"直接写成"试验成功"。
2. 【失败不降权】III 期失败是行业级事件。不能因为消息是坏的就把分压低。
   这是与 BD 雷达（失败没价值）的根本区别。
3. 【来源分层】注册库统计量（p值/CI）> 公司公告 > 媒体转述。
   同一件事因来源不同，可信度差一倍。
4. 【"未找到"≠"不存在"】查不到写 verify_todo，不下否定结论，绝不编造数值。
5. 【排除非临床】纯 BD 交易/人事/融资不是本雷达的内容，各有归属。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

from openai import OpenAI

from common import (RawItem, SCORED_DIR, RAW_DIR, load_yaml, read_json,
                    write_json, today_cst)

# 临床信号（给 LLM 一个显式提示，不做硬性拦截）。
#
# ⚠️ 正则一定有漏判，所以只作为参考字段，最终由 LLM 判断。
# 实测漏判案例（noise.py 里同源）：
#   "NEJM：CAR-T新星 anito-cel——38 例全部应答，近 80% 完全缓解，
#    3 年生存 65%" —— 当时关键词表里只有"生存期"没有"生存"，
#    只有"应答率"没有"应答"，于是整条被判"无临床信号"。
CLINICAL_SIGNAL = re.compile(
    r"临床|III期|II期|I期|3期|2期|1期|三期|二期|一期|"
    r"主要终点|次要终点|终点|达到|未达到|"
    r"疗效|安全性|不良事件|耐受性|缓解|应答|"
    r"生存|无进展|疾病进展|"
    r"顶线|数据读出|中期分析|入组|获批|批准|"
    r"上市申请|受理|突破性疗法|优先审评|"
    r"ORR|PFS|OS\b|DOR|CR\b|PR\b|"
    r"\bphase\s*[123]\b|\bclinical\s+(?:trial|study|data|results?)\b|"
    r"\bprimary\s+endpoint\b|\bmet\s+the\b|\bmissed\b|\btopline\b|"
    r"\befficacy\b|\bsafety\b|\bapproval\b|\bapproved\b", re.I)


SYSTEM_PROMPT = """你是临床研究（clinical development）领域的资深分析师。
你的任务是给新闻条目打分，识别出真正值得深度报道的**临床数据与进展**。

这个雷达不做 BD 交易（那是另一个雷达），只做：
  · 临床试验结果读出（达到/未达到终点）
  · 试验终止/暂停/失败
  · 关键进展节点（入组完成、期中分析、启动注册试验）
  · 监管审批与上市申请（作为临床进展的结果）

# 评分维度（总分 10 分）

## 1. 终点结果 (0-3.0) —— 权重最高
- 主要终点达到且统计学显著 = 3.0
- **主要终点未达到（明确失败）= 2.8**
- 因安全性问题终止 = 3.0
- 达到但未达预设阈值 / 部分达到 = 1.5
- 疗效数据积极但未明确终点结论 = 1.2
- 仅有进展（入组完成/试验启动）= 0.5
- 完全无法判定 = 0.6

## 2. 阶段与规模 (0-2.0)
- III 期 / 注册试验 = 2.0（样本量 >500 再加 0.3，但不超过 2.0）
- II 期 = 1.2
- I 期 = 0.5
- 研究者发起 / 回顾性研究 = 0.3
- 前瞻性队列 / 真实世界研究 = 0.4

## 3. 未满足需求 (0-1.5)
- 无有效疗法（胰腺癌、胶质母细胞瘤等）= 1.5
- 现有疗法有限、预后差 = 1.0
- 已有多种疗法但仍有空间 = 0.6
- 竞争激烈（PD-1、GLP-1、IL-17 等）= 0.3

## 4. 首创性 (0-1.5)
- 全球首个同机制/靶点 = 1.5
- 全球首个同适应症 = 1.2
- 国产首个 = 1.0
- 国内前三家 = 0.6
- me-too / 仿制 = 0.2

## 5. 数据成熟度与可信度 (0-2.0) —— 可信度闸门
- 顶刊正式发表（NEJM / Lancet / Nature Medicine / JAMA）= 2.0
- 学术会议口头报告（ASCO / ESMO / ASH 全会）= 1.5
- 会议摘要 / 壁报 = 1.2
- 注册库有统计检验（p 值 / 置信区间）= 1.3
- 公司顶线公告 = 1.0
- 注册库仅裸数值、无统计检验 = 0.6
- 仅有媒体转述、无原始出处 = 0.5

同一件事因来源不同分值可差一倍——这是刻意的，
防止把"媒体转述"当成"确证结论"。

# 必须执行的核查（历史教训，务必遵守）

**① 疗效数据 ≠ 达到终点（最容易犯的错）**
"38 例全部应答，近 80% 完全缓解，3 年生存 65%" 是**疗效数据**；
"达到主要终点"是**统计结论**。两者不是一回事。

终点是否达到取决于试验**预设的统计假设**（α 消耗、缺失数据处理、
非劣界值等），这些不在新闻里。所以：
- 标题只说疗效数字、没说达到终点 → 归入"疗效数据积极"，最多 1.2 分
- 只有在有统计检验（p 值/CI）或权威来源明确表述时才给高分
- reasoning 里必须写明你依据的是"统计结论"还是"疗效数字"

**② 失败和终止不降权，反而要重视**
这是临床雷达与 BD 雷达的根本区别。
III 期失败是行业级事件，可能影响整个赛道。
"未达到主要终点""因安全性终止""中期分析无效"都应给高分（2.5-3.0）。
不要因为它是坏消息就压低分数。

**③ 来源分层（重要）**
条目里可能带 `verdict` 字段（来自注册库与措辞的自动判定）：
- `basis=statistical`：注册库有 p 值/CI，**可信**
- `basis=registry_status`：注册库状态记录，**可信**
- `basis=textual`：新闻措辞，可信度中等
- `basis=numeric`：仅裸数值推断，**不可信，必须降权**

有可信统计量的，`data_maturity` 给 1.3 以上；
只有媒体措辞的，不超过 1.0。

**④ "未找到"≠"不存在"**
查不到使用"未找到公开数据"，不要下否定结论，写进 `verify_todo`。
绝对不能编造终点结果、数值、样本量。

**⑤ 排除非临床内容**
本雷达不做：
- 纯 BD 交易（授权/收购/合作，无临床数据）→ flags 加 "not_clinical"，score ≤ 3.0
- 人事变动 → flags 加 "personnel"，score ≤ 1.0
- 融资/IPO → score ≤ 2.0
- 综述、预告、盘点类（无具体数据）→ score ≤ 4.0

**⑥ 时效性**
条目带 `published` 和 `age_days`：
- 超过 7 天：降外溢影响分，reasoning 里注明"非近日"
- 超过 14 天：直接降 2 分以上
- 未知：按内容评分，但注明"发布日期未知"
不要把旧数据读出当成今天的新闻。

**⑦ 扣分项（在 reasoning 里说明）**
- 单臂无对照：-0.5
- 替代终点（如 ORR 代替 OS）：-0.3
- 样本量 < 100：-0.5
- 亚组 / 事后分析：-0.5
- 仅动物或体外数据：score ≤ 2.0

**⑧ 中国 / 全球分开**
条目要标出属于哪个栏目：
- `china`：中国原研、中国申办方、中国主要站点、NMPA/CDE 相关
- `global`：其余
在 reasoning 里说明依据。

**⑨ reasoning 只写"为什么值得报道"，不写事实结论（2026-10-17 新增）**

`reasoning` 会被**原样喂给成稿阶段**当"选题理由"。你在这里下的每一个
事实判断，成稿都会当成既成事实继续扩写——下游所有环节都在你的前提上做加法。
**评分只负责选材，不负责定稿。**

- ❌ "Retlirafusp alfa（同类首创靶向 **PD-L1/VEGF** 的双功能抗体）Ⅲ 期改善 OS"
      —— 机制是错的（实为 PD-L1/TGF-β），而这条错误结论被成稿放大成了
      整篇 5000 字的错误叙事，还连带编出企业归属和分子代号
- ❌ "该药已获 FDA 批准" / "这是首个口服 GLP-1"（素材没说的结论）
- ✅ "Ⅲ 期 HER2- 胃癌试验报告 OS 改善，中国原研，属明确临床获益信号；
      机制与监管状态需在成稿前核实"

**如果机制、企业归属、批准状态这些你并不确定，就不要在 reasoning 里替它下结论，
写进 `verify_todo`。** 素材（content）为空时尤其如此——那时你手里只有一个标题。

# 输出格式（严格 JSON，不要任何解释文字）

返回一个 JSON 数组，每个元素：
{
  "id": "条目的 id（原样返回）",
  "score": 0-10 的数字,
  "channel": "china" | "global",
  "cluster_key": "同药或同靶点的归并键，格式如 'drug:retatrutide' 或 'target:glp-1r'。无法判断填空字符串",
  "flags": ["not_clinical" | "personnel" | "halted" | "failed" | "met" |
            "positive_data" | "stale" | "single_arm" | "surrogate" |
            "small_n" | "subgroup" | "preclinical"],
  "endpoint_verdict": "met" | "missed" | "halted" | "partial" |
                      "positive_data" | "unclear" | "none",
  "verdict_basis": "statistical" | "registry_status" | "textual" |
                   "numeric" | "none",
  "confidence": 0-1 的数字,
  "reasoning": "一到两句话，说明为什么给这个分。必须点明依据的是统计结论还是疗效数字",
  "verify_todo": ["需要核实的事项，没有就空数组"],
  "article_type": "readout" | "halt" | "landscape" | "milestone"
}

`cluster_key` 很重要：多条讲同一个药（或同靶点）的条目会被合并成
一篇横向综述，而不是写成 N 篇重复快报。判断依据是药名/靶点，
不是适应症——同一个药做多个适应症仍算同一簇。

article_type 说明：
- readout：单试验结果读出
- halt：终止 / 失败复盘
- landscape：同靶点或同适应症多个试验，适合横向综述
- milestone：入组完成 / 期中分析 / 启动注册试验等节点

score < 6.0 的条目也要返回，但会被选题阶段丢弃——
这样做是为了让选题层看到全量分布。
"""



def _client(cfg: dict, all_providers: dict) -> tuple[OpenAI, dict]:
    """
    按配置创建客户端，自动选择 provider。
    all_providers 来自 scoring.yaml 顶层的 providers 表。
    返回 (client, 生效参数)，生效参数可能被 providers 表覆盖。
    """
    name = cfg.get("provider", "deepseek")
    provs = all_providers or {}

    p = provs.get(name)
    if p:
        base_url = p.get("base_url")
        model = p.get("model")
        max_tokens = p.get("max_tokens", cfg.get("max_tokens", 8000))
        json_mode = p.get("json_mode", False)
        env_names = [p.get("env_key", ""), "LLM_API_KEY", "OPENAI_API_KEY"]
    else:
        base_url = cfg.get("base_url")
        model = cfg.get("model")
        max_tokens = cfg.get("max_tokens", 8000)
        json_mode = cfg.get("json_mode", False)
        env_names = ["LLM_API_KEY", "DEEPSEEK_API_KEY", "MINIMAX_API_KEY",
                     "OPENAI_API_KEY"]

    api_key = None
    for env in env_names:
        if env and os.getenv(env):
            api_key = os.getenv(env)
            break
    if not api_key:
        raise RuntimeError(
            f"缺少 API key。请设置 LLM_API_KEY（provider={name}）")

    eff = {"model": model, "max_tokens": max_tokens, "json_mode": json_mode,
           "temperature": cfg.get("temperature", 0.3),
           "base_url": base_url, "provider": name}

    print(f"[llm] provider={name} model={model}")
    print(f"     base_url={base_url} json_mode={json_mode}")
    return OpenAI(api_key=api_key, base_url=base_url,
                  timeout=180.0, max_retries=2), eff


def _strip_think(text: str) -> str:
    """
    MiniMax 等模型默认开启思考，内容里会混入 <think>...</think>。
    必须剥掉，否则 JSON 解析失败。
    """
    if not text:
        return ""
    # 情况1：完整的 <think>...</think> 前缀
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S)
    # 情况2：未闭合的截断式思考（max_tokens 用尽时常见）
    if "</think>" not in text and "<think>" in text:
        text = text.split("<think>", 1)[1]
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
    # 情况3：残留的空标签
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def _build_items(items: list[RawItem],
                 enrich: dict | None = None) -> list[dict[str, Any]]:
    """
    构造给 LLM 的 payload。

    enrich 是 {item_id: {"verdict":..., "registry":...}}，
    来自 registry.py + verdict.py 的一手材料与终点判定。
    有它就带上，让评分能按"来源层级"赋权——
    有统计量的和只有媒体措辞的，data_maturity 分值差一倍。
    """
    enrich = enrich or {}
    out = []
    for it in items:
        ex = enrich.get(it.id) or {}
        v = ex.get("verdict") or {}
        reg = ex.get("registry") or {}
        best = reg.get("best") or {}
        out.append({
            "id": it.id,
            "title": it.title,
            "source": it.source,
            "lang": it.lang,
            "url": it.url,
            "published": it.published or "未知",
            "age_days": (round(it.age_hours / 24, 1)
                         if it.age_hours >= 0 else "未知"),
            "access": it.access,
            # 素材量：0 表示只有标题、没有正文，写作时极易脑补
            "material_chars": len(it.content or it.summary or ""),
            # 临床信号：仅作提示，最终由你判断
            "clinical_signal": bool(CLINICAL_SIGNAL.search(it.title)),
            # 终点判定（来自 verdict.py），字段缺失表示尚未做一手材料检索
            "endpoint_verdict": v.get("verdict") or "",
            "verdict_basis": v.get("basis") or "",
            "verdict_confidence": v.get("confidence"),
            "halt_reason": v.get("halt_reason") or "",
            # 注册库（来自 registry.py）
            "registry_nct": best.get("nct") or "",
            "registry_phase": best.get("phase") or "",
            "registry_status": best.get("status") or "",
            "registry_enrollment": best.get("enrollment"),
            "registry_has_results": best.get("has_results"),
            "registry_note": reg.get("note") or "",
            "summary": (it.summary or it.content)[:800],
            "multi_source": [t for t in it.topics if t.startswith("also:")],
        })
    return out


def _parse(text: str) -> list[dict]:
    """LLM 返回可能带 markdown 包裹或前后废话，做健壮解析"""
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 优先整段 JSON 数组
    i, j = text.find("["), text.rfind("]")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except Exception:
            pass

    # 退化：逐个捞对象（支持嵌套括号）
    objs, depth, start, in_str, esc = [], 0, -1, False, False
    for k, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = k
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    o = json.loads(text[start:k + 1])
                    if isinstance(o, dict) and "id" in o:
                        objs.append(o)
                except Exception:
                    pass
                start = -1
    return objs


def _prefilter(items: list[RawItem], cfg: dict) -> list[RawItem]:
    """
    规则层预筛：明显不属于本雷达的直接丢弃，省 LLM 调用。

    实测（药时代 RSS）10 条里混着两类非临床内容，必须先剔掉：
      · 纯 BD 交易   "三迭纪与强生达成合作"              ← 归 BD 雷达
      · 人事变动     "百济神州高管履新吉利德…"            ← 两边都不做

    ⚠️ 只在**标题**上判定，且只在"没有临床信号"时才丢。
    因为交易公告经常同时含临床数据（"XX授权YYY，III期数据亮眼"），
    那种应该留下——所以条件是"命中排除词 且 无临床信号"。
    """
    excl = cfg.get("exclude_keywords", {}) or {}
    groups = {}
    for k in ("pure_bd", "personnel", "pure_financing"):
        groups[k] = [w.lower() for w in (excl.get(k) or [])]

    out = []
    for it in items:
        t = it.title.lower()
        has_clinical = bool(CLINICAL_SIGNAL.search(it.title))
        hit = next((k for k, ws in groups.items()
                    if ws and any(w in t for w in ws)), None)
        # 命中排除词但没有临床信号 → 丢弃
        if hit and not has_clinical:
            continue
        out.append(it)
    return out


def _call_llm(client: OpenAI, eff: dict, system: str, user: str,
              retries: int = 2) -> str:
    """统一调用入口，自动处理 json_mode 降级和 think 标签"""
    kwargs = {
        "model": eff["model"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": eff["temperature"],
        "max_tokens": eff["max_tokens"],
    }
    if eff.get("json_mode"):
        kwargs["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(**kwargs)
            content = r.choices[0].message.content or ""
            # 若 json_mode 生效但内容被 think 污染，退回普通模式
            if not content.strip() and eff.get("json_mode"):
                kwargs.pop("response_format", None)
                eff["json_mode"] = False
                continue
            return _strip_think(content)
        except Exception as e:
            msg = str(e)
            # 不支持 response_format 就降级
            if "response_format" in msg or "json_object" in msg.lower():
                kwargs.pop("response_format", None)
                eff["json_mode"] = False
                last_err = e
                continue
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    if last_err:
        raise last_err
    return ""


def score_items(items: list[RawItem], cfg: dict, client: OpenAI,
                eff: dict) -> list[dict]:
    batch = cfg.get("batch_size", 30)
    retries = cfg.get("max_retries", 2)
    results: list[dict] = []

    pre = _prefilter(items, cfg)
    print(f"[score] 预筛后 {len(pre)} 条（原始 {len(items)}）")

    for i in range(0, len(pre), batch):
        chunk = pre[i:i + batch]
        user = f"请为以下 {len(chunk)} 条临床数据新闻打分。\n" \
               f"只输出 JSON 数组，不要任何其他文字。\n\n" + \
               json.dumps(_build_items(chunk), ensure_ascii=False, indent=1)
        try:
            txt = _call_llm(client, eff, SYSTEM_PROMPT, user, retries)
            parsed = _parse(txt)
            results.extend(parsed)
            print(f"  [batch {i//batch + 1}] 返回 {len(txt)} 字符，"
                  f"解析 {len(parsed)}/{len(chunk)}")
            time.sleep(1.0)
        except Exception as e:
            print(f"  [batch {i//batch + 1}] 失败: {type(e).__name__}: {e}")
    return results


def run(date: str | None = None) -> list[dict]:
    date = date or today_cst()
    cfg = load_yaml("scoring.yaml")

    path = RAW_DIR / f"{date}.deduped.json"
    raw_items = read_json(path, [])
    if not raw_items:
        raw_items = read_json(RAW_DIR / f"{date}.json", [])
    if not raw_items:
        print(f"[score] 没有待评分数据: {path}")
        return []

    items = [RawItem.from_dict(d) for d in raw_items]
    client, eff = _client(cfg["llm"], cfg.get("providers", {}))
    scored = score_items(items, cfg, client, eff)

    # 回填原始信息
    by_id = {i.id: i for i in items}
    for s in scored:
        src = by_id.get(s.get("id", ""))
        if src:
            s["title"] = src.title
            s["url"] = src.url
            s["sources"] = [src.source] + [
                t.split(":", 1)[1] for t in src.topics if t.startswith("also:")]
            s["content"] = src.content
            s["entity_tags"] = list(src.entity_tags)
            s["summary"] = src.summary
            s["lang"] = src.lang
            # ⚠️ published 必须回填。实测踩坑：漏了这个字段后，
            # 成稿模型拿不到发布日期，于是**自己编年份**——
            # 2026-10-09 的新闻被写成"2025年10月8日"。
            # 日期是临床数据稿的核心事实（读者据此判断时效性），
            # 模型看不到就一定会猜。
            s["published"] = src.published
            s["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    write_json(SCORED_DIR / f"{date}.json", scored)

    print(f"\n[score] 完成 {len(scored)} 条 -> data/scored/{date}.json")
    if scored:
        thr = cfg["score"]["must_publish"]
        hot = [s for s in scored if s.get("score", 0) >= thr]
        print(f"  >= {thr} 分：{len(hot)} 条")
        for s in scored[:8]:
            print(f"    {s.get('score', 0):>4} | {s.get('title', '')[:62]}")
    return scored


if __name__ == "__main__":
    # Windows 控制台默认 GBK，输出特殊符号会崩
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    run(sys.argv[1] if len(sys.argv) > 1 else None)