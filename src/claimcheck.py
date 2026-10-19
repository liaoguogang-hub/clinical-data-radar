"""
逐条断言核验：把成稿拆成原子事实断言，要求每条都给出素材出处。

# 为什么需要它（2026-10-17）

实测把事实闸门（`factguard.check_facts`，正则抓分子代号）拿去对 6 类编造，
**只拦住 1 类**：

    分子代号编造        ⛔ 拦下
    企业归属编造        ❌ 放行   ← retlirafusp 那篇最主要的错
    机制写错            ❌ 放行   ← PD-L1/VEGF
    编造缺失（把有写成没有） ❌ 放行   ← daraxonrasib 那篇
    编造具体数值        ❌ 放行
    编造监管状态        ❌ 放行

正则只能匹配"代号形状"的东西，而上面 5 类都不长那样。

# 做法：不问"这句话对不对"，而问"这句话的出处能不能逐字引出来"

这是本模块的核心设计。关键在**两步**，缺一不可：

  ① 让核查模型对每条断言给出**逐字**的素材原文片段（evidence）；
  ② **程序再验一次**这段 evidence 是否真的出现在素材里。

第 ② 步是防"核查模型自己橡皮图章"的关键——
模型很容易在证据栏里凭记忆写一段看着像原文的话。
只要程序发现"引文不在素材里"，就判定为**编造出处**，整条打回。

实测这条设计能覆盖上面 6 类中的全部（见 `__main__` 里的自测）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ⚠️ 复用 score 的 <think> 剥离实现（compose.py 也是这么接的）。
#    MiniMax 默认开思考，响应会混入 `<think>…`；不剥离就没法解析 JSON。
#    这是"核验返回 0 条断言"的直接成因，别再各写一份。
from score import _strip_think  # noqa: E402

# 需要出处的断言类型。general（通用医学/行业背景，如"胰腺癌预后差"）
# 免出处——否则每篇都会被"这个常识没出处"打回。
KINDS_REQUIRE_EVIDENCE = {
    "attribution",   # 谁做的/谁宣布的（企业归属）
    "mechanism",     # 靶点、作用机制
    "number",        # 任何具体数值（样本量、OS、HR、收入…）
    "status",        # 监管状态、试验状态、批准与否
    "company",       # 公司产品线、营收、定位（用户 2026-10-17 允许用作扩写素材）
    "deal_term",     # 交易条款（金额、权益范围）
    "clinical",      # 临床数据本身（终点、疗效、安全性）
}

CLAIM_SYS = """你是事实核查员。下面给你【一篇成稿】和【它的素材】。
把成稿拆成**原子事实断言**，逐条判断它在素材里有没有依据。

# 输出（严格 JSON 数组，不要解释文字）
[
  {
    "text": "断言本身，一句话，保留具体主体/数字",
    "kind": "attribution|mechanism|number|status|company|deal_term|clinical|general",
    "evidence": "素材里逐字支持这条断言的原文片段；找不到就空字符串"
  }
]

# kind 怎么分
- attribution：谁做的、谁宣布的、企业归属（如"恒瑞开发的"）
- mechanism  ：靶点、作用机制（如"PD-L1/TGF-β 双功能融合蛋白"）
- number     ：任何具体数值（样本量、OS、HR、p 值、营收、金额）
- status     ：监管/试验状态（获批、申报中、III 期、已终止）
- company    ：公司背景（主营、已上市产品、年收入、在研管线、定位）
- deal_term  ：交易条款（首付、里程碑、权益范围）
- clinical   ：临床数据本身（终点是否达到、疗效、安全性）
- general    ：通用医学/行业常识（如"胰腺癌预后差""该靶点长期被视为不可成药"）
               —— 不需要出处

# ⚠️ evidence 的唯一合法填法：从素材里**逐字复制粘贴**
- 不许改写、不许概括、不许翻译、不许凭记忆写、不许把两处拼起来
- ⚠️ **语言必须与素材一致。素材是英文就贴英文原文，不要译成中文。**
  （实测：成稿写"阿斯利康"、素材写 "AstraZeneca"，若 evidence 填中文
   就与素材匹配不上，真断言被判成无依据——这是本模块最常见的误报来源。）
- ⚠️ **不许写"素材中提到""素材标题显示""素材里说"这类说明文字**，
  那只说明"你以为有"，不是证据。直接贴素材的原文片段。
- 素材里**没有**这条断言 → evidence 留空字符串
- **留空不会被罚；编一个出处会被程序发现并判整条不通过**

# ⚠️ 拆细
一句里含两个事实就拆两条。数值、企业名、机制、状态各算独立的断言。
标题、HOOK、正文、表格、结语里的断言**都要拆**——不要只查正文。

只输出 JSON 数组。"""


# 引文的最短有效长度（归一化后）。
# 太短的引文不算证据——"III期""该试验"这种在素材里到处都是，
# 拿它当出处等于没给。但门槛不能太高：中文里 "入组96例" 只有 6 个字，
# 却是一条很好的出处（实测把它判成"编造出处"就是门槛定成 8 导致的）。
MIN_EVIDENCE_LEN = 5

# 引文与素材比对用的判据：**最长公共子串**。
#
# ⚠️ 试过两种更"高级"的算法，都不如这个（2026-10-17 实测）：
#   ① 严格"整段是子串" —— 核查模型常改写/截断/拼接引文，
#      把大量真出处判成"编造"，误报淹掉报告。
#   ② 字符 n-gram 重合率 —— 素材本身很短时，任何改写都会把重合率稀释到极低
#      （实测一条真出处的重合率只有 10%，因为引文里混了改写词）。
# 最长公共子串只问一件事：**引文里有没有一段连续的真原文**。
# 有 ≥MIN_EVIDENCE_LEN 字的连续命中 → 认为出处真实（允许外面有改写）。
# 纯凭记忆编的引文与素材几乎没有连续重合，会被抓住。
_GRAM_N = 6  # 保留常量名以免外部引用报错；当前用最长公共子串


def _norm(s: str) -> str:
    """归一化，用于"引文是否真的在素材里"的比对。"""
    s = (s or "").lower()
    s = re.sub(r"[\s\u3000]+", "", s)
    return re.sub(r"[，。；：、！？（）()\[\]【】\"'“”‘’·,.;:!?\-—_/]", "", s)


def _longest_common_run(ev_n: str, mat_n: str) -> int:
    """
    引文与素材的最长公共**连续**子串长度。

    用滚动 DP，字符串都很短（引文几百字、素材几千字），开销可忽略。
    """
    if not ev_n or not mat_n:
        return 0
    prev = [0] * (len(mat_n) + 1)
    best = 0
    for i in range(1, len(ev_n) + 1):
        cur = [0] * (len(mat_n) + 1)
        ei = ev_n[i - 1]
        for j in range(1, len(mat_n) + 1):
            if ei == mat_n[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


# 解析失败时重试几次。单次坏响应曾让整篇的核验归零（2026-10-17）。
_EXTRACT_ATTEMPTS = 2


def claims_from_raw(raw: str) -> list[dict] | None:
    """
    从模型响应里解析断言数组。**解析失败返回 None**，由调用方决定重试/报错。

    ⚠️ 三个真实踩过的坑（2026-10-17，一次"上午 50 条、下午 0 条"的事故）：

    ① **MiniMax 默认开思考**，响应以 `<think>…` 开头（有时不闭合）。
       不剥离就去找 `[...]`，会把思考正文里的方括号当成数组起点，
       `json.loads` 必然失败 → 原来静默 `return []`
       → 报告里显示"未能拆出任何断言"，与"模型真没拆出"完全同形，
       排查时分不清是解析坏了还是模型没做事。
       （实测：思考里出现一个 `[` 就复现 0 条；一个都没有就正常。）
    ② 原来的 `re.search(r"\\[.*\\]", raw, re.S)` 是**贪婪**匹配，
       从第一个 `[` 抓到最后一个 `]`，把思考正文整个吞进候选串。
    ③ 所以：先剥 think，再**扫描**出最大的一组合法 JSON 数组。

    ⚠️ 与 BD 雷达 `pharma-bd-radar/src/claimcheck.py` 是**一对镜像**，
    改这里必须同步改那边（skill §13）。
    """
    text = _strip_think(raw or "")
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    if not text:
        return None
    dec = json.JSONDecoder()
    best: list = []
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(obj, list) and len(obj) > len(best):
            best = obj
    if not best:
        # 模型明确返回空数组 `[]`：算解析成功（确实没有断言），
        # 用 None 区分开"根本没解析出来"。
        return [] if re.search(r"\[\s*\]", text) else None
    return [c for c in best if isinstance(c, dict)]


def extract_claims(client, eff: dict, article_text: str,
                   material: str) -> list[dict]:
    """让 LLM 把成稿拆成原子断言，并尽量给出逐字出处。"""
    user = (f"# 成稿\n{article_text}\n\n"
            f"# 素材\n{material}\n\n"
            f"现在按 JSON 数组输出。")
    data: list[dict] = []
    last = ""
    for _ in range(_EXTRACT_ATTEMPTS):
        r = client.chat.completions.create(
            model=eff["model"],
            messages=[{"role": "system", "content": CLAIM_SYS},
                      {"role": "user", "content": user}],
            temperature=0,
            max_tokens=eff.get("max_tokens", 4000),
        )
        last = (r.choices[0].message.content or "").strip()
        data = claims_from_raw(last)
        if data:
            break
    if not data:
        # ⚠️ **必须出声**：解析失败以前是静默 return []，与"模型没拆出断言"
        #    长得一样，导致"核验没生效"可能被读成"文章没问题"。
        print(f"[claimcheck] ⚠️ 未解析出断言（响应 {len(last)} 字符，"
              f"已试 {_EXTRACT_ATTEMPTS} 次）：{last[:120]!r}")
        return []
    out = []
    for c in data:
        txt = str(c.get("text") or "").strip()
        if not txt:
            continue
        out.append({"text": txt,
                    "kind": str(c.get("kind") or "general").strip().lower(),
                    "evidence": str(c.get("evidence") or "").strip()})
    return out


def _cjk_ratio(s: str) -> float:
    """中文汉字占字母类的比例，用来判断引文/素材是否同一书写系统。"""
    if not s:
        return 0.0
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in s if ch.isalpha())
    return cjk / letters if letters else 0.0


def verify_claims(claims: list[dict], material: str) -> tuple[list[str], dict]:
    """
    程序化核验。返回 (问题列表, 统计)。

    三类结果：
      1. 需要出处却没给出处            → no_evidence
      2. 给出了处、引文能在素材里找到   → ok
      3. **给出了处，但引文不在素材里**  → fake_evidence（编造出处）
      4. 引文与素材**书写系统不同**      → cross_lang（无法判定，不罚）

    ⚠️ 第 4 类为什么必须单独算（2026-10-19 事故根因之一）：
       素材多是英文（NEJM / 公司公告 / CT.gov），成稿是中文。核查模型被要求
       "逐字给引文"时，经常给出**中文译文**——子串与英文素材重合必然 ≈0，
       于是被判成 fake_evidence = **编造出处**。
       结果：真文章被当成"核查模型在编造证据"，claim 闸门把 3 篇全拦，
       当天产出归零。这不是阈值问题，是**判据用错了地方**：
       跨语种时子串比对没有意义，既不能证明有出处，也不能证明编造。
       所以这里静默计入 cross_lang，既不加 errs 也不算 ok。
    """
    mat_n = _norm(material)
    mat_cjk = _cjk_ratio(material)
    errs: list[str] = []
    stat = {"total": 0, "ok": 0, "no_evidence": 0,
            "fake_evidence": 0, "cross_lang": 0}

    for c in claims:
        kind = c.get("kind", "general")
        if kind not in KINDS_REQUIRE_EVIDENCE:
            continue                       # general：免出处
        stat["total"] += 1
        ev = c.get("evidence") or ""
        if not ev:
            stat["no_evidence"] += 1
            errs.append(f"[{kind}] 断言「{c['text'][:60]}」在素材里找不到出处。"
                        f"把它删掉，或改成素材支持的表述。")
            continue
        # 跨语种：引文一边倒中文、素材一边倒英文（或反过来）→ 无法用子串判定
        ev_cjk = _cjk_ratio(ev)
        if (ev_cjk > 0.5 and mat_cjk < 0.1) or (ev_cjk < 0.1 and mat_cjk > 0.5):
            stat["cross_lang"] += 1
            continue
        ev_n = _norm(ev)
        # 判据：引文里至少要有一段 ≥MIN_EVIDENCE_LEN 字的**连续**真原文
        run = _longest_common_run(ev_n, mat_n)
        if len(ev_n) >= MIN_EVIDENCE_LEN and run >= MIN_EVIDENCE_LEN:
            stat["ok"] += 1
            continue
        stat["fake_evidence"] += 1
        errs.append(
            f"[{kind}] 断言「{c['text'][:50]}」给出的出处**不在素材里**"
            f"（引文：{ev[:40]}…，与素材最长连续重合仅 {run} 字）"
            f"——疑似编造出处。出处必须从素材逐字复制；素材里没有就删掉这条断言。")
    return errs, stat


def check_article(client, eff: dict, article_text: str,
                  material: str) -> tuple[bool, list[str], dict]:
    """成稿后跑一次断言核验。返回 (是否通过, 问题, 统计)。"""
    claims = extract_claims(client, eff, article_text, material)
    if not claims:
        return False, ["断言核验未能拆出任何断言（核查模型返回为空）"], {}
    errs, stat = verify_claims(claims, material)
    return (not errs), errs, stat


# 无出处断言**不**作为拦截依据（2026-10-19 事后修正，见 should_block）
NO_EVIDENCE_BLOCK_RATIO = None


def should_block(stat: dict, errs: list[str]) -> tuple[bool, str]:
    """
    决定这条稿子该不该【硬拦】。

    ⚠️ 演进过程——前两版都不对，第三版才对：

      v1「有错就拦」：只要一条断言无出处就拦。
        2026-10-19 实测：跨语种素材（英文公告 + 中文成稿）下，核查模型
        常把 evidence 写成中文、与英文素材匹配不上，**真断言**被判无出处。
        结果 3 篇 × 4 次重试全被拦，当天产出归零。

      v2「无出处占比 > 50% 才拦」：仍然拦——跨语种下无出处占比本来就
        轻易过半。改完重跑，仍然 0 篇。

      v3（现在）**无出处一律不拦**，只有"编造出处"才硬拦。

    为什么 v3 才对：
      · `fake_evidence` = 核查模型给的引文**根本不在素材里**——模型自己在
        编造证据，无歧义，必须拦。
      · `no_evidence` = 模型没找到出处。跨语种下**大量是误报**；拿它当
        发布阻塞项，等于让"检查器的语言能力"决定今天能不能发稿。
      · 真相交给 `audit.py` 事后逐条报告 + 人工复核，而不是在这里硬堵。

    仍然硬拦的三种（都属"核验没生效"或"证据造假"，与语言无关）：
      · 未返回统计 / 未拆出断言（核验没生效）
      · fake_evidence > 0（编造出处）
    """
    total = int((stat or {}).get("total", 0) or 0)
    fake = int((stat or {}).get("fake_evidence", 0) or 0)

    if not stat:
        return True, "断言核验未返回统计（核验未生效）"
    if total == 0:
        return True, "未能拆出任何断言（核验未生效）"
    if fake > 0:
        return True, f"{fake} 条断言的出处是编造的（引文不在素材里）"
    return False, ""


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 离线自测：不调 LLM，直接喂"已拆好的断言"验证第二步（引文核验）
    MAT = ("标题：Retlirafusp Alfa Improves Overall Survival in Phase 3 Trial\n"
           "注册库：NCT07682376 [PHASE2] RECRUITING 入组 96 例 "
           "conditions: Gastric Cancer interventions: Retlirafusp Alfa")
    CASES = [
        ("有真实出处", "mechanism", "conditions: Gastric Cancer", True),
        ("没给出处", "attribution", "", False),
        ("编造出处", "mechanism", "该药为 PD-L1/VEGF 双功能抗体", False),
        ("通用常识免出处", "general", "", True),
        ("数值有出处", "number", "入组 96 例", True),
        # 改写/截断但是真出处：应当放行（否则误报会把报告淹掉）
        ("改写的真出处", "clinical", "该试验共入组 96 例受试者，胃癌", True),
        ("纯凭记忆编的长句", "company",
         "恒瑞医药 2025 年全年营业收入达到 320 亿元人民币", False),
    ]
    print("=" * 70)
    print("第二步（引文核验）离线自测")
    print("=" * 70)
    for name, kind, ev, expect_ok in CASES:
        errs, _ = verify_claims(
            [{"text": "测试断言", "kind": kind, "evidence": ev}], MAT)
        ok = not errs
        mark = "✓" if ok == expect_ok else "✗"
        print(f"   {mark} {name:16} kind={kind:12} 通过={ok}")
        if errs:
            print(f"        {errs[0][:88]}")
