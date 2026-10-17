"""
公司背景补全：给成稿素材补一段**可引用的**公司与产品信息。

# 为什么需要它（2026-10-17 用户定的规则）

素材不足时要扩写，首选方向是「介绍相关公司」。但**扩写必须有出处**——
否则"年收入 320 亿""主要产品 XX"就是下一个胡编入口
（retlirafusp 那篇的"君实/Coherus 有北美商业化能力"就是这么编出来的）。

所以本模块只做一件事：**把真实存在、可引用的公司公开信息取回来放进素材**，
让成稿能写、且断言核验（claimcheck）能验。

# 数据源与各自的边界（务必如实告知用户）

| 源 | 给什么 | 边界 |
|---|---|---|
| **openFDA Drugs@FDA** | 已上市产品（美国批准） | 只有美国批准记录。实测 BEIGENE 有、HENGRUI/INNOCARE/JUNSHI 都 404 |
| **ClinicalTrials.gov** | 在研管线（试验数/期别/适应症） | 只覆盖注册在 CT.gov 的试验；中国试验常只登 CDE |
| **公司官网** | 主营/产品/定位 | 需在 `config/company_sites.yaml` 配 URL；**抓回后必须验证页面确实属于该公司**，否则不用 |

⚠️ **年收入 / 中国已上市产品（NMPA 批文）本模块【不提供】**——
需要年报/交易所披露或 NMPA 注册库，是另外的数据源。素材里没有，
断言核验就会拦下相关表述；这是刻意的：宁可不写，不要编。
"""
from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

import httpx
import pypdf

from common import ROOT, load_yaml, DATA_DIR  # noqa: F401

UA = {"User-Agent": "Mozilla/5.0 (compatible; clinical-data-radar/1.0)"}
OPENFDA = "https://api.fda.gov/drug/drugsfda.json"


def _cfg() -> dict:
    p = ROOT / "config" / "company_sites.yaml"
    if not p.exists():
        return {}
    try:
        return load_yaml("company_sites.yaml") or {}
    except Exception:
        return {}


# ---------- 源 A：药品注册库（美国已上市产品）----------
def us_approved_products(client: httpx.Client, sponsor_en: str) -> list[dict]:
    """
    按申办方查美国已批准产品。

    ⚠️ openFDA 的 sponsor_name 是**大写**存的，查询也必须大写，
    否则一律 404（实测 "Hengrui" 404 / "BEIGENE" 3 条）。
    404 表示"没有记录"，是正常结果，不是错误。
    """
    q = (sponsor_en or "").strip().upper()
    if len(q) < 3:
        return []
    try:
        r = client.get(OPENFDA, params={
            "search": f'sponsor_name:"{q}"', "limit": 20}, timeout=25)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    try:
        d = r.json()
    except Exception:
        return []
    out = []
    for it in (d.get("results") or []):
        prods = it.get("products") or []
        brands = [p.get("brand_name") for p in prods
                  if p.get("brand_name") and p.get("marketing_status") == "Prescription"]
        if not brands:
            brands = [p.get("brand_name") for p in prods if p.get("brand_name")]
        if not brands:
            continue
        subs = it.get("submissions") or []
        dates = sorted(s.get("submission_status_date", "") for s in subs
                       if s.get("submission_status_date"))
        out.append({
            "brand": brands[0],
            "appno": it.get("application_number", ""),
            "sponsor": it.get("sponsor_name", ""),
            "date": (dates[-1] if dates else "")[:8],
        })
    # 去重（同一 brand 多个申请号）
    seen, uniq = set(), []
    for o in out:
        if o["brand"] in seen:
            continue
        seen.add(o["brand"])
        uniq.append(o)
    return uniq


# ---------- 源 B：在研管线（CT.gov 申办方组合）----------
CT_BASE = "https://clinicaltrials.gov/api/v2/studies"


def pipeline_summary(client: httpx.Client, sponsor_en: str) -> dict:
    """按申办方拉 CT.gov 组合，给出试验数/期别分布/主要适应症。"""
    if not sponsor_en or len(sponsor_en) < 3:
        return {}
    try:
        r = client.get(CT_BASE, params={
            "query.spons": sponsor_en, "pageSize": 40}, timeout=30)
        d = r.json() if r.status_code == 200 else {}
    except Exception:
        return {}
    trials = d.get("studies") or []
    if not trials:
        return {}
    from collections import Counter
    phases, conds, years = Counter(), Counter(), []
    for t in trials:
        ps = (t.get("protocolSection") or {})
        dm = ps.get("designModule") or {}
        for ph in (dm.get("phases") or []):
            phases[ph] += 1
        for c in ((ps.get("conditionsModule") or {}).get("conditions") or []):
            conds[c] += 1
        sd = ((ps.get("statusModule") or {}).get("startDateStruct") or {})
        if sd.get("date"):
            years.append(sd["date"][:4])
    return {
        "n": len(trials),
        "phases": phases.most_common(5),
        "conditions": conds.most_common(8),
        "start_range": (min(years), max(years)) if years else ("", ""),
    }


# ---------- 源 C：公司官网 ----------
def _site_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|nav|footer|header|svg)[^>]*>.*?</\1>",
                  " ", html)
    txt = re.sub(r"(?s)<[^>]+>", " ", html)
    txt = (txt.replace("&nbsp;", " ").replace("&amp;", "&")
              .replace("&quot;", '"').replace("&#39;", "'"))
    return re.sub(r"\s+", " ", txt).strip()


def official_site(client: httpx.Client, zh: str, en: str,
                  limit: int = 3000) -> dict:
    """
    抓公司官网的一段简介文本。

    ⚠️ **必须验证页面确实属于这家公司**：配错 URL 会抓到别家站点，
    然后把别人的产品或营收当成本公司的——那是比没有材料严重得多的错误。
    验证方式：页面正文里出现公司中文名或英文名，否则整段丢弃。
    """
    sites = _cfg().get("sites") or {}
    url = sites.get(zh) or sites.get(en)
    if not url:
        return {}
    try:
        r = client.get(url, timeout=25, follow_redirects=True)
    except Exception:
        return {}
    if r.status_code != 200 or len(r.text) < 500:
        return {}
    txt = _site_text(r.text)
    if len(txt) < 200:
        return {}
    names = [n for n in (zh, en) if n]
    if not any(n in txt for n in names):
        # 页面里没有这家公司的名字 → 大概率配错 URL，宁可不给
        return {"rejected": True, "url": url,
                "why": "页面正文未出现该公司名，疑似 URL 配错，已丢弃"}
    return {"url": url, "text": txt[:limit]}


# ---------- 源 D：公司年报 ----------
#
# 用户 2026-10-17 追加要求：营收/产品这类信息要有出处 → 用年报。
# 两条路，覆盖不同上市地：
#   · 美股/ADS 申报主体 → SEC EDGAR XBRL（**结构化**，不用解 PDF）
#   · A 股              → 巨潮资讯的年报 PDF（要下载 + 抽文本）
#
# ⚠️ 严格失败：读不到就【不写】，绝不猜。用户明确要求
#    "如果真的读不到内容，那就不写了"。
CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = ("https://data.sec.gov/api/xbrl/companyconcept/"
             "CIK{cik}/{tax}/{tag}.json")
SEC_HEADERS = {"User-Agent": "clinical-data-radar/1.0 "
                             "(research; contact: radar@example.com)"}

# ⚠️ 必须同时支持 **us-gaap 和 ifrs-full** 两套分类（2026-10-17 实测）：
#    美国本土公司（ABBV/LLY/PFE）走 us-gaap；
#    但**外国私人发行人（20-F）常用 IFRS**，营收不在 us-gaap 下——
#    实测 AstraZeneca（AZN）us-gaap 全部 404，而
#    ifrs-full/Revenue 有 FY2025 = 587.39 亿美元。
_REV_TAGS = (
    ("us-gaap", "Revenues"),
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    ("us-gaap", "SalesRevenueNet"),
    ("ifrs-full", "Revenue"),
    ("ifrs-full", "RevenueFromContractsWithCustomers"),
)

# ⚠️ 币种不能一律当美元：IFRS 申报主体可能用 EUR/GBP 报告。
_CUR_ZH = {"USD": "美元", "EUR": "欧元", "GBP": "英镑",
           "CNY": "人民币", "JPY": "日元"}


def _sec_cik(client: httpx.Client, ticker: str) -> str:
    """ticker -> 10 位 CIK。"""
    try:
        r = client.get(SEC_TICKERS, headers=SEC_HEADERS, timeout=30)
        d = r.json()
    except Exception:
        return ""
    t = (ticker or "").strip().upper()
    for v in (d.values() if isinstance(d, dict) else []):
        if str(v.get("ticker", "")).upper() == t:
            return f"{int(v['cik_str']):010d}"
    return ""


def annual_report_us(client: httpx.Client, ticker: str) -> dict:
    """SEC XBRL 取最近一个财年的营业收入（美元）。读不到返回 {}。"""
    cik = _sec_cik(client, ticker)
    if not cik:
        return {}
    for tax, tag in _REV_TAGS:
        try:
            r = client.get(SEC_FACTS.format(cik=cik, tax=tax, tag=tag),
                           headers=SEC_HEADERS, timeout=30)
            if r.status_code != 200:
                continue
            units_all = (r.json().get("units") or {})
        except Exception:
            continue
        cur = next((k for k in ("USD", "EUR", "GBP", "CNY", "JPY")
                    if units_all.get(k)), "")
        if not cur:
            continue
        units = units_all.get(cur) or []
        ann = [x for x in units
               if x.get("form") in ("20-F", "10-K", "40-F")
               and x.get("fp") == "FY" and x.get("end")]
        if not ann:
            continue
        # ⚠️ 同一个 end 日期会有**多条**：全年 + 第四季度（2026-10-17 实测）
        #    实测 AbbVie Revenues：
        #      frame=CY2025    → 611.60 亿美元（全年，对）
        #      frame=CY2025Q4  → 166.18 亿美元（仅 Q4，错）
        #    按 end 排序取最后一条会拿到 Q4 当全年。
        #    这个错比"取不到"危险得多：错数字进了素材，
        #    claimcheck 会把它当"有出处"放行。
        # 先只留【纯年度】frame（CY + 4 位年），没有再按 end 取最大值兜底。
        pure = [x for x in ann
                if not x.get("frame")
                or re.fullmatch(r"CY\d{4}", str(x["frame"]))]
        if pure:
            ann = pure
        else:
            best: dict = {}
            for x in ann:      # 同一 end 只留最大值（全年 > 任一单季）
                if x["end"] not in best or x["val"] > best[x["end"]]["val"]:
                    best[x["end"]] = x
            ann = list(best.values())
        ann.sort(key=lambda x: x["end"])
        last = ann[-1]
        # ⚠️ 新鲜度闸门（2026-10-17 实测）：AMGN 用 Revenues 标签时
        #    "纯年度"条目只剩到 2020-12-31，不设闸门就会把**6 年前**的
        #    营收当成当前年报数字写进素材。过期数据同样是有害错误，
        #    不只是"取不到"。超过 2 年判过期，继续试下一个标签。
        if int(last["end"][:4]) < date.today().year - 2:
            continue
        return {"source": f"SEC XBRL {tax}/{tag}",
                "period": last["end"], "value": last["val"],
                "currency": cur, "ticker": ticker.upper()}
    return {}


def annual_report_a(client: httpx.Client, zh_name: str,
                    max_pages: int = 25) -> dict:
    """
    巨潮资讯取 A 股年报，抽「营业收入」与「归母净利润」。

    ⚠️ 只抽年报（**排除半年度报告和摘要**）；抽前 max_pages 页
    （「主要会计数据」在开头，全本 200+ 页没必要）。
    抽不到就返回 {}，由调用方如实写"未取到"。
    """
    import re as _re
    try:
        r = client.post(CNINFO_QUERY, data={
            "pageNum": 1, "pageSize": 30, "column": "szse",
            "tabName": "fulltext",
            "searchkey": f"{zh_name} 年度报告",
            "isHLtitle": "true",
        }, headers={"Referer": "http://www.cninfo.com.cn/",
                    "Content-Type":
                    "application/x-www-form-urlencoded; charset=UTF-8"},
            timeout=35)
        anns = (r.json() or {}).get("announcements") or []
    except Exception:
        return {}
    # 只要年度报告正本：排除"摘要""半年度""季度"
    cand = []
    for a in anns:
        t = _re.sub(r"<[^>]+>", "", a.get("announcementTitle") or "")
        if "年度报告" not in t:
            continue
        if any(k in t for k in ("摘要", "半年度", "季度", "英文")):
            continue
        if a.get("adjunctUrl"):
            cand.append((t, a["adjunctUrl"]))
    if not cand:
        return {}
    title, url = cand[0]
    try:
        pr = client.get(CNINFO_STATIC + url, timeout=90)
        if pr.status_code != 200 or len(pr.content) < 10000:
            return {}
        buf = io.BytesIO(pr.content)
        rd = pypdf.PdfReader(buf)
        text = ""
        for i in range(min(max_pages, len(rd.pages))):
            try:
                text += (rd.pages[i].extract_text() or "")
            except Exception:
                continue
    except Exception:
        return {}
    if len(text) < 500:
        return {}

    def _pick(kw: str) -> int | None:
        """
        取关键词后出现的金额。

        ⚠️ 取**第一个**匹配，不取最大值（2026-10-17 实测）：
           「归属于上市公司股东的净利润」后面紧跟的是
           「归属于上市公司股东的净资产」（61.2B）再才是净利润（7.7B），
           按"取最大"会把净资产当净利润。而年报的「主要会计数据」表里
           **本期数在最前**，所以第一个才是对的。
        窗口放到 40 字：关键词与数字之间常夹着别的表头文字。
        """
        m = _re.search(kw + r"[^\d]{0,40}?([\d,]{7,}\.\d{2})", text)
        if not m:
            return None
        try:
            return int(float(m.group(1).replace(",", "")))
        except Exception:
            return None

    rev = _pick("营业收入")
    np_ = _pick("归属于上市公司股东的净利润")
    if rev is None and np_ is None:
        return {}
    return {"source": f"巨潮资讯《{title}》",
            "revenue_cny": rev, "net_profit_cny": np_,
            "url": CNINFO_STATIC + url}


def _resolve_ticker(client: httpx.Client, en: str) -> str:
    """
    英文公司名 -> 美股 ticker。

    先试"名字本身就是 ticker"，再试在 SEC 公司表里按标题匹配。
    匹配不到就返回空（→ 走 A 股年报那条路，或如实报"未取到"）。
    """
    if not en:
        return ""
    up = en.strip().upper()
    try:
        r = client.get(SEC_TICKERS, headers=SEC_HEADERS, timeout=30)
        d = r.json()
    except Exception:
        return ""
    rows = list(d.values()) if isinstance(d, dict) else []
    for v in rows:                      # 1) 直接就是 ticker
        if str(v.get("ticker", "")).upper() == up:
            return up
    key = re.sub(r"[^A-Z]", "", up)
    if len(key) >= 4:                   # 2) 公司名包含在 SEC 标题里
        for v in rows:
            if key in re.sub(r"[^A-Z]", "", str(v.get("title", "")).upper()):
                return str(v.get("ticker", "")).upper()
    return ""


def _annual_report_line(client: httpx.Client, zh: str, en: str) -> str:
    """
    年报营收那一行。**读不到就如实说读不到，绝不写数字。**

    用户 2026-10-17 明确要求："如果真的读不到内容，那就不写了"。
    """
    # ① 美股 / ADS 申报主体 → SEC XBRL
    if en:
        tk = _resolve_ticker(client, en)
        if tk:
            d = annual_report_us(client, tk)
            if d:
                v = d["value"] / 1e8
                unit = _CUR_ZH.get(d.get("currency", "USD"), "美元")
                return (f"- 年报营收（来源：{d['source']}，ticker {d['ticker']}）："
                        f"截至 {d['period']} 财年，营业收入约 {v:,.2f} 亿{unit}")
    # ② A 股 → 巨潮年报 PDF
    if zh:
        d = annual_report_a(client, zh)
        if d:
            seg = []
            if d.get("revenue_cny"):
                seg.append(f"营业收入约 {d['revenue_cny']/1e8:,.2f} 亿元")
            if d.get("net_profit_cny"):
                seg.append(f"归母净利润约 {d['net_profit_cny']/1e8:,.2f} 亿元")
            if seg:
                return f"- 年报（来源：{d['source']}）：" + "；".join(seg)
    return ("- 年报：**未取到**（SEC XBRL 与巨潮资讯均未读到可引用数据）。"
            "→ 不要写任何营收/利润数字。")


# ---------- 汇总 ----------
def company_brief(zh: str, en: str) -> str:
    """
    返回一段可直接放进成稿素材的公司背景（带来源标注）。
    任何源取不到就如实说明取不到，**绝不编**。

    ⚠️ 刻意做成**同步**：`compose.build_material()` 是同步函数，
    而生成流程已经在 async 上下文里（不能再 asyncio.run）。
    所以这里用 httpx 的同步 Client，不引入事件循环。
    """
    if not (zh or en):
        return ""
    lines = [f"【公司背景（可引用，来自公开注册库/官网）】{zh or en}"]
    got_any = False
    try:
        with httpx.Client(headers=UA, follow_redirects=True) as c:
            # A. 美国已上市产品
            if en:
                prods = us_approved_products(c, en)
                if prods:
                    got_any = True
                    lines.append(
                        f"- 已上市产品（美国 FDA 批准，来源：openFDA Drugs@FDA，"
                        f"检索 sponsor_name={en.upper()}）：")
                    for p in prods[:8]:
                        lines.append(f"    · {p['brand']}（{p['appno']}，"
                                     f"{p['date']}）")
                else:
                    lines.append(f"- 已上市产品：openFDA 未检索到"
                                 f"（sponsor_name={en.upper()}）美国批准记录")
                # B. 在研管线
                pipe = pipeline_summary(c, en)
                if pipe:
                    got_any = True
                    lines.append(f"- 在研管线（来源：ClinicalTrials.gov，"
                                 f"申办方 {en}）：共 {pipe['n']} 项注册试验")
                    if pipe["phases"]:
                        lines.append("    · 期别分布：" + "、".join(
                            f"{k} {v}项" for k, v in pipe["phases"]))
                    if pipe["conditions"]:
                        lines.append("    · 主要适应症：" + "、".join(
                            f"{k} {v}项" for k, v in pipe["conditions"][:6]))
            # C. 官网
            site = official_site(c, zh, en)
            if site.get("text"):
                got_any = True
                lines.append(f"- 公司官网自述（来源：{site['url']}）：")
                lines.append(f"    {site['text'][:1500]}")
            elif site.get("rejected"):
                lines.append(f"- 官网：{site['why']}")

            # D. 年报（营收/净利）—— 用户 2026-10-17 追加
            #    ⚠️ 读不到就如实写"未取到"，**不写任何数字**。
            lines.append(_annual_report_line(c, zh, en))
    except Exception as e:
        lines.append(f"- 取公司背景时异常：{type(e).__name__}")

    if not got_any:
        lines.append("- 未取到任何可引用的公司背景材料"
                     "（美国批准记录/注册试验/官网均无）")
    lines.append("⚠️ 以上是本段提供的**全部**公司信息。"
                 "没有列出的（中国 NMPA 批文、市场份额、产品定价等）"
                 "一律不要写——素材里没有出处，断言核验会拦。")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for zh, en in [("百济神州", "BeiGene"), ("恒瑞医药", "Hengrui")]:
        print("=" * 70)
        print(company_brief(zh, en))
