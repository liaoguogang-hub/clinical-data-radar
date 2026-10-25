"""
当天推文的可信度核查报告（并推送飞书）。

# 为什么要有这个

成稿阶段的断言核验（compose + claimcheck）只拦"生成时就跑"的稿子。
但两类情况漏在外面：
  · 早期生成的旧稿（核验上线前发的）
  · 人写成稿 / 手工重写进去的稿

所以再提供一个**事后独立审计**：对某天已落地的每一篇，重建它的素材，
重新拆断言、重新核验，出一份可读报告，并推到飞书。

用法：
    python audit.py                 # 今天
    python audit.py 2026-10-17      # 指定日期
    python audit.py 2026-10-17 --no-notify
    python audit.py 2026-10-17 --report-only   # 只出报告，不调 LLM 核验
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

import claimcheck  # noqa: E402
import compose  # noqa: E402
import score  # noqa: E402
from common import SCORED_DIR, load_yaml, today_cst  # noqa: E402
from compose import POSTS_DIR  # noqa: E402
from render import parse_content, content_text  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "data" / "audit"


def _plan_items(date: str) -> dict:
    """lead_id(前8) -> 选题条目"""
    plan = json.loads(
        (SCORED_DIR / f"{date}.plan.json").read_text(encoding="utf-8")) \
        if (SCORED_DIR / f"{date}.plan.json").exists() else {}
    out = {}
    for it in (plan.get("plan") or []):
        out[(it.get("lead_id") or "")[:8]] = it
    return out


def _day_articles(date: str) -> list[Path]:
    d = POSTS_DIR / date[:4] / date[5:7] / date[8:10]
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.txt") if not p.name.startswith("_"))


def audit(date: str, do_notify: bool = True,
          report_only: bool = False) -> dict:
    items = _day_articles(date)
    plans = _plan_items(date)
    style_cfg = load_yaml("style_guide.yaml")
    cfg = load_yaml("scoring.yaml")

    client = eff = None
    if not report_only:
        gcfg = cfg.get("gen_llm") or cfg["llm"]
        client, eff = score._client(gcfg, cfg.get("providers", {}))

    print(f"\n=== 可信度核查 {date}（{len(items)} 篇）===")
    rows = []
    for p in items:
        lead = re.search(r"-([0-9a-f]{8})\.txt$", p.name)
        lead = lead.group(1) if lead else p.stem
        txt = p.read_text(encoding="utf-8")
        title = ""
        m = re.search(r"^TITLE:\s*(.+)$", txt, re.M)
        if m:
            title = m.group(1).strip()

        plan = plans.get(lead)
        material = compose.build_material(date, plan) if plan else ""
        if not material:
            rows.append({"lead": lead, "title": title, "verdict": "无素材可核",
                         "detail": "选题计划里找不到该条，无法重建素材",
                         "stat": {}, "bad": []})
            print(f"  [{lead}] ⚠️  无素材可核")
            continue

        if report_only:
            rows.append({"lead": lead, "title": title, "verdict": "未核验",
                         "detail": "report-only 模式", "stat": {}, "bad": []})
            continue

        doc = parse_content(txt)
        body = content_text(doc) + " " + (doc.get("title") or "")
        try:
            claims = claimcheck.extract_claims(client, eff, body, material)
            errs, stat = claimcheck.verify_claims(claims, material)
        except Exception as e:
            claims, errs, stat = [], [f"核验异常 {type(e).__name__}: {e}"], {}

        # ── 补两道确定性/充分性防线 ────────────────────────────────────
        #
        # ⚠️ 为什么必须补（2026-10-17 实测，且已造成一次错报）：
        #    LLM 断言的**抽取数量与证据填写**在多次运行间波动极大。
        #    同一篇已知有编造的稿子，三次核查分别得到
        #      「2 条无依据 / 24 条无依据 / ✅ 可信 7/7 有出处」。
        #    最后那次给了一篇含"君实/Coherus/CytomX/PD-L1-VEGF"编造的稿子
        #    **"可信"结论**，而这张卡片已经推到了飞书群。
        #    一个会对已知坏稿说"可信"的报告，比没有报告更危险。
        #
        # 两道补救：
        #   ① 叠加【确定性】检查（正则抓分子代号）——不依赖 LLM，不会波动
        #   ② 断言太少就判"核验不充分"，**不给"可信"结论**
        from factguard import check_facts
        det = check_facts(body, material)
        if det:
            errs = list(errs) + [f"[确定性检查] {e}" for e in det]

        n_claims = len(claims)
        # 一篇正常长度的稿子，原子断言应在 10 条以上。
        # 抽得太少说明核查没真正展开，此时"没发现问题"没有意义。
        min_claims = max(5, len(body) // 300)
        insufficient = n_claims < min_claims

        if not claims:
            errs = list(errs) + ["未能拆出任何断言（核查模型返回为空）——"
                                 "核验未生效，不能据此判为可信"]
        elif insufficient:
            errs = list(errs) + [
                f"核验不充分：全文 {len(body)} 字只拆出 {n_claims} 条断言"
                f"（至少应有 {min_claims} 条）。核查未真正展开，"
                f"**不能判为可信**，需人工复核"]

        if not errs:
            verdict = "✅ 可信"
        elif insufficient or not claims:
            verdict = f"⚠️ 核验不充分（{n_claims} 条断言）"
        else:
            verdict = f"⚠️ {len(errs)} 条无依据"
        rows.append({"lead": lead, "title": title, "verdict": verdict,
                     "detail": "", "stat": stat, "bad": errs[:8]})
        print(f"  [{lead}] {verdict}  "
              f"断言 {stat.get('ok',0)}/{stat.get('total',0)} 有出处  {title[:34]}")
        for e in errs[:3]:
            print(f"        · {e[:100]}")

    result = {"date": date, "rows": rows}
    _write_report(result)
    if do_notify:
        _notify(result)
    return result


def _write_report(res: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date = res["date"]
    lines = [f"# 可信度核查报告 · {date}", "",
             f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
             "",
             "对当天每篇成稿重建素材，把文章拆成原子断言，逐条要求"
             "**素材逐字出处**；引文经程序比对，不在素材里的判为编造出处。",
             "",
             "| 稿件 | 结论 | 有出处/总断言 |", "|---|---|---|"]
    for r in res["rows"]:
        st = r.get("stat") or {}
        ratio = (f"{st.get('ok',0)}/{st.get('total',0)}" if st else "—")
        lines.append(f"| {r['title'][:42]} | {r['verdict']} | {ratio} |")
    lines.append("")
    for r in res["rows"]:
        if not r.get("bad"):
            continue
        lines += [f"## {r['title'][:60]}", f"`{r['lead']}`", ""]
        for e in r["bad"]:
            lines.append(f"- {e}")
        lines.append("")
    p = REPORT_DIR / f"credibility_{date}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[audit] 报告已写 -> {p}")
    return p


def _notify(res: dict) -> bool:
    """推飞书卡片。复用 publish 的签名逻辑。

    ⚠️ 2026-10-25 架构调整：云端默认**不发**飞书。
    可信度核查与改稿的最终口径在本机 publisher（推草稿箱前逐篇核验），
    云端再发一张核查卡会出现两套口径。需要时设 RADAR_FEISHU_NOTIFY=1 打开。
    """
    if os.getenv("RADAR_FEISHU_NOTIFY", "").lower() not in ("1", "true", "yes"):
        print("[audit] 飞书推送已跳过（云端雷达不发飞书；"
              "核查口径由本机 publisher 统一回报）")
        return False
    url = os.getenv("FEISHU_WEBHOOK")
    if not url:
        print("[audit] 未配置 FEISHU_WEBHOOK，跳过推送")
        return False
    from publish import _feishu_sign  # 复用已验证的签名实现

    bad = [r for r in res["rows"] if r.get("bad")]
    n = len(res["rows"])
    body = [f"**{res['date']}**　共 {n} 篇，"
            f"{'全部通过 ✅' if not bad else f'{len(bad)} 篇有问题 ⚠️'}", ""]
    for r in res["rows"]:
        mark = "⚠️" if r.get("bad") else "✅"
        st = r.get("stat") or {}
        extra = (f"（{st.get('ok',0)}/{st.get('total',0)} 有出处）"
                 if st else "")
        body.append(f"{mark} {r['title'][:44]}{extra}")
    if bad:
        body.append("")
        body.append("**问题摘要**")
        for r in bad[:3]:
            for e in r["bad"][:2]:
                body.append(f"· {e[:70]}")

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text",
                          "content": f"可信度核查 · {res['date']}"},
                "template": "red" if bad else "green",
            },
            "elements": [
                {"tag": "div",
                 "text": {"tag": "lark_md", "content": "\n".join(body)[:3800]}},
                {"tag": "hr"},
                {"tag": "note", "elements": [
                    {"tag": "plain_text",
                     "content": "逐条断言核验 · 引文经程序比对"}]},
            ],
        },
    }
    secret = os.getenv("FEISHU_SECRET")
    if secret:
        ts = str(int(time.time()))
        payload["timestamp"] = ts
        payload["sign"] = _feishu_sign(ts, secret)
    try:
        r = httpx.post(url, json=payload, timeout=20)
        d = r.json()
        if d.get("code") == 0:
            print("[audit] 飞书推送成功")
            return True
        print(f"[audit] 飞书推送失败 code={d.get('code')} msg={d.get('msg')}")
        return False
    except Exception as e:
        print(f"[audit] 飞书推送异常 {type(e).__name__}: {str(e)[:100]}")
        return False


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    date = args[0] if args else today_cst()
    audit(date,
          do_notify="--no-notify" not in sys.argv,
          report_only="--report-only" in sys.argv)


if __name__ == "__main__":
    main()
