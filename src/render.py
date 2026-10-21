"""
渲染层：把 LLM 输出的轻量标记文本，渲染成固定模板的 HTML。

设计动机（实测结论）：
让 LLM 一次完成"写内容 + 写 HTML + 复制 CSS"三件事，成功率只有 ~1/3。
失败全部集中在格式层：标签不配平、CSS 丢失、输出截断。

所以把职责拆开：
  LLM  → 只写内容和轻量标记（不写任何 HTML 标签）
  代码 → 负责所有标签、CSS、转义

这样标签错误和 CSS 丢失从架构上不可能发生。

轻量标记格式（LLM 只需输出纯文本 + 行首标记）：
  KICKER: 栏目名
  TITLE: 主标题
  SUB: 副标题
  HOOK: 钩子（一段）
  ## 01 小节标题
  正文段落……
  ### 子标题
  TABLE: 列1 | 列2 | 列3
  行1 | 行2 | 行3
  ENDTABLE
  BOX w 标签： | 内容
  BOX r 标签： | 内容
  ENDCARD 标题 | 正文
  FN 1. 来源……
正文里可用 **加粗**，渲染时会转成 <strong>。
"""
from __future__ import annotations

import html as _html
import re
import sys

# ============================================================
# 固定 CSS（2430 字符，与既有推文风格一致）
# ============================================================
CSS = """:root{--bg:#fff;--soft:#f7f7f5;--ink:#1a1a1a;--ink2:#4a4a4a;--faint:#8a8a8a;--rule:#e6e6e3;--bronze:#9c6b3c;--deep:#6b4a2a;--red:#a93226;--green:#4a7c4e;--amber:#f5ecd9;--rose:#f7e8e6;--mint:#e8f0e6;--serif:"Noto Serif SC",Georgia,"Songti SC",serif;--mono:"JetBrains Mono",Menlo,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--serif);font-size:16px;line-height:1.9;color:var(--ink);background:var(--bg)}
.wrap{max-width:780px;margin:0 auto;padding:60px 24px 90px}
p{margin-bottom:18px;text-align:justify}
.kicker{font-size:12px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--bronze);margin-bottom:12px}
h1{font-size:clamp(25px,3.6vw,34px);font-weight:700;color:var(--red);line-height:1.35;margin-bottom:14px}
.sub{font-size:15px;color:var(--ink2);margin-bottom:34px;font-style:italic}
h2{font-size:21px;font-weight:600;color:var(--deep);margin:44px 0 16px;padding-top:12px;border-top:1px solid var(--rule)}
h2 .n{color:var(--red);font-family:var(--mono);font-size:15px;margin-right:9px}
h3{font-size:17px;font-weight:600;margin:26px 0 10px}
.hook{background:var(--red);color:#fff;padding:26px 30px;margin:22px 0 32px;font-size:18px;line-height:1.75}
.hook strong{color:#ffd166}
.box{background:var(--soft);border:1px solid var(--rule);padding:15px 19px;margin:20px 0;font-size:14px;color:var(--ink2)}
.box b{color:var(--deep)}
.box p{margin-bottom:8px;text-align:left}
.w{background:var(--amber);border-left:3px solid #b88a3e}
.g{background:var(--mint);border-left:3px solid var(--green)}
.r{background:var(--rose);border-left:3px solid var(--red)}
table{width:100%;border-collapse:collapse;margin:20px 0;font-size:13px}
th{background:var(--deep);color:#fff;text-align:left;padding:9px 11px;font-weight:600;font-size:12.5px}
td{padding:9px 11px;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:nth-child(even){background:var(--soft)}
.m{font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:12.5px}
.hl{color:var(--red);font-weight:700}
.foot{font-size:12px;color:var(--faint);margin-top:7px}
strong{font-weight:700}
ul{margin:12px 0 18px 24px}
li{margin-bottom:8px}
.fn{font-size:12px;color:var(--faint);margin-top:38px;line-height:1.8;text-align:left}
.end{background:var(--deep);color:#fff;padding:24px 28px;margin:40px 0 22px}
.end h4{font-size:18px;font-weight:600;margin-bottom:8px}
@media(max-width:720px){.wrap{padding:30px 16px 60px}}"""


# ============================================================
# 解析
# ============================================================
RE_KICKER = re.compile(r"^KICKER\s*[:：]\s*(.+)$", re.I)
RE_TITLE = re.compile(r"^TITLE\s*[:：]\s*(.+)$", re.I)
RE_SUB = re.compile(r"^SUB\s*[:：]\s*(.+)$", re.I)
RE_HOOK = re.compile(r"^HOOK\s*[:：]\s*(.+)$", re.I)
RE_H2 = re.compile(r"^##\s+(\d{1,2})?\s*[.、:：]?\s*(.+)$")
RE_H3 = re.compile(r"^###\s+(.+)$")
RE_TABLE = re.compile(r"^TABLE\s*[:：]\s*(.*)$", re.I)
RE_ENDTABLE = re.compile(r"^ENDTABLE\s*$", re.I)
RE_BOX = re.compile(r"^BOX\s+([wgr])\s+(.+?)\s*[|｜]\s*(.+)$", re.I)
RE_ENDCARD = re.compile(r"^ENDCARD\s+(.+?)\s*[|｜]\s*(.+)$", re.I)
RE_FN = re.compile(r"^FN\s+(.+)$", re.I)


def _inline(text: str) -> str:
    """转义 HTML，再把 **加粗** 和 `代码` 转成标签。"""
    t = _html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r'<span class="m">\1</span>', t)
    return t


def _split_row(line: str) -> list[str]:
    parts = re.split(r"\s*[|｜]\s*", line.strip())
    return [p.strip() for p in parts]


def _merge_split_markers(text: str) -> str:
    """
    把「标记词单独一行、内容在下一行」的写法合并回一行。

    # 为什么需要这个（2026-10-15 事故）

    提示词里已经反复强调 `ENDCARD 标题 | 内容`，还给了 ❌/✅ 对照，
    但模型仍然经常写成两行：

        ENDCARD
        真正的看点 | 这不是一次普通的读出，而是赛道的分水岭。

    RE_ENDCARD 要求标记后必须有竖线，所以第一行匹配不上，
    掉进下面的"裸标记行"分支 → marker_errors → 整篇作废。
    **结果就是内容其实完全写对了，仅仅因为换了个行就被判死。**

    实测那天 3 篇 × 4 次重试 = 12 次机会全部耗在这一个格式问题上，
    最后 0 篇成稿 —— 而选题明明有 3 篇。

    # 修法：在解析前做一次"缝合"，而不是报错

    宽松只放在【解析】这一层，【输出】那层仍然严格：
    缝合后照样走 RE_ENDCARD，生成的还是正规结语卡，
    读者看到的 HTML 不受影响。
    真正内容缺失（下一行也没有 `|`）的，仍然照旧报错。

    只对 ENDCARD / TABLE / BOX / FN 这些"标签 | 内容"型标记做，
    TITLE/HOOK/KICKER 那几个后面接的是多行正文，不能这么缝。

    # 2026-10-21 扩展：裸 ENDCARD + 下一行【没有竖线】

    实测 2026-10-21 有一篇稿子把 `ENDCARD` 甩在【正文中段】两次
    （都在一个 BOX 之后），每次后面跟着一段正文，整篇从头到尾没有结语卡，
    于是同时吃了三条错：两条"标记行格式不全" + 一条"缺少 ENDCARD 结语卡"。

    这次扩展按**位置**分两种处理，绝不吞正文：

    · 位置在【全文末尾】（后面只剩 FN 脚注/空行）→
      下一行那段就是结语本身，把它提升成正规结语卡
      （标题沿用最后一个小节标题，没有就用"结语"）。
      这是真正的可恢复场景：内容本来就在，只是标记写歪了。

    · 位置在【正文中段】→ 它不是结语卡（模型把它当成小节收尾用了）。
      只丢这个标记词，**不吞后面的正文**（吞了等于删正文），
      真正的病根"全文没有合规结语卡"交给 validate_content 报错，
      并在提示里说清楚"结语卡必须放在全文最后"。
    """
    out: list[str] = []
    lines = text.split("\n")

    # ---- 先定位"全文最后一行有实质内容的行"（FN 脚注不算内容）----
    # 用来判断裸 ENDCARD 是在末尾还是在正文中段。
    last_content = -1
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or re.match(r"^FN\b", s, re.I):
            continue
        last_content = i

    # ---- 最后一个小节的标题（给补出来的结语卡当标题用）----
    end_title = ""
    for raw in lines:
        m = re.match(r"^##\s+(?:\d+\s+)?(.+?)\s*$", raw.strip())
        if m and m.group(1).strip():
            end_title = m.group(1).strip()

    # 只处理这三个「标记 标题 | 内容」型的；TITLE/HOOK/KICKER 后面接的是
    # 多行正文，不能这么缝（缝了反而把正文吞掉）。
    for i, raw in enumerate(lines):
        s = raw.strip()
        m = re.match(r"^(ENDCARD|TABLE|BOX)\s*$", s, re.I)
        if m:
            kind = m.group(1).upper()
            # ⚠️ 必须**跳过空行**再找内容（2026-10-19 补充）：
            #    原来只看紧邻的下一行，于是
            #        ENDCARD
            #        <空行>
            #        真正的看点 | 内容……
            #    这种形态找不nxt → 不缝合 → 掉进"裸标记"分支 → 整篇作废。
            #    实测 10-19 有一篇 2318 字（已过 2100 字数下限）
            #    就因为这一处被判死。
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                is_marker = re.match(
                    r"^(ENDCARD|ENDTABLE|TABLE|BOX|FN|KICKER|TITLE|SUB|HOOK)\b",
                    nxt, re.I)
                # 下一行必须有竖线，且不是又一个标记行，才值得缝
                if "|" in nxt and not is_marker:
                    out.append(f"{kind} {nxt}")
                    lines[j] = ""            # 内容行已消费
                    for k in range(i + 1, j):  # 中间的空行也一并清掉
                        lines[k] = ""
                    continue
                # ---- 2026-10-21：裸 ENDCARD + 下一行没有竖线 ----
                if kind == "ENDCARD" and not is_marker:
                    if j == last_content:
                        # 在全文末尾 → 这一段就是结语，提升成正规结语卡
                        title = end_title or "结语"
                        out.append(f"ENDCARD {title} | {nxt}")
                        lines[j] = ""
                        for k in range(i + 1, j):
                            lines[k] = ""
                        continue
                    # 在正文中段 → 只丢标记词，**不吞正文**
                    continue
        out.append(raw)
    return "\n".join(out)


def parse_content(text: str) -> dict:
    """
    把轻量标记文本解析成结构化内容。
    容错：标记缺失时退化为普通段落，绝不抛异常。
    """
    # 剥掉可能的 markdown 代码块包裹
    text = re.sub(r"^\s*```(?:[a-z]*)?\s*\n", "", text.strip())
    text = re.sub(r"\n\s*```\s*$", "", text)

    doc: dict = {
        "kicker": "", "title": "", "sub": "", "hook": "",
        "sections": [], "end": None, "footnotes": [],
        "loose": [],           # 标记之外的散段落
    }
    cur_sec: dict | None = None
    in_table = False
    table_head: list[str] = []
    table_rows: list[list[str]] = []

    def _flush_table():
        nonlocal in_table, table_head, table_rows
        if in_table and (table_head or table_rows):
            blk = {"type": "table", "head": table_head, "rows": table_rows}
            _append_block(blk)
        in_table, table_head, table_rows = False, [], []

    def _append_block(blk: dict):
        if cur_sec is not None:
            cur_sec["blocks"].append(blk)
        else:
            doc["loose"].append(blk)

    # ⚠️ 先缝合"标记单独成行、内容在下一行"的写法（见 _merge_split_markers 的说明）。
    #    不做这一步的话，模型换行写的 ENDCARD 会被判成裸标记 → 整篇作废。
    for raw in _merge_split_markers(text).split("\n"):
        line = raw.rstrip()
        s = line.strip()
        if not s:
            continue

        if in_table:
            if RE_ENDTABLE.match(s):
                _flush_table()
            else:
                row = _split_row(s)
                if not table_head:
                    table_head = row
                else:
                    table_rows.append(row)
            continue

        if m := RE_KICKER.match(s):
            doc["kicker"] = m.group(1).strip(); continue
        if m := RE_TITLE.match(s):
            doc["title"] = m.group(1).strip(); continue
        if m := RE_SUB.match(s):
            doc["sub"] = m.group(1).strip(); continue
        if m := RE_HOOK.match(s):
            doc["hook"] = m.group(1).strip(); continue

        if m := RE_TABLE.match(s):
            in_table = True
            inline_head = m.group(1).strip()
            if inline_head:
                table_head = _split_row(inline_head)
            continue
        if RE_ENDTABLE.match(s):
            _flush_table(); continue

        if m := RE_BOX.match(s):
            _append_block({"type": "box", "variant": m.group(1).lower(),
                           "label": m.group(2).strip(),
                           "text": m.group(3).strip()})
            continue
        if m := RE_ENDCARD.match(s):
            doc["end"] = {"h4": m.group(1).strip(), "text": m.group(2).strip()}
            continue
        if m := RE_FN.match(s):
            doc["footnotes"].append(m.group(1).strip()); continue

        if m := RE_H3.match(s):
            _append_block({"type": "h3", "text": m.group(1).strip()})
            continue
        if m := RE_H2.match(s):
            cur_sec = {"n": (m.group(1) or "").strip(),
                       "title": m.group(2).strip(), "blocks": []}
            doc["sections"].append(cur_sec)
            continue

        # ---- 光秃秃的标记行：必须报错，不能当普通段落 ----
        #
        # ⚠️ 实测事故：模型把结尾写成了单独一行 `ENDCARD`（漏了「标题 | 内容」），
        #    而上面的 RE_ENDCARD 要求有竖线，于是**不匹配**，
        #    这行就掉进"普通段落"分支，最终输出成 <p>ENDCARD</p>——
        #    稿子发到公众号，读者看到的是两个大写的 ENDCARD。
        #    校验层当时只查 TITLE/HOOK/小节数/脚注，没查结语卡，也没拦住。
        #
        #    现在显式识别"像标记但格式不全"的行，记进 marker_errors，
        #    交给 validate_content 判失败并触发重写。
        #    （静默降级比报错危险得多——错误会一路流到读者眼前。）
        bare = re.match(r"^(ENDCARD|ENDTABLE|ENDCARD|TABLE|BOX|FN|KICKER|"
                        r"TITLE|SUB|HOOK)\b\s*$", s, re.I)
        if bare:
            kind = bare.group(1).upper()
            # ⚠️ 裸 ENDCARD 不再记 marker_errors（2026-10-21）。
            #    理由：裸 ENDCARD 本身没有信息量，只有两种情况：
            #      ① 重复的裸标记——doc["end"] 已有内容，本来就该忽略
            #         （2026-10-19 实测：模型写完合规结语卡，末尾再甩一个
            #          裸 ENDCARD，为此把一篇 2318 字的合格稿判死）；
            #      ② 模型把结语卡甩在正文中段（2026-10-21 实测甩了两次）。
            #    两种情况真正的病根都是同一个：**全文没有合规的结语卡**，
            #    而 validate_content 的"缺少 ENDCARD 结语卡"已经能说清楚，
            #    这次还补上了"必须放在全文最后、FN 之前"。
            #    再叠一条"标记行格式不全：'ENDCARD'"只会把重写提示搞得更啰嗦，
            #    模型却看不出到底该改什么（2026-10-21：4 次重试没改对）。
            #    注意：裸 ENDCARD 绝不能流进正文（那是 <p>ENDCARD</p>，
            #    读者会看到两个大写 ENDCARD），这里直接丢弃标记词即可。
            if kind == "ENDCARD":
                continue
            doc.setdefault("marker_errors", []).append(
                f"标记行格式不全：{s!r}（应写成 `关键词 标题 | 内容`）")
            continue

        # 普通段落
        _append_block({"type": "p", "text": s})

    _flush_table()
    return doc


# ============================================================
# 渲染
# ============================================================
def _render_blocks(blocks: list[dict]) -> list[str]:
    out: list[str] = []
    for b in blocks:
        t = b.get("type")
        if t == "p":
            out.append(f"<p>{_inline(b['text'])}</p>")
        elif t == "h3":
            out.append(f"<h3>{_inline(b['text'])}</h3>")
        elif t == "box":
            v = b.get("variant", "w")
            label = _inline(b.get("label", ""))
            txt = _inline(b.get("text", ""))
            lbl = f"<b>{label}</b>" if label else ""
            out.append(f'<div class="box {v}"><p>{lbl}{txt}</p></div>')
        elif t == "table":
            head = b.get("head") or []
            rows = b.get("rows") or []
            if not head and not rows:
                continue
            parts = ['<table>']
            if head:
                parts.append("<thead><tr>"
                             + "".join(f"<th>{_inline(c)}</th>" for c in head)
                             + "</tr></thead>")
            parts.append("<tbody>")
            for r in rows:
                parts.append("<tr>"
                             + "".join(f"<td>{_inline(c)}</td>" for c in r)
                             + "</tr>")
            parts.append("</tbody></table>")
            out.append("".join(parts))
    return out


def render_html(doc: dict, lang: str = "zh-CN") -> str:
    """把结构化内容渲染成完整 HTML（标签与 CSS 由代码保证）"""
    body: list[str] = ['<article class="wrap">']

    if doc.get("kicker"):
        body.append(f'<div class="kicker">{_inline(doc["kicker"])}</div>')
    body.append(f'<h1>{_inline(doc.get("title", ""))}</h1>')
    if doc.get("sub"):
        body.append(f'<p class="sub">{_inline(doc["sub"])}</p>')
    if doc.get("hook"):
        body.append(f'<div class="hook">{_inline(doc["hook"])}</div>')

    body.extend(_render_blocks(doc.get("loose") or []))

    for sec in doc.get("sections") or []:
        n = sec.get("n") or ""
        num = f'<span class="n">{n}</span>' if n else ""
        body.append(f"<h2>{num}{_inline(sec.get('title',''))}</h2>")
        body.extend(_render_blocks(sec.get("blocks") or []))

    end = doc.get("end")
    if end:
        body.append('<div class="end">'
                    f'<h4>{_inline(end.get("h4",""))}</h4>'
                    f'<p>{_inline(end.get("text",""))}</p>'
                    "</div>")

    if doc.get("footnotes"):
        items = "<br>\n".join(_inline(x) for x in doc["footnotes"])
        body.append(f'<p class="fn"><strong>来源与核查口径：</strong><br>\n{items}</p>')

    body.append("</article>")

    return (f'<!DOCTYPE html>\n<html lang="{lang}">\n<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f"<title>{_html.escape(doc.get('title',''))}</title>\n"
            f"<style>\n{CSS}\n</style>\n</head>\n<body>\n"
            + "\n".join(body)
            + "\n</body>\n</html>\n")


# ============================================================
# 内容度量（质检用）
# ============================================================
def content_text(doc: dict, include_tables: bool = False) -> str:
    """取正文纯文本。默认不含表格和脚注，与推文字数口径一致。"""
    parts: list[str] = []
    if doc.get("hook"):
        parts.append(doc["hook"])
    for b in (doc.get("loose") or []):
        if b.get("type") == "p":
            parts.append(b["text"])
    for sec in (doc.get("sections") or []):
        for b in (sec.get("blocks") or []):
            if b.get("type") in ("p", "h3"):
                parts.append(b["text"])
            elif b.get("type") == "box":
                parts.append(b.get("text", ""))
            elif b.get("type") == "table" and include_tables:
                parts.extend(b.get("head") or [])
                for r in (b.get("rows") or []):
                    parts.extend(r)
    if doc.get("end"):
        parts.append(doc["end"].get("text", ""))
    return "\n".join(parts)


def cn_count(s: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", s))


def body_chars(doc: dict) -> int:
    return cn_count(content_text(doc, include_tables=False))


def structure_report(doc: dict) -> dict:
    return {
        "body_chars": body_chars(doc),
        "sections": len(doc.get("sections") or []),
        "tables": sum(1 for s in (doc.get("sections") or [])
                      for b in s["blocks"] if b["type"] == "table")
                  + sum(1 for b in (doc.get("loose") or [])
                        if b["type"] == "table"),
        "boxes": sum(1 for s in (doc.get("sections") or [])
                     for b in s["blocks"] if b["type"] == "box"),
        "footnotes": len(doc.get("footnotes") or []),
        "has_hook": bool(doc.get("hook")),
        "has_end": doc.get("end") is not None,
        "has_title": bool(doc.get("title")),
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import json
    src = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    d = parse_content(src)
    print(json.dumps(structure_report(d), ensure_ascii=False, indent=1))
    if len(sys.argv) > 2:
        open(sys.argv[2], "w", encoding="utf-8").write(render_html(d))
        print("rendered ->", sys.argv[2])