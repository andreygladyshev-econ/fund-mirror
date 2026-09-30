"""Записка .md -> .html -> .pdf (Chrome без окна). Поддерживает заголовки, абзацы, **жирный**, *курсив*, списки,
таблицы и рисунки ![подпись](путь). python3 md2pdf.py ЗАПИСКА_v10.md -> ЗАПИСКА_v10.pdf рядом"""
import html
import re
import subprocess
import sys
from pathlib import Path

CSS = """
@page { size: A4; margin: 18mm 18mm 18mm 20mm; }
body { font-family: 'PT Serif', 'Times New Roman', serif; font-size: 11pt; line-height: 1.42; color: #111; }
h1 { font-family: 'PT Sans', Arial, sans-serif; font-size: 16pt; line-height: 1.25; margin: 0 0 10pt; }
h2 { font-family: 'PT Sans', Arial, sans-serif; font-size: 12.5pt; margin: 14pt 0 4pt; break-after: avoid; }
table { break-inside: avoid; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 6pt 16pt; padding: 0; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6pt 0 8pt; font-family: 'PT Sans', Arial, sans-serif; }
th, td { border-top: 0.6pt solid #999; border-bottom: 0.6pt solid #999; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f3f3f1; }
figure { margin: 8pt 0 10pt; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-family: 'PT Sans', Arial, sans-serif; font-size: 9pt; color: #444; margin-top: 3pt; }
.lit p { text-align: left; font-size: 10pt; margin: 0 0 2pt; }
"""


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"(\d) (?=\d{3}\b)", "\\1\u00a0", t)          # 3 007: неразрывный пробел в числах
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(https://[^\s<]+[^\s<.,;)])", r'<a href="\1">\1</a>', t)   # ссылки кликабельны в PDF
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", t)
    return t


def convert(md, base):
    out, para, lit = [], [], False
    lines = md.splitlines()
    i = 0

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()
    while i < len(lines):
        l = lines[i].rstrip()
        if not l.strip():
            flush()
        elif l.startswith("# "):
            flush(); out.append(f"<h1>{inline(l[2:])}</h1>")
        elif l.startswith("## "):
            flush()
            lit = l[3:].strip() == "Литература"
            out.append(f"<h2>{inline(l[3:])}</h2>" + ("<div class='lit'>" if lit else ""))
        elif m := re.match(r"!\[(.*?)\]\((.*?)\)", l.strip()):
            flush()
            src = (base / m.group(2)).resolve().as_uri()
            cap, _, width = m.group(1).partition("|")          # ![Подпись|80](путь): ширина рисунка в процентах
            style = f" style='width:{width}%'" if width else ""
            out.append(f"<figure><img src='{src}'{style}><figcaption>{inline(cap)}</figcaption></figure>")
        elif l.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s\-:|]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[1:]
            out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</table>")
            continue
        elif re.match(r"^(\d+\.|-) ", l):
            flush()
            tag = "ol" if l[0].isdigit() else "ul"
            items = []
            while i < len(lines) and re.match(r"^(\d+\.|-) ", lines[i]):
                item = re.sub(r"^(\d+\.|-) ", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    item += " " + lines[i].strip()
                    i += 1
                items.append(f"<li>{inline(item)}</li>")
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue
        elif lit:
            flush(); out.append(f"<p>{inline(l)}</p>")
        else:
            para.append(l.strip())
        i += 1
    flush()
    if lit:
        out.append("</div>")
    return "<!doctype html><html lang='ru'><meta charset='utf-8'><style>" + CSS + "</style><body>" + "\n".join(out) + "</body></html>"


src = Path(sys.argv[1]).resolve()
htm, pdf = src.with_suffix(".html"), src.with_suffix(".pdf")
htm.write_text(convert(src.read_text(encoding="utf-8"), src.parent), encoding="utf-8")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", htm.as_uri()],
               check=True, capture_output=True)
print(pdf)
