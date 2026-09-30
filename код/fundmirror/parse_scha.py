"""Разбор ежемесячной справки о стоимости чистых активов ПИФ (форма 0420502) в список позиций.

Справка содержит полный поимённый состав фонда (проверено на ОПИФ «Мои акции», 31.08.2026: сумма позиций совпала
с итогом раздела «Ценные бумаги» до рубля). Строка позиции узнаётся по ISIN; в ней — название эмитента, ISIN,
количество бумаг и стоимость (последние два числа строки).

python3 parse_scha.py <файл.xlsx|.pdf> -> печать позиций и суммы
"""
import re
import subprocess
import sys
from pathlib import Path

ISIN = re.compile(r"\b(RU[0-9A-Z]{10}|[A-Z]{2}[0-9A-Z]{9}[0-9])\b")


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _rows(path):
    """Строки всех листов Excel (.xlsx через openpyxl, старый .xls через xlrd)."""
    if Path(path).suffix.lower() == ".xls":
        import xlrd
        for sh in xlrd.open_workbook(path).sheets():
            for i in range(sh.nrows):
                yield sh.row_values(i)
        return
    import openpyxl
    for ws in openpyxl.load_workbook(path, data_only=True, read_only=True).worksheets:
        yield from ws.iter_rows(values_only=True)


def from_xlsx(path):
    """Строка позиции акции — есть ISIN и рег. номер выпуска акций (облигации и прочее отсекаются по номеру)."""
    out = []
    for row in _rows(path):
        cells = [c for c in row if c not in (None, "")]
        isin = next((m.group(1) for c in cells if isinstance(c, str) and (m := ISIN.fullmatch(c.strip()))), None)
        reg = next((m.group(1) for c in cells if isinstance(c, str) and (m := REG.fullmatch(c.strip()))), None)
        if not isin or not reg:
            continue
        k = next(i for i, c in enumerate(cells) if isinstance(c, str) and REG.fullmatch(c.strip()))
        nums = [v for c in cells[k + 1:] if (not isinstance(c, str) or NUMS.fullmatch(c.strip())) and (v := _num(c)) is not None]
        name = next((c for c in cells if isinstance(c, str) and len(c) > 6 and not ISIN.fullmatch(c.strip()) and not REG.fullmatch(c.strip())), "")
        if len(nums) >= 2:
            out.append({"isin": isin, "reg": reg, "name": name.strip(), "qty": nums[-2], "value": nums[-1]})
    return out


REG = re.compile(r"(?<![\w-])([12]-\d{2}-\d{5}-[A-Z](?:-\d{3}[A-Z]?)?|[12]\d{7}B|\d{5}-[A-Z])(?![\w-])")   # акции: 1 — обыкн., 2 — прив.; 4… — облигации
NUMS = re.compile(r"\d{1,3}(?:[ \xa0]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?")   # число, записанное в ячейке текстом
NUM = re.compile(r"(?<![\w.,])\d{1,3}(?:[ \xa0]\d{3})*(?:,\d+)?(?![\w])")


def from_pdf(path):
    """Текстовый слой PDF (pdftotext -layout). Ячейки таблицы переносятся по строкам, поэтому ключ бумаги —
    государственный регистрационный номер выпуска (в той же строке, что количество и стоимость); ISIN — если
    целиком помещается в строку. Количество и стоимость — два последних числа строки после рег. номера.
    ponytail: вёрстка у УК разная; при несходящейся сумме справку разбирает языковая модель (см. README)."""
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
    out = []
    lines = txt.splitlines()
    for k, line in enumerate(lines):
        m = REG.search(line)
        if not m:
            continue
        tail = line[m.end():]
        tail_clean = re.sub(r"\b[A-Z]{2}[0-9A-Z]{4,11}\b", " ", tail)          # обрывки ISIN не считать числами
        tail_clean = re.sub(r"[Уу]ровень\s*\d", " ", tail_clean)               # «Уровень 1» иерархии справедливой стоимости
        nums = [_num(x) for x in NUM.findall(tail_clean)]
        nums = [n for n in nums if n is not None]
        if len(nums) == 1 and k > 0:
            # крупная стоимость не влезла в ячейку: «1 060 574 174,» строкой выше, дробная часть строкой ниже
            up = lines[k - 1]
            w = re.search(r"(\d{1,3}(?:[ \xa0]\d{3})+),(\d*)[ \t]", up[m.end():] + " ")
            if w:
                col = m.end() + w.start()
                nxt = lines[k + 1][col:col + 25] if k + 1 < len(lines) else ""
                d = re.match(r"\s*(\d+)\b", nxt)
                nums.append(_num(w.group(1) + "," + w.group(2) + (d.group(1) if d else "")))
        if len(nums) >= 2:
            isin = ISIN.search(line)
            out.append({"isin": isin.group(1) if isin else "", "reg": m.group(1), "name": line[:m.start()].strip()[-60:],
                        "qty": nums[-2], "value": nums[-1]})
    return out


def totals(path):
    """Итоги строк 02 (российские эмитенты), 03 (иностранные) и СЧА — из первой копии отчёта."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        txt = subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout
    else:
        txt = "\n".join(" ".join(str(c) for c in r if c not in (None, "")) for r in _rows(p))
    out = {}
    for code, key in (("02", "ru"), ("03", "foreign")):
        m = re.search(r"эмитентов\s*[–-]\s*вс[её]го\s+" + code + r"\s+([\d \xa0]+(?:[.,]\d+)?)", txt)
        out[key] = (_num(m.group(1)) if m else 0.0) or 0.0
    m = re.search(r"(?<![\d.])02\.07(?![\d.])[ \t\xa0]+(\d[\d \xa0]*(?:[.,]\d+)?)", txt)   # код строки «акции российских АО – всего»
    out["shares"] = (_num(m.group(1)) if m else 0.0) or 0.0
    return out


def parse(path):
    """Позиции без повторов (в части справок отчёт напечатан дважды — берём первое вхождение бумаги)."""
    p = Path(path)
    raw = from_xlsx(p) if p.suffix.lower() in (".xlsx", ".xlsm", ".xls") else from_pdf(p)
    seen, out = set(), []
    for x in raw:
        k = x.get("reg") or x.get("isin")
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


if __name__ == "__main__":
    pos = parse(sys.argv[1])
    for x in pos:
        print(f"{x['isin']}  {x['qty']:>14,.0f}  {x['value']:>16,.2f}  {x['name'][:60]}")
    print("позиций:", len(pos), "сумма:", f"{sum(x['value'] for x in pos):,.2f}")
