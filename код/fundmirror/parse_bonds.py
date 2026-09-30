"""Облигации из справок СЧА (Excel): ISIN, количество, стоимость; сверка суммы с итогами строк справки
(строки второго уровня разделов 02 и 03, в названии которых «облигации» / «государственные» / «муниципальные»
ценные бумаги — коды у разных форм справки различаются). Справка
принимается, если расхождение ≤ 0,5%.
python3 parse_bonds.py [опись_облигации.csv опись.csv ...] -> данные/составы_фондов/облигации.csv, облигации_проверка.csv
"""
import csv
import re
import sys
from datetime import datetime

import subprocess

from parse_scha import _rows as _xl_rows


def _rows(path):
    """Excel — строки листов; PDF — строки pdftotext -layout, разбитые на ячейки по 2+ пробелам."""
    if str(path).lower().endswith(".pdf"):
        t = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
        for line in t.splitlines():
            yield re.split(r"\s{2,}", line.strip())
        return
    yield from _xl_rows(path)
from robust import ROOT

ISIN = re.compile(r"^(RU000A[0-9A-Z]{6}|RU[0-9]{9}[0-9A-Z]|SU\d{5}RMFS\d)$")
TOT = re.compile(r"^0[23]\.\d\d$")                          # разделы 2.х и 3.х второго уровня
BOND = re.compile(r"облигац|государственные ценные бумаги|муниципальные ценные бумаги", re.I)


def num(x):
    if isinstance(x, (int, float)):
        return float(x)
    if re.fullmatch(r"\d{10}|\d{12,13}|\d{15}", str(x).strip()):   # ИНН / ОГРН, а не количество
        return None
    s = str(x).replace("\xa0", "").replace(" ", "").replace(",", ".")
    return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def parse(path):
    pos, tot, inb, prev = {}, {}, False, ""
    for row in _rows(path):
        cells = [c for c in row if c not in (None, "")]
        strs = [str(c).strip() for c in cells]
        head = " ".join(strs[:2])
        if head.startswith("Расшифровк") or "Подраздел" in head or re.match(r"\d\.\d{1,2}\.\s+[А-Я]", head):   # позиции берём только из подразделов облигаций
            inb = bool(BOND.search(head)) and not re.search(r"расписк|паи|пае|акци|вексел|сертификат", head, re.I)
        for i, s in enumerate(strs):
            label = " ".join(strs[:i]) if i else prev              # в PDF код строки бывает на отдельной строке
            if TOT.match(s) and BOND.search(label) and i + 1 < len(cells) and num(cells[i + 1]) is not None and s not in tot:
                tot[s] = num(cells[i + 1])
        prev = " ".join(strs)
        k = next((i for i, s in enumerate(strs) if ISIN.match(s)), None)
        if k is None or not inb:
            continue
        rest = cells[k + 1:]
        mat = next((str(c)[:10] for c in rest if isinstance(c, datetime) or re.fullmatch(r"\d{4}[.\-]\d\d[.\-]\d\d.*|\d\d\.\d\d\.\d{4}", str(c).strip())), "")
        nums = [num(c) for c in rest if not isinstance(c, datetime) and num(c) is not None and not re.fullmatch(r"\d\d\.\d\d\.\d{4}|\d{4}\.\d\d\.\d\d", str(c).strip())]
        if len(nums) >= 2:
            q, v = nums[0], nums[1]
            isin = strs[k]
            if isin in pos:                          # одна бумага двумя строками (разные места хранения) — складываем
                pos[isin] = (pos[isin][0] + q, pos[isin][1] + v, mat)
            else:
                pos[isin] = (q, v, mat)
    return pos, tot


PDF_OK = {"ВИМ"}                                         # PDF с целыми номерами выпусков в строке


def main(invs):
    out, chk = [], []
    for inv in invs:
        for r in csv.DictReader(open(ROOT / inv, encoding="utf-8")):
            p = ROOT / r["файл"]
            if not str(p).lower().endswith((".xlsx", ".xls", ".pdf")) or (str(p).lower().endswith(".pdf") and r["ук"] not in PDF_OK):
                continue
            try:
                pos, tot = parse(p)
            except Exception as e:  # noqa: BLE001
                chk.append({"ук": r["ук"], "фонд": r["фонд"], "дата": r["дата"], "сумма": 0, "итог": 0, "сошлось": False, "ошибка": str(e)[:80]})
                continue
            s, t = sum(v for _, v, _ in pos.values()), sum(tot.values())
            ok = t > 0 and abs(s - t) <= 0.005 * t
            chk.append({"ук": r["ук"], "фонд": r["фонд"], "дата": r["дата"], "сумма": round(s), "итог": round(t), "сошлось": ok, "ошибка": ""})
            if ok:
                out += [{"ук": r["ук"], "фонд": r["фонд"], "дата": r["дата"], "isin": i, "qty": q, "value": v, "погашение": m}
                        for i, (q, v, m) in pos.items()]
    for name, rows in (("облигации.csv", out), ("облигации_проверка.csv", chk)):
        with open(ROOT / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    ok = [c for c in chk if c["сошлось"] and c["итог"] > 0]
    print(f"справок: {len(chk)}; сошлось: {len(ok)}; без облигаций/не сошлось: {len(chk) - len(ok)}; позиций: {len(out)}")
    from collections import Counter
    bad = Counter(c["фонд"] for c in chk if not c["сошлось"])
    print("не сошлось по фондам:", dict(bad))


if __name__ == "__main__":
    # PDF берём только у УК, где номера выпусков не разорваны по строкам (ВИМ); остальные — через ИИ-читателя
    main(sys.argv[1:] or ["опись_облигации.csv", "опись.csv"])
