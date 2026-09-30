"""Разбор всех скачанных справок (опись.csv) в таблицу позиций и сверка с итогами справки.
python3 parse_all.py -> данные/составы_фондов/позиции.csv, проверка.csv
Месяц годен, если сумма разобранных позиций акций совпадает с итогом строки 02.07 с точностью 0,5%. Если правила
не справились (нестандартная вёрстка), берётся ответ языковой модели из llm_positions/ — тоже только при совпадении.
"""
import csv
import hashlib
import json
from pathlib import Path

from llm_read import qty_flags
from parse_scha import parse, totals

ROOT = Path(__file__).resolve().parents[2] / "данные" / "составы_фондов"

if __name__ == "__main__":
    inv = list(csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8")))
    inv = list({(r["ук"], r["фонд"], r["дата"]): r for r in inv}.values())   # на одну дату бывает две справки (исправленная) — берём последнюю
    pos_rows, chk = [], []
    for r in inv:
        f = ROOT / r["файл"]
        try:
            pos, tot = parse(f), totals(f)
        except Exception as e:  # noqa: BLE001
            chk.append({**{k: r[k] for k in ("ук", "фонд", "дата")}, "позиций": 0, "сумма": 0, "итог_ру": 0, "итог_акции": 0, "итог_ин": 0, "ок": f"ошибка {e}"[:60], "источник": ""})
            continue
        s = sum(x["value"] for x in pos)
        ok = tot["shares"] > 0 and abs(s - tot["shares"]) <= 0.005 * tot["shares"]
        src = "правила"
        c = ROOT / "llm_positions" / (hashlib.md5(r["файл"].encode()).hexdigest() + ".json")
        if not ok and c.exists() and (j := json.loads(c.read_text()))["сошлось"] and not qty_flags(j["позиции"], r["дата"]):
            # запасной разбор языковой моделью — принимается только при совпадении суммы с итогом 02.07 (llm_read.py)
            # и второй сверке: количество × цена биржи ≈ стоимость у каждой позиции
            pos = [{"reg": x.get("reg", ""), "isin": "", "qty": float(x["qty"]), "value": float(x["value"]), "name": ""} for x in j["позиции"]]
            s, ok, src = j["сумма"], True, "модель"
        chk.append({"ук": r["ук"], "фонд": r["фонд"], "дата": r["дата"], "позиций": len(pos), "сумма": round(s),
                    "итог_ру": round(tot["ru"]), "итог_акции": round(tot["shares"]), "итог_ин": round(tot["foreign"]), "ок": ok, "источник": src})
        for x in pos:
            pos_rows.append({"ук": r["ук"], "фонд": r["фонд"], "дата": r["дата"], "reg": x.get("reg", ""), "isin": x.get("isin", ""),
                             "qty": x["qty"], "value": x["value"], "имя": x["name"][:60]})
    for name, rows in (("позиции.csv", pos_rows), ("проверка.csv", chk)):
        with open(ROOT / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    good = sum(1 for c in chk if c["ок"] is True)
    print(f"справок {len(chk)}, сошлось {good}, позиций {len(pos_rows)}")
    for c in chk:
        if c["ок"] is not True:
            print("  не сошлось:", c["фонд"][:40], c["дата"], c["позиций"], c["сумма"], c.get("итог_акции"), c["итог_ин"], c["ок"])
