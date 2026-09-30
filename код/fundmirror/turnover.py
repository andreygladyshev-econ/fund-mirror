"""Дневной оборот (₽) бумаг TQBR на концах месяцев — для фильтра ликвидности подсказок. Отдельный файл, базовые цены не
трогает. python3 turnover.py -> данные/составы_фондов/обороты.csv"""
import csv
from datetime import date, timedelta

from prices import ISS, J, ROOT, month_ends, table


def main():
    out = ROOT / "обороты.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["дата", "торговый_день", "secid", "оборот_руб"])
        for d in month_ends(date(2021, 9, 1), date.today() - timedelta(days=1)):
            for back in range(8):
                dd = min(d, date.today() - timedelta(days=1)) - timedelta(days=back)
                rows, start = [], 0
                while True:
                    part = table(J(f"{ISS}/history/engines/stock/markets/shares/boards/TQBR/securities.json?iss.meta=off&date={dd}&start={start}"
                                   "&history.columns=SECID,VALUE"), "history")
                    rows += part
                    if len(part) < 100:
                        break
                    start += 100
                if sum(1 for r in rows if r["VALUE"]) > 50:
                    w.writerows([d.isoformat(), dd.isoformat(), r["SECID"], r["VALUE"] or 0] for r in rows)
                    print(d, dd, len(rows), flush=True)
                    break


if __name__ == "__main__":
    main()
