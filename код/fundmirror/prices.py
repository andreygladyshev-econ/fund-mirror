"""Цены и дивиденды акций Мосбиржи для анализа сделок фондов.
- соответствие «рег. номер выпуска → тикер»: справочник TQBR (текущие бумаги) + поиск ISS по номеру для остальных;
- цены закрытия на даты справок (конец месяца) и на промежуточные концы месяцев: история TQBR по дате (все бумаги
  сразу, 3 запроса на дату);
- дивиденды: smart-lab.ru/q/{тикер}/dividend/ (дата отсечки и сумма на акцию; ISS перестал их отдавать);
- дробления и консолидации: ISS /statistics/engines/stock/splits (цены в ISS не скорректированы на них).
python3 prices.py -> данные/составы_фондов/тикеры.csv, цены.csv, дивиденды.csv, дробления.csv
"""
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "данные" / "составы_фондов"
ISS = "https://iss.moex.com/iss"


def J(url):
    for k in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=90))
        except Exception:  # noqa: BLE001
            time.sleep(4 * (k + 1))
    return None


def table(d, name):
    t = d.get(name, {}) if d else {}
    return [dict(zip(t.get("columns", []), r)) for r in t.get("data", [])]


def month_ends(d0, d1):
    out, d = [], date(d0.year, d0.month, 1)
    while d <= d1:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        out.append(nxt - timedelta(days=1))
        d = nxt
    return out


def close_on(day):
    """Цены закрытия TQBR на последний торговый день не позже day (до 7 дней назад)."""
    for back in range(8):
        dd = day - timedelta(days=back)
        rows, start = [], 0
        while True:
            d = J(f"{ISS}/history/engines/stock/markets/shares/boards/TQBR/securities.json?iss.meta=off&date={dd}&start={start}"
                  "&history.columns=SECID,TRADEDATE,CLOSE,LEGALCLOSEPRICE")
            part = table(d, "history")
            rows += part
            if len(part) < 100:
                break
            start += 100
        px = {r["SECID"]: r["CLOSE"] or r["LEGALCLOSEPRICE"] for r in rows if (r["CLOSE"] or r["LEGALCLOSEPRICE"])}
        if len(px) > 50:
            return dd, px
    return day, {}


def splits():
    rows = table(J(f"{ISS}/statistics/engines/stock/splits.json?iss.meta=off"), "splits")
    with open(ROOT / "дробления.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["дата", "secid", "до", "после"])
        for r in rows:
            w.writerow([r["tradedate"], r["secid"], r["before"], r["after"]])
    print("дроблений:", len(rows))


def main():
    splits()
    pos = list(csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")))
    regs = sorted({p["reg"] for p in pos if p["reg"]})
    cur = table(J(f"{ISS}/engines/stock/markets/shares/boards/TQBR/securities.json?iss.meta=off&iss.only=securities"
                  "&securities.columns=SECID,REGNUMBER,ISIN,SHORTNAME"), "securities")
    reg2 = {r["REGNUMBER"]: r["SECID"] for r in cur if r.get("REGNUMBER")}
    for rg in regs:
        if rg not in reg2:
            found = [r for r in table(J(f"{ISS}/securities.json?iss.meta=off&q={rg}&securities.columns=secid,regnumber,isin,primary_boardid"), "securities")
                     if r.get("regnumber") == rg and (r.get("primary_boardid") or "").startswith("TQ")]
            if found:
                reg2[rg] = found[0]["secid"]
    with open(ROOT / "тикеры.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reg", "secid"])
        for rg in regs:
            w.writerow([rg, reg2.get(rg, "")])
    print(f"рег. номеров {len(regs)}, найдено тикеров {sum(1 for r in regs if reg2.get(r))}", flush=True)
    dates = sorted({date.fromisoformat(p["дата"]) for p in pos})
    last = date.today() - timedelta(days=1)                   # последний завершённый торговый день (было зашито 25.09.2026)
    ends = sorted(set(month_ends(dates[0], last)) | set(dates))
    with open(ROOT / "цены.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["дата", "торговый_день", "secid", "close"])
        for d in ends:
            td, px = close_on(min(d, last))                    # конец текущего месяца ещё не наступил — цена последнего дня (см. торговый_день)
            for s, c in px.items():
                w.writerow([d.isoformat(), td.isoformat(), s, c])
            print(d, td, len(px), flush=True)
    dividends(sorted({reg2[r] for r in regs if reg2.get(r)}))                   # только бумаги из портфелей фондов


def smartlab_divs(secid):
    """История выплат с smart-lab.ru/q/<тикер>/dividend/ (ISS перестал отдавать /securities/<тикер>/dividends:
    с осени 2026 там карточка бумаги). Строка: тикер, дата T-1, дата отсечки, период, дивиденд, …"""
    import re
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    t = None
    for page in [secid] + ([secid[:-1]] if secid.endswith("P") else []):   # у привилегированных нет своей страницы
        for k in range(3):
            try:
                t = urllib.request.urlopen(urllib.request.Request(f"https://smart-lab.ru/q/{page}/dividend/", headers={"User-Agent": "Mozilla/5.0"}),
                                           timeout=60, context=ctx).read().decode("utf-8", "ignore")
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    t = t or ""                               # страницы нет — выплат не было (не сбой сети)
                    break
                time.sleep(3 * (k + 1))
            except Exception:  # noqa: BLE001
                time.sleep(3 * (k + 1))
        if t:
            break
    if t is None:
        return None
    out = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S):
        c = [re.sub(r"<[^>]+>|\s+", " ", x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if len(c) >= 5 and c[0] == secid and re.fullmatch(r"\d\d\.\d\d\.\d{4}", c[2]):
            v = re.sub(r"[^\d,.]", "", c[4]).replace(",", ".")
            if v:
                out.append((f"{c[2][6:]}-{c[2][3:5]}-{c[2][:2]}", float(v)))
    return out


def dividends(tick):
    n_ok, rows = 0, []
    for s in tick:
        d = smartlab_divs(s)
        if d is None:
            print("дивиденды не получены:", s, flush=True)
            continue
        n_ok += 1
        rows += [(s, day, v, "RUB") for day, v in sorted(set(d))]
        time.sleep(0.3)
    assert n_ok > 0.8 * len(tick) and rows, f"дивиденды получены лишь по {n_ok} из {len(tick)} тикеров — расчёт без дивидендов неверен"
    with open(ROOT / "дивиденды.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["secid", "реестр", "дивиденд", "валюта"])
        w.writerows(rows)
    print(f"дивиденды: {n_ok} из {len(tick)} тикеров, выплат {len(rows)}")


if __name__ == "__main__":
    main()
