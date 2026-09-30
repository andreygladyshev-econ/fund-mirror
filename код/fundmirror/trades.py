"""«Хорошо покупают, плохо продают» на российских фондах акций. Методика — Akepanidtaworn, Di Mascio,
Imas, Schmidt (JF 2023) в месячной версии (как у Chen, Jegadeesh, Wermers 2000 на квартальных составах).

Для соседних справок фонда (d0, d1; разрыв ≤ 45 дней; обе прошли сверку сумм):
  продажа  — число акций уменьшилось (частично или полностью), покупка — выросло (в том числе новая позиция);
  дробления акций отсекаются (кратное изменение числа при обратном изменении цены);
  доходность бумаги от d1 на h = 1, 3, 6 месяцев — рост цены плюс дивиденды, цена без которых (дата отсечки) в окне;
  контрфакт продажи — средняя доходность бумаг, которые фонд держал на d0 и не продавал; контрфакт покупки — средняя
  доходность бумаг, которые держал и не докупал.
  второй вариант (Dp) — контрфакт весь портфель на d0 без самой бумаги: не зависит от того, как часто фонд торгует;
  Dm, Dpm — то же, но только среди бумаг той же терцили прошлой доходности (контроль моментума).
  D = доходность(проданной или купленной) − контрфакт. Для продаж D > 0 значит «продали то, что потом росло лучше
  оставленного» — ошибка продажи; для покупок D > 0 — навык покупки.
Погрешность — бутстрэп по месяцам (все фонды торгуют на одном рынке в один месяц).
Эвристика продаж: доля продаж из крайних квинтилей прошлой доходности (3 мес. до d0) внутри портфеля.

python3 trades.py -> данные/составы_фондов/сделки.csv (и сводка в консоль)
"""
import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "данные" / "составы_фондов"
H = (1, 3, 6)
MIN_POOL = 5                      # меньше бумаг в контрфакте — сравнение превращается в шум (событие не считаем)


def load():
    ok = {(r["фонд"], r["дата"]) for r in csv.DictReader(open(ROOT / "проверка.csv", encoding="utf-8")) if r["ок"] == "True"}
    tick = {r["reg"]: r["secid"] for r in csv.DictReader(open(ROOT / "тикеры.csv", encoding="utf-8")) if r["secid"]}
    hold = defaultdict(lambda: defaultdict(float))
    for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")):
        if (p["фонд"], p["дата"]) in ok and p["reg"] in tick:
            hold[(p["фонд"], p["дата"])][tick[p["reg"]]] += float(p["qty"])
    px = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "цены.csv", encoding="utf-8")):
        px[r["дата"]][r["secid"]] = float(r["close"])
    div = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "дивиденды.csv", encoding="utf-8")):
        if r["реестр"] and r["дивиденд"] and r["валюта"] in ("RUB", "SUR"):
            div[r["secid"]].append((ex_date(r["реестр"]), float(r["дивиденд"])))
    # цены ISS не скорректированы на дробления: приводим цены и число бумаг до даты дробления к новым акциям
    for r in csv.DictReader(open(ROOT / "дробления.csv", encoding="utf-8")):
        s, t, k = r["secid"], r["дата"], float(r["после"]) / float(r["до"])
        for d in px:
            if d < t and s in px[d]:
                px[d][s] /= k
        # дивиденды не трогаем: smart-lab уже даёт их на одну текущую акцию (проверено: GMKN, PLZL, VTBR, T, TRNFP)
        for (f, d), h in hold.items():
            if d < t and s in h:
                h[s] *= k
    return hold, px, div


def ex_date(record):
    """Первый день торгов без дивиденда (цена падает в этот день) по дате реестра. При расчётах T+2 (до 31.07.2023)
    это рабочий день перед датой реестра, при T+1 — сама дата реестра. Дивиденд относится к тому окну доходности,
    в котором цена его уже не содержит; праздники не учитываются (только выходные)."""
    if record >= "2023-07-31":
        return record
    d = date.fromisoformat(record) - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def _eom(d, add):
    """Календарный конец месяца, отстоящего от месяца даты d на add месяцев (строка ISO)."""
    y, m = int(d[:4]), int(d[5:7]) + add
    y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return (nxt - timedelta(days=1)).isoformat()


def fwd(s, d1, months, px, div, dates=None):
    """Полная доходность бумаги s от даты справки d1 до конца календарного месяца через months месяцев."""
    d2 = _eom(d1, months)
    if d2 not in px or d1 not in px:
        return None
    p1, p2 = px[d1].get(s), px[d2].get(s)
    if not p1 or not p2:
        return None
    dv = sum(v for day, v in div[s] if d1 < day <= d2)
    return (p2 + dv) / p1 - 1


def past(s, d0, px, dates=None, months=3):
    db = _eom(d0, -months)
    if db not in px or d0 not in px:
        return None
    p0, pb = px[d0].get(s), px[db].get(s)
    return p0 / pb - 1 if p0 and pb else None


def month_key(d):
    return d[:7]


def main():
    hold, px, div = load()
    dates = sorted(px)
    ev = []
    funds = defaultdict(list)
    for (f, d) in hold:
        funds[f].append(d)
    for f, ds in funds.items():
        ds.sort()
        for d0, d1 in zip(ds, ds[1:]):
            if (date.fromisoformat(d1) - date.fromisoformat(d0)).days > 45 or d0 not in px or d1 not in px:
                continue
            h0, h1 = hold[(f, d0)], hold[(f, d1)]
            names = sorted(set(h0) | set(h1))                         # порядок строк не зависит от хеширования
            chg = {}
            for s in names:
                q0, q1 = h0.get(s, 0.0), h1.get(s, 0.0)
                if q0 > 0 and q1 > 0:
                    r = q1 / q0
                    pr = (px[d1].get(s) or 0) / (px[d0].get(s) or 1)
                    k = round(r) if r >= 1 else round(1 / r)
                    if k >= 2 and abs((r if r >= 1 else 1 / r) - k) < 1e-6 and pr and abs(pr * (r if r >= 1 else 1 / r) - 1) < 0.35:
                        continue                                     # дробление/консолидация, не сделка
                if abs(q1 - q0) > 1e-9 * max(q0, q1, 1):
                    chg[s] = q1 - q0
            if not chg:
                continue
            held = [s for s in h0 if s not in chg or chg[s] > 0]      # держали и не продавали
            kept = [s for s in h0 if s not in chg]                    # держали без изменений
            pst = {s: past(s, d0, px, dates) for s in h0}
            pvals = [v for v in pst.values() if v is not None]
            q = np.quantile(pvals, [0.2, 0.8]) if len(pvals) >= 5 else None
            for s, dq in chg.items():
                side = "продажа" if dq < 0 else "покупка"
                pool = held if side == "продажа" else [x for x in kept if x != s]
                row = {"фонд": f, "d0": d0, "d1": d1, "secid": s, "сторона": side,
                       "полностью": side == "продажа" and h1.get(s, 0) == 0, "новая": side == "покупка" and h0.get(s, 0) == 0,
                       "объём_руб": abs(dq) * (px[d0].get(s) or px[d1].get(s) or 0)}
                pp = pst.get(s)
                # признаки для модели «пожалеем ли о продаже» (всё известно на дату решения d0)
                v0 = {x: q * (px[d0].get(x) or 0) for x, q in h0.items()}
                tot = sum(v0.values()) or 1
                p12 = [px[_eom(d0, -k)].get(s) for k in range(0, 13) if _eom(d0, -k) in px]
                p12 = [x for x in p12 if x]
                rets = np.diff(np.log(p12[::-1])) if len(p12) >= 7 else []
                row.update({"прошл1": past(s, d0, px, dates, 1), "прошл3": pp, "прошл12": past(s, d0, px, dates, 12),
                            "прошл3_портф": float(np.mean(pvals)) if pvals else None, "вес": v0.get(s, 0) / tot,
                            "держали_мес": sum(1 for d in ds if d <= d0 and s in hold[(f, d)]),
                            "вол12": float(np.std(rets)) if len(rets) else None})
                row["крайний_квинтиль"] = (None if q is None or pp is None else bool(pp <= q[0] or pp >= q[1]))
                # контроль моментума: контрфакт только из бумаг той же терцили прошлой доходности внутри портфеля
                t3 = np.quantile(pvals, [1 / 3, 2 / 3]) if len(pvals) >= 6 else None
                tc = (lambda v: None if v is None or t3 is None else int(v > t3[0]) + int(v > t3[1]))
                port = [x for x in h0 if x != s]                          # весь портфель на d0 без самой бумаги
                for hmo in H:
                    r = fwd(s, d1, hmo, px, div, dates)
                    for tag, pl in (("", pool), ("p", port)):
                        cf = [v for x in pl if x != s and (v := fwd(x, d1, hmo, px, div, dates)) is not None]
                        row[f"D{tag}{hmo}"] = (r - float(np.mean(cf))) if r is not None and len(cf) >= MIN_POOL else None
                        cm = [v for x in pl if x != s and tc(pst.get(x)) is not None and tc(pst.get(x)) == tc(pp)
                              and (v := fwd(x, d1, hmo, px, div, dates)) is not None]
                        row[f"D{tag}m{hmo}"] = (r - float(np.mean(cm))) if r is not None and len(cm) >= 3 and tc(pp) is not None else None
                ev.append(row)
    with open(ROOT / "сделки.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev[0]))
        w.writeheader()
        w.writerows(ev)
    rng = np.random.default_rng(5)
    res = {}
    for side in ("продажа", "покупка"):
        for hmo in H:
            rows = [e for e in ev if e["сторона"] == side and e[f"D{hmo}"] is not None]
            if not rows:
                continue
            by = defaultdict(list)
            for e in rows:
                by[month_key(e["d1"])].append(e[f"D{hmo}"])
            ms = list(by)
            m = float(np.mean([e[f"D{hmo}"] for e in rows]))
            boot = []
            for _ in range(2000):
                pick = rng.choice(len(ms), len(ms))
                vals = [v for i in pick for v in by[ms[i]]]
                boot.append(np.mean(vals))
            res[f"{side} {hmo} мес."] = {"событий": len(rows), "месяцев": len(ms), "D_средняя_пп": m * 100,
                                         "95%": [float(np.percentile(boot, 2.5)) * 100, float(np.percentile(boot, 97.5)) * 100],
                                         "доля_D>0": float(np.mean([e[f"D{hmo}"] > 0 for e in rows]))}
            r = res[f"{side} {hmo} мес."]
            print(f"{side:<8} {hmo} мес.: событий {r['событий']:5d}, месяцев {r['месяцев']:3d}, D = {r['D_средняя_пп']:+.2f} п.п. "
                  f"[{r['95%'][0]:+.2f}; {r['95%'][1]:+.2f}], доля D>0 {r['доля_D>0']:.0%}", flush=True)
    print("по фондам (3 мес.): продажи D / покупки D, п.п.; число событий")
    res["по_фондам"] = {}
    for f in sorted({e["фонд"] for e in ev}):
        sd = [e["D3"] for e in ev if e["фонд"] == f and e["сторона"] == "продажа" and e["D3"] is not None]
        bd = [e["D3"] for e in ev if e["фонд"] == f and e["сторона"] == "покупка" and e["D3"] is not None]
        res["по_фондам"][f] = {"продажи": [float(np.mean(sd)) * 100 if sd else None, len(sd)], "покупки": [float(np.mean(bd)) * 100 if bd else None, len(bd)]}
        print(f"  {f[:55]:<55} продажи {np.mean(sd)*100 if sd else float('nan'):+6.2f} ({len(sd):4d})  покупки {np.mean(bd)*100 if bd else float('nan'):+6.2f} ({len(bd):4d})")
    ext = [e for e in ev if e["сторона"] == "продажа" and e["крайний_квинтиль"] is not None]
    if ext:
        res["продажи_из_крайних_квинтилей"] = float(np.mean([e["крайний_квинтиль"] for e in ext]))
        print(f"продаж из крайних квинтилей прошлой доходности: {res['продажи_из_крайних_квинтилей']:.0%} (при случайном выборе 40%)")


if __name__ == "__main__":
    main()
