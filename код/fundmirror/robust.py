"""Проверки устойчивости результата trades.py на сделки.csv (горизонт 3 мес., бутстрэп по месяцам).
  - D — против бумаг, которые держали без сделки; Dp — против всего портфеля на начало месяца;
    Dm, Dpm — то же внутри терцили прошлой доходности (контроль моментума);
  - без механических фондов («Трендовые»); полные / частичные продажи; новые позиции / докупки;
  - взвешивание по объёму сделки в рублях; по УК.
python3 robust.py [горизонт=3] -> печать и данные/составы_фондов/устойчивость.json
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "данные" / "составы_фондов"
MECH = ("Трендовые",)                                     # фонды, торгующие по механическому правилу
INDEX = ("SBMX", "SIPO", "Индекс МосБиржи", "Индекс дивидендных", "Индекс акций роста", "Т-Капитал Дивидендные")   # TDIV следует индексу TDIVRX   # «Индекс МосБиржи» — ВИМ и АК Барс                                  # индексные фонды — контроль: навыка быть не должно (D ≈ 0)
# Перерегистрация иностранных компаний в Россию (МКПАО): бумага «впервые» появляется в разделе российских акций
# справки, хотя фонд держал компанию и раньше (в разделе иностранных эмитентов) — это не решение управляющего.
REDOM = {"YDEX", "HEAD", "X5", "OZON", "CNRU", "RAGR", "T", "MDMG", "LENT", "FIXR", "GLRX", "ETLN"}
UNSURE = ("SBBC", "SBSC", "Дивидендные акции. Россия", "Анализ акций", "ПСБ – Российские акции", "Т-Капитал Акции роста")   # TITR: был индексным, стал активным (дата неизвестна)                                 # БПИФ, стратегия не подтверждена (вероятно индексные) — вне основного анализа и контроля


def boot(rows, col, w=False, n=2000, seed=5):
    rows = [r for r in rows if r[col] not in ("", None)]
    if len(rows) < 20:
        return None
    by = defaultdict(list)
    for r in rows:
        by[r["d1"][:7]].append((float(r[col]), float(r["объём_руб"]) if w else 1.0))
    ms = list(by)

    def mean(pairs):
        v = np.array(pairs)
        return float((v[:, 0] * v[:, 1]).sum() / v[:, 1].sum()) if v[:, 1].sum() > 0 else float("nan")

    m = mean([p for k in ms for p in by[k]])
    rng = np.random.default_rng(seed)
    b = [mean([p for i in rng.choice(len(ms), len(ms)) for p in by[ms[i]]]) for _ in range(n)]
    return {"n": len(rows), "D": m * 100, "lo": float(np.nanpercentile(b, 2.5)) * 100, "hi": float(np.nanpercentile(b, 97.5)) * 100}


def selective(ev):
    """Единое определение (27.09) для записки и «Зеркала»: активные фонды (без индексных, неуточнённых, механических),
    без перерегистраций (обмен бумаг — не решение); сделка «выборочная», если в этом месяце фонд сделал сделки той же
    стороны не более чем по половине позиций (иначе — приток/отток денег)."""
    from collections import Counter
    act = [e for e in ev if not any(k in e["фонд"] for k in INDEX + UNSURE + MECH) and e["secid"] not in REDOM]
    nh = Counter()
    for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")):
        nh[(p["фонд"], p["дата"])] += 1
    ns = Counter((e["фонд"], e["d1"], e["сторона"]) for e in act)
    return [e for e in act if ns[(e["фонд"], e["d1"], e["сторона"])] <= 0.5 * nh.get((e["фонд"], e["d0"]), 1)]


def canon():
    """Канонические цифры «проблемы» для записки: python3 robust.py canon -> проблема_канон.json"""
    import trades as T
    ev = list(csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8")))
    uk = {r["фонд"]: r["ук"] for r in csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8"))}
    _, px, _ = T.load()
    sel = selective(ev)
    seasoned = lambda e: bool(px.get(T._eom(e["d1"], -6), {}).get(e["secid"]))       # торговалась ≥ 6 мес. — не IPO
    B = [e for e in sel if e["сторона"] == "покупка"]
    S = [e for e in sel if e["сторона"] == "продажа"]
    idx = [e for e in ev if any(k in e["фонд"] for k in INDEX)]
    rows = {"выборочные покупки": boot(B, "Dp3"), "выборочные покупки, контроль моментума": boot(B, "Dpm3"),
            "новые позиции без IPO": boot([e for e in B if e["новая"] == "True" and seasoned(e)], "Dp3"),
            "новые позиции: IPO": boot([e for e in B if e["новая"] == "True" and not seasoned(e)], "Dp3"),
            "выборочные продажи": boot(S, "Dp3"), "выборочные полные выходы": boot([e for e in S if e["полностью"] == "True"], "Dp3"),
            "контроль: индексные, покупки": boot([e for e in idx if e["сторона"] == "покупка"], "Dp3"),
            "контроль: индексные, продажи": boot([e for e in idx if e["сторона"] == "продажа"], "Dp3")}
    per = {u: boot([e for e in B if uk.get(e["фонд"]) == u], "Dp3") for u in sorted({uk.get(e["фонд"]) for e in B})}
    rows["УК со значимым навыком покупок"] = f"{sum(1 for b in per.values() if b and b['lo'] > 0)} из {len(per)}"
    for k, v in rows.items():
        print(f"{k:<42} " + (v if isinstance(v, str) else "мало данных" if v is None else f"{v['D']:+.2f} [{v['lo']:+.2f}; {v['hi']:+.2f}] n={v['n']}"))
    for u, b in per.items():
        print(f"   {u:<18} " + ("мало данных" if b is None else f"{b['D']:+.2f} [{b['lo']:+.2f}; {b['hi']:+.2f}] n={b['n']}"))
    (ROOT / "проблема_канон.json").write_text(json.dumps({"итог": rows, "по_УК": per}, ensure_ascii=False, indent=1))


def main():
    h = sys.argv[1] if len(sys.argv) > 1 else "3"
    ev = list(csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8")))
    uk = {r["фонд"]: r["ук"] for r in csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8"))}
    for e in ev:
        e["ук"] = uk.get(e["фонд"], "?")
    idx = [e for e in ev if any(k in e["фонд"] for k in INDEX)]
    uns = [e for e in ev if any(k in e["фонд"] for k in UNSURE)]
    ev = [e for e in ev if not any(k in e["фонд"] for k in INDEX + UNSURE)]   # основной анализ — только активные фонды
    S = [e for e in ev if e["сторона"] == "продажа"]
    B = [e for e in ev if e["сторона"] == "покупка"]
    nm = lambda rows: [e for e in rows if not any(k in e["фонд"] for k in MECH)]
    tests = {}
    for side, rows in (("продажи", S), ("покупки", B)):
        for tag, name in (("D", "держали без сделки"), ("Dp", "весь портфель")):
            tests[f"{side}: {name}"] = (rows, tag, False)
            tests[f"{side}: {name}, контроль моментума"] = (rows, tag + ("m" if tag == "D" else "m"), False)
            tests[f"{side}: {name}, без механических"] = (nm(rows), tag, False)
            tests[f"{side}: {name}, взвешено по объёму"] = (rows, tag, True)
        full = "полностью" if side == "продажи" else "новая"
        tests[f"{side}: весь портфель, {'полные' if side == 'продажи' else 'новые позиции'}"] = ([e for e in rows if e[full] == "True"], "Dp", False)
        tests[f"{side}: весь портфель, {'частичные' if side == 'продажи' else 'докупки'}"] = ([e for e in rows if e[full] != "True"], "Dp", False)
    for u in sorted({e["ук"] for e in ev}):
        for tag in ("D", "Dp", "Dpm"):
            tests[f"продажи: {u} [{tag}]"] = ([e for e in S if e["ук"] == u], tag, False)
            tests[f"покупки: {u} [{tag}]"] = ([e for e in B if e["ук"] == u], tag, False)
    for side, name in (("продажа", "продажи"), ("покупка", "покупки")):
        for tag in ("Dp", "Dpm"):
            tests[f"КОНТРОЛЬ, индексные фонды: {name} [{tag}]"] = ([e for e in idx if e["сторона"] == side], tag, False)
    for side, name in (("продажа", "продажи"), ("покупка", "покупки")):
        tests[f"не уточнено (SBBC, SBSC, ДОХОДЪ-2): {name} [Dp]"] = ([e for e in uns if e["сторона"] == side], "Dp", False)
    for f in sorted({e["фонд"] for e in ev} | {e["фонд"] for e in idx}):
        for side, name in (("продажа", "продажи"), ("покупка", "покупки")):
            tests[f"фонд {f[:32]}: {name} [Dp]"] = ([e for e in ev + idx if e["фонд"] == f and e["сторона"] == side], "Dp", False)
    out = {}
    for k, (rows, col, w) in tests.items():
        r = boot(rows, f"{col}{h}", w)
        out[k] = r
        print(f"{k:<62} " + ("мало данных" if r is None else f"n={r['n']:5d}  D={r['D']:+6.2f} п.п. [{r['lo']:+6.2f}; {r['hi']:+6.2f}]"))
    (ROOT / f"устойчивость_{h}м.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    canon() if sys.argv[1:] == ["canon"] else main()
