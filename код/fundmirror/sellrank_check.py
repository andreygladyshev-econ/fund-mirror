"""Проверки устойчивости «справки к продаже» на всех 12 месяцах (2025 II + 2026 I), без новых прогонов моделей.
1. Бутстрэп по месяцам и по УК×месяц (фонды одной УК зависимы).
2. Без каждой УК по очереди — не держится ли всё на одной команде.
3. Что добавляет ИИ сверх простых правил: случаи, где модель и правило выбирают разное, — кто прав.
4. «Чистые» продажи: без крупных позиций (доля > 7% — возможен лимит) и без выходов.
5. Экономика на уровне портфеля: рубли выборочных продаж × разница доходностей, в % от СЧА акций в год.
python3 sellrank_check.py [deepseek|qwen]   -> печать + данные/составы_фондов/sellrank_check_<модель>.json
"""
import csv
import json
import re
import sys
from collections import defaultdict

import numpy as np

from robust import INDEX, MECH, ROOT, UNSURE

uk = {r["фонд"]: r["ук"] for r in csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8"))}
vol = {(e["фонд"], e["d1"], e["secid"]): float(e["объём_руб"] or 0) for e in csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8"))
       if e["сторона"] == "продажа"}


def feats(text):
    """Признаки каждой бумаги из текста справки (ровно то, что видела модель)."""
    out = {}
    for L, body in re.findall(r"Бумага ([АБВГ]): (.*?)(?=\n\nБумага [АБВГ]:|\Z)", text, re.S):
        num = lambda s: None if s is None else float(s.replace("−", "-"))
        g = lambda pat: (m := re.search(pat, body)) and m.group(1)
        last = lambda name: (m := re.search(rf"^{name}[^:\n]*:([^\n]*)", body, re.M)) and [x.strip() for x in m.group(1).split("|")][-1]
        f = lambda s: num(s) if s and re.fullmatch(r"-?[\d.]+", s) else None
        out[L] = {"доля": num(g(r"доля в портфеле ([\d.]+)%")), "p12": num(g(r"за 12 мес\. ([+-][\d.]+)%")),
                  "p3": num(g(r"за 3 мес\. ([+-][\d.]+)%")), "yld": num(g(r"доходность за 12 мес\. ([\d.]+)%")),
                  "pe": f(last("P/E")), "fcf": f(last("FCF ")), "np": f(last("Чистая прибыль"))}
    return out


def composite(fe):
    """Составное правило из самокритики №6 (пороги заданы до проверки на 2025 г.); ничья — по росту за 12 мес."""
    def pts(x):
        return ((x["pe"] is not None and x["pe"] > 8) or (x["np"] is not None and x["np"] < 0)) + \
               (x["yld"] is None or x["yld"] < 3) + (x["p12"] is not None and x["p12"] > 30) + \
               (x["fcf"] is not None and x["fcf"] < 0) + (x["np"] is not None and x["np"] < 0)
    return max(fe, key=lambda L: (pts(fe[L]), fe[L]["p12"] if fe[L]["p12"] is not None else -1e9))



def main():
    M = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    SETS = [("", f"sellrank_answers_{M}.jsonl"), ("_2025h2", f"sellrank_answers_2025h2_{M}.jsonl")]
    rows = []
    for tag, fn in SETS:
        cases = json.loads((ROOT / f"sellrank_cases{tag}.json").read_text())
        for a in map(json.loads, open(ROOT / fn, encoding="utf-8")):
            c = cases[a["i"]]
            if a["ответ"] is None:
                continue
            r = {b["буква"]: b["доходность_3м"] for b in c["бумаги"]}
            mean = np.mean(list(r.values()))
            ex = lambda L: (mean - r[L]) * 100
            fe = feats(c["текст"])
            pm = next(b for b in c["бумаги"] if b["продана"])
            r12 = max(c["бумаги"], key=lambda b: -1e9 if b["рост_12м"] is None else b["рост_12м"])["буква"]
            rows.append({"m": c["d1"][:7], "uk": uk.get(c["фонд"], "?"), "вид": c["вид"], "доля": fe[pm["буква"]]["доля"] or 0,
                         "model": a["ответ"], "pm": pm["буква"], "rule12": r12, "comp": composite(fe),
                         "e_model": ex(a["ответ"]), "e_pm": ex(pm["буква"]), "e_rule12": ex(r12), "e_comp": ex(composite(fe)),
                         "руб": vol.get((c["фонд"], c["d1"], pm["secid"]), 0),
                         # рубли той же продажи, перенаправленные на бумагу модели: выигрыш = руб × (r_проданной − r_модели)
                         "gain": (r[pm["буква"]] - r[a["ответ"]])})
    print(f"модель: {M}; случаев: {len(rows)}; месяцев: {len({r['m'] for r in rows})}; УК: {len({r['uk'] for r in rows})}")
    rng = np.random.default_rng(5)


    def boot(rs, key, by="m", n=4000):
        """Среднее и 95% интервал; ресэмплинг кластеров (месяцев или УК×месяц), а не отдельных случаев."""
        g = defaultdict(list)
        for r in rs:
            g[r[by] if by != "ukm" else (r["uk"], r["m"])].append(key(r))
        cl = list(g.values())
        bs = []
        for _ in range(n):
            pick = [cl[i] for i in rng.integers(0, len(cl), len(cl))]
            bs.append(np.mean([x for c in pick for x in c]))
        v = [key(r) for r in rs]
        return np.mean(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


    fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}; {t[2]:+.2f}]"
    res = {}
    print("\n1. Модель минус управляющий, п.п. за 3 мес.")
    d = lambda r: r["e_model"] - r["e_pm"]
    for by, name in (("m", "кластеры — месяцы"), ("ukm", "кластеры — УК×месяц"), ("uk", "кластеры — УК (10, грубо)")):
        res[f"разница_{by}"] = boot(rows, d, by)
        print(f"   {name:<28} {fmt(res[f'разница_{by}'])}")
    print("   модель против среднего из 4:", fmt(boot(rows, lambda r: r["e_model"])), "; управляющий:", fmt(boot(rows, lambda r: r["e_pm"])))

    print("\n2. Без каждой УК (модель минус управляющий; кластеры — месяцы)")
    res["без_УК"] = {}
    for u in sorted({r["uk"] for r in rows}):
        sub = [r for r in rows if r["uk"] != u]
        res["без_УК"][u] = boot(sub, d, n=1500)
        print(f"   без {u:<16} (случаев {len(rows) - len(sub):>3}) {fmt(res['без_УК'][u])}")
    own = defaultdict(list)
    for r in rows:
        own[r["uk"]].append(d(r))
    print("   по УК (случаев, среднее):", ", ".join(f"{u} {len(v)}:{np.mean(v):+.1f}" for u, v in sorted(own.items())))

    print("\n3. Что добавляет модель сверх правил")
    for rule, name in (("rule12", "«продать самое выросшее за 12 мес.»"), ("comp", "составное правило (5 признаков)")):
        agree = [r for r in rows if r["model"] == r[rule]]
        dis = [r for r in rows if r["model"] != r[rule]]
        res[f"сверх_{rule}"] = {"совпадение": len(agree) / len(rows), "правило": boot(rows, lambda r: r["e_" + rule]),
                                "расходятся_модель_минус_правило": boot(dis, lambda r: r["e_model"] - r["e_" + rule])}
        print(f"   {name}: правило в целом {fmt(res[f'сверх_{rule}']['правило'])}; совпадает с моделью в {len(agree) / len(rows):.0%}")
        print(f"      где расходятся ({len(dis)}): модель минус правило {fmt(res[f'сверх_{rule}']['расходятся_модель_минус_правило'])}")
        for h in ("2025", "2026"):
            s = [r for r in rows if r["m"].startswith(h)]
            print(f"      {h}: модель {np.mean([r['e_model'] for r in s]):+.2f}, правило {np.mean([r['e_' + rule] for r in s]):+.2f}")

    print("\n4. «Чистые» продажи (модель минус управляющий)")
    for name, f in (("сокращения, доля ≤ 7%", lambda r: r["вид"] == "сокращение" and r["доля"] <= 7),
                    ("все, кроме доли > 7%", lambda r: r["доля"] <= 7), ("выходы", lambda r: r["вид"] == "выход")):
        s = [r for r in rows if f(r)]
        res["чистые_" + name] = boot(s, d)
        print(f"   {name:<24} случаев {len(s):>3}: {fmt(res['чистые_' + name])}")

    print("\n5. Экономика на уровне портфеля")
    w = [r for r in rows if r["руб"] > 0]
    amt = np.array([r["руб"] for r in w])
    g = np.array([r["gain"] for r in w]) * 100
    print(f"   выигрыш на рубль продажи: взвешенно по рублям {np.average(g, weights=amt):+.2f} п.п. за 3 мес.; простое среднее {g.mean():+.2f}")
    # доля выборочных продаж в портфеле акций в месяц: по всем активным фондам
    pos = defaultdict(float)
    for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")):
        pos[(p["фонд"], p["дата"])] += float(p["value"] or 0)
    ev = [e for e in csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8")) if not any(k in e["фонд"] for k in INDEX + UNSURE + MECH)]
    nh, ns = defaultdict(int), defaultdict(int)
    for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")):
        nh[(p["фонд"], p["дата"])] += 1
    for e in ev:
        if e["сторона"] == "продажа":
            ns[(e["фонд"], e["d1"])] += 1
    sold, nav = defaultdict(float), {}
    for e in ev:
        k = (e["фонд"], e["d1"])
        nav[k] = pos.get((e["фонд"], e["d0"]), 0)
        if e["сторона"] == "продажа" and ns[k] <= 0.5 * nh.get((e["фонд"], e["d0"]), 1) and e["d1"] >= "2025-07":
            sold[k] += float(e["объём_руб"] or 0)
    turn = [sold[k] / nav[k] for k in nav if nav[k] > 0 and k[1] >= "2025-07" and k[1] < "2026-07"]
    t_med, t_mean = np.median(turn), np.mean(turn)
    per = np.average(g, weights=amt)
    res["экономика"] = {"п.п._на_рубль_3м": per, "оборот_выборочных_продаж_мес_медиана": t_med, "среднее": t_mean,
                        "годовых_медиана": per * t_med * 12, "годовых_среднее": per * t_mean * 12}
    print(f"   выборочные продажи в месяц, доля СЧА акций: медиана {t_med:.1%}, среднее {t_mean:.1%} (фонд-месяцы 07.2025–06.2026)")
    print(f"   → при полном следовании справке: {per * t_med * 12:+.2f}% годовых (медианный фонд), {per * t_mean * 12:+.2f}% (средний)")
    print("     (каждый месяц: выигрыш на рубль × доля проданного; окна по 3 мес. перекрываются, поэтому ×12)")
    (ROOT / f"sellrank_check_{M}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
