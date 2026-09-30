"""Устойчивая справка к продаже — метод из проекта по отчётам Сбера (устойчивость оценок языковой модели):
каждый случай спрашивается K раз с перестановкой порядка четырёх бумаг (буквы А–Г назначаются заново), температура 0.
Итог — самый частый выбор; случай «хрупкий», если выборы расходятся (модель не уверена — решение человеку).
python3 permute.py run <модель> <openrouter|local> [K] [N] [effort]  -> данные/составы_фондов/perm_<метка>.jsonl
python3 permute.py score <файл.jsonl>
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from robust import ROOT

LET = "АБВГ"
CT = os.environ.get("PERM_TAG", "")          # период случаев: "" — 01–06.2026, "_2025h2" — 07–12.2025 и т. д.


def split(text):
    head, rest = text.split("\n\nБумага А:", 1)
    blocks = re.split(r"\n\n(?=Бумага [БВГ]:)", "Бумага А:" + rest)
    return head, [re.sub(r"^Бумага [АБВГ]:", "", b) for b in blocks]


BRIEF = os.environ.get("PERM_BRIEF")          # краткий режим для локальной модели: ≤ 5 предложений, без рассуждений (≈ 50 с)


def permuted(text, perm):
    head, blocks = split(text)
    if BRIEF:
        head = head.replace("Рассуди по существу (оценка, динамика прибыли, долг, дивиденды, перегретость цены), затем ответь последней строкой",
                            "Рассуди кратко (не больше 5 предложений: оценка, динамика прибыли, долг, дивиденды, перегретость цены), затем ответь последней строкой")
    return head + "\n\n" + "\n\n".join(f"Бумага {LET[j]}:{blocks[perm[j]]}" for j in range(4)) + (" /no_think" if BRIEF else "")


def run(model, provider, k, n, effort, tag):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from llm_read import ask
    cases = json.loads((ROOT / f"sellrank_cases{CT}.json").read_text())[:n]
    out = ROOT / f"perm_{tag}{'_brief' if BRIEF else ''}{CT}.jsonl"
    done = {(a["i"], a["k"]) for a in map(json.loads, open(out, encoding="utf-8"))} if out.exists() else set()
    lock = Lock()

    def one(job):
        i, kk = job
        perm = list(range(4))
        if kk:
            random.Random(1000 * i + kk).shuffle(perm)            # прогон 0 — исходный порядок
        try:
            ans = ask(model, permuted(cases[i]["текст"], perm), effort=effort, provider=provider)
        except Exception as e:  # noqa: BLE001
            print("ошибка", i, kk, str(e)[:60], flush=True)
            return
        m = re.findall(r"ПРОДАТЬ:\s*\**\s*([АБВГ])", ans or "")
        orig = LET[perm[LET.index(m[-1])]] if m else None
        with lock, open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"i": i, "k": kk, "ответ": orig}, ensure_ascii=False) + "\n")

    jobs = [(i, kk) for i in range(len(cases)) for kk in range(k) if (i, kk) not in done]
    with ThreadPoolExecutor(1 if provider == "local" else 16) as ex:
        list(ex.map(one, jobs))


def score(fn):
    cases = json.loads((ROOT / f"sellrank_cases{CT}.json").read_text())
    by = defaultdict(list)
    for a in map(json.loads, open(ROOT / fn, encoding="utf-8")):
        if a["ответ"]:
            by[a["i"]].append((a["k"], a["ответ"]))
    rng = np.random.default_rng(5)

    def ex(c, L):
        r = {b["буква"]: b["доходность_3м"] for b in c["бумаги"]}
        return (np.mean(list(r.values())) - r[L]) * 100

    def ci(v):
        v = np.array(v)
        b = [rng.choice(v, len(v)).mean() for _ in range(3000)]
        return f"{v.mean():+.2f} [{np.percentile(b, 2.5):+.2f}; {np.percentile(b, 97.5):+.2f}] n={len(v)}"
    full = {i: v for i, v in by.items() if len(v) >= 5}
    rows = []
    for i, v in full.items():
        c = cases[i]
        cnt = Counter(a for _, a in v)
        top, f = cnt.most_common(1)[0]
        pm = next(b["буква"] for b in c["бумаги"] if b["продана"])
        first = dict(v).get(0, v[0][1])
        rows.append({"share": f / len(v), "maj": ex(c, top), "single": ex(c, first), "pm": ex(c, pm),
                     "avg": np.mean([ex(c, a) for _, a in v]), "maj_vs_pm": ex(c, top) - ex(c, pm)})
    print(f"случаев с ≥5 прогонами: {len(rows)}")
    print(f"все 5 прогонов одинаковы: {np.mean([r['share'] == 1 for r in rows]):.0%}; ≥4 из 5: {np.mean([r['share'] >= .8 for r in rows]):.0%}")
    print("один прогон (исходный порядок):", ci([r["single"] for r in rows]))
    print("в среднем по одиночным прогонам:", ci([r["avg"] for r in rows]))
    print("самый частый ответ (голосование):", ci([r["maj"] for r in rows]))
    print("управляющий:", ci([r["pm"] for r in rows]))
    for name, f in (("устойчивые (5 из 5)", lambda r: r["share"] == 1), ("почти (4 из 5)", lambda r: .8 <= r["share"] < 1),
                    ("хрупкие (≤3 из 5)", lambda r: r["share"] < .8)):
        s = [r for r in rows if f(r)]
        if s:
            print(f"  {name}: голосование {ci([r['maj'] for r in s])}; против управляющего {ci([r['maj_vs_pm'] for r in s])}")


def deep(*specs, rules_fn="правила_wf_A.json"):
    """Что даёт колебание сверх правила. Бутстрэп по месяцам (как у главных цифр), а не по случаям.
    Разница везде: доходность проданной управляющим минус доходность выбора (п.п. за 3 мес.; > 0 — выбор лучше)."""
    import rulegen as R
    rules = [r["правило"] for r in json.loads((ROOT / rules_fn).read_text())]
    pts = lambda f: sum(c["баллы"] for rule in rules for c in rule if f.get(c["признак"]) is not None and
                        ((c["знак"] == ">" and f[c["признак"]] > c["порог"]) or (c["знак"] == "<" and f[c["признак"]] < c["порог"])))
    rows = []
    for spec in specs:  # «файл» или «файл@метка_периода» (несколько — вместе)
        fn, _, ct = spec.partition("@")
        rows += _deep_rows(fn, ct or CT, pts, R)
    _deep_report(" + ".join(specs), rows)


def levels(*specs, rules_fn="правила_wf_A.json"):
    """Уровни: чем заменить выбор управляющего. Разница — проданная управляющим минус выбранная (п.п. за 3 мес.),
    усреднение по ВСЕМ продажам (где уровень молчит — 0: остаётся выбор управляющего); бутстрэп по месяцам."""
    import rulegen as R
    rules = [r["правило"] for r in json.loads((ROOT / rules_fn).read_text())]
    pts = lambda f: sum(c["баллы"] for rule in rules for c in rule if f.get(c["признак"]) is not None and
                        ((c["знак"] == ">" and f[c["признак"]] > c["порог"]) or (c["знак"] == "<" and f[c["признак"]] < c["порог"])))
    rows = [x for spec in specs for x in _deep_rows(spec.partition("@")[0], spec.partition("@")[2] or CT, pts, R)]
    rng = np.random.default_rng(3)
    dv = lambda x, L: x["r"][x["pm"]] - x["r"][L]
    lv = [("0. случайная бумага из четырёх", lambda x: x["r"][x["pm"]] - np.mean(list(x["r"].values())), lambda x: True),
          ("1. правило по числам, всегда", lambda x: dv(x, x["rule"]), lambda x: True),
          ("2. только ИИ, всегда", lambda x: dv(x, x["top"]), lambda x: True),
          ("2б. только ИИ, когда он устойчив", lambda x: dv(x, x["top"]), lambda x: x["stable"]),
          ("3. правило + ИИ: советуем, когда согласны", lambda x: dv(x, x["rule"]), lambda x: x["top"] == x["rule"]),
          ("4. то же + ИИ устойчив", lambda x: dv(x, x["rule"]), lambda x: x["top"] == x["rule"] and x["stable"])]
    ms = sorted({x["m"] for x in rows})
    print(f"{' + '.join(specs)}: продаж {len(rows)}, месяцев {len(ms)}. Управляющий = 0.")
    print(f"{'уровень':<44}{'доля продаж с советом':>22}{'выигрыш на совет':>28}{'на все продажи':>28}")
    for name, f, on in lv:
        g = {m: [(f(x), on(x)) for x in rows if x["m"] == m] for m in ms}
        per_all = lambda idx: np.mean([v if o else 0.0 for j in idx for v, o in g[ms[j]]])
        per_on = lambda idx: np.mean([v for j in idx for v, o in g[ms[j]] if o])
        bs = [(per_all(i), per_on(i)) for i in (rng.integers(0, len(ms), len(ms)) for _ in range(2000))]
        fa, fo = np.array([b[0] for b in bs]), np.array([b[1] for b in bs])
        full = range(len(ms))
        cov = np.mean([on(x) for x in rows])
        print(f"{name:<44}{cov:>21.0%} {per_on(full):>+10.2f} [{np.nanpercentile(fo, 2.5):+.2f}; {np.nanpercentile(fo, 97.5):+.2f}]"
              f"{per_all(full):>+13.2f} [{np.percentile(fa, 2.5):+.2f}; {np.percentile(fa, 97.5):+.2f}]")


def _deep_rows(fn, ct, pts, R):
    cases = json.loads((ROOT / f"sellrank_cases{ct}.json").read_text())
    by = defaultdict(list)
    for a in map(json.loads, open(ROOT / fn, encoding="utf-8")):
        if a["ответ"]:
            by[a["i"]].append(a["ответ"])
    rows = []
    for i, v in by.items():
        if len(v) < 5:
            continue
        c = cases[i]
        r = {b["буква"]: b["доходность_3м"] * 100 for b in c["бумаги"]}
        fe = R.feats2(c["текст"])
        key = lambda L: (pts(fe[L]), fe[L]["p12"] if fe[L]["p12"] is not None else -1e9)
        order = sorted(fe, key=key, reverse=True)
        top, f = Counter(v).most_common(1)[0]
        pm = next(b["буква"] for b in c["бумаги"] if b["продана"])
        rows.append({"m": c["d1"][:7], "stable": f == len(v), "share": f / len(v), "top": top, "rule": order[0], "pm": pm,
                     "margin": pts(fe[order[0]]) - pts(fe[order[1]]), "r": r})
    return rows


def _deep_report(fn, rows):
    rng = np.random.default_rng(7)

    def ci(sel, pick):
        s = [x for x in rows if sel(x)]
        g = defaultdict(list)
        for x in s:
            g[x["m"]].append(x["r"][x["pm"]] - x["r"][pick(x)])
        ms = list(g)
        if len(s) < 5:
            return f"n={len(s)}"
        bs = [np.mean([d for j in rng.integers(0, len(ms), len(ms)) for d in g[ms[j]]]) for _ in range(3000)]
        return f"{np.mean([d for d in sum(g.values(), [])]):+.2f} [{np.percentile(bs, 2.5):+.2f}; {np.percentile(bs, 97.5):+.2f}] n={len(s)}"

    def diff(sel_a, sel_b, pick):  # разница двух групп, бутстрэп по месяцам
        g = defaultdict(lambda: ([], []))
        for x in rows:
            d = x["r"][x["pm"]] - x["r"][pick(x)]
            if sel_a(x):
                g[x["m"]][0].append(d)
            elif sel_b(x):
                g[x["m"]][1].append(d)
        ms = list(g)
        est = lambda idx: np.mean([d for j in idx for d in g[ms[j]][0]]) - np.mean([d for j in idx for d in g[ms[j]][1]])
        bs = [est(rng.integers(0, len(ms), len(ms))) for _ in range(3000)]
        return f"{est(range(len(ms))):+.2f} [{np.nanpercentile(bs, 2.5):+.2f}; {np.nanpercentile(bs, 97.5):+.2f}]"

    st, fr = (lambda x: x["stable"]), (lambda x: not x["stable"])
    model, rule = (lambda x: x["top"]), (lambda x: x["rule"])
    agree = lambda x: x["top"] == x["rule"]
    print(f"{fn}: случаев {len(rows)}, месяцев {len({x['m'] for x in rows})}, устойчивых 5/5 {np.mean([x['stable'] for x in rows]):.0%}")
    print("1) выбор МОДЕЛИ против управляющего")
    print("   устойчивые", ci(st, model), "| хрупкие", ci(fr, model), "| разница", diff(st, fr, model))
    print("2) выбор ПРАВИЛА против управляющего — подсказывает ли колебание модели, когда верить правилу?")
    print("   все", ci(lambda x: True, rule), "| модель устойчива", ci(st, rule), "| хрупка", ci(fr, rule), "| разница", diff(st, fr, rule))
    print("3) модель и правило согласны / нет")
    print(f"   согласны {np.mean([agree(x) for x in rows]):.0%}: правило", ci(agree, rule), "| не согласны: правило",
          ci(lambda x: not agree(x), rule), "модель", ci(lambda x: not agree(x), model))
    print("   согласны И модель устойчива: правило", ci(lambda x: agree(x) and x["stable"], rule),
          "| остальные", ci(lambda x: not (agree(x) and x["stable"]), rule))
    flag = lambda x: x["stable"] and x["top"] != x["pm"]
    both = lambda x: agree(x) and x["stable"]
    print("   H2 разница (согласны и устойчива − остальные):", diff(both, lambda x: not both(x), rule),
          "| согласны, но модель хрупка: правило", ci(lambda x: agree(x) and not x["stable"], rule),
          "| внутри согласных устойчивая − хрупкая:", diff(both, lambda x: agree(x) and not x["stable"], rule))
    print("   H3 разница (флаг − без флага, выигрыш замены на выбор модели):", diff(flag, lambda x: not flag(x), model))
    print("4) не просто ли «очевидный случай»? Отрыв правила (баллы 1-го минус 2-го) у устойчивых и хрупких:",
          f"{np.mean([x['margin'] for x in rows if x['stable']]):.2f} / {np.mean([x['margin'] for x in rows if not x['stable']]):.2f}")
    hi = lambda x: x["margin"] > np.median([y["margin"] for y in rows])
    for nm, sel in (("отрыв большой", hi), ("отрыв малый", lambda x: not hi(x))):
        print(f"   {nm}: правило при устойчивой модели", ci(lambda x: sel(x) and x["stable"], rule),
              "| при хрупкой", ci(lambda x: sel(x) and not x["stable"], rule))
    print("5) «красный флаг» продаже: модель устойчиво называет ДРУГУЮ бумагу, чем продал управляющий")
    flag = lambda x: x["stable"] and x["top"] != x["pm"]
    print(f"   доля продаж с флагом {np.mean([flag(x) for x in rows]):.0%}: выигрыш замены", ci(flag, model),
          "| без флага", ci(lambda x: not flag(x), model))
    # второй метод для H1 — плацебо: метки «устойчив» перемешиваются внутри месяца; доля случайных разниц ≥ настоящей
    d = np.array([x["r"][x["pm"]] - x["r"][x["rule"]] for x in rows])
    s = np.array([x["stable"] for x in rows])
    ms = np.array([x["m"] for x in rows])
    gap = lambda lab: d[lab].mean() - d[~lab].mean()
    real, fake = gap(s), []
    for _ in range(5000):
        lab = s.copy()
        for m in set(ms):
            j = np.where(ms == m)[0]
            lab[j] = rng.permutation(lab[j])
        fake.append(gap(lab))
    print(f"6) плацебо для H1 (правило: устойчивая − хрупкая): разница {real:+.2f}, p = {np.mean(np.array(fake) >= real):.3f} (одностор.)")
    # контроль «очевидного случая»: та же разница внутри трёх групп по отрыву правила, среднее по группам
    mg = np.array([x["margin"] for x in rows])
    t = np.digitize(mg, np.percentile(mg, [33.3, 66.7]))
    per = [d[(t == k) & s].mean() - d[(t == k) & ~s].mean() for k in range(3)
           if ((t == k) & s).any() and ((t == k) & ~s).any()]
    print(f"   с поправкой на отрыв правила (среднее по третям): {np.mean(per):+.2f} (по третям: {', '.join(f'{p:+.2f}' for p in per)})")


if __name__ == "__main__":
    if sys.argv[1] == "levels":
        levels(*sys.argv[2:])
    elif sys.argv[1] == "deep":
        deep(*sys.argv[2:])
    elif sys.argv[1] == "run":
        a = sys.argv + [None] * 6
        model, prov = a[2], a[3]
        k, n, eff = int(a[4] or 5), int(a[5] or 300), a[6] or "none"
        run(model, prov, k, n, eff, f"{'local' if prov == 'local' else model.split('/')[-1]}_{eff}")
    else:
        score(sys.argv[2])
