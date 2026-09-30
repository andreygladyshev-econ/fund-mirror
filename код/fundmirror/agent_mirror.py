"""ИИ читает зеркало рынка (план проверки — в README). Каждый квартал модель получает дату, ставку и
сводку зеркала: какие признаки слабой бумаги помогали выбрать, что продать, за последний год и за год до него; пишет
правило на квартал. Вариант Б без модели: версии прогона agent_rate отбираются по году продаж.
python3 agent_mirror.py run | score"""
import json
import re
import sys

import numpy as np

import agent_rate as G
import rule as A
import rulegen as R
from robust import ROOT

OUT = ROOT / "агент_зеркало"
SIG = [("debt_ebitda", 1, "высокий Долг/EBITDA"), ("fcf_neg", 1, "отрицательный свободный денежный поток"),
       ("np_neg", 1, "убыток"), ("roe", -1, "низкий ROE"), ("np_g", -1, "падение чистой прибыли"),
       ("rev_g", -1, "падение выручки"), ("pe", 1, "высокий P/E"), ("pb", 1, "высокий P/B"),
       ("ev_ebitda", 1, "высокий EV/EBITDA"), ("p12", 1, "сильный рост цены за 12 мес."),
       ("p12", -1, "сильное падение цены за 12 мес."), ("p3", 1, "сильный рост цены за 3 мес."),
       ("yld", -1, "низкая дивидендная доходность")]


def lift(cases, f, s):
    """Насколько бумага с самым выраженным признаком отстала от случайной бумаги того же портфеля, п.п.; число случаев."""
    d = []
    for x in cases:
        v = {L: x["fe"][L][f] for L in "АБВГ" if x["fe"][L][f] is not None}
        if len(set(v.values())) > 1:
            L = max(v, key=lambda k: s * v[k])
            d.append((np.mean(list(x["r"].values())) - x["r"][L]) * 100)
    return (float(np.mean(d)), len(d)) if d else (None, 0)


def summary(ALL, q):
    last = [x for x in ALL if A.shift(q, -15) <= x["m"] <= A.shift(q, -4)]
    prev = [x for x in ALL if A.shift(q, -27) <= x["m"] <= A.shift(q, -16)]
    fmt = lambda t: "н/д" if t[0] is None or t[1] < 20 else f"{t[0]:+.1f} ({t[1]})"
    lines = []
    for f, s, name in SIG:
        vals = [x["fe"][L][f] for x in last for L in "АБВГ" if x["fe"][L][f] is not None]
        qs = "/".join(f"{v:.3g}" for v in np.percentile(vals, [25, 50, 75])) if vals else "н/д"
        lines.append(f"- {name} [{f}{'>' if s > 0 else '<'}]: последний год {fmt(lift(last, f, s))}; год до него "
                     f"{fmt(lift(prev, f, s))}; квартили {f} за год {qs}")
    return "\n".join(lines), len(last)


def prompt(q, table, n):
    feats = "\n".join(f"- {k}: {v}" for k, v in R.FEAT.items())
    return f"""Ты — количественный аналитик управляющей компании (российский рынок акций). Сегодня {q}-01. Ключевая ставка
Банка России сегодня {G.rate(q)}%, полгода назад была {G.rate(A.shift(q, -6))}%.
Ниже сводка «зеркала рынка» — продаж паевых фондов с уже известным результатом ({n} продаж за последний год). Для
каждого признака показано, насколько бумага портфеля, у которой признак выражен сильнее всего, за следующие 3 месяца
отставала от случайной бумаги того же портфеля (с поправкой на похожие бумаги), в процентных пунктах; плюс — признак
помогал выбрать, что продать, минус — вредил. В скобках — число случаев.
{table}

Составь ПРОЗРАЧНОЕ правило выбора бумаги для продажи на следующий квартал: не больше 6 условий, каждое — признак,
знак сравнения (">" или "<"), порог и баллы (от 1 до 3). Продаётся бумага с наибольшей суммой баллов (при отсутствии
данных условие не срабатывает). Опирайся на сводку и на экономическую логику при нынешней ставке; не доверяй признаку
только потому, что он сработал на малом числе случаев или в одном году. Признаки:
{feats}
Сначала коротко (3–6 предложений) объясни логику, затем выдай правило последним блоком строго в формате JSON:
[{{"признак": "...", "знак": ">", "порог": 0, "баллы": 1}}, ...]"""


def run():
    from llm_read import ask
    OUT.mkdir(exist_ok=True)
    ALL = G.load_all()
    for k, q in enumerate(A.QS):
        f = OUT / f"квартал_{q}.json"
        if f.exists():
            continue
        table, n = summary(ALL, q)
        rules, texts = [], []
        for j in range(3):
            out = ask("qwen/qwen3.8-27b", prompt(q, table, n) + " /no_think", effort="none", provider="local") or ""
            texts.append(out)
            m = re.findall(r"\[\s*\{.*?\}\s*\]", out, re.S)
            try:
                rule = [c for c in json.loads(m[-1]) if c.get("признак") in R.FEAT and c.get("знак") in (">", "<")]
                if rule:
                    rules.append(rule)
            except Exception:  # noqa: BLE001
                pass
        f.write_text(json.dumps({"квартал": q, "ставка": G.rate(q), "сводка": table, "продаж_в_сводке": n,
                                 "правило": rules, "ответы_модели": texts}, ensure_ascii=False, indent=1))
        print(q, f"ставка {G.rate(q)}", "версий", len(rules), flush=True)


def score():
    ALL = G.load_all()
    prior_rules = [r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())]
    prior = A.picker(prior_rules)
    # вариант Б: версии прогона agent_rate, отбор по году продаж
    active, chosen_b = prior_rules, {}
    for q in A.QS:
        d = json.loads((G.OUT / f"квартал_{q}.json").read_text())
        year = [x for x in ALL if A.shift(q, -15) <= x["m"] <= A.shift(q, -4)]
        g_new = A.gain(year, A.picker(d["новое"])) if d["новое"] else float("nan")
        g_old = A.gain(year, A.picker(active))
        if d["новое"] and g_new > g_old:
            active = d["новое"]
        chosen_b[q] = (A.picker(active), max(v for v in (g_new, g_old) if v == v) >= 0)
    # конфигурация месячного цикла (mirror_cycle.py): версии М отбираются по году продаж (проверка после основных);
    # «цикл» — замена при любом превосходстве, «цикл_порог» — только при превосходстве на A.MARGIN п.п. (введено позже)
    chosen = {}
    for key, margin in (("цикл", 0.0), ("цикл_порог", A.MARGIN)):
        active, chosen[key] = prior_rules, {}
        for q in A.QS:
            cand = json.loads((OUT / f"квартал_{q}.json").read_text())["правило"]
            year = [x for x in ALL if A.shift(q, -15) <= x["m"] <= A.shift(q, -4)]
            g_cur, g_new = A.gain(year, A.picker(active)), (A.gain(year, A.picker(cand)) if cand else float("nan"))
            if cand and g_new > g_cur + margin:
                active = cand
                g_cur = g_new
            chosen[key][q] = (A.picker(active), g_cur >= 0)
    rows = []
    for q in A.QS:
        dm = json.loads((OUT / f"квартал_{q}.json").read_text())
        pm_ = A.picker(dm["правило"]) if dm["правило"] else prior
        pb, on_b = chosen_b[q]
        for x in [x for x in ALL if q <= x["m"] <= A.shift(q, 2)]:
            g = lambda L: (x["r"][x["pm"]] - x["r"][L]) * 100
            row = {"m": x["m"], "q": q, "М": g(pm_(x)), "Б": g(pb(x)) if on_b else 0.0, "неизменное": g(prior(x))}
            for key in chosen:
                pc, on_c = chosen[key][q]
                row[key] = g(pc(x)) if on_c else 0.0
            rows.append(row)
    def boot(key):
        return list(A.block_boot([(r["m"], key(r)) for r in rows])[:3])
    res = {"продаж": len(rows), "М против управляющего": boot(lambda r: r["М"]),
           "М минус неизменное": boot(lambda r: r["М"] - r["неизменное"]),
           "Б против управляющего": boot(lambda r: r["Б"]), "Б минус неизменное": boot(lambda r: r["Б"] - r["неизменное"]),
           "цикл (М + отбор по году) против управляющего": boot(lambda r: r["цикл"]),
           "цикл минус неизменное": boot(lambda r: r["цикл"] - r["неизменное"]),
           "цикл с порогом замены 1 п.п. против управляющего": boot(lambda r: r["цикл_порог"]),
           "цикл с порогом минус неизменное": boot(lambda r: r["цикл_порог"] - r["неизменное"]),
           "неизменное против управляющего": boot(lambda r: r["неизменное"]),
           "по кварталам (М, Б, неизменное)": {q: [round(float(np.mean([r[k] for r in rows if r["q"] == q])), 2)
                                                  for k in ("М", "Б", "неизменное")] for q in A.QS}}
    (OUT / "итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "PTSans" in f.replace(" ", "") or "PT_Sans" in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "PT Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False})
    roman = {"01": "I", "04": "II", "07": "III", "10": "IV"}
    qv = res["по кварталам (М, Б, неизменное)"]
    x = np.arange(len(A.QS))
    num = lambda v: f"{v:+.1f}".replace(".", ",").replace("-", "−")
    fig, ax = plt.subplots(figsize=(7.0, 3.5), facecolor="white")
    ax.bar(x - 0.2, [qv[q][2] for q in A.QS], 0.38, color="#b7b5ad",
           label=f"неизменное правило, в среднем {num(res['неизменное против управляющего'][0])}")
    ax.bar(x + 0.2, [qv[q][0] for q in A.QS], 0.38, color="#2a78d6",
           label=f"агент, читающий зеркало, в среднем {num(res['М против управляющего'][0])}")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(x, [f"{roman[q[5:]]}\n{q[:4]}" for q in A.QS], fontsize=9, color="#52514e")
    ax.tick_params(axis="both", length=0, colors="#52514e")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "−")))
    ax.grid(axis="y", color="#eceae4", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel("выигрыш от продажи по подсказке\nвместо выбора управляющего, п.п.", color="#52514e")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, f"Квартал продажи; результат за 3 месяца за вычетом роста похожих бумаг; {res['продаж']} продаж 07.2023–06.2026. "
             "Агент в начале квартала видел только дату, ставку и сводку зеркала за прошлые годы.", fontsize=7.8,
             color="#52514e", wrap=True)
    fig.subplots_adjust(bottom=0.24, top=0.97, left=0.13, right=0.99)
    fig.savefig(ROOT.parents[1] / "рисунки" / "агент_зеркало.png", dpi=220, facecolor="white")


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else score()
