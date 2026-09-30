"""Агент «Зеркало рынка» (план проверки записан до запуска, 29.09.2026): каждый квартал модель видит только дату, ключевую ставку и
продажи рынка с известным исходом за последние 12 месяцев, предлагает версию правила; программа оставляет версию, лучшую
на последних известных продажах, или выключает подсказки. Исходы — очищенные доходности (мерка Х), отчётность *_fix.
python3 agent_rate.py run | score"""
import contextlib
import io
import json
import random
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import agent_replay as A
import rulegen as R
from robust import ROOT

OUT = ROOT / "агент_ставка"
# действующая ключевая ставка с даты (день после решения); 2022–2023 — история Банка России, дальше — key_rate_changes.txt
RATE = [("2022-07-26", 8.0), ("2022-09-20", 7.5), ("2023-07-24", 8.5), ("2023-08-15", 12.0), ("2023-09-18", 13.0),
        ("2023-10-30", 15.0), ("2023-12-18", 16.0)]
for line in (ROOT.parents[0] / "key_rate_changes.txt").read_text().splitlines():
    if m := re.fullmatch(r"(\d\d)\.(\d\d)\.(\d{4}) ([\d.]+)", line.strip()):
        RATE.append((f"{m[3]}-{m[2]}-{m[1]}", float(m[4])))
RATE.sort()


def rate(month):
    """Ставка на первое число месяца «ГГГГ-ММ»."""
    return [v for d, v in RATE if d <= month + "-01"][-1]


def load_all():
    sys.argv = ["adj_walk.py"]
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(str(Path(__file__).with_name("adj_walk.py")), run_name="x")
    return g["load"](tuple(t + "_fix" for t in ("_r1", "_r2", "_r3", "_2025h2", "")), True)


def prompt(sample, q):
    lines = []
    for k, x in enumerate(sample, 1):
        lines.append(f"Случай {k} (решение {x['m']}). Худшей за 3 мес. оказалась бумага {x['worst']}.")
        for L in "АБВГ":
            f = x["fe"][L]
            lines.append(f"  {L}: " + "; ".join(f"{n}={'н/д' if f[n] is None else round(f[n], 2)}" for n in R.FEAT))
    feats_desc = "\n".join(f"- {k}: {v}" for k, v in R.FEAT.items())
    return f"""Ты — количественный аналитик управляющей компании (российский рынок акций). Сегодня {q}-01. Ключевая ставка
Банка России сегодня {rate(q)}%, полгода назад была {rate(A.shift(q, -6))}%.
Ниже {len(sample)} реальных случаев продаж фондов за последний год: в портфеле 4 бумаги (обезличены), известны признаки
на дату решения и какая из 4 бумаг показала ХУДШУЮ доходность за следующие 3 месяца по сравнению с похожими бумагами
(того же размера и той же доходности за прошлый год) — её и надо было продать.
Признаки:
{feats_desc}

Найди закономерности, которые подходят для нынешних условий (учти уровень ставки и её направление), и предложи
ПРОЗРАЧНОЕ правило для выбора, какую бумагу продавать: не больше 6 условий, каждое — признак, знак сравнения (">" или
"<"), порог и баллы (от 1 до 3). Продаётся бумага с наибольшей суммой баллов (при отсутствии данных условие не
срабатывает). Правило должно быть экономически осмысленным и понятным риск-менеджеру; избегай подгонки под отдельные
случаи.
Сначала коротко (3–6 предложений) объясни логику, затем выдай правило последним блоком строго в формате JSON:
[{{"признак": "...", "знак": ">", "порог": 0, "баллы": 1}}, ...]

Данные:
{chr(10).join(lines)}"""


def derive(pool, seed, q):
    from llm_read import ask
    rules, texts = [], []   # 40 примеров: при 60 запрос ≈ 29 тыс. токенов и ≈ 14 мин (исправление до результатов)
    for j in range(3):
        rng = random.Random(seed * 10 + j)
        out = ask("qwen/qwen3.8-27b", prompt(rng.sample(pool, min(40, len(pool))), q) + " /no_think", effort="none",
                  provider="local") or ""
        texts.append(out)
        m = re.findall(r"\[\s*\{.*?\}\s*\]", out, re.S)
        try:
            rule = [c for c in json.loads(m[-1]) if c.get("признак") in R.FEAT and c.get("знак") in (">", "<")]
            if rule:
                rules.append(rule)
        except Exception:  # noqa: BLE001
            pass
    return rules, texts


def run():
    OUT.mkdir(exist_ok=True)
    ALL = load_all()
    active = [r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())]
    for k, q in enumerate(A.QS):
        f = OUT / f"квартал_{q}.json"
        if f.exists():
            active = json.loads(f.read_text())["действует"]
            continue
        pool = [x for x in ALL if A.shift(q, -15) <= x["m"] <= A.shift(q, -4)]   # исход за месяц m известен в конце m+3
        val = [x for x in ALL if A.shift(q, -6) <= x["m"] <= A.shift(q, -4)]
        new, texts = derive(pool, k, q)
        g_new = A.gain(val, A.picker(new)) if new else float("nan")
        g_old = A.gain(val, A.picker(active))
        choice, why = (new, "заменено новым") if new and g_new > g_old else (active, "оставлено действующее")
        best = max(v for v in (g_new, g_old) if v == v)
        on = best >= 0
        f.write_text(json.dumps({"квартал": q, "ставка": rate(q), "ставка_полгода_назад": rate(A.shift(q, -6)),
                                 "примеров_в_пуле": len(pool), "проверочных": len(val), "новое": new,
                                 "проверка_новое": g_new, "проверка_действующее": g_old, "решение": why,
                                 "сигнал_включён": on, "действует": choice, "ответы_модели": texts},
                                ensure_ascii=False, indent=1))
        active = choice
        print(q, f"ставка {rate(q)}", why, "новое", round(g_new, 2), "старое", round(g_old, 2), "вкл" if on else "ВЫКЛ", flush=True)


def score():
    ALL = load_all()
    prior = A.picker([r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())])
    rows = []
    for q in A.QS:
        d = json.loads((OUT / f"квартал_{q}.json").read_text())
        pa = A.picker(d["действует"])
        for x in [x for x in ALL if q <= x["m"] <= A.shift(q, 2)]:
            ag = pa(x) if d["сигнал_включён"] else x["pm"]
            rows.append({"m": x["m"], "q": q, "агент": (x["r"][x["pm"]] - x["r"][ag]) * 100,
                         "неизменное": (x["r"][x["pm"]] - x["r"][prior(x)]) * 100})
    rng = np.random.default_rng(4)

    def boot(key, H=3):
        by = defaultdict(list)
        for r in rows:
            by[r["m"]].append(key(r))
        ms = sorted(by)
        est = [np.mean([v for st in rng.integers(0, len(ms), -(-len(ms) // H)) for k in range(H) for v in by[ms[(st + k) % len(ms)]]])
               for _ in range(2000)]
        return [round(float(np.mean([key(r) for r in rows])), 2), round(float(np.percentile(est, 2.5)), 2),
                round(float(np.percentile(est, 97.5)), 2)]
    res = {"продаж": len(rows), "агент против управляющего": boot(lambda r: r["агент"]),
           "неизменное против управляющего": boot(lambda r: r["неизменное"]),
           "агент минус неизменное": boot(lambda r: r["агент"] - r["неизменное"]),
           "по кварталам": {q: [round(float(np.mean([r["агент"] for r in rows if r["q"] == q])), 2),
                                round(float(np.mean([r["неизменное"] for r in rows if r["q"] == q])), 2)] for q in A.QS}}
    res["смен_правила"] = sum(json.loads((OUT / f"квартал_{q}.json").read_text())["решение"] == "заменено новым" for q in A.QS)
    res["выключений"] = sum(not json.loads((OUT / f"квартал_{q}.json").read_text())["сигнал_включён"] for q in A.QS)
    (OUT / "итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # Рисунок: по кварталам — неизменное правило и агент
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
    qv = res["по кварталам"]
    x = np.arange(len(A.QS))
    fig, ax = plt.subplots(figsize=(7.0, 3.5), facecolor="white")
    ax.bar(x - 0.2, [qv[q][1] for q in A.QS], 0.38, color="#b7b5ad", label=f"неизменное правило, в среднем {res['неизменное против управляющего'][0]:+.1f}".replace(".", ","))
    ax.bar(x + 0.2, [qv[q][0] for q in A.QS], 0.38, color="#2a78d6", label=f"агент, пересобирающий правило, в среднем {res['агент против управляющего'][0]:+.1f}".replace(".", ","))
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(x, [f"{roman[q[5:]]}\n{q[:4]}" for q in A.QS], fontsize=9, color="#52514e")
    ax.tick_params(axis="both", length=0, colors="#52514e")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "−")))
    ax.grid(axis="y", color="#eceae4", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel("выигрыш от продажи по подсказке\nвместо выбора управляющего, п.п.", color="#52514e")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, f"Квартал продажи; результат за 3 месяца за вычетом роста похожих бумаг; {res['продаж']} продаж 07.2023–06.2026. "
             "Агент в начале квартала знал только дату, ставку и уже известные исходы.", fontsize=7.8, color="#52514e", wrap=True)
    fig.subplots_adjust(bottom=0.24, top=0.97, left=0.13, right=0.99)
    fig.savefig(ROOT.parents[1] / "рисунки" / "р8_4_агент.png", dpi=220, facecolor="white")


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else score()
