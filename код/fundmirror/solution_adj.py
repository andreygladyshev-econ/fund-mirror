"""Числа решения для записки (30.09) на очищенных доходностях (мерка Х) и исправленной отчётности: проданная бумага
против трёх случайных бумаг того же портфеля; правило «без данных» (правила_prior.json: модель писала его только из
знаний, без данных и без указания периода) против выбора управляющего и против жребия, по периодам и кварталам;
экономика с пределом «заменить не больше позиции подсказанной бумаги». Модель не вызывается.
python3 solution_adj.py -> рисунки/р8_2_правило.png, рисунки/числа_р8.json"""
import contextlib
import csv
import io
import json
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.argv = ["adj_walk.py"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(Path(__file__).with_name("adj_walk.py")), run_name="x")
load = g["load"]
import agent_replay as A  # noqa: E402
import rulegen as R  # noqa: E402
from robust import ROOT  # noqa: E402

TEST = tuple(t + "_fix" for t in ("_r2", "_r3", "_2025h2", ""))


def boot(rows, val, H=3, seed=4):
    by = defaultdict(list)
    for x in rows:
        by[x["m"]].append(val(x))
    ms = sorted(by)
    rng = np.random.default_rng(seed)
    est = [np.mean([d for st in rng.integers(0, len(ms), -(-len(ms) // H)) for k in range(H) for d in by[ms[(st + k) % len(ms)]]])
           for _ in range(2000)]
    return {"D": round(float(np.mean([d for v in by.values() for d in v])), 2), "lo": round(float(np.percentile(est, 2.5)), 2),
            "hi": round(float(np.percentile(est, 97.5)), 2), "n": sum(map(len, by.values())),
            "мес_в_плюсе": f"{sum(np.mean(v) > 0 for v in by.values())}/{len(ms)}"}


prior = A.picker([r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())])
avg = lambda x: np.mean(list(x["r"].values()))
te, tr = load(TEST, True), load(("_r1_fix",), True)
ALL = tr + te
for x in ALL:
    x["prior"], x["q"] = prior(x), f"{x['m'][:5]}{(int(x['m'][5:]) - 1) // 3 * 3 + 1:02d}"
gap = lambda x: (x["r"][x["pm"]] - x["r"][x["prior"]]) * 100
nums = {"проданная против 3 других бумаг портфеля, все годы": boot(ALL, lambda x: (x["r"][x["pm"]] - np.mean([v for L, v in x["r"].items() if L != x["pm"]])) * 100),
        "управляющий против жребия из 4, все годы": boot(ALL, lambda x: (x["r"][x["pm"]] - avg(x)) * 100),
        "правило без данных, все годы": boot(ALL, gap),
        "правило без данных против жребия, все годы": boot(ALL, lambda x: (avg(x) - x["r"][x["prior"]]) * 100),
        "правило без данных, все годы без 2026": boot([x for x in ALL if x["m"] < "2026"], gap),
        "правило без данных, 07.2023–06.2026": boot(te, gap)}
for lo, hi in (("2022-10", "2023-06"), ("2023-07", "2024-06"), ("2024-07", "2025-06"), ("2025-07", "2026-06")):
    sub = [x for x in ALL if lo <= x["m"] <= hi]
    nums[f"правило без данных {lo}…{hi}"] = boot(sub, gap)
    nums[f"правило без данных против жребия {lo}…{hi}"] = boot(sub, lambda x: (avg(x) - x["r"][x["prior"]]) * 100)
    nums[f"управляющий против жребия {lo}…{hi}"] = boot(sub, lambda x: (avg(x) - x["r"][x["pm"]]) * 100)
qs = defaultdict(list)
for x in ALL:
    qs[x["q"]].append(gap(x))
nums["prior по кварталам"] = {q: [round(float(np.mean(v)), 2), len(v)] for q, v in sorted(qs.items())}

# Экономика: выигрыш на рубль продажи с пределом «заменить не больше позиции подсказанной бумаги»
nav, amt = defaultdict(float), {}
for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")):
    nav[(p["фонд"], p["дата"])] += float(p["value"] or 0)
for e in csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8")):
    if e["сторона"] == "продажа":
        amt[(e["фонд"], e["d0"], e["d1"], e["secid"])] = float(e["объём_руб"] or 0)
case = {x["text"]: c for t in TEST + ("_r1_fix",) for x, c in zip(R.load((t,)), json.loads((ROOT / f"sellrank_cases{t}.json").read_text()))}
for k in ("prior",):
    num = unc = den = 0.0
    for x in ALL:
        c = case[x["text"]]
        a = amt.get((c["фонд"], c["d0"], c["d1"], next(b["secid"] for b in c["бумаги"] if b["продана"])))
        N = nav.get((c["фонд"], c["d0"]))
        if not a or not N:
            continue
        L = x[k]
        share = float(re.search(rf"Бумага {L}: доля в портфеле ([\d.]+)%", x["text"]).group(1)) / 100
        d = gap(x)
        num += (min(a, share * N) if L != x["pm"] else a) * d
        unc += a * d
        den += a
    per = num / den
    # доля выборочных продаж в СЧА акций за месяц: медиана 2,7%, среднее 4,1% (sellrank_check.py, 07.2025–06.2026)
    nums[f"экономика {k}"] = {"п.п._на_рубль_3м_с_пределом": round(per, 2), "без_предела": round(unc / den, 2),
                              "годовых_медиана_2.7%": round(per * 0.027 * 12, 2), "годовых_среднее_4.1%": round(per * 0.041 * 12, 2)}
share = lambda x, L: float(re.search(rf"Бумага {L}: доля в портфеле ([\d.]+)%", x["text"]).group(1))
for k in ("prior",):
    dif = [x for x in ALL if x[k] != x["pm"]]
    nums[f"доли позиций {k}: подсказка / проданная, %"] = [round(float(np.mean([share(x, x[k]) for x in dif])), 1),
                                                         round(float(np.mean([share(x, x["pm"]) for x in dif])), 1), len(dif)]
assert nums["правило без данных, все годы"]["n"] == len(ALL) == sum(v[1] for v in nums["prior по кварталам"].values())

# Рисунок 3. Правило без данных против управляющего по кварталам
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for f in font_manager.findSystemFonts():
    if "PTSans" in f.replace(" ", "") or "PT_Sans" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "PT Sans", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK, MUTED, BLUE, GREY = "#0b0b0b", "#52514e", "#2a78d6", "#b7b5ad"
fmt = lambda v: "0,0" if abs(v) < 0.05 else f"{v:+.1f}".replace(".", ",").replace("-", "−")
roman = {"01": "I", "04": "II", "07": "III", "10": "IV"}
ql = list(nums["prior по кварталам"])
vals = [nums["prior по кварталам"][q][0] for q in ql]
fig, ax = plt.subplots(figsize=(7.0, 3.6), facecolor="white")
ax.bar(range(len(ql)), vals, width=0.62, color=[BLUE if v > 0.5 else GREY for v in vals])
for i, v in enumerate(vals):
    ax.text(i, v + (0.2 if v >= 0 else -0.2), fmt(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK,
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
m = nums["правило без данных, все годы"]
ax.hlines(m["D"], -0.45, len(ql) - 0.55, color=INK, lw=1, ls=(0, (4, 3)))
ax.set_xlim(-0.6, len(ql) + 1.5)
ax.text(len(ql) - 0.4, m["D"], f"в среднем\n{fmt(m['D'])} п.п.", ha="left", va="center", fontsize=10, fontweight="bold", color=INK)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(len(ql)), [f"{roman[q[5:]]}\n{q[:4]}" for q in ql], fontsize=9)
ax.tick_params(axis="x", length=0)
ax.set_ylim(min(0, min(vals)) - 1.3, max(vals) + 1.2)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "−")))
ax.grid(axis="y", color="#eceae4", lw=0.7)
ax.set_axisbelow(True)
ax.set_ylabel("выигрыш от продажи по подсказке\nвместо выбора управляющего, п.п.", color=MUTED)
fig.text(0.01, 0.01, f"Квартал продажи; результат за 3 месяца за вычетом роста похожих бумаг. {str(m['n'])[:-3]} {str(m['n'])[-3:]} продаж {ql[0][5:]}.{ql[0][:4]}–06.2026. "
         f"95% интервал среднего: {fmt(m['lo'])}…{fmt(m['hi'])}. Исходная версия правила, без обновлений.",
         fontsize=7.8, color=MUTED, wrap=True)
fig.subplots_adjust(bottom=0.25, top=0.97, left=0.13, right=0.98)
OUT = ROOT.parents[1] / "рисунки"
fig.savefig(OUT / "р8_2_правило.png", dpi=220, facecolor="white")
(OUT / "числа_р8.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1))
print(json.dumps(nums, ensure_ascii=False, indent=1))
