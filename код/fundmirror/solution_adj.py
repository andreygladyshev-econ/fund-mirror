"""Числа решения для записки (разделы 3–6): проданная бумага против трёх случайных бумаг того же портфеля; исходное
правило «без данных» (правила_prior.json) против выбора управляющего и против жребия — по периодам и кварталам, на всех
1 139 продажах и без крупных сокращений; экономика с пределом «заменить не больше позиции подсказанной бумаги».
Доходности очищены так же, как в оценке сделок (cases.py). Модель не вызывается.
python3 solution_adj.py -> рисунки/рис5_правило.png, рисунки/числа_правило.json"""
import csv
import json
from collections import defaultdict

import numpy as np

import cases as C
import rule as A
from robust import INDEX, MECH, ROOT, UNSURE

ALL = C.load()
prior = A.picker([r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())])
avg = lambda x: np.mean(list(x["r"].values()))
for x in ALL:
    x["prior"], x["q"] = prior(x), f"{x['m'][:5]}{(int(x['m'][5:]) - 1) // 3 * 3 + 1:02d}"
gap = lambda x: (x["r"][x["pm"]] - x["r"][x["prior"]]) * 100               # правило против управляющего
vs_rand = lambda x: (avg(x) - x["r"][x["prior"]]) * 100                     # правило против жребия из 4
pm_vs_3 = lambda x: (x["r"][x["pm"]] - np.mean([v for L, v in x["r"].items() if L != x["pm"]])) * 100
mgr_vs_rand = lambda x: (avg(x) - x["r"][x["pm"]]) * 100                    # минус — управляющий хуже жребия


def boot(rows, val):
    d, lo, hi, n = A.block_boot([(x["m"], val(x)) for x in rows])
    by = defaultdict(list)
    for x in rows:
        by[x["m"]].append(val(x))
    return {"D": d, "lo": lo, "hi": hi, "n": n, "мес_в_плюсе": f"{sum(np.mean(v) > 0 for v in by.values())}/{len(by)}"}


small = [x for x in ALL if not x["крупное"]]
nums = {"проданная против 3 других бумаг портфеля, все годы": boot(ALL, pm_vs_3),
        "управляющий против жребия из 4, все годы": boot(ALL, lambda x: -mgr_vs_rand(x)),
        "правило без данных, все годы": boot(ALL, gap),
        "правило без данных против жребия, все годы": boot(ALL, vs_rand),
        "правило без данных, все годы без 2026": boot([x for x in ALL if x["m"] < "2026"], gap),
        "правило без данных, 07.2023–06.2026": boot([x for x in ALL if x["m"] >= "2023-07"], gap),
        "крупных сокращений (от 7% портфеля) в выборке": sum(x["крупное"] for x in ALL),
        "без крупных сокращений: проданная против 3 других": boot(small, pm_vs_3),
        "без крупных сокращений: правило без данных": boot(small, gap),
        "без крупных сокращений: правило против жребия": boot(small, vs_rand)}
for lo, hi in (("2022-10", "2023-06"), ("2023-07", "2024-06"), ("2024-07", "2025-06"), ("2025-07", "2026-06")):
    for tag, rows in (("", ALL), ("без крупных сокращений: ", small)):
        sub = [x for x in rows if lo <= x["m"] <= hi]
        nums[f"{tag}правило без данных {lo}…{hi}"] = boot(sub, gap)
        nums[f"{tag}правило без данных против жребия {lo}…{hi}"] = boot(sub, vs_rand)
        nums[f"{tag}управляющий против жребия {lo}…{hi}"] = boot(sub, mgr_vs_rand)
qs = defaultdict(list)
for x in ALL:
    qs[x["q"]].append(gap(x))
nums["prior по кварталам"] = {q: [round(float(np.mean(v)), 2), len(v)] for q, v in sorted(qs.items())}

# Экономика: выигрыш на рубль продажи с пределом «заменить не больше позиции подсказанной бумаги»
nav, amt, npos = defaultdict(float), {}, defaultdict(int)
for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")):
    nav[(p["фонд"], p["дата"])] += float(p["value"] or 0)
    npos[(p["фонд"], p["дата"])] += 1
ev = list(csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8")))
for e in ev:
    if e["сторона"] == "продажа":
        amt[(e["фонд"], e["d0"], e["d1"], e["secid"])] = float(e["объём_руб"] or 0)
num = unc = den = 0.0
for x in ALL:
    a, N = amt.get((x["фонд"], x["d0"], x["d1"], x["sold"])), nav.get((x["фонд"], x["d0"]))
    if not a or not N:
        continue
    L, d = x["prior"], gap(x)
    num += (min(a, x["fe"][L]["доля"] / 100 * N) if L != x["pm"] else a) * d
    unc += a * d
    den += a
per = num / den
# доля выборочных продаж в стоимости акций фонда за месяц: активные фонды, фонд-месяцы 07.2025–06.2026
act = [e for e in ev if not any(k in e["фонд"] for k in INDEX + UNSURE + MECH)]
ns = defaultdict(int)
for e in act:
    ns[(e["фонд"], e["d1"])] += e["сторона"] == "продажа"
sold, base = defaultdict(float), {}
for e in act:
    k = (e["фонд"], e["d1"])
    if "2025-07" <= e["d1"][:7] <= "2026-06":
        base[k] = nav.get((e["фонд"], e["d0"]), 0)
        if e["сторона"] == "продажа" and ns[k] <= 0.5 * npos.get((e["фонд"], e["d0"]), 1):
            sold[k] += float(e["объём_руб"] or 0)
turn = [sold[k] / v for k, v in base.items() if v > 0]
t_med, t_mean = float(np.median(turn)), float(np.mean(turn))
nums["экономика prior"] = {"п.п._на_рубль_3м_с_пределом": round(per, 2), "без_предела": round(unc / den, 2),
                           "выборочные_продажи_в_месяц_медиана_%": round(t_med * 100, 2), "среднее_%": round(t_mean * 100, 2),
                           "годовых_медиана": round(per * t_med * 12, 2), "годовых_среднее": round(per * t_mean * 12, 2)}
dif = [x for x in ALL if x["prior"] != x["pm"]]
nums["доли позиций prior: подсказка / проданная, %"] = [round(float(np.mean([x["fe"][x["prior"]]["доля"] for x in dif])), 1),
                                                     round(float(np.mean([x["fe"][x["pm"]]["доля"] for x in dif])), 1), len(dif)]
assert nums["правило без данных, все годы"]["n"] == len(ALL) == sum(v[1] for v in nums["prior по кварталам"].values())

# Рисунок 5. Правило без данных против управляющего по кварталам
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
fig.savefig(OUT / "рис5_правило.png", dpi=220, facecolor="white")
(OUT / "числа_правило.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1))
print(json.dumps(nums, ensure_ascii=False, indent=1))
