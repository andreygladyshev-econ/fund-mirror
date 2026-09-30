"""Рисунок 1 и числа раздела 3: купленные и проданные бумаги активных и индексных фондов против похожих бумаг за 3 мес.
после сделки (одна выборка для покупок, продаж и разрыва; продажи без крупных сокращений); разрыв «купленные минус
проданные» при трёх способах подбора похожих бумаг (по обороту торгов, 3 и 5 групп; по весу в индексе МосБиржи).
python3 figure_problem.py -> рисунки/рис1_покупки_продажи.png, рисунки/числа_проблема.json"""
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in font_manager.findSystemFonts():
    if "PTSans" in f.replace(" ", "") or "PT_Sans" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "PT Sans", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#0b0b0b"})
INK, MUTED, SURF, ACT, IDXC = "#0b0b0b", "#52514e", "#ffffff", "#2a78d6", "#b7b5ad"
ROOT_ = Path(__file__).resolve().parents[2]
OUT = ROOT_ / "рисунки"
num = lambda v: f"{v:+.1f}".replace(".", ",").replace("-", "−")


def bench(nb):
    sys.argv = ["mirror_bench.py", "0.5", nb]
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(Path(__file__).with_name("mirror_bench.py")))


def flip(b):
    return {"D": -b["D"], "lo": -b["hi"], "hi": -b["lo"], "n": b["n"]}


nums = {}
g = bench("3")
rows, bb, diff = g["rows"], g["bb"], g["diff"]
small = lambda x: x["вид"] != "сокращение ≥ 7%"
panel = []
for who, flag in (("активные фонды", False), ("индексные фонды (контроль)", True)):
    sub = [x for x in rows if x["индексный"] == flag]
    buys = [x for x in sub if x["сторона"] == "покупка"]
    sells = [x for x in sub if x["сторона"] == "продажа" and small(x)]
    b, s = bb(buys, "Х"), flip(bb(sells, "Х"))
    d = diff(buys, sells)
    panel.append((who, b, s, {"D": d[0], "lo": d[1], "hi": d[2]}))
    nums[who] = {"покупки": [round(b["D"], 2), round(b["lo"], 2), round(b["hi"], 2), b["n"]],
                 "продажи": [round(s["D"], 2), round(s["lo"], 2), round(s["hi"], 2), s["n"]],
                 "проданные: рост сверх похожих": [round(-s["D"], 2), round(-s["hi"], 2), round(-s["lo"], 2), s["n"]],
                 "купленные − проданные": [round(v, 2) for v in d]}

# Рисунок 1. «Гантель»: покупки и продажи относительно похожих бумаг; разрыв подписан
BUY, SELL = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(7.0, 4.1), facecolor=SURF)
xs = {0: "Активные фонды", 1: "Индексные фонды\n(контроль)"}
for k, (who, b, s, d) in enumerate(panel):
    x = k * 1.35
    faded = k == 1
    for v, c, dx, lab in ((b, BUY, -0.09, "купленные"), (flip(s), SELL, 0.09, "проданные")):
        cc = c if not faded else "#b7b5ad"
        ax.plot([x + dx, x + dx], [v["lo"], v["hi"]], color=cc, lw=2, alpha=0.55, solid_capstyle="round")
        ax.scatter([x + dx], [v["D"]], s=150, color=cc, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(x + dx + (0.06 if dx > 0 else -0.06), v["D"] - (0.35 if dx > 0 else 0), f"{lab}\n{num(v['D'])}" + (": как похожие" if lab == "проданные" and not faded else ""), ha="left" if dx > 0 else "right",
                va="center", fontsize=9.5, color=INK if not faded else MUTED, bbox=dict(fc="white", ec="none", pad=0.8), zorder=4)
    top, bot = max(b["D"], flip(s)["D"]), min(b["D"], flip(s)["D"])
    cc = INK if not faded else MUTED  # скобка линиями: короткую стрелку matplotlib рисует крестиком
    ax.plot([x + 0.4, x + 0.4], [bot, top], color=cc, lw=1.2)
    for yy in (bot, top):
        ax.plot([x + 0.37, x + 0.43], [yy, yy], color=cc, lw=1.2)
    ax.text(x + 0.48, (top + bot) / 2, f"разрыв\n{num(d['D'])} п.п.", va="center", fontsize=10.5 if not faded else 9.5,
            fontweight="bold" if not faded else "normal", color=cc)
ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.set_xticks([0, 1.35], [xs[0], xs[1]])
ax.tick_params(axis="x", length=0, labelsize=10.5)
ax.set_xlim(-0.6, 2.25)
ax.spines["left"].set_visible(True)
ax.set_ylabel("рост бумаги за 3 месяца после сделки\nсверх роста похожих бумаг, п.п.", color=MUTED)
ax.grid(axis="y", color="#eceae4", lw=0.7)
nb_, ns_ = nums["активные фонды"]["покупки"][3], nums["активные фонды"]["продажи"][3]
sp = lambda n: f"{n:,}".replace(",", " ")
fig.text(0.01, 0.01, f"Активные фонды: {sp(nb_)} покупок и {sp(ns_)} продаж без крупных сокращений. "
         "Линии: 95% доверительный интервал.", fontsize=7.8, color=MUTED, wrap=True)
fig.subplots_adjust(bottom=0.19, top=0.97, left=0.15, right=0.98)
fig.savefig(OUT / "рис1_покупки_продажи.png", dpi=220, facecolor=SURF)
plt.close(fig)

# Итог «купленные против проданных» при трёх способах подобрать похожие бумаги
res = {"по обороту торгов, 3 группы": nums["активные фонды"]["купленные − проданные"],
       "по обороту торгов, 5 групп": None, "по весу в индексе МосБиржи": None}
idxres = {"по обороту торгов, 3 группы": nums["индексные фонды (контроль)"]["купленные − проданные"]}
for nb, lab in (("5", "по обороту торгов, 5 групп"), ("w", "по весу в индексе МосБиржи")):
    g2 = bench(nb)
    sub = lambda flag: ([x for x in g2["rows"] if x["индексный"] == flag and x["сторона"] == "покупка"],
                        [x for x in g2["rows"] if x["индексный"] == flag and x["сторона"] == "продажа" and small(x)])
    res[lab] = [round(v, 2) for v in g2["diff"](*sub(False))]
    idxres[lab] = [round(v, 2) for v in g2["diff"](*sub(True))]
nums["итог по способам"] = {"активные": res, "индексные": idxres}
nums["покупки: активные − индексные"] = [round(v, 2) for v in g["res"]["покупки: активные − индексные"]]
# контроль сравнения с индексом МосБиржи: у индексных фондов «результат» должен быть нулевым (минус у продаж — «выигрыш»)
nums["индексные фонды против индекса МосБиржи"] = {k: [round(g["res"][f"индексные (плацебо) | {k}"]["Б"][c], 2) for c in ("D", "lo", "hi")]
                                                  for k in ("покупка", "продажа", "выход", "сокращение < 7%")}
(OUT / "числа_проблема.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1))
print(json.dumps(nums, ensure_ascii=False, indent=1))
