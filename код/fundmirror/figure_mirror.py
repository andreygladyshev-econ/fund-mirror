"""«Зеркало рынка»: какие признаки слабой бумаги работали в разные периоды ставки (29.09, записка v8).
Для каждой продажи из 4 бумаг портфеля (проданная + 3 случайные) признак выбирает самую «слабую»; клетка — насколько она
затем отстала от случайной из тех же 4 (очищенные доходности, мерка Х; блочный бутстрэп по месяцам).
python3 figure_mirror.py -> рисунки/р8_3_зеркало.png, рисунки/числа_р8_зеркало.json"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import agent_rate as G
from robust import ROOT

SIG = [("высокий долг (Долг/EBITDA)", "debt_ebitda", 1), ("отрицательный денежный поток", "fcf_neg", 1),
       ("убыток", "np_neg", 1), ("низкая рентабельность (ROE)", "roe", -1), ("падение прибыли", "np_g", -1),
       ("дорогая оценка (P/E)", "pe", 1), ("сильный рост цены за год", "p12", 1), ("низкие дивиденды", "yld", -1)]
PER = [("2022-10", "2023-06", "10.2022–06.2023\nставка 7,5%"), ("2023-07", "2024-06", "07.2023–06.2024\nставка 8,5–16%"),
       ("2024-07", "2025-06", "07.2024–06.2025\nставка 16–21%"), ("2025-07", "2026-06", "07.2025–06.2026\nставка 20–14%")]
MIN_N = 60


def pick(x, f, s):
    v = {L: x["fe"][L][f] for L in "АБВГ" if x["fe"][L][f] is not None}
    return max(v, key=lambda L: s * v[L]) if len(set(v.values())) > 1 else None


def boot(by, H=3, seed=4):
    ms = sorted(by)
    rng = np.random.default_rng(seed)
    est = [np.mean([d for st in rng.integers(0, len(ms), -(-len(ms) // H)) for k in range(H) for d in by[ms[(st + k) % len(ms)]]])
           for _ in range(2000)]
    return float(np.mean([d for v in by.values() for d in v])), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


ALL = G.load_all()
res = {}
for name, f, s in SIG:
    for lo, hi, lab in PER:
        by = defaultdict(list)
        for x in ALL:
            if lo <= x["m"] <= hi and (L := pick(x, f, s)):
                by[x["m"]].append((np.mean(list(x["r"].values())) - x["r"][L]) * 100)   # плюс — признак выбрал отстающую
        n = sum(map(len, by.values()))
        res[f"{name} | {lo}"] = [round(v, 2) for v in boot(by)] + [n] if n >= MIN_N else [None, None, None, n]

for f in font_manager.findSystemFonts():
    if "PTSans" in f.replace(" ", "") or "PT_Sans" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "PT Sans", "font.size": 10})
INK, MUTED, POS, NEG, NA = "#0b0b0b", "#52514e", (0x2a / 255, 0x78 / 255, 0xd6 / 255), (0xeb / 255, 0x68 / 255, 0x34 / 255), "#f1f0ec"
fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor="white")
ax.set_xlim(0, len(PER))
ax.set_ylim(len(SIG), 0)
ax.axis("off")
for i, (name, _, _) in enumerate(SIG):
    ax.text(-0.05, i + 0.5, name, ha="right", va="center", fontsize=9.8, color=INK)
    for j, (lo, _, _) in enumerate(PER):
        d, a, b, n = res[f"{name} | {lo}"]
        if d is None:
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.06), 0.94, 0.88, color=NA))
            ax.text(j + 0.5, i + 0.5, "мало данных", ha="center", va="center", fontsize=8, color=MUTED)
            continue
        sig = a > 0 or b < 0
        base = POS if d > 0 else NEG
        alpha = 0.5 + min(abs(d) / 12, 0.45) if sig else 0.06 + min(abs(d) / 12, 0.22)
        ax.add_patch(plt.Rectangle((j + 0.03, i + 0.06), 0.94, 0.88, color=base, alpha=alpha))
        txt = f"{d:+.1f}".replace(".", ",").replace("-", "−")
        ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=10, fontweight="bold" if sig else "normal",
                color="white" if alpha > 0.55 else INK)
for j, (_, _, lab) in enumerate(PER):
    ax.text(j + 0.5, -0.12, lab, ha="center", va="bottom", fontsize=9, color=MUTED, linespacing=1.3)
fig.text(0.01, 0.015, "Клетка: насколько бумага, отмеченная признаком, отстала за 3 месяца от случайной бумаги того же портфеля, п.п.\n"
         "Синий: признак помогал выбрать, что продать; оранжевый: вредил. Насыщенный цвет и жирные цифры: значимо (95%).",
         fontsize=7.8, color=MUTED)
fig.subplots_adjust(left=0.33, right=0.99, top=0.86, bottom=0.12)
fig.savefig(ROOT.parents[1] / "рисунки" / "р8_3_зеркало.png", dpi=220, facecolor="white")
(ROOT.parents[1] / "рисунки" / "числа_р8_зеркало.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1))
