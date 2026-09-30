"""Рисунок 4 из выпуска месячного цикла (mirror_cycle.py): список кандидатов на продажу для одного фонда (бумаги
обезличены) и блок «что сейчас показывает зеркало» — последняя квартальная проверка правила и главное из версии,
предложенной ИИ-аналитиком. Всё берётся из выпуска и зеркало_состояние.json. python3 figure_card.py [ГГГГ-ММ] -> рисунки/рис4_карточка.png"""
import csv
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

import rule as A
from agent_rate import rate
from robust import ROOT

MONTH = sys.argv[1] if len(sys.argv) > 1 else "2026-08"
FUND = "ВИМ – Акции"                                        # пример портфеля; в рисунке фонд и бумаги не называются
for f in font_manager.findSystemFonts():
    if "PTSans" in f.replace(" ", "") or "PT_Sans" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "PT Sans"})
OUT = ROOT.parents[1] / "рисунки"
rows = [r for r in csv.DictReader(open(ROOT.parent / "зеркало" / "выпуски" / MONTH / "списки.csv", encoding="utf-8")) if r["фонд"] == FUND]
st = json.loads((ROOT / "зеркало_состояние.json").read_text())
chk = [h for h in st["история"] if h["месяц"] <= MONTH][-1]
num = lambda v: f"{float(v):.1f}".replace(".", ",").replace(",0", "")
pct = lambda v: f"{v:g}".replace(".", ",")
sign = lambda v: f"{v:+.1f}".replace(".", ",").replace("-", "−")
MON = {"01": "январь", "02": "февраль", "03": "март", "04": "апрель", "05": "май", "06": "июнь", "07": "июль", "08": "август",
       "09": "сентябрь", "10": "октябрь", "11": "ноябрь", "12": "декабрь"}
when = lambda m: f"{MON[m[5:]]} {m[:4]}"

INK, MUTED, BLUE, PALE, LINE = "#0b0b0b", "#52514e", "#2a78d6", "#eaf2fc", "#d9d7d0"
fig, ax = plt.subplots(figsize=(8.8, 3.7), facecolor="white")
ax.set_xlim(0, 11.8)
ax.set_ylim(1.05, 5.7)
ax.axis("off")
ax.add_patch(FancyBboxPatch((0.05, 1.1), 11.7, 4.55, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=LINE, lw=1.1))
ax.add_patch(FancyBboxPatch((0.05, 5.02), 11.7, 0.63, boxstyle="round,pad=0.02,rounding_size=0.12", fc=BLUE, ec=BLUE))
ax.text(0.3, 5.34, f"Зеркало рынка · кандидаты на продажу · {when(MONTH)}", color="white", fontsize=11.5, fontweight="bold", va="center")
ax.text(11.5, 5.34, "фонд акций, бумаги обезличены", color="white", fontsize=8.5, va="center", ha="right")
x0 = 0.3
ax.text(x0, 4.7, f"№   Бумага       Доля    Балл из {num(rows[0]['из'])}    Что отмечает правило", color=MUTED, fontsize=8.3,
        fontweight="bold", va="center")
for k, r in enumerate(rows[:5]):
    y = 4.3 - k * 0.52
    if k == 0:
        ax.add_patch(FancyBboxPatch((0.2, y - 0.22), 7.75, 0.44, boxstyle="round,pad=0.01,rounding_size=0.06", fc=PALE, ec=PALE))
    ax.text(x0, y, str(k + 1), color=INK, fontsize=9.5, va="center")
    ax.text(x0 + 0.35, y, f"Бумага {'АБВГД'[k]}", color=INK, fontsize=9.5, va="center")
    ax.text(x0 + 1.5, y, f"{num(r['доля_%'])}%", color=INK, fontsize=9.5, va="center")
    ax.text(x0 + 2.55, y, num(r["балл"]), color=INK, fontsize=9.5, fontweight="bold", va="center")
    parts = r["что_отмечает"].split(" · ")                      # все сработавшие условия: их баллы дают балл строки
    txt = " · ".join(parts) if len(parts) <= 3 else " · ".join(parts[:3]) + "\n" + " · ".join(parts[3:])
    ax.text(x0 + 3.05, y, txt, color=INK, fontsize=7.7 if len(parts) <= 3 else 7.2, va="center", linespacing=1.15)
rest = len(rows) - 5
word = "позиция" if rest % 10 == 1 and rest % 100 != 11 else "позиции" if rest % 10 in (2, 3, 4) and rest % 100 not in (12, 13, 14) else "позиций"
ax.text(x0, 4.3 - 5 * 0.52, f"… еще {rest} {word} от 0,5% портфеля с меньшим баллом", color=MUTED, fontsize=8.5, va="center")
ax.plot([8.2, 8.2], [1.45, 4.85], color=LINE, lw=1)
xr = 8.45
ax.text(xr, 4.7, "ЧТО СЕЙЧАС ПОКАЗЫВАЕТ ЗЕРКАЛО", color=BLUE, fontsize=8.3, fontweight="bold", va="center")
NAME = {("fcf_neg", ">"): "денежный поток < 0", ("np_neg", ">"): "убыток", ("yld", "<"): "низкие дивиденды",
        ("debt_ebitda", ">"): "высокий долг", ("roe", "<"): "низкий ROE", ("np_g", "<"): "падение прибыли",
        ("rev_g", "<"): "падение выручки", ("pe", ">"): "высокий P/E", ("pb", ">"): "высокий P/B",
        ("ev_ebitda", ">"): "высокий EV/EBITDA", ("p12", ">"): "рост цены за год", ("p12", "<"): "падение цены за год",
        ("p3", ">"): "рост цены за 3 мес."}
verdict = ("правило обновлено." if chk["заменено"] else
           f"разница меньше {A.MARGIN:g} п.п.:\nправило оставлено.")
ax.text(xr, 4.4, f"Проверка правила, {when(chk['месяц'])}:\nна {chk['продаж_за_год']} продажах рынка за год\n"
        f"версия ИИ-аналитика {sign(chk['новое'])} п.п.,\nдействующее правило {sign(chk['действующее'])} п.п.;\n{verdict}",
        color=INK, fontsize=8.4, va="top", linespacing=1.3)
ver = chk["новая_версия"][0] if chk["новая_версия"] else []
r0, r1 = rate(A.shift(chk["месяц"], -6)), rate(chk["месяц"])
groups = {}
for c in sorted(ver, key=lambda c: -c["баллы"])[:3]:
    groups.setdefault(c["баллы"], []).append(NAME.get((c["признак"], c["знак"]), c["признак"]))
feat_txt = ", ".join(f"{', '.join(v)} ({'по ' if len(v) > 1 else ''}{k})" for k, v in groups.items())
wrapped = textwrap.fill(f"главное в предложенной версии: {feat_txt}.", 34)
move = "снизилась" if r1 < r0 else "выросла" if r1 > r0 else "не менялась"
rate_txt = f"ставка {move} до {pct(r1)}% (полгода назад {pct(r0)}%)" if r1 != r0 else f"ставка {pct(r1)}%, за полгода не менялась"
ax.text(xr, 2.8, f"Записка ИИ-аналитика, {when(chk['месяц'])}:", color=MUTED, fontsize=8.3, fontweight="bold", va="top")
ax.text(xr, 2.47, textwrap.fill(rate_txt, 34) + ";\n" + wrapped,
        color=INK, fontsize=8.3, va="top", linespacing=1.3)
fig.savefig(OUT / "рис4_карточка.png", dpi=220, bbox_inches="tight", facecolor="white")
print("ok", len(rows))
