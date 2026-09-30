"""Рисунок 3 записки — схема ИИ-агента «Зеркало рынка»: где работают ИИ, данные всего рынка, методика и человек.
python3 figure_scheme.py -> рисунки/рис3_схема.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

for f in font_manager.findSystemFonts():
    if "PTSans" in f.replace(" ", "") or "PT_Sans" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "PT Sans"})
ROOT_ = Path(__file__).resolve().parents[2]
OUT = ROOT_ / "рисунки" if (ROOT_ / "рисунки").exists() else ROOT_ / "документы" / "рисунки"
INK, MUTED = "#0b0b0b", "#52514e"
ROLE = {"данные рынка": "#2a78d6", "ИИ": "#eb6834", "методика": "#1baf7a", "человек": "#52514e"}
steps = [
    ("ИИ", "Чтение справок", "шаблоны\nи ИИ-читатель\nпереводят новые\nсправки фондов\nв таблицы; суммы\nи цены сверяются"),
    ("данные рынка", "База продаж\nрынка", "восстанавливаются\nпродажи фондов\nза месяц;\nк продажам\n3-месячной давности\nдописывается\nих результат"),
    ("ИИ", "ИИ-аналитик", "читает сводку\nзеркала и ставку;\nпишет записку\nкомитету;\nпредлагает\nверсию правила"),
    ("методика", "Проверка\nправила", "сравнивает версии\nна продажах рынка\nза год; правило\nменяется, только\nесли новая\nверсия лучше"),
    ("методика", "Список\nкандидатов", "по каждому\nпортфелю: позиции\nот 0,5%, сверху\nсамые слабые\nпо баллу правила"),
    ("человек", "Управляющий", "когда нужно\nосвободить деньги,\nсмотрит в верх\nсписка\nи решает сам"),
]
fig, ax = plt.subplots(figsize=(10.4, 3.1), facecolor="white")
ax.set_xlim(0, 11.55)
ax.set_ylim(0.25, 2.72)
ax.axis("off")
w, gap = 1.63, 0.29
for i, (role, head, body) in enumerate(steps):
    x = 0.1 + i * (w + gap)
    c = ROLE[role]
    ax.add_patch(FancyBboxPatch((x, 0.3), w, 1.95, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=c, lw=1.8))
    ax.add_patch(FancyBboxPatch((x, 1.93), w, 0.32, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec=c, lw=1.8))
    ax.text(x + w / 2, 2.09, role.upper(), ha="center", va="center", color="white", fontsize=7.8, fontweight="bold")
    ax.text(x + w / 2, 1.62, head, ha="center", va="center", color=INK, fontsize=9.2, fontweight="bold", linespacing=1.1)
    ax.text(x + w / 2, 0.92, body, ha="center", va="center", color=MUTED, fontsize=7.9, linespacing=1.35)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.03, 1.3), xytext=(x + w + 0.03, 1.3),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
for i0, i1, lab in ((0, 1, "ДАННЫЕ · каждый месяц"), (2, 3, "ЗНАНИЯ · раз в квартал"), (4, 5, "РЕШЕНИЯ · каждый месяц")):
    xa, xb = 0.1 + i0 * (w + gap), 0.1 + i1 * (w + gap) + w
    ax.plot([xa + 0.05, xb - 0.05], [2.4, 2.4], color=INK, lw=1.2)
    ax.text((xa + xb) / 2, 2.52, lab, ha="center", va="bottom", color=INK, fontsize=8.6, fontweight="bold")
fig.savefig(OUT / "рис3_схема.png", dpi=220, bbox_inches="tight", facecolor="white")
print("ok")
