"""Общее для проверки правила и агента: баллы правила, выбор бумаги, кварталы, блочный бутстрэп по месяцам.
Правило — список версий; версия — список условий {"признак", "знак" (">"/"<"), "порог", "баллы"}. Баллы бумаги —
сумма по всем версиям (исходное правило — три ответа модели); продаётся бумага с наибольшей суммой, при равенстве —
с наибольшим ростом цены за 12 мес."""
from collections import defaultdict

import numpy as np

QS = [f"{y}-{m:02d}" for y in (2023, 2024, 2025, 2026) for m in (1, 4, 7, 10) if "2023-07" <= f"{y}-{m:02d}" <= "2026-04"]
MARGIN = 1.0              # новая версия правила заменяет действующую, только если лучше на 1 п.п. и больше


def shift(m, k):
    """Месяц «ГГГГ-ММ», сдвинутый на k месяцев."""
    y, mm = int(m[:4]), int(m[5:7]) + k
    y, mm = y + (mm - 1) // 12, (mm - 1) % 12 + 1
    return f"{y}-{mm:02d}"


def points(rules, f):
    """Сумма баллов бумаги с признаками f по всем версиям правила (нет данных — условие не срабатывает)."""
    return sum(c["баллы"] for rule in rules for c in rule if f.get(c["признак"]) is not None and
               ((c["знак"] == ">" and f[c["признак"]] > c["порог"]) or (c["знак"] == "<" and f[c["признак"]] < c["порог"])))


def picker(rules):
    """Функция: случай продажи -> буква бумаги, которую правило предлагает продать."""
    return lambda x: max(x["fe"], key=lambda L: (points(rules, x["fe"][L]), x["fe"][L]["p12"] if x["fe"][L]["p12"] is not None else -1e9))


def gain(rows, pick):
    """Средний выигрыш подсказки против выбора управляющего, п.п. за 3 мес."""
    return float(np.mean([(x["r"][x["pm"]] - x["r"][pick(x)]) * 100 for x in rows])) if rows else float("nan")


def block_boot(pairs, H=3, seed=4, B=2000):
    """Среднее и 95% интервал: блочный бутстрэп по месяцам (кольцевые блоки из H подряд идущих месяцев выборки).
    pairs — список (месяц, значение). Генератор создаётся заново при каждом вызове, поэтому одна и та же величина
    в разных скриптах получает один и тот же интервал. Возвращает (среднее, низ, верх, n)."""
    by = defaultdict(list)
    for m, v in pairs:
        by[m].append(v)
    ms = sorted(by)
    rng = np.random.default_rng(seed)
    est = [np.mean([v for st in rng.integers(0, len(ms), -(-len(ms) // H)) for k in range(H) for v in by[ms[(st + k) % len(ms)]]])
           for _ in range(B)]
    return (round(float(np.mean([v for _, v in pairs])), 2), round(float(np.percentile(est, 2.5)), 2),
            round(float(np.percentile(est, 97.5)), 2), len(pairs))
