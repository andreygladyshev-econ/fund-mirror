"""ИИ как исследователь: языковая модель сама выводит прозрачное правило продажи из истории, где исходы известны,
а правило проверяется на годах, которых модель не видела.
Обучение: 07.2023–06.2025 (случаи _r2, _r3 — два года, 800 продаж). Проверка: 07.2022–06.2023 (_r1, низкая ставка),
07–12.2025 (_2025h2), 01–06.2026 (''). Бумаги обезличены; модели даются только признаки и какая из 4 оказалась худшей.
Правило — JSON: до 6 условий {"признак", "знак" (">"/"<"), "порог", "баллы"}; продать бумагу с наибольшей суммой
баллов, ничья — по росту за 12 мес. Для сравнения: ручной чек-лист (придуман по 2026 г.) и логистическая регрессия
(классический количественный подход) на тех же признаках и тех же годах.
python3 rulegen.py gen [K]     # K предложений модели (OpenRouter, DeepSeek V4 Flash) -> данные/составы_фондов/правила_ии.json
python3 rulegen.py eval        # оценка всех правил на обучении и на проверке
"""
import json
import random
import re
import sys
from collections import defaultdict

import numpy as np

from robust import ROOT
from sellrank_check import composite, feats

TRAIN, TEST = ("_r2", "_r3"), ("_r1", "_2025h2", "")
NAMES = {"_r1": "07.2022–06.2023 (ставка 7,5–8,5%)", "_r2": "07.2023–06.2024", "_r3": "07.2024–06.2025",
         "_2025h2": "07–12.2025", "": "01–06.2026"}
FEAT = {"доля": "доля в портфеле, %", "p1": "цена за 1 мес., %", "p3": "цена за 3 мес., %", "p12": "цена за 12 мес., %",
        "yld": "дивидендная доходность за 12 мес., %", "pe": "P/E", "pb": "P/B", "ev_ebitda": "EV/EBITDA",
        "debt_ebitda": "Долг/EBITDA", "roe": "ROE, %", "fcf_neg": "FCF отрицательный (1/0)", "np_neg": "убыток (1/0)",
        "rev_g": "рост выручки за год, %", "np_g": "рост чистой прибыли за год, %"}
OUT = ROOT / "правила_ии.json"


def num(x):
    x = (x or "").replace(" ", "").replace("%", "").replace("−", "-")
    return float(x) if re.fullmatch(r"-?\d+(\.\d+)?", x) else None


def feats2(text):
    base = feats(text)
    out = {}
    for L, body in re.findall(r"Бумага ([АБВГ]): (.*?)(?=\n\nБумага [АБВГ]:|\Z)", text, re.S):
        def series(name):
            m = re.search(rf"^{re.escape(name)}\s*[,:][^\n]*?:([^\n]*)$", body, re.M) or re.search(rf"^{re.escape(name)}:([^\n]*)$", body, re.M)
            return [num(v.strip()) for v in m.group(1).split("|")] if m else []
        last = lambda n: next((v for v in reversed(series(n)) if v is not None), None)
        def growth(n):
            s = [v for v in series(n) if v is not None]
            return (s[-1] / s[-2] - 1) * 100 if len(s) >= 2 and s[-2] > 0 else None
        b = base[L]
        out[L] = {"доля": b["доля"], "p1": b.get("p1"), "p3": b["p3"], "p12": b["p12"], "yld": b["yld"], "pe": b["pe"],
                  "pb": last("P/B") if last("P/B") is not None else last("P/BV"), "ev_ebitda": last("EV/EBITDA"),
                  "debt_ebitda": last("Долг/EBITDA"), "roe": last("ROE"),
                  "fcf_neg": None if b["fcf"] is None else float(b["fcf"] < 0), "np_neg": None if b["np"] is None else float(b["np"] < 0),
                  "rev_g": growth("Выручка"), "np_g": growth("Чистая прибыль")}
        m = re.search(r"цена за 1 мес\. ([+-][\d.]+)%", body)
        out[L]["p1"] = float(m.group(1)) if m else None
    return out


def load(tags):
    rows = []
    for t in tags:
        for c in json.loads((ROOT / f"sellrank_cases{t}.json").read_text()):
            r = {b["буква"]: b["доходность_3м"] for b in c["бумаги"]}
            rows.append({"tag": t, "m": c["d1"][:7], "fe": feats2(c["текст"]), "r": r, "text": c["текст"],
                         "pm": next(b["буква"] for b in c["бумаги"] if b["продана"]),
                         "worst": min(r, key=r.get)})
    return rows


def apply_rule(rule, fe):
    def pts(x):
        s = 0.0
        for c in rule:
            v = x.get(c["признак"])
            if v is not None and ((c["знак"] == ">" and v > c["порог"]) or (c["знак"] == "<" and v < c["порог"])):
                s += c["баллы"]
        return s
    return max(fe, key=lambda L: (pts(fe[L]), fe[L]["p12"] if fe[L]["p12"] is not None else -1e9))


def score(rows, pick, seed=4):
    """Разница: доходность проданной управляющим минус доходность выбора (п.п. за 3 мес.), бутстрэп по месяцам."""
    by = defaultdict(list)
    for x in rows:
        by[x["m"]].append((x["r"][x["pm"]] - x["r"][pick(x)]) * 100)
    v = [d for ds in by.values() for d in ds]
    ms = list(by)
    rng = np.random.default_rng(seed)
    bs = [np.mean([d for i in rng.integers(0, len(ms), len(ms)) for d in by[ms[i]]]) for _ in range(2000)]
    return np.mean(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def prompt(sample):
    lines = []
    for k, x in enumerate(sample, 1):
        lines.append(f"Случай {k}. Худшей за 3 мес. оказалась бумага {x['worst']}.")
        for L in "АБВГ":
            f = x["fe"][L]
            lines.append(f"  {L}: " + "; ".join(f"{n}={'н/д' if f[n] is None else round(f[n], 2)}" for n in FEAT))
    feats_desc = "\n".join(f"- {k}: {v}" for k, v in FEAT.items())
    return f"""Ты — количественный аналитик управляющей компании (российский рынок акций, 2023–2025 гг., ключевая ставка
высокая). Ниже {len(sample)} реальных случаев: в портфеле фонда 4 бумаги (обезличены), известны признаки на дату решения и
какая из 4 бумаг показала ХУДШУЮ полную доходность за следующие 3 месяца (её и надо было продать).
Признаки:
{feats_desc}

Найди в данных закономерности и предложи ПРОЗРАЧНОЕ правило для выбора, какую бумагу продавать: не больше 6 условий,
каждое — признак, знак сравнения (">" или "<"), порог и баллы (от 1 до 3). Продаётся бумага с наибольшей суммой
баллов (при отсутствии данных условие не срабатывает). Правило должно быть экономически осмысленным и понятным
риск-менеджеру; избегай подгонки под отдельные случаи.
Сначала коротко (3–6 предложений) объясни логику, затем выдай правило последним блоком строго в формате JSON:
[{{"признак": "...", "знак": ">", "порог": 0, "баллы": 1}}, ...]

Данные:
{chr(10).join(lines)}"""


def prompt_prior():
    """Без данных: правило только из знаний модели (проверка «не подогнано ли под наши данные»)."""
    feats_desc = "\n".join(f"- {k}: {v}" for k, v in FEAT.items())
    return f"""Ты — количественный аналитик управляющей компании (российский рынок акций). В портфеле фонда 4 бумаги
(обезличены), нужно продать одну — ту, что, вероятно, покажет ХУДШУЮ полную доходность в следующие 3 месяца. Данных
об исходах у тебя нет — опирайся только на знания о том, какие характеристики компаний предвещают слабую доходность.
Признаки, доступные на дату решения:
{feats_desc}

Предложи ПРОЗРАЧНОЕ правило: не больше 6 условий, каждое — признак, знак (">" или "<"), порог и баллы (от 1 до 3).
Продаётся бумага с наибольшей суммой баллов. Сначала коротко (3–6 предложений) объясни логику, затем выдай правило
последним блоком строго в формате JSON:
[{{"признак": "...", "знак": ">", "порог": 0, "баллы": 1}}, ...]"""


def gen(k):
    """Модель и объём выборки — через окружение: RG_MODEL, RG_PROV (openrouter|local), RG_N, RG_OUT (файл правил)."""
    import os
    from llm_read import ask
    model, prov = os.environ.get("RG_MODEL", "deepseek/deepseek-v4-flash"), os.environ.get("RG_PROV", "openrouter")
    n, out_f = int(os.environ.get("RG_N", 220)), ROOT / os.environ.get("RG_OUT", OUT.name)
    train = load(tuple(os.environ["RG_TRAIN"].split(",")) if os.environ.get("RG_TRAIN") else TRAIN)   # «только вперёд»: другое окно
    mode = os.environ.get("RG_MODE", "")          # "" — как есть; shuffle — ответы перемешаны; prior — без данных
    rules = json.loads(out_f.read_text()) if out_f.exists() else []
    for j in range(k):
        rng = random.Random(100 + len(rules) + j)
        sample = rng.sample(train, n)
        if mode == "shuffle":
            sample = [dict(x, worst=rng.choice("АБВГ")) for x in sample]
        text = (prompt_prior() if mode == "prior" else prompt(sample)) + (" /no_think" if os.environ.get("RG_NOTHINK") else "")   # локальная модель: без развёрнутого рассуждения
        out = ask(model, text, effort=os.environ.get("RG_EFFORT", "low"), provider=prov) or ""
        m = re.findall(r"\[\s*\{.*?\}\s*\]", out, re.S)
        try:
            rule = [c for c in json.loads(m[-1]) if c.get("признак") in FEAT and c.get("знак") in (">", "<")]
        except Exception:  # noqa: BLE001
            print("не разобрал ответ", j)
            continue
        rules.append({"id": len(rules), "правило": rule, "логика": out[: out.rfind("[")].strip()[-1500:]})
        out_f.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
        print("правило", len(rules) - 1, rule, flush=True)


def logit(train):
    """Классический подход: логистическая регрессия «худшая ли бумага» на тех же признаках (пропуски — медиана + флаг)."""
    keys = list(FEAT)
    X = [[x["fe"][L][k] for k in keys] for x in train for L in "АБВГ"]
    med = [np.nanmedian([np.nan if r[i] is None else r[i] for r in X]) for i in range(len(keys))]

    def vec(f):
        v = [f[k] for k in keys]
        return [med[i] if v[i] is None else float(np.clip(v[i], -500, 500)) for i in range(len(keys))] + [float(x is None) for x in v]
    Xt = np.array([vec(x["fe"][L]) for x in train for L in "АБВГ"])
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    y = [int(L == x["worst"]) for x in train for L in "АБВГ"]
    Z, y = np.c_[np.ones(len(Xt)), (Xt - mu) / sd], np.array(y, float)
    w = np.zeros(Z.shape[1])
    for _ in range(3000):                                  # градиентный спуск с L2-штрафом (аналог C=0.1)
        p = 1 / (1 + np.exp(-Z @ w))
        w -= 0.1 * (Z.T @ (p - y) / len(y) + 0.01 * np.r_[0, w[1:]])
    return lambda x: max("АБВГ", key=lambda L: float(np.r_[1, (np.array(vec(x["fe"][L])) - mu) / sd] @ w))


def evaluate():
    import os
    train, test = load(TRAIN), load(TEST)
    rules = json.loads((ROOT / os.environ.get("RG_OUT", OUT.name)).read_text())
    cands = [("ручной чек-лист (придуман по 2026 г.)", lambda x: composite(feats(x["text"]))),
             ("логистическая регрессия", logit(train))]
    cands += [(f"правило ИИ №{r['id']}", (lambda rule: lambda x: apply_rule(rule, x["fe"]))(r["правило"])) for r in rules]
    fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}; {t[2]:+.2f}]"
    res = {}
    print(f"{'':<40}{'обучение 2023–25':<24}" + "".join(f"{NAMES[t][:16]:<24}" for t in TEST) + "проверка вместе")
    for name, pick in cands:
        tr = score(train, pick)
        te = [score([x for x in test if x["tag"] == t], pick) for t in TEST]
        al = score(test, pick)
        res[name] = {"обучение": tr, "проверка": dict(zip(TEST, te)), "проверка_вместе": al}
        print(f"{name:<40}{fmt(tr):<24}" + "".join(f"{fmt(t):<24}" for t in te) + fmt(al))
    (ROOT / "правила_ии_оценка.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return res


if __name__ == "__main__":
    gen(int(sys.argv[2]) if len(sys.argv) > 2 else 6) if sys.argv[1] == "gen" else evaluate()
