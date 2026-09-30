"""ИИ-агент по истории: каждый квартал сам выводит правило по уже известным исходам, сравнивает с действующим, выбирает,
применяет к продажам следующего квартала (план проверки записан до запуска). python3 agent_replay.py run | score"""
import json
import random
import re
import sys
from datetime import date

import numpy as np

import rulegen as R
from robust import ROOT

OUT = ROOT / "агент"
QS = [f"{y}-{m:02d}" for y in (2023, 2024, 2025, 2026) for m in (1, 4, 7, 10) if "2023-07" <= f"{y}-{m:02d}" <= "2026-04"]
ALL = R.load(("_r1", "_r2", "_r3", "_2025h2", ""))


def shift(m, k):
    y, mm = int(m[:4]), int(m[5:7]) + k
    y, mm = y + (mm - 1) // 12, (mm - 1) % 12 + 1
    return f"{y}-{mm:02d}"


def picker(rules):
    pts = lambda f: sum(c["баллы"] for rule in rules for c in rule if f.get(c["признак"]) is not None and
                        ((c["знак"] == ">" and f[c["признак"]] > c["порог"]) or (c["знак"] == "<" and f[c["признак"]] < c["порог"])))
    return lambda x: max(x["fe"], key=lambda L: (pts(x["fe"][L]), x["fe"][L]["p12"] if x["fe"][L]["p12"] is not None else -1e9))


def gain(rows, pick):
    return float(np.mean([(x["r"][x["pm"]] - x["r"][pick(x)]) * 100 for x in rows])) if rows else float("nan")


def derive(pool, seed):
    from llm_read import ask
    rules = []
    for j in range(3):
        rng = random.Random(seed * 10 + j)
        out = ask("qwen/qwen3.8-27b", R.prompt(rng.sample(pool, min(60, len(pool)))) + " /no_think", effort="none", provider="local") or ""
        m = re.findall(r"\[\s*\{.*?\}\s*\]", out, re.S)
        try:
            rule = [c for c in json.loads(m[-1]) if c.get("признак") in R.FEAT and c.get("знак") in (">", "<")]
            if rule:
                rules.append(rule)
        except Exception:  # noqa: BLE001
            pass
    return rules


def run():
    OUT.mkdir(exist_ok=True)
    active = None
    for k, q in enumerate(QS):
        f = OUT / f"квартал_{q}.json"
        if f.exists():
            active = json.loads(f.read_text())["действует"]
            continue
        pool = [x for x in ALL if x["m"] <= shift(q, -4)]   # исход справки за месяц m известен в конце m+3
        val = [x for x in ALL if shift(q, -6) <= x["m"] <= shift(q, -4)]
        new = derive(pool, k)
        g_new = gain(val, picker(new)) if new else float("nan")
        g_old = gain(val, picker(active)) if active else float("nan")
        if active and (not new or g_old >= g_new):
            choice, why = active, "оставлено действующее"
        else:
            choice, why = new, "заменено новым"
        best = max([v for v in (g_new, g_old) if v == v] or [float("nan")])
        on = not (best == best and best < 0)
        f.write_text(json.dumps({"квартал": q, "обучение_случаев": len(pool), "новое": new, "проверка_новое": g_new,
                                 "проверка_действующее": g_old, "решение": why, "сигнал_включён": on, "действует": choice},
                                ensure_ascii=False, indent=1))
        active = choice
        print(q, why, "новое", round(g_new, 2), "старое", round(g_old, 2) if g_old == g_old else "—", "включён" if on else "ВЫКЛ", flush=True)


def score():
    rng = np.random.default_rng(4)
    frozen = picker([r["правило"] for r in json.loads((ROOT / "правила_wf_A.json").read_text())])
    rows = []
    for k, q in enumerate(QS):
        d = json.loads((OUT / f"квартал_{q}.json").read_text())
        pa = picker(d["действует"])
        pool = [x for x in ALL if x["m"] <= shift(q, -4)]   # исход справки за месяц m известен в конце m+3
        lg = R.logit(pool)
        for x in [x for x in ALL if q <= x["m"] <= shift(q, 2)]:
            ag = pa(x) if d["сигнал_включён"] else x["pm"]
            rows.append({"m": x["m"], "q": q, "агент": (x["r"][x["pm"]] - x["r"][ag]) * 100,
                         "застывшее": (x["r"][x["pm"]] - x["r"][frozen(x)]) * 100,
                         "регрессия": (x["r"][x["pm"]] - x["r"][lg(x)]) * 100})
    from collections import defaultdict
    def boot(key):
        by = defaultdict(list)
        for r in rows:
            by[r["m"]].append(key(r))
        ms = list(by)
        bs = [np.mean([v for i in rng.integers(0, len(ms), len(ms)) for v in by[ms[i]]]) for _ in range(3000)]
        return f"{np.mean([key(r) for r in rows]):+.2f} [{np.percentile(bs, 2.5):+.2f}; {np.percentile(bs, 97.5):+.2f}]"
    print(f"продаж {len(rows)}, кварталов {len(QS)}")
    for k in ("агент", "застывшее", "регрессия"):
        print(f"  {k:<10} против управляющего {boot(lambda r: r[k])}")
    print(f"  агент минус застывшее (парно) {boot(lambda r: r['агент'] - r['застывшее'])}")
    for q in QS:
        v = [r["агент"] for r in rows if r["q"] == q]
        print(f"    {q}: агент {np.mean(v):+.2f} (n {len(v)})")


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else score()
