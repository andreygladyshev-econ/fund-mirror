"""Неоднородность по фондам и компаниям (30.09, по вопросу о корректности рисунка 2 и главного результата).
1) доля случаев у крупнейших компаний; 2) главный результат (исходное правило против выбора управляющего и против
жребия): бутстрэп по фондам, равные веса фондов, исключение каждой компании по очереди; 3) рисунок 2: меняет ли знак
значимая клетка, если убрать любую одну компанию.
python3 hetero_check.py -> неоднородность.json"""
import csv
import json
from collections import Counter, defaultdict

import numpy as np

import agent_rate as G
import agent_replay as A
from figure_mirror import PER, SIG, pick, res as HEAT
from robust import ROOT

ALL = G.load_all()
uk = {r["фонд"]: r["ук"] for r in csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8"))}
for x in ALL:
    x["ук"] = uk.get(x["фонд"], x["фонд"])
prior = A.picker([r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())])
g_man = lambda x: (x["r"][x["pm"]] - x["r"][prior(x)]) * 100
g_rnd = lambda x: (np.mean(list(x["r"].values())) - x["r"][prior(x)]) * 100
rng = np.random.default_rng(5)


def by_fund(g, rows):
    fs = defaultdict(list)
    for x in rows:
        fs[x["фонд"]].append(g(x))
    keys = list(fs)
    est = [np.mean([v for k in rng.choice(keys, len(keys)) for v in fs[k]]) for _ in range(2000)]
    return {"среднее": round(float(np.mean([g(x) for x in rows])), 2),
            "95% бутстрэп по фондам": [round(float(np.percentile(est, q)), 2) for q in (2.5, 97.5)],
            "равные веса фондов": round(float(np.mean([np.mean(v) for v in fs.values()])), 2)}


out = {"случаев": len(ALL), "фондов": len({x["фонд"] for x in ALL}),
       "доля компаний": {k: round(v / len(ALL), 3) for k, v in Counter(x["ук"] for x in ALL).most_common()}}
for name, g in (("правило против управляющего", g_man), ("правило против жребия", g_rnd)):
    out[name] = by_fund(g, ALL)
    out[name]["без одной компании: мин и макс"] = [round(f(float(np.mean([g(x) for x in ALL if x["ук"] != u])) for u in out["доля компаний"]), 2)
                                                   for f in (min, max)]
flips = []
for nm, f, s in SIG:
    for lo, hi, _ in PER:
        d, a, b, n = HEAT[f"{nm} | {lo}"]
        if d is None or not (a > 0 or b < 0):
            continue
        for u in out["доля компаний"]:
            v = [(np.mean(list(x["r"].values())) - x["r"][L]) * 100 for x in ALL
                 if x["ук"] != u and lo <= x["m"] <= hi and (L := pick(x, f, s))]
            if np.sign(np.mean(v)) != np.sign(d):
                flips.append(f"{nm} | {lo} без {u}: {np.mean(v):+.2f}")
out["рисунок 2: значимые клетки, меняющие знак без одной компании"] = flips
(ROOT / "неоднородность.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
