"""Диагностика прогона agent_rate (после результатов, не предрег.): откуда проигрыш агента неизменному правилу.
python3 agent_diag.py"""
import json
from collections import defaultdict
import numpy as np
import agent_rate as G, agent_replay as A
from robust import ROOT
ALL = G.load_all()
prior = A.picker([r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())])
D = {q: json.loads((G.OUT / f"квартал_{q}.json").read_text()) for q in A.QS}
def boot(rows, H=3, seed=4):
    by = defaultdict(list)
    for m, v in rows: by[m].append(v)
    ms = sorted(by); rng = np.random.default_rng(seed)
    est = [np.mean([v for st in rng.integers(0, len(ms), -(-len(ms)//H)) for k in range(H) for v in by[ms[(st+k) % len(ms)]]]) for _ in range(2000)]
    return round(float(np.mean([v for _, v in rows])), 2), round(float(np.percentile(est, 2.5)), 2), round(float(np.percentile(est, 97.5)), 2)
def run(pick_for_q):
    rows = []
    for q in A.QS:
        pk = pick_for_q(q)
        for x in [x for x in ALL if q <= x["m"] <= A.shift(q, 2)]:
            L = pk(x) if pk else x["pm"]
            rows.append((x["m"], (x["r"][x["pm"]] - x["r"][L]) * 100))
    return boot(rows)
print("агент как есть                 ", run(lambda q: A.picker(D[q]["действует"]) if D[q]["сигнал_включён"] else None))
print("агент без выключателя          ", run(lambda q: A.picker(D[q]["действует"])))
print("всегда новая версия квартала   ", run(lambda q: A.picker(D[q]["новое"])))
print("неизменное правило             ", run(lambda q: prior))
# выключатель по окну в год: неизменное правило против управляющего на исходах, известных к началу квартала (q-15..q-4)
for q in A.QS:
    w = [x for x in ALL if A.shift(q, -15) <= x["m"] <= A.shift(q, -4)]
    print(q, "неизменное на последнем году:", round(A.gain(w, prior), 2), "на квартале проверки агента:", round(A.gain([x for x in ALL if A.shift(q, -6) <= x["m"] <= A.shift(q, -4)], prior), 2))
