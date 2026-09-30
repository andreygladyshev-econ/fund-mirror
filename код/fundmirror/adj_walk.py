"""Регрессия «только вперёд» на доходностях, очищенных от размера (ликвидности) и инерции (мерка Х, DGTW). Ревью К6.
python3 adj_walk.py"""
import json
import runpy
import sys
from collections import defaultdict

import numpy as np

import rulegen as R
from robust import ROOT

sys.argv = ["mirror_bench.py", "0.5"]
g = runpy.run_path("mirror_bench.py")
dgtw = g["dgtw"]


def load(tags, adj):
    out = []
    for t in tags:
        raw = json.loads((ROOT / f"sellrank_cases{t}.json").read_text())
        for x, c in zip(R.load((t,)), raw):
            if adj:
                r = {}
                for b in c["бумаги"]:
                    p = dgtw(b["secid"], c["d0"], c["d1"])
                    if p is None:
                        break
                    r[b["буква"]] = b["доходность_3м"] - p
                if len(r) < 4:
                    continue
                x = dict(x, r=r, worst=min(r, key=r.get), фонд=c["фонд"])
            out.append(x)
    return out


def bb(rows, pick, H=3, seed=4):
    by = defaultdict(list)
    for x in rows:
        by[x["m"]].append((x["r"][x["pm"]] - x["r"][pick(x)]) * 100)
    ms = sorted(by)
    rng = np.random.default_rng(seed)
    est = [np.mean([d for st in rng.integers(0, len(ms), -(-len(ms) // H)) for k in range(H) for d in by[ms[(st + k) % len(ms)]]])
           for _ in range(2000)]
    return np.mean([d for v in by.values() for d in v]), np.percentile(est, 2.5), np.percentile(est, 97.5), sum(map(len, by.values()))


for adj in (False, True):
    tr, te = load(("_r1",), adj), load(("_r2", "_r3", "_2025h2", ""), adj)
    lg = R.logit(tr)
    d, lo, hi, n = bb(te, lg)
    d2, lo2, hi2, n2 = bb([x for x in te if not x["m"].startswith("2026")], lg)
    print(f"{'очищенные (мерка Х)' if adj else 'сырые доходности   '}: регрессия против управляющего {d:+.2f} [{lo:+.2f}; {hi:+.2f}] n={n}; "
          f"без 2026 {d2:+.2f} [{lo2:+.2f}; {hi2:+.2f}]")
