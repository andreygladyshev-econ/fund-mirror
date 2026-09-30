"""Продолжают ли работать признаки слабой бумаги, которые работали последний год (план записан до расчёта, 30.09.2026).
Весь рынок: каждый конец месяца все бумаги из портфелей фондов с отчётностью, доходность за 3 мес. минус доходность
похожих бумаг. python3 signal_momentum.py panel | test"""
import contextlib
import functools
import io
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import agent_replay as A
import rulegen as R
import sellrank as S
from agent_mirror import SIG
from robust import ROOT

PANEL = ROOT / "панель_признаков.json"


def panel():
    sys.argv = ["mirror_bench.py", "0.5", "3"]
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(str(Path(__file__).with_name("mirror_bench.py")))
    fwd, dgtw, px, div, hold = g["fwd"], g["dgtw"], g["px"], g["div"], g["hold"]
    S.fund_table = functools.lru_cache(maxsize=None)(S.fund_table)
    universe = sorted({x for h in hold.values() for x in h})
    ends = sorted({d for d in px if "2022-07" <= d[:7] <= "2026-06" and d == max(k for k in px if k[:7] == d[:7])})
    out = {}
    for d in ends:
        rows = {}
        for x in universe:
            if not px[d].get(x) or not S.pit_block(x, d):
                continue
            r, b = fwd(x, d), dgtw(x, d, d)
            if r is None or b is None:
                continue
            f = R.feats2("Бумага А: " + S.block(x, d, 0.0, px, div))["А"]
            rows[x] = {"r": r - b, "f": f}
        out[d[:7]] = rows
        print(d, len(rows), flush=True)
    PANEL.write_text(json.dumps(out, ensure_ascii=False))


def weak_set(rows, f, s):
    v = {x: row["f"][f] for x, row in rows.items() if row["f"].get(f) is not None}
    if len(v) < 30:
        return None
    if set(v.values()) <= {0, 1, 0.0, 1.0}:
        w = {x for x, a in v.items() if a == 1}
    else:
        cut = np.quantile([s * a for a in v.values()], 2 / 3)
        w = {x for x, a in v.items() if s * a > cut}
    return (w, set(v)) if 5 <= len(w) <= len(v) - 5 else None


def usefulness(rows, weak, all_):
    rest = [rows[x]["r"] for x in all_ - weak]
    return (np.mean(rest) - np.mean([rows[x]["r"] for x in weak])) * 100


def test():
    P = json.loads(PANEL.read_text())
    months = sorted(P)
    use = {}                                                # (признак, месяц) -> польза
    for i, (f, s, name) in enumerate(SIG):
        for m in months:
            ws = weak_set(P[m], f, s)
            if ws:
                use[(i, m)] = usefulness(P[m], *ws)
    hist = {}
    for (i, m) in use:
        past = [use[(i, k)] for k in months if A.shift(m, -15) <= k <= A.shift(m, -4) and (i, k) in use]
        if len(past) >= 6:
            hist[(i, m)] = float(np.mean(past))
    prior_rules = [r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())]
    pts = lambda fe, rules: sum(c["баллы"] for rule in rules for c in rule if fe.get(c["признак"]) is not None and
                               ((c["знак"] == ">" and fe[c["признак"]] > c["порог"]) or (c["знак"] == "<" and fe[c["признак"]] < c["порог"])))

    def composite(rows, score):
        sc = {x: score(x) for x in rows}
        vals = np.array(list(sc.values()))
        cut = np.quantile(vals, 2 / 3)
        weak = {x for x, v in sc.items() if v > cut}
        if len(weak) < 5:
            weak = {x for x, v in sc.items() if v >= cut}
        return usefulness(rows, weak, set(rows)) if 5 <= len(weak) <= len(rows) - 5 else None
    by_month = defaultdict(dict)
    for m in months:
        on = [i for i in range(len(SIG)) if hist.get((i, m), -1) > 0]
        have = [i for i in range(len(SIG)) if (i, m) in hist]
        if not have:
            continue
        ws = {i: weak_set(P[m], SIG[i][0], SIG[i][1]) for i in on}
        adapt = composite(P[m], lambda x: sum(1 for i in on if ws[i] and x in ws[i][0])) if on else None
        fixed = composite(P[m], lambda x: pts(P[m][x]["f"], prior_rules))
        by_month[m] = {"pairs": [(hist[(i, m)], use[(i, m)]) for i in have],
                       "все": float(np.mean([use[(i, m)] for i in have])),
                       "включённые": float(np.mean([use[(i, m)] for i in on])) if on else None,
                       "подвижное": adapt, "исходное": fixed}
    ms = sorted(by_month)
    rng = np.random.default_rng(4)

    def boot(stat, H=3, B=2000):
        est = []
        for _ in range(B):
            pick = [ms[(st + k) % len(ms)] for st in rng.integers(0, len(ms), -(-len(ms) // H)) for k in range(H)]
            est.append(stat(pick))
        return round(float(stat(ms)), 3), round(float(np.percentile(est, 2.5)), 3), round(float(np.percentile(est, 97.5)), 3)

    def slope(sel):
        xy = np.array([p for m in sel for p in by_month[m]["pairs"]])
        x, y = xy[:, 0] - xy[:, 0].mean(), xy[:, 1]
        return float((x * y).sum() / (x * x).sum())
    diff = lambda a, b: lambda sel: np.mean([by_month[m][a] - by_month[m][b] for m in sel if by_month[m][a] is not None and by_month[m][b] is not None])
    res = {"месяцев": len(ms), "пар признак×месяц": sum(len(by_month[m]["pairs"]) for m in ms),
           "наклон пользы на историю": boot(slope),
           "включённые минус все признаки, п.п.": boot(diff("включённые", "все")),
           "подвижное минус исходное правило, п.п.": boot(diff("подвижное", "исходное")),
           "исходное правило, польза, п.п.": boot(lambda sel: np.mean([by_month[m]["исходное"] for m in sel if by_month[m]["исходное"] is not None])),
           "подвижное, польза, п.п.": boot(lambda sel: np.mean([by_month[m]["подвижное"] for m in sel if by_month[m]["подвижное"] is not None]))}
    (ROOT / "моментум_признаков.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    panel() if sys.argv[1] == "panel" else test()
