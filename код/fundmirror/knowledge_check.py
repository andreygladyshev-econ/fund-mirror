"""Знают ли управляющие про финансовый стресс: где в своём портфеле по баллу правила стоят купленные и проданные бумаги.
Перцентиль балла бумаги среди всех позиций фонда на дату d0 (1 — самая «стрессовая», 0,5 — как случайно); интервал —
блочный бутстрэп по месяцам. python3 knowledge_check.py -> знание_стресса.json"""
import contextlib
import csv
import functools
import io
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import rule as A
import rulegen as R
import sellrank as S
import trades as T
from robust import ROOT, selective

S.fund_table = functools.lru_cache(maxsize=None)(S.fund_table)
hold, px, div = T.load()
rules = [r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())]   # исходное правило записки
pts = lambda f: A.points(rules, f)


@functools.lru_cache(maxsize=None)
def score(x, d0):
    if not px.get(d0, {}).get(x) or not S.pit_block(x, d0):
        return None
    return pts(R.feats2("Бумага А: " + S.block(x, d0, 0.0, px, div))["А"])


ev = selective(list(csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8"))))
# та же выборка, что в разделе 3 записки: сделки активных фондов с известным исходом, без сокращений позиций от 7%
sys.argv = ["mirror_bench.py", "0.5", "3"]
with contextlib.redirect_stdout(io.StringIO()):
    MB = runpy.run_path(str(Path(__file__).with_name("mirror_bench.py")))["rows"]
KEEP = {(x["фонд"], x["secid"], x["d0"], x["сторона"]) for x in MB if not x["индексный"] and x["вид"] != "сокращение ≥ 7%"}
ev = [e for e in ev if (e["фонд"], e["secid"], e["d0"], e["сторона"]) in KEEP]
res = defaultdict(list)
for e in ev:
    f, d0, s = e["фонд"], e["d0"], e["secid"]
    port = [x for x in hold.get((f, d0), {}) if x != s]
    sc = [score(x, d0) for x in port]
    sc = [v for v in sc if v is not None]
    me = score(s, d0)
    if me is None or len(sc) < 5:
        continue
    pct = (sum(v < me for v in sc) + 0.5 * sum(v == me for v in sc)) / len(sc)
    kind = "покупка" if e["сторона"] == "покупка" else ("выход" if e["полностью"] == "True" else "сокращение")
    res[(kind, d0[:4])].append((e["d1"][:7], pct))
    res[(kind, "все")].append((e["d1"][:7], pct))
out = {}
for kind in ("покупка", "сокращение", "выход"):
    for y in ("все", "2022", "2023", "2024", "2025", "2026"):
        v = res.get((kind, y), [])
        if len(v) < 20:
            continue
        pairs = [(m, x * 100) for m, x in v]                      # в процентах: block_boot округляет до сотых
        d, lo, hi, n = A.block_boot(pairs)
        out[f"{kind}|{y}"] = [round(d / 100, 3), round(lo / 100, 3), round(hi / 100, 3), n]
        print(f"{kind:<11} {y:<4} n={n:>5}  перцентиль «стрессовости» {d / 100:.3f} [{lo / 100:.3f}; {hi / 100:.3f}]  (0,5 — как случайно)")
(ROOT / "знание_стресса.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
