"""Случаи продаж с очищенными доходностями (разделы 4 и 6 записки, агент, месячный цикл).
Случай — выборочная продажа активного фонда: проданная бумага и три случайные бумаги того же портфеля, которые фонд в
этом месяце не трогал, с отчётностью у всех четырёх (sellrank.py build). Доходность каждой бумаги за 3 мес. после
справки очищается вычитанием средней доходности похожих бумаг (та же треть по обороту торгов и по росту за 12 мес.,
mirror_bench.dgtw); случай без похожих хотя бы для одной бумаги отбрасывается."""
import contextlib
import functools
import io
import json
import runpy
import sys
from pathlib import Path

import rulegen as R
from robust import ROOT

# файлы sellrank_cases<тег>.json: 10.2022–06.2023, 07.2023–06.2024, 07.2024–06.2025, 07–12.2025, 01–06.2026
TAGS = ("_r1", "_r2", "_r3", "_2025h2", "")
BIG = 7.0                  # крупное сокращение: частичная продажа позиции от 7% портфеля (раздел 7 записки)


@functools.lru_cache(maxsize=None)
def _dgtw():
    argv, sys.argv = sys.argv, ["mirror_bench.py", "0.5", "3"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(str(Path(__file__).with_name("mirror_bench.py")))["dgtw"]
    finally:
        sys.argv = argv


def load(tags=TAGS):
    dgtw = _dgtw()
    out = []
    for t in tags:
        raw = json.loads((ROOT / f"sellrank_cases{t}.json").read_text())
        for x, c in zip(R.load((t,)), raw):
            r = {}
            for b in c["бумаги"]:
                p = dgtw(b["secid"], c["d0"], c["d1"])
                if p is None:
                    break
                r[b["буква"]] = b["доходность_3м"] - p
            if len(r) < 4:
                continue
            share = x["fe"][x["pm"]]["доля"]
            out.append(dict(x, r=r, worst=min(r, key=r.get), фонд=c["фонд"], d0=c["d0"], d1=c["d1"], вид=c["вид"],
                            sold=next(b["secid"] for b in c["бумаги"] if b["продана"]),
                            крупное=c["вид"] == "сокращение" and share is not None and share >= BIG))
    return out
