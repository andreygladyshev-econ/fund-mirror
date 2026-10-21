"""Все числа проверки правила на одной выборке: продажи с отчетностью по четырем бумагам БЕЗ крупных сокращений
(частичная продажа позиции от 7% портфеля). Те же скрипты (solution_adj, figure_mirror, hetero_check, agent_mirror score),
но cases.load отбрасывает крупные сокращения; модель не вызывается (версии правила агента — сохраненные).
Старые файлы не трогаются: всё пишется в данные/.../чистая_выборка/.
python3 clean716.py"""
import functools
import json
import sys
import types
from pathlib import Path

import cases as C
from robust import ROOT

OUT = ROOT / "чистая_выборка"
OUT.mkdir(exist_ok=True)
_load = C.load


@functools.wraps(_load)
def load(*a, **k):
    return [x for x in _load(*a, **k) if not x["крупное"]]


C.load = load
HERE = Path(__file__).parent


def run(name, subs, as_module=None, main_args=None):
    src = (HERE / f"{name}.py").read_text()
    for a, b in subs:
        assert a in src, (name, a)
        src = src.replace(a, b)
    mod = types.ModuleType(as_module or f"_{name}")
    mod.__file__ = str(HERE / f"{name}.py")
    if as_module:
        sys.modules[as_module] = mod
    if main_args:
        sys.argv = [name] + main_args
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


fig = 'ROOT.parents[1] / "рисунки"'
run("figure_mirror", [(f'{fig} / "рис2_признаки.png"', 'OUT_C / "рис2_признаки.png"'),
                      (f'{fig} / "числа_признаки.json"', 'OUT_C / "числа_признаки.json"'),
                      ("from robust import ROOT", f"from robust import ROOT\nOUT_C = ROOT / 'чистая_выборка'")],
    as_module="figure_mirror")
run("solution_adj", [('OUT = ROOT.parents[1] / "рисунки"', 'OUT = ROOT / "чистая_выборка"')])
run("hetero_check", [('ROOT / "неоднородность.json"', 'ROOT / "чистая_выборка" / "неоднородность.json"')])
am = run("agent_mirror", [('(OUT / "итог.json")', '(ROOT / "чистая_выборка" / "агент_итог.json")'),
                          (f'{fig} / "агент_зеркало.png"', 'ROOT / "чистая_выборка" / "агент_зеркало.png"')])
am.score()
n = json.loads((OUT / "числа_правило.json").read_text())["правило без данных, все годы"]["n"]
assert n == 716, n   # self-check: та же выборка, что «без крупных сокращений» в старом расчете
print("OK, продаж:", n)
