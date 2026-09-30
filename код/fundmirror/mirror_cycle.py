"""Ежемесячный цикл «Зеркала рынка»: данные → знания → решения (записка, раздел 5, рисунок 3).
python3 mirror_cycle.py ГГГГ-ММ [--fetch] [--llm] [--quarter]
  --fetch    докачать справки всех УК, прочитать их (шаблоны и ИИ-читатель с двумя проверками), обновить цены и сделки
  --llm      в квартальном контуре спросить ИИ-аналитика (локальная модель): записка комитету и новая версия правила
  --quarter  выполнить квартальный контур не в начале квартала
Выход: данные/зеркало/выпуски/ГГГГ-ММ/сводка.md, списки.md, списки.csv; состояние правила — зеркало_состояние.json"""
import contextlib
import csv
import functools
import io
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

import agent_mirror as M
import agent_replay as A
import rulegen as R
import sellrank as S
import trades as T
from robust import INDEX, MECH, ROOT, UNSURE

HERE = Path(__file__).parent
STATE = ROOT / "зеркало_состояние.json"
BASE = ("_r1_fix", "_r2_fix", "_r3_fix", "_2025h2_fix", "_fix")        # продажи 10.2022–06.2026 с известным исходом
UKS = ("tcap", "arsagera", "sistema", "vim", "alfa", "dohod", "psb", "akbars", "rshb")   # «Первая» и «Ингосстрах» — через браузер
MONTH = sys.argv[1]
MIN_SHARE = 0.005                                               # позиции меньше 0,5% портфеля денег не освобождают
OUT = ROOT.parent / "зеркало" / "выпуски" / MONTH
num = lambda v, k=1: f"{0.0 if abs(v) < 0.05 else v:+.{k}f}".replace(".", ",").replace("-", "−")


def run(*args, env=None):
    subprocess.run([sys.executable, *args], cwd=HERE, check=True, env=env)


def data():
    """Контур «данные»: новые справки → позиции → сделки; продажи, исход которых стал известен, — в зеркало."""
    if "--fetch" in sys.argv:
        for uk in UKS:
            subprocess.run([sys.executable, "fetch.py", uk], cwd=HERE)        # сбой одного сайта не останавливает цикл
        run("parse_all.py")                                                   # шаблоны; модель — из кэша llm_positions
        if "--llm" in sys.argv:                                               # ИИ-читатель: справки, не прошедшие сверку
            import hashlib
            from llm_read import batch
            inv = {(r["ук"], r["фонд"], r["дата"]): r["файл"] for r in csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8"))}
            bad = {}
            for r in csv.DictReader(open(ROOT / "проверка.csv", encoding="utf-8")):
                f = inv.get((r["ук"], r["фонд"], r["дата"]))
                if r["ок"] != "True" and f and not (ROOT / "llm_positions" / (hashlib.md5(f.encode()).hexdigest() + ".json")).exists():
                    bad.setdefault(r["ук"], set()).add(f)
            for uk, files in bad.items():
                batch(uk, "qwen/qwen3.8-27b", "local", only=files)
            run("parse_all.py")                                               # ответ модели принят, только если прошёл обе сверки
        for script in ("prices.py", "trades.py"):
            run(script)
    m_out = A.shift(MONTH, -3)                                 # исход продаж этого месяца известен к концу MONTH
    known = {x["m"] for t in BASE + tags_new() for x in R.load((t,))}
    if m_out in known or m_out <= "2026-06":
        return m_out, -1                                       # продажи этого месяца уже в зеркале
    import os
    env = dict(os.environ, SR_TAG=f"_m{m_out}", SR_FROM=m_out, SR_TO=m_out)
    run("sellrank.py", "build", "100000", env=env)
    return m_out, len(json.loads((ROOT / f"sellrank_cases_m{m_out}.json").read_text()))


def tags_new():
    return tuple(p.stem.replace("sellrank_cases", "") for p in sorted(ROOT.glob("sellrank_cases_m*.json")))


def mirror():
    sys.argv, argv = ["adj_walk.py"], sys.argv
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(str(HERE / "adj_walk.py"), run_name="x")
    sys.argv = argv
    return g["load"](BASE + tags_new(), True)


def knowledge(ALL):
    """Контур «знания»: сводка зеркала, записка ИИ-аналитика, новая версия правила, проверка по году продаж."""
    prior = [r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())]
    st = json.loads(STATE.read_text()) if STATE.exists() else {"правило": prior, "действует_с": "исходная версия",
                                                                 "включено": True, "история": []}
    year = [x for x in ALL if A.shift(MONTH, -15) <= x["m"] <= A.shift(MONTH, -4)]
    table, n = M.summary(ALL, MONTH)
    memo, new = "", []
    if "--llm" in sys.argv:
        from llm_read import ask
        for _ in range(3):
            out = ask("qwen/qwen3.8-27b", M.prompt(MONTH, table, n) + " /no_think", effort="none", provider="local") or ""
            memo = memo or out.split("```")[0].strip()
            m = re.findall(r"\[\s*\{.*?\}\s*\]", out, re.S)
            try:
                rule = [c for c in json.loads(m[-1]) if c.get("признак") in R.FEAT and c.get("знак") in (">", "<")]
                new += [rule] if rule else []
            except Exception:  # noqa: BLE001
                pass
    g_cur = A.gain(year, A.picker(st["правило"]))
    g_new = A.gain(year, A.picker(new)) if new else float("nan")
    changed = bool(new) and g_new > g_cur
    if changed:
        st["правило"], st["действует_с"] = new, MONTH
    st["включено"] = max(v for v in (g_cur, g_new) if v == v) >= 0
    st["история"].append({"месяц": MONTH, "продаж_за_год": len(year), "действующее": round(g_cur, 2),
                          "новое": None if g_new != g_new else round(g_new, 2), "заменено": changed,
                          "включено": st["включено"], "записка": memo, "новая_версия": new})
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return st, table, n, memo


LABEL = {"fcf_neg": lambda v, t: "денежный поток < 0", "np_neg": lambda v, t: "убыток",
         "debt_ebitda": lambda v, t: f"долг {v:.1f} EBITDA".replace(".", ","), "roe": lambda v, t: "капитал < 0" if v < -100 else f"ROE {v:.0f}%",
         "yld": lambda v, t: "нет дивидендов" if v == 0 else f"дивиденды {v:.1f}%".replace(".", ","), "np_g": lambda v, t: "прибыль падает",
         "pe": lambda v, t: f"P/E {v:.0f}", "pb": lambda v, t: f"P/B {v:.1f}".replace(".", ",")}


def why(fe, rules):
    """Какие условия действующего правила сработали — по убыванию баллов, без повторов."""
    conds = {}
    for rule in rules:
        for c in rule:
            v = fe.get(c["признак"])
            if v is not None and ((c["знак"] == ">" and v > c["порог"]) or (c["знак"] == "<" and v < c["порог"])):
                conds[c["признак"]] = max(conds.get(c["признак"], (0, v, c["порог"]))[0], c["баллы"]), v, c["порог"]
    order = sorted(conds.items(), key=lambda kv: -kv[1][0])
    return " · ".join(LABEL.get(f, lambda v, t: f"{f} {v:.1f}")(v, t) for f, (_, v, t) in order)


def decisions(st):
    """Контур «решения»: по каждому активному фонду — бумаги, упорядоченные по баллу действующего правила."""
    hold, px, div = T.load()
    S.fund_table = functools.lru_cache(maxsize=None)(S.fund_table)
    rules = st["правило"]
    pts = lambda fe: sum(c["баллы"] for rule in rules for c in rule if fe.get(c["признак"]) is not None and
                         ((c["знак"] == ">" and fe[c["признак"]] > c["порог"]) or (c["знак"] == "<" and fe[c["признак"]] < c["порог"])))
    lists = {}
    for (f, d), h in sorted(hold.items()):
        if not d.startswith(MONTH) or any(k in f for k in INDEX + UNSURE + MECH):
            continue
        val = {x: q * (px.get(d, {}).get(x) or 0) for x, q in h.items()}
        tot = sum(val.values()) or 1
        rows = []
        for x in h:
            if val[x] / tot >= MIN_SHARE and px.get(d, {}).get(x) and S.pit_block(x, d):
                fe = R.feats2("Бумага А: " + S.block(x, d, val[x] / tot, px, div))["А"]
                rows.append({"secid": x, "доля": round(val[x] / tot * 100, 2), "балл": round(pts(fe) / len(rules), 1),
                             "p12": fe.get("p12"), "долг_ebitda": fe.get("debt_ebitda"), "fcf_отр": fe.get("fcf_neg"),
                             "roe": fe.get("roe"), "рост_прибыли": fe.get("np_g"), "что_отмечает": why(fe, rules)})
        rows.sort(key=lambda r: (r["балл"], r["p12"] if r["p12"] is not None else -1e9), reverse=True)   # как picker
        if rows:
            lists[(f, d)] = rows
    return lists


def teams():
    """Покупки и продажи каждой УК за последний год с известным исходом против похожих бумаг (мерка раздела 3)."""
    sys.argv, argv = ["mirror_bench.py", "0.5", "3"], sys.argv
    with contextlib.redirect_stdout(io.StringIO()):
        rows = runpy.run_path(str(HERE / "mirror_bench.py"))["rows"]
    sys.argv = argv
    uk = {p["фонд"]: p["ук"] for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8"))}
    last = max(r["m"] for r in rows)
    lo = A.shift(last, -11)
    act = [r for r in rows if lo <= r["m"] <= last and not r["индексный"] and r["вид"] != "сокращение ≥ 7%"
           and not any(k in r["фонд"] for k in UNSURE + MECH)]
    out = {}
    for name in sorted({uk.get(r["фонд"], "?") for r in act}) + ["весь рынок"]:
        sub = act if name == "весь рынок" else [r for r in act if uk.get(r["фонд"]) == name]
        b = [r["Х"] for r in sub if r["сторона"] == "покупка"]
        s_ = [r["Х"] for r in sub if r["сторона"] == "продажа"]
        out[name] = (len(b), float(np.mean(b)) if b else None, len(s_), float(np.mean(s_)) if s_ else None)
    return lo, last, out


def report(m_out, added, st, table, n, memo, lists, ALL):
    OUT.mkdir(parents=True, exist_ok=True)
    year = [x for x in ALL if A.shift(MONTH, -15) <= x["m"] <= A.shift(MONTH, -4)]
    pk = A.picker(st["правило"])
    vs_rand = float(np.mean([(np.mean(list(x["r"].values())) - x["r"][pk(x)]) * 100 for x in year])) if year else float("nan")
    with open(OUT / "списки.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["фонд", "дата", "место", "secid", "доля_%", "балл", "из", "что_отмечает", "долг_ebitda", "fcf_отр", "roe",
                    "рост_прибыли", "рост_цены_12м"])
        top = np.mean([sum(c["баллы"] for c in rule if c["баллы"] > 0) for rule in st["правило"]])
        for (f, d), rows in lists.items():
            for k, r in enumerate(rows, 1):
                w.writerow([f, d, k, r["secid"], r["доля"], r["балл"], round(top, 1), r["что_отмечает"], r["долг_ebitda"], r["fcf_отр"],
                            r["roe"], r["рост_прибыли"], r["p12"]])
    md = [f"# Кандидаты на продажу · {MONTH}", "", "Позиции от 0,5% портфеля.",
          f"Правило действует с: {st['действует_с']}; подсказки {'включены' if st['включено'] else 'ВЫКЛЮЧЕНЫ'}. "
          "Когда нужно освободить деньги, продаётся бумага сверху списка, но не больше её позиции; решение за управляющим.", ""]
    for (f, d), rows in lists.items():
        md += [f"## {f} · {d}", "", "| Место | Бумага | Доля, % | Балл | Что отмечает правило |", "|---|---|---|---|---|"]
        fmt = lambda v: "н/д" if v is None else f"{v:.1f}".replace(".", ",")
        for k, r in enumerate(rows[:5], 1):
            md.append(f"| {k} | {r['secid']} | {fmt(r['доля'])} | {fmt(r['балл'])} | {r['что_отмечает'] or 'нет'} |")
        md += [f"Ещё {len(rows) - 5} бумаг: в списки.csv." if len(rows) > 5 else "", ""]
    (OUT / "списки.md").write_text("\n".join(md), encoding="utf-8")
    last = st["история"][-1] if st["история"] else {}
    note = {-1: f"0 (продажи {m_out} уже в зеркале)", 0: f"0 (исход продаж {m_out} ещё неизвестен)"}.get(added, str(added))
    sv = [f"# Зеркало рынка · выпуск {MONTH}", "",
          "## Данные", f"- Активных фондов со справкой за месяц: {len(lists)}.",
          f"- Продаж с известным исходом в зеркале: {len(ALL)} (по {max(x['m'] for x in ALL)}); добавлено в этом цикле: {note}.", "",
          "## Правило", f"- Действует с: {st['действует_с']}; подсказки {'включены' if st['включено'] else 'выключены'}.",
          f"- Квартальная проверка {last.get('месяц', 'не проводилась')} на продажах рынка за год ({last.get('продаж_за_год')}): "
          f"прежнее правило {num(last['действующее']) if last.get('действующее') is not None else 'н/д'} п.п. за квартал против выбора "
          f"управляющих, новая версия {num(last['новое']) if last.get('новое') is not None else 'не предлагалась'}; "
          f"заменено: {'да' if last.get('заменено') else 'нет'}.",
          f"- Действующее правило на том же году против случайной бумаги того же портфеля: {num(vs_rand)} п.п."
          + (" Правило выбрано на этом же году, поэтому оценка завышена; честная проверка идёт вперёд во времени."
             if st["действует_с"] == MONTH else ""), "",
          "## Какие признаки помогали выбрать, что продать", "Насколько бумага с самым выраженным признаком отставала от "
          f"случайной бумаги того же портфеля, п.п. (последний год; год до него), {n} продаж:", "", table, ""]
    if memo:
        sv += ["## Записка ИИ-аналитика", "", memo, ""]
    lo, hi, tm = teams()
    f1 = lambda v: "н/д" if v is None else num(v)
    sv += [f"## Покупки и продажи команд на фоне рынка ({lo}…{hi}, исход известен)", "",
           "Насколько бумага за 3 месяца после сделки выросла быстрее похожих, п.п.: для покупки плюс хорошо, для продажи "
           "плюс значит, что продали бумагу, которая потом обогнала похожие. Сокращения позиций от 7% не оцениваются. По "
           "одной компании оценка шумная (интервал 3–6 п.п.); точный отчёт строится по внутренним сделкам компании.", "",
           "| Компания | Покупок | Купленные против похожих | Продаж | Проданные против похожих |", "|---|---|---|---|---|"]
    sv += [f"| {k} | {v[0]} | {f1(v[1])} | {v[2]} | {f1(v[3])} |" for k, v in tm.items()] + [""]
    sv += ["## Списки кандидатов на продажу", "", "списки.md (первые 5 бумаг по каждому фонду), списки.csv (все бумаги)."]
    (OUT / "сводка.md").write_text("\n".join(sv), encoding="utf-8")


if __name__ == "__main__":
    m_out, added = data()
    ALL = mirror()
    quarter = MONTH[5:] in ("01", "04", "07", "10") or "--quarter" in sys.argv
    if quarter:
        st, table, n, memo = knowledge(ALL)
    else:
        prior = [r["правило"] for r in json.loads((ROOT / "правила_prior.json").read_text())]
        st = json.loads(STATE.read_text()) if STATE.exists() else {"правило": prior, "действует_с": "исходная версия",
                                                                     "включено": True, "история": []}
        (table, n), memo = M.summary(ALL, MONTH), ""
    lists = decisions(st)
    report(m_out, added, st, table, n, memo, lists, ALL)
    print(f"выпуск {MONTH}: фондов {len(lists)}; правило с {st['действует_с']}; подсказки {'вкл' if st['включено'] else 'ВЫКЛ'} -> {OUT}")
