"""Проверка «справки к продаже»: умеет ли языковая модель выбрать, какую из 4 бумаг портфеля разумнее продать?
Случаи: реальные ВЫБОРОЧНЫЕ продажи активных фондов (в месяце продано ≤ 50% позиций — не механика оттока),
выходы и сокращения, с датой справки в 2026 г. (результат за 3 мес.
известен к сентябрю 2026 — после даты обучения большинства моделей). К проданной бумаге добавляются 3 случайные
бумаги того же портфеля, которые фонд в этом месяце не трогал. Бумаги ОБЕЗЛИЧЕНЫ («Бумага А–Г»): модель видит только
то, что было известно на дату решения — годовые МСФО с датой публикации ≤ даты решения (smart-lab), изменение цены
за 1/3/12 мес., дивиденды за 12 мес., долю в портфеле.
Истина: какая из 4 бумаг за 3 мес. после справки показала худшую полную доходность (её и надо было продать).
Сравнение: модель; реальный выбор управляющего (проданная бумага); простое правило без ИИ (продать бумагу с
наибольшим ростом за 12 мес. — самый сильный признак модели «плохой продажи»); случайный выбор — 25%.
python3 sellrank.py build [N]                 # собрать случаи -> данные/составы_фондов/sellrank_cases.json
python3 sellrank.py run <модель> <провайдер>  # прогон модели -> sellrank_answers.jsonl
SR_LIVE=1 SR_TAG=_live SR_FROM=2026-08 SR_TO=2026-08 python3 sellrank.py build 200   # живая проверка (исход после сдачи)
python3 sellrank.py score [файл.jsonl]        # итог; ответы по моделям: sellrank_answers_deepseek.jsonl, _qwen.jsonl;
                                              # период 2025 II: SR_TAG=_2025h2 python3 sellrank.py score sellrank_answers_2025h2_deepseek.jsonl
"""
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

import trades as T
from robust import INDEX, MECH, ROOT, UNSURE

import os
TAG = os.environ.get("SR_TAG", "")                         # другой период/набор: SR_TAG=_2025h2 SR_FROM=2025-07 SR_TO=2025-12
PERIOD = (os.environ.get("SR_FROM", "2026-01"), os.environ.get("SR_TO", "2026-06"))
CASES, ANS, FCACHE = ROOT / f"sellrank_cases{TAG}.json", ROOT / os.environ.get("SR_OUT", f"sellrank_answers{TAG}.jsonl"), ROOT / "smartlab_f"
LET = "АБВГ"
PROMPT = """Ты — аналитик управляющей компании. В портфеле фонда акций четыре позиции; нужно продать одну (освободить
деньги). Какую продажа обойдётся дешевле всего, то есть какая из четырёх бумаг, по-твоему, покажет ХУДШУЮ доходность
(с дивидендами) в следующие 3 месяца? Бумаги обезличены; даны только факты, известные на дату решения {date}.
Рассуди по существу (оценка, динамика прибыли, долг, дивиденды, перегретость цены), затем ответь последней строкой
строго в формате: ПРОДАТЬ: <буква>

{blocks}"""


def fund_table(secid):
    """Годовые МСФО со smart-lab с датами публикации (кэш на диске)."""
    FCACHE.mkdir(exist_ok=True)
    c = FCACHE / f"{secid}.json"
    if c.exists() and json.loads(c.read_text()):              # пустой кэш (разовый сбой сети) — скачать заново (ревью С7)
        return json.loads(c.read_text())
    from fetch import get
    t = (get(f"https://smart-lab.ru/q/{secid}/f/y/") or b"").decode("utf-8", "ignore")
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S):
        c2 = [re.sub(r"<[^>]+>|\s+|&nbsp;", " ", x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if len(c2) > 3:
            rows.append(c2)
    c.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def pit_block(secid, date):
    """Только годы, отчёт по которым опубликован не позже даты решения; без названия компании."""
    rows = fund_table(secid) or (fund_table(secid[:-1]) if secid.endswith("P") else [])   # префы — отчётность обыкновенных
    hdr = next((r for r in rows if any(re.fullmatch(r"20\d\d", x) for x in r)), None)
    dates = next((r for r in rows if r[0].startswith("Дата отчета")), None)
    if not hdr or not dates:
        return ""
    # строка лет в таблице smart-lab короче строк дат и значений на `off` ячеек (проверено: 1 во всех таблицах кэша);
    # до 28.09 столбцы брались по индексам строки лет, из-за чего последний год (подписанный «LTM» со сдвигом)
    # выпадал, а подпись «год публикации − 1» была неверна у отчитавшихся с опозданием (2022 г. — в 2024 г.)
    off = len(dates) - len(hdr)
    cols = [i for i in range(len(dates)) if 0 <= i - off < len(hdr) and re.fullmatch(r"20\d\d", hdr[i - off])
            and re.fullmatch(r"\d\d\.\d\d\.\d{4}", dates[i]) and f"{dates[i][6:]}-{dates[i][3:5]}-{dates[i][:2]}" <= date][-3:]
    keep = ("Выручка", "EBITDA", "Чистая прибыль ,", "FCF", "Чистый долг", "Див.выплата", "P/E", "EV/EBITDA", "P/B", "Долг/EBITDA", "ROE")
    out = ["год: " + " | ".join(hdr[i - off] for i in cols)] if cols else []
    for r in rows:
        if r[0].startswith(keep) and cols:
            out.append(r[0].split(",")[0] + ("," + r[0].split(",")[1] if "," in r[0] else "") + ": " + " | ".join(r[i] if i < len(r) else "" for i in cols))
    return "\n".join(out)


def block(x, d0, share, px, div):
    """Описание одной бумаги на дату решения d0 — только то, что было известно (общее для проверки и выпуска «Зеркала»)."""
    p1, p3, p12 = (T.past(x, d0, px, None, k) for k in (1, 3, 12))
    dv = sum(v for d, v in div[x] if T._eom(d0, -12) < d <= d0)
    yld = dv / px[d0][x] * 100 if px[d0].get(x) else None
    f_ = lambda v: "н/д" if v is None else f"{v * 100:+.1f}%"
    return (f"доля в портфеле {share * 100:.1f}%; цена за 1 мес. {f_(p1)}, за 3 мес. {f_(p3)}, "
            f"за 12 мес. {f_(p12)}; дивидендная доходность за 12 мес. {'н/д' if yld is None else f'{yld:.1f}%'}.\n"
            f"Годовая отчётность (МСФО, опубликована до даты решения):\n{pit_block(x, d0)}")


def build(n):
    hold, px, div = T.load()
    ev = [e for e in csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8"))
          if not any(k in e["фонд"] for k in INDEX + UNSURE + MECH) and e["сторона"] == "продажа"
          and PERIOD[0] <= e["d1"][:7] <= PERIOD[1]]
    nh, ns = {}, {}
    for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")):
        nh[(p["фонд"], p["дата"])] = nh.get((p["фонд"], p["дата"]), 0) + 1
    for e in ev:
        ns[(e["фонд"], e["d1"])] = ns.get((e["фонд"], e["d1"]), 0) + 1
    ev = [e for e in ev if ns[(e["фонд"], e["d1"])] <= 0.5 * nh.get((e["фонд"], e["d0"]), 1)]   # только выборочные решения
    rng = random.Random(7)
    rng.shuffle(ev)
    traded = {}
    for e in csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8")):
        traded.setdefault((e["фонд"], e["d1"]), set()).add(e["secid"])
    cases = []
    for e in ev:
        f, d0, d1, s = e["фонд"], e["d0"], e["d1"], e["secid"]
        h0 = hold.get((f, d0), {})
        alt = [x for x in h0 if x != s and x not in traded.get((f, d1), set())]
        fw = lambda x: T.fwd(x, d1, 3, px, div)
        live = os.environ.get("SR_LIVE")                        # живая проверка: исход ещё неизвестен
        alt = [x for x in alt if (live or fw(x) is not None) and px[d0].get(x) and pit_block(x, d0)]
        if (not live and fw(s) is None) or len(alt) < 3 or not pit_block(s, d0):
            continue
        group = [s] + rng.sample(alt, 3)
        rng.shuffle(group)
        v0 = {x: q * (px[d0].get(x) or 0) for x, q in h0.items()}
        tot = sum(v0.values()) or 1
        blocks, meta = [], []
        for L, x in zip(LET, group):
            blocks.append(f"Бумага {L}: " + block(x, d0, v0.get(x, 0) / tot, px, div))
            meta.append({"буква": L, "secid": x, "продана": x == s, "доходность_3м": fw(x), "рост_12м": T.past(x, d0, px, None, 12)})
        cases.append({"фонд": f, "d0": d0, "d1": d1, "вид": "выход" if e["полностью"] == "True" else "сокращение",
                      "текст": PROMPT.format(date=d0, blocks="\n\n".join(blocks)), "бумаги": meta})
        if len(cases) >= n:
            break
    if CASES.exists() and json.loads(CASES.read_text()):          # ответы моделей ссылаются на номера случаев
        sys.exit(f"{CASES.name} уже есть — не перезаписываю (удалите вручную, если нужно пересобрать)")
    CASES.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print("случаев:", len(cases), "; выходов:", sum(c["вид"] == "выход" for c in cases))


def run(model, provider, named=False):
    """named=True — контроль «памяти модели»: те же случаи, но с тикерами. Если так модель угадывает заметно лучше,
    она вспоминает будущее, и обезличенному результату верить нельзя."""
    from llm_read import ask
    global ANS
    if named:
        ANS = ROOT / f"sellrank_answers{TAG}_named.jsonl"
    cases = json.loads(CASES.read_text())
    done = {json.loads(l)["i"] for l in open(ANS, encoding="utf-8")} if ANS.exists() else set()
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    lock = Lock()

    def one(i):
        c = cases[i]
        text = c["текст"]
        if named:
            for b in c["бумаги"]:
                text = text.replace(f"Бумага {b['буква']}:", f"Бумага {b['буква']} (тикер Мосбиржи {b['secid']}):")
        try:
            out = ask(model, text, effort="low", provider=provider)
        except Exception as e:  # noqa: BLE001
            print("ошибка", i, str(e)[:80], flush=True)
            return
        m = re.findall(r"ПРОДАТЬ:\s*\**\s*([АБВГ])", out or "")
        with lock, open(ANS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"i": i, "модель": model, "ответ": m[-1] if m else None, "текст": (out or "")[-1500:]}, ensure_ascii=False) + "\n")
        print(i, m[-1] if m else "—", flush=True)

    todo = [i for i in range(len(cases)) if i not in done][: int(os.environ.get("SR_LIMIT", 10 ** 6))]
    with ThreadPoolExecutor(int(os.environ.get("SR_WORKERS", 16))) as ex:   # локальная модель — 1 поток
        list(ex.map(one, todo))


def score(path=None):
    cases = json.loads(CASES.read_text())
    ans = [json.loads(l) for l in open(path or ANS, encoding="utf-8")]
    rng = np.random.default_rng(3)

    def worst(c):
        return min(c["бумаги"], key=lambda b: b["доходность_3м"])["буква"]

    def excess(c, L):
        """Насколько выбранная бумага отстала от среднего по четырём (п.п.; плюс = продали худшее — хорошо)."""
        r = {b["буква"]: b["доходность_3м"] for b in c["бумаги"]}
        return (np.mean(list(r.values())) - r[L]) * 100

    rows = []
    for a in ans:
        c = cases[a["i"]]
        if a["ответ"] is None:
            continue
        pm = next(b["буква"] for b in c["бумаги"] if b["продана"])
        rule = max(c["бумаги"], key=lambda b: -1e9 if b["рост_12м"] is None else b["рост_12м"])["буква"]
        rows.append({"model": a["ответ"] == worst(c), "pm": pm == worst(c), "rule": rule == worst(c),
                     "e_model": excess(c, a["ответ"]), "e_pm": excess(c, pm), "e_rule": excess(c, rule), "вид": c["вид"]})
    n = len(rows)
    print(f"случаев с ответом: {n} (случайный выбор угадывает худшую бумагу в 25%)")
    for k, name in (("model", "модель"), ("pm", "управляющий (реальная продажа)"), ("rule", "правило: продать самое выросшее за 12 мес.")):
        hit = np.mean([r[k] for r in rows])
        e = np.array([r["e_" + k] for r in rows])
        b = [rng.choice(e, n).mean() for _ in range(3000)]
        print(f"{name:<44} угадал худшую {hit:.0%}; выигрыш против среднего из 4: {e.mean():+.2f} п.п. [{np.percentile(b, 2.5):+.2f}; {np.percentile(b, 97.5):+.2f}]")
    d = np.array([r["e_model"] - r["e_pm"] for r in rows])
    b = [rng.choice(d, n).mean() for _ in range(3000)]
    print(f"модель минус управляющий: {d.mean():+.2f} п.п. [{np.percentile(b, 2.5):+.2f}; {np.percentile(b, 97.5):+.2f}]")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "local", named="named" in sys.argv)
    else:                                            # score [файл ответов], напр. score sellrank_answers_qwen.jsonl
        f = next((a for a in sys.argv[2:] if a.endswith(".jsonl")), None)
        score(ROOT / f if f else (ROOT / f"sellrank_answers{TAG}_named.jsonl" if "named" in sys.argv else None))
