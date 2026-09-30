"""Случаи продаж для проверки правила: реальная ВЫБОРОЧНАЯ продажа активного фонда (в месяце продано не больше половины
позиций — не механика оттока) и 3 случайные бумаги того же портфеля, которые фонд в этом месяце не трогал. Бумаги
ОБЕЗЛИЧЕНЫ («Бумага А–Г»): в описании только то, что было известно на дату решения, — годовые МСФО с датой публикации
не позже даты решения (smart-lab), изменение цены за 1/3/12 мес., дивиденды за 12 мес., доля в портфеле. Для каждой
бумаги сохраняется полная доходность за 3 мес. после справки (исход).
SR_TAG=_r1 SR_FROM=2022-10 SR_TO=2023-06 python3 sellrank.py build N   # собрать случаи -> sellrank_cases<тег>.json
python3 sellrank.py rebuild [тег …]   # пересобрать описания и исходы тех же бумаг на текущих данных (цены, дивиденды, МСФО)
"""
import csv
import json
import os
import random
import re
import sys

import trades as T
from robust import INDEX, MECH, ROOT, UNSURE

TAG = os.environ.get("SR_TAG", "")                         # другой период/набор: SR_TAG=_2025h2 SR_FROM=2025-07 SR_TO=2025-12
PERIOD = (os.environ.get("SR_FROM", "2026-01"), os.environ.get("SR_TO", "2026-06"))
CASES, FCACHE = ROOT / f"sellrank_cases{TAG}.json", ROOT / "smartlab_f"
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
    if c.exists() and json.loads(c.read_text()):              # пустой кэш (разовый сбой сети) — скачать заново
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
    # столбец года выбирается по дате публикации отчёта, поэтому отчитавшиеся с опозданием подписаны своим годом
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


def rebuild(tags):
    """Те же бумаги, даты и буквы; описания и исходы заново на текущих данных (например, после исправления дивидендов)."""
    hold, px, div = T.load()
    for tag in tags:
        f = ROOT / f"sellrank_cases{tag}.json"
        cases, out, changed = json.loads(f.read_text()), [], 0
        for c in cases:
            d0, d1 = c["d0"], c["d1"]
            h0 = hold.get((c["фонд"], d0), {})
            v0 = {x: q * (px[d0].get(x) or 0) for x, q in h0.items()}
            tot = sum(v0.values()) or 1
            blocks = [f"Бумага {b['буква']}: " + block(b["secid"], d0, v0.get(b["secid"], 0) / tot, px, div) for b in c["бумаги"]]
            meta = [dict(b, доходность_3м=T.fwd(b["secid"], d1, 3, px, div), рост_12м=T.past(b["secid"], d0, px, None, 12))
                    for b in c["бумаги"]]
            new = dict(c, текст=PROMPT.format(date=d0, blocks="\n\n".join(blocks)), бумаги=meta)
            changed += new != c
            out.append(new)
        f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"{tag or '(2026)'}: случаев {len(out)}, изменилось {changed}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif sys.argv[1] == "rebuild":
        rebuild(sys.argv[2:] or ["_r1", "_r2", "_r3", "_2025h2", ""])
    else:
        sys.exit(__doc__)
