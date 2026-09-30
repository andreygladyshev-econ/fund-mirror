"""Оценка сделок: доходность бумаги за H мес. после справки минус индекс МосБиржи (Б), минус свой портфель, взвешенный
по стоимости (В), и минус похожие бумаги (Х, характеристический бенчмарк по Daniel и др., 1997); контроль — индексные
фонды; блочный бутстрэп по месяцам. python3 mirror_bench.py [порог_выборочности=0.5] [группы: 3 | 5 | w] [H=3] — вывод в консоль"""
import csv
import sys
from collections import Counter, defaultdict

import numpy as np

import trades as T
from robust import INDEX, MECH, REDOM, ROOT, UNSURE

H = int(sys.argv[3]) if len(sys.argv) > 3 else 3                    # срок исхода, мес.
BENCH = ("Фонд Топ Российских акций - SBMX", "ВИМ – Индекс МосБиржи")
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
NB = sys.argv[2] if len(sys.argv) > 2 else "3"   # группы размера и инерции: 3 (основная), 5, или "w" — размер по весу в индексе
hold, px, div = T.load()
ev = list(csv.DictReader(open(ROOT / "сделки.csv", encoding="utf-8")))

# веса индекса по месяцу: средний состав двух индексных фондов по стоимости
bw = defaultdict(Counter)
for (f, d), h in hold.items():
    if f in BENCH and d in px:
        v = {x: q * (px[d].get(x) or 0) for x, q in h.items()}
        tot = sum(v.values())
        if tot:
            for x, val in v.items():
                bw[d[:7]][x] += val / tot
fw = {}
def fwd(x, d1):
    if (x, d1) not in fw:
        fw[(x, d1)] = T.fwd(x, d1, H, px, div)
    return fw[(x, d1)]
def wavg(w, d1):
    r = [(wt, fwd(x, d1)) for x, wt in w.items() if wt > 0]
    r = [(a, b) for a, b in r if b is not None]
    s = sum(a for a, _ in r)
    return sum(a * b for a, b in r) / s if s > 0.5 * sum(w.values()) else None     # покрыто ≥ половины веса
bench = {}

# характеристический бенчмарк (DGTW 1997): та же треть по ликвидности (средний оборот на 3 концах месяца) и по
# доходности за 12 мес.; доходность корзины — равновзвешенная, без самой бумаги
turn = defaultdict(dict)
for r_ in csv.DictReader(open(ROOT / "обороты.csv", encoding="utf-8")):
    turn[r_["дата"][:7]][r_["secid"]] = float(r_["оборот_руб"])
chars = {}
def buckets(d0):
    if d0 not in chars:
        ms = [T._eom(d0, -k)[:7] for k in range(3)]
        c = {}
        for x in px[d0]:
            t = [turn[m][x] for m in ms if x in turn[m]]
            p12 = T.past(x, d0, px, None, 12)
            if t and p12 is not None:
                c[x] = (np.mean(t), p12)
        if len(c) < 30:
            chars[d0] = None
        elif NB == "w":                          # размер по весу в индексе: ≥ 1,5% / в индексе / вне индекса
            w = bw.get(d0[:7], {})
            tw = sum(w.values()) or 1
            qm = np.percentile([v[1] for v in c.values()], [100 / 3, 200 / 3])
            chars[d0] = {x: (2 if w.get(x, 0) / tw >= 0.015 else 1 if w.get(x, 0) > 0 else 0, int(np.searchsorted(qm, v[1])))
                         for x, v in c.items()}
        else:
            nb = int(NB)
            cuts = [100 * k / nb for k in range(1, nb)]
            qt = np.percentile([v[0] for v in c.values()], cuts)
            qm = np.percentile([v[1] for v in c.values()], cuts)
            chars[d0] = {x: (int(np.searchsorted(qt, v[0])), int(np.searchsorted(qm, v[1]))) for x, v in c.items()}
    return chars[d0]
def dgtw(s, d0, d1):
    b = buckets(d0)
    if not b or s not in b:
        return None
    peers = [fwd(x, d1) for x, k in b.items() if k == b[s] and x != s]
    peers = [v for v in peers if v is not None]
    return float(np.mean(peers)) if len(peers) >= 5 else None

nh = Counter((p["фонд"], p["дата"]) for p in csv.DictReader(open(ROOT / "позиции.csv", encoding="utf-8")))
ns = Counter((e["фонд"], e["d1"], e["сторона"]) for e in ev)
rows = []
for e in ev:
    f, s, d0, d1 = e["фонд"], e["secid"], e["d0"], e["d1"]
    if s in REDOM or any(k in f for k in UNSURE + MECH) or ns[(f, d1, e["сторона"])] > TH * nh.get((f, d0), 1):
        continue
    r = fwd(s, d1)
    if r is None or d0[:7] not in bw:
        continue
    if d1 not in bench:
        bench[d1] = wavg(bw[d0[:7]], d1)
    h0 = hold.get((f, d0), {})
    own = wavg({x: q * (px[d0].get(x) or 0) for x, q in h0.items() if x != s}, d1)
    ch = dgtw(s, d0, d1)
    if bench[d1] is None or own is None or ch is None:
        continue
    w = float(e["вес"]) if e["вес"] not in ("", "None") else 0
    rows.append({"новая": e["новая"] == "True", "m": d1[:7], "индексный": any(k in f for k in INDEX), "сторона": e["сторона"], "фонд": f, "secid": s,
                 "вид": ("покупка" if e["сторона"] == "покупка" else "выход" if e["полностью"] == "True"
                         else "сокращение ≥ 7%" if w >= 0.07 else "сокращение < 7%"),
                 "Б": (r - bench[d1]) * 100, "В": (r - own) * 100, "Х": (r - ch) * 100, "руб": float(e["объём_руб"] or 0), "d0": d0})

ms_all = sorted({x["m"] for x in rows})
rng = np.random.default_rng(17)
IDX = [rng.integers(0, len(ms_all), -(-len(ms_all) // H)) for _ in range(2000)]   # начала блоков (кольцевые)


def bb(sub, col):
    """Среднее и 95% интервал: блочный бутстрэп по месяцам, блок = срок исхода."""
    by = defaultdict(list)
    for x in sub:
        by[x["m"]].append(x[col])
    if sum(map(len, by.values())) < 20:
        return None
    est = []
    for starts in IDX:
        v = [d for st in starts for k in range(H) for d in by.get(ms_all[(st + k) % len(ms_all)], [])]
        if v:
            est.append(np.mean(v))
    allv = [d for vs in by.values() for d in vs]
    return {"D": float(np.mean(allv)), "lo": float(np.percentile(est, 2.5)), "hi": float(np.percentile(est, 97.5)), "n": len(allv)}


fmt = lambda b: "мало" if b is None else f"{b['D']:+.2f} [{b['lo']:+.2f}; {b['hi']:+.2f}] n={b['n']}"
res = {}
print(f"порог выборочности {TH:.0%}; Б — против индекса МосБиржи; Х — против бумаг той же ликвидности и инерции (DGTW)")
for who, flag in (("активные", False), ("индексные (плацебо)", True)):
    print(f"\n{who}")
    for kind in ("покупка", "новая позиция", "докупка", "продажа", "выход", "сокращение < 7%", "сокращение ≥ 7%"):
        sub = [x for x in rows if x["индексный"] == flag and (
            x["сторона"] == kind if kind in ("покупка", "продажа") else
            (x["сторона"] == "покупка" and x["новая"] == (kind == "новая позиция")) if kind in ("новая позиция", "докупка") else x["вид"] == kind)]
        b, v, h = bb(sub, "Б"), bb(sub, "В"), bb(sub, "Х")
        res[f"{who} | {kind}"] = {"Б": b, "В": v, "Х": h}
        print(f"  {kind:<18} Б: {fmt(b):<34} Х: {fmt(h)}")
# асимметрия покупки − продажи у активных (парный блочный бутстрэп)
act = [x for x in rows if not x["индексный"]]
def diff(a, b, col="Х"):
    """Разность средних двух подвыборок; блочный бутстрэп по тем же блокам месяцев."""
    A, B = defaultdict(list), defaultdict(list)
    for x in a:
        A[x["m"]].append(x[col])
    for x in b:
        B[x["m"]].append(x[col])
    full = np.mean([v for vs in A.values() for v in vs]) - np.mean([v for vs in B.values() for v in vs])
    est = []
    for starts in IDX:
        ms = [ms_all[(st + k) % len(ms_all)] for st in starts for k in range(H)]
        va, vb = [v for m in ms for v in A.get(m, [])], [v for m in ms for v in B.get(m, [])]
        if va and vb:
            est.append(np.mean(va) - np.mean(vb))
    return [float(full), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


for who, flag in (("активные", False), ("индексные", True)):
    sub = [x for x in rows if x["индексный"] == flag]
    d = diff([x for x in sub if x["сторона"] == "покупка"], [x for x in sub if x["сторона"] == "продажа" and x["вид"] != "сокращение ≥ 7%"])
    res[f"купленные − проданные без сокращений ≥ 7% | {who}"] = d
    print(f"купленные − проданные (без сокращений ≥ 7%), {who}: {d[0]:+.2f} [{d[1]:+.2f}; {d[2]:+.2f}]")
d = diff([x for x in rows if not x["индексный"] and x["сторона"] == "покупка"], [x for x in rows if x["индексный"] and x["сторона"] == "покупка"])
res["покупки: активные − индексные"] = d
print(f"покупки, активные − индексные: {d[0]:+.2f} [{d[1]:+.2f}; {d[2]:+.2f}]")
COL = "Х"
by = defaultdict(lambda: ([], []))
for x in act:
    by[x["m"]][0 if x["сторона"] == "покупка" else 1].append(x[COL])
def gap(starts):
    b = [d for st in starts for k in range(H) for d in by.get(ms_all[(st + k) % len(ms_all)], ([], []))[0]]
    s = [d for st in starts for k in range(H) for d in by.get(ms_all[(st + k) % len(ms_all)], ([], []))[1]]
    return np.mean(b) - np.mean(s)
g = [gap(st) for st in IDX]
full = np.mean([d for v in by.values() for d in v[0]]) - np.mean([d for v in by.values() for d in v[1]])
print(f"\nасимметрия у активных (покупки − продажи), мерка {COL}: {full:+.2f} [{np.percentile(g, 2.5):+.2f}; {np.percentile(g, 97.5):+.2f}]")
res["асимметрия_Х"] = [float(full), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]
