"""Точность ИИ-читателя справок против разбора правилами (эталон) + двойная сверка (сумма и количество × цена).
Предрег. 28.09. python3 reader_accuracy.py run   -> данные/составы_фондов/читатель/*.json (локальная модель)
               python3 reader_accuracy.py score"""
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict

from llm_read import PROMPT, ROOT, ask, text_of
from parse_scha import parse, totals

OUT = ROOT / "читатель"
MODEL = "qwen/qwen3.8-27b"
LAT = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")
norm = lambda s: re.sub(r"[\s\-]", "", str(s or "")).upper().translate(LAT)


def sample():
    ok = [r for r in csv.DictReader(open(ROOT / "проверка.csv", encoding="utf-8")) if r["ок"] == "True" and r["источник"] == "правила"]
    inv = {(r["фонд"], r["дата"]): r["файл"] for r in csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8"))}
    by = defaultdict(list)
    for r in ok:
        by[r["ук"]].append(r)
    rng = random.Random(31)
    return [(r["ук"], r["фонд"], r["дата"], ROOT / inv[(r["фонд"], r["дата"])]) for uk in sorted(by) for r in rng.sample(by[uk], min(3, len(by[uk])))]


def run():
    OUT.mkdir(exist_ok=True)
    for uk, f, d, p in sample():
        c = OUT / (hashlib.md5(str(p).encode()).hexdigest() + ".json")
        if c.exists():
            continue
        t0 = time.time()
        try:
            out = ask(MODEL, PROMPT.format(text=text_of(p)) + "\n/no_think", effort="none", provider="local")
            m = re.search(r"\[.*\]", out, re.S)
            rows = json.loads(m.group(0)) if m else []
        except Exception as e:  # noqa: BLE001
            print("ошибка", uk, f, d, str(e)[:80], flush=True)
            continue
        c.write_text(json.dumps({"ук": uk, "фонд": f, "дата": d, "файл": str(p.relative_to(ROOT)), "позиции": rows,
                                 "секунд": round(time.time() - t0)}, ensure_ascii=False))
        print(f"{uk:<22} {d} {len(rows):3d} поз. {time.time() - t0:5.0f} с", flush=True)


def score():
    tick = {norm(r["reg"]): r["secid"] for r in csv.DictReader(open(ROOT / "тикеры.csv", encoding="utf-8")) if r["secid"]}
    px = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "цены.csv", encoding="utf-8")):
        px[r["дата"]][r["secid"]] = float(r["close"])
    S = defaultdict(float)
    per_uk = defaultdict(lambda: [0, 0])
    for c in sorted(OUT.glob("*.json")):
        a = json.loads(c.read_text())
        p = ROOT / a["файл"]
        truth = {}
        for x in parse(p):
            for k in (norm(x["reg"]), norm(x["isin"])):
                if k:
                    truth[k] = x
        uniq = {id(x): x for x in truth.values()}
        model = [x for x in a["позиции"] if isinstance(x, dict)]
        tot = totals(p)["shares"]
        msum = sum(float(x.get("value") or 0) for x in model)
        sum_ok = bool(tot) and abs(msum - tot) <= 0.005 * tot
        S["справок"] += 1
        S["сверка суммы прошла"] += sum_ok
        found = {}
        for x in model:
            t = truth.get(norm(x.get("reg")))
            if t is None:
                S["лишних позиций"] += 1
                continue
            found[id(t)] = (x, t)
        S["позиций эталона"] += len(uniq)
        S["найдено"] += len(found)
        S["позиций модели"] += len(model)
        bad_doc = not all(id(t) in found for t in uniq.values())
        for x, t in found.values():
            q_ok = abs(float(x.get("qty") or 0) - t["qty"]) <= 0.001 * max(t["qty"], 1)
            v_ok = abs(float(x.get("value") or 0) - t["value"]) <= 0.005 * max(t["value"], 1)
            S["верно количество"] += q_ok
            S["верна стоимость"] += v_ok
            err = not (q_ok and v_ok)
            bad_doc |= err
            s = tick.get(norm(t["reg"])) or tick.get(norm(t["isin"]))
            pr = px.get(a["дата"], {}).get(s) if s else None
            if pr:
                flag = abs(float(x.get("qty") or 0) * pr - float(x.get("value") or 0)) > 0.10 * max(float(x.get("value") or 0), 1)
                S["позиций с ценой"] += 1
                S["ошибок с ценой"] += err
                S["ошибка поймана сверкой количества"] += err and flag
                S["ложная тревога сверки количества"] += (not err) and flag
        S["справок с ошибкой"] += bad_doc
        S["справка с ошибкой прошла сверку суммы"] += bad_doc and sum_ok
        S["секунд"] += a["секунд"]
        per_uk[a["ук"]][0] += 1
        per_uk[a["ук"]][1] += not bad_doc
    n = S["справок"]
    print(f"справок {n:.0f} (УК {len(per_uk)}); среднее время {S['секунд'] / max(n, 1):.0f} с на справку")
    print(f"полнота {S['найдено'] / S['позиций эталона']:.1%} ({S['найдено']:.0f} из {S['позиций эталона']:.0f}); "
          f"точность {S['найдено'] / max(S['позиций модели'], 1):.1%} (лишних {S['лишних позиций']:.0f})")
    print(f"среди найденных: количество верно {S['верно количество'] / S['найдено']:.1%}, стоимость верна {S['верна стоимость'] / S['найдено']:.1%}")
    print(f"справок без единой ошибки {1 - S['справок с ошибкой'] / n:.1%}; прошли сверку суммы {S['сверка суммы прошла'] / n:.1%}; "
          f"справок с ошибкой, прошедших сверку суммы: {S['справка с ошибкой прошла сверку суммы']:.0f} из {S['справок с ошибкой']:.0f}")
    if S["ошибок с ценой"]:
        print(f"сверка количества × цена: поймала {S['ошибка поймана сверкой количества']:.0f} из {S['ошибок с ценой']:.0f} ошибочных позиций; "
              f"ложных тревог {S['ложная тревога сверки количества']:.0f} из {S['позиций с ценой'] - S['ошибок с ценой']:.0f} верных")
    print("по УК (справок без ошибок / всего):", {k: f"{v[1]}/{v[0]}" for k, v in sorted(per_uk.items())})


HARD = ROOT / "читатель_рсхб"


def hard_sample():
    inv = [r for r in csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8")) if r["ук"] == "РСХБ"]
    return random.Random(41).sample(inv, 15)


def hard_run():
    HARD.mkdir(exist_ok=True)
    for r in hard_sample():
        p = ROOT / r["файл"]
        c = HARD / (hashlib.md5(r["файл"].encode()).hexdigest() + ".json")
        if c.exists():
            continue
        t0 = time.time()
        try:
            out = ask(MODEL, PROMPT.format(text=text_of(p)) + "\n/no_think", effort="none", provider="local")
            m = re.search(r"\[.*\]", out, re.S)
            rows = json.loads(m.group(0)) if m else []
        except Exception as e:  # noqa: BLE001
            print("ошибка", r["фонд"], r["дата"], str(e)[:80], flush=True)
            continue
        c.write_text(json.dumps({"фонд": r["фонд"], "дата": r["дата"], "файл": r["файл"], "позиции": rows,
                                 "секунд": round(time.time() - t0)}, ensure_ascii=False))
        print(f"{r['дата']} {len(rows):3d} поз. {time.time() - t0:5.0f} с", flush=True)


def hard_score():
    from llm_read import CACHE, qty_flags
    ok = {"локальная": 0, "облачная": 0}
    agree = total = n = 0
    for c in sorted(HARD.glob("*.json")):
        a = json.loads(c.read_text())
        p = ROOT / a["файл"]
        tot = totals(p)["shares"]
        cloud_f = CACHE / c.name
        cloud = json.loads(cloud_f.read_text())["позиции"] if cloud_f.exists() else None
        n += 1
        line = [a["дата"]]
        for name, rows in (("локальная", a["позиции"]), ("облачная", cloud)):
            if rows is None:
                continue
            rows = [x for x in rows if isinstance(x, dict)]
            s = sum(float(x.get("value") or 0) for x in rows)
            sum_ok, fl = bool(tot) and abs(s - tot) <= 0.005 * tot, qty_flags(rows, a["дата"])
            ok[name] += sum_ok and not fl
            line.append(f"{name}: сумма {'ок' if sum_ok else f'{s / tot:.3f} от итога'}, позиций с ошибкой количества {len(fl)}")
        print("   " + " | ".join(line))
        if cloud:
            cl = {norm(x.get("reg")): x for x in cloud if isinstance(x, dict)}
            for x in (y for y in a["позиции"] if isinstance(y, dict)):
                total += 1
                y = cl.get(norm(x.get("reg")))
                agree += bool(y) and abs(float(x.get("qty") or 0) - float(y.get("qty") or 0)) <= 0.001 * max(float(y.get("qty") or 0), 1) \
                    and abs(float(x.get("value") or 0) - float(y.get("value") or 0)) <= 0.005 * max(float(y.get("value") or 0), 1)
    print(f"справок РСХБ {n}: прошли обе сверки — локальная {ok['локальная']}/{n}, облачная {ok['облачная']}/{n}; "
          f"позиции локальной, совпавшие с облачной: {agree}/{total}")


if __name__ == "__main__":
    {"run": run, "score": score, "hard": hard_run, "hard_score": hard_score}[sys.argv[1]]()
