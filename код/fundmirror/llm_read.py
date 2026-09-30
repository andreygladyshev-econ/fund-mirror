"""ИИ-читатель справок о СЧА: извлечение таблицы акций языковой моделью и две проверки её ответа.
Модель получает только участок справки с таблицей акций (pdftotext -layout или строки Excel) и возвращает JSON
[{"reg": "...", "qty": число, "value": число}]. Ответ принимается (parse_all.py), только если 1) сумма позиций совпала с
итогом строки 02.07 справки («акции российских АО — всего») с точностью 0,5% и 2) у каждой позиции количество × цена
Мосбиржи на дату справки ≈ стоимость (±10%). Не прошедшее проверку разбирает человек. Точность — reader_accuracy.py.
python3 llm_read.py batch <УК> [модель] [local|openrouter]   # все справки УК -> данные/составы_фондов/llm_positions/
В месячном цикле (mirror_cycle.py --fetch --llm) batch вызывается только для справок, не прошедших сверку шаблонов.
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from parse_scha import _rows, totals

ROOT = Path(__file__).resolve().parents[2] / "данные" / "составы_фондов"
PROMPT = """Ниже — фрагмент ежемесячной справки о стоимости чистых активов паевого фонда (форма 0420502), раздел с
акциями российских акционерных обществ. Выпиши ВСЕ позиции акций. Для каждой: государственный регистрационный номер
выпуска (например 1-02-00077-A или 10301481B), количество бумаг, стоимость в рублях. Числа — без пробелов, дробная
часть через точку. Облигации, депозиты и прочее не включай. Ответ — только JSON-массив
[{{"reg": "...", "qty": 0, "value": 0.0}}], без пояснений.

{text}"""


def text_of(path):
    """Только участок с расшифровкой акций: от первой до последней строки с ISIN/рег. номером акции (±12 строк)."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        t = subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout
    else:
        t = "\n".join(" | ".join(str(c) for c in r if c not in (None, "")) for r in _rows(p))
    L = t.splitlines()
    idx = [i for i, l in enumerate(L) if re.search(r"RU0|[12]-\d\d-\d{5}|[12]\d{7}B", l) and not re.search(r"4B0|облигац", l, re.I)]
    if not idx:
        return t[:20000]
    part = L[max(0, idx[0] - 12): idx[-1] + 12]
    return "\n".join(re.sub(r" {3,}", "  ", l) for l in part if re.search(r"\d", l))   # строки без цифр — обрывки названий


def ask(model, text, effort="low", provider="local"):
    """provider: local — LM Studio (порт 50255); openrouter — ключ из ~/.openrouter_key (в код и вывод не попадает)."""
    import urllib.request
    if provider == "openrouter":
        url, key = "https://openrouter.ai/api/v1/chat/completions", Path.home().joinpath(".openrouter_key").read_text().strip()
        body = {"model": model, "temperature": 0, "max_tokens": 16000, "messages": [{"role": "user", "content": text}]}
        if effort == "none":
            body["reasoning"] = {"enabled": False}         # извлечение таблицы не требует рассуждений — в разы быстрее
        elif effort in ("low", "medium", "high"):
            body["reasoning"] = {"effort": effort}
        hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    else:
        url, hdr = "http://localhost:50255/v1/chat/completions", {"Content-Type": "application/json"}
        body = {"model": model, "temperature": 0, "reasoning_effort": effort, "messages": [{"role": "user", "content": text}]}
    for _ in range(2):                                  # пустой ответ (всё ушло на рассуждения) — одна повторная попытка
        with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), hdr), timeout=1800) as r:
            out = json.load(r)["choices"][0]["message"].get("content")
        if out:
            return out
    return ""


CACHE = ROOT / "llm_positions"
_PX = {}


def qty_flags(rows, date, tol=0.10):
    """Вторая сверка ответа модели: количество × цена Мосбиржи на дату справки ≈ стоимость (± tol).
    Возвращает позиции, где расхождение больше допуска; позиции без биржевой цены не проверяются."""
    if not _PX:
        lat = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")
        _PX["norm"] = lambda v: re.sub(r"[\s\-]", "", str(v or "")).upper().translate(lat)
        _PX["tick"] = {_PX["norm"](r["reg"]): r["secid"] for r in csv.DictReader(open(ROOT / "тикеры.csv", encoding="utf-8")) if r["secid"]}
        px = {}
        for r in csv.DictReader(open(ROOT / "цены.csv", encoding="utf-8")):
            px.setdefault(r["дата"], {})[r["secid"]] = float(r["close"])
        _PX["px"] = px
    prices = _PX["px"].get(date, {})
    bad = []
    for x in rows:
        s = _PX["tick"].get(_PX["norm"](x.get("reg")))
        p, q, v = prices.get(s), float(x.get("qty") or 0), float(x.get("value") or 0)
        if p and v and abs(q * p - v) > tol * v:
            bad.append(x)
    return bad


def batch(uk, model, provider, only=None):
    """Все справки УК: модель извлекает позиции; ответ принимается, только если сумма совпала с итогом 02.07.
    Результат — данные/составы_фондов/llm_positions/<файл>.json (читает parse_all как запасной разбор)."""
    import hashlib
    CACHE.mkdir(exist_ok=True)
    inv = [r for r in csv.DictReader(open(ROOT / "опись.csv", encoding="utf-8")) if r["ук"] == uk and (only is None or r["файл"] in only)]
    ok_n = 0
    for r in inv:
        p = ROOT / r["файл"]
        c = CACHE / (hashlib.md5(r["файл"].encode()).hexdigest() + ".json")
        if c.exists():
            ok_n += json.loads(c.read_text())["сошлось"]
            continue
        tot = totals(p)["shares"]
        try:
            out = ask(model, PROMPT.format(text=text_of(p)), effort="none", provider=provider)
            m = re.search(r"\[.*\]", out, re.S)
            rows = json.loads(m.group(0)) if m else []
        except Exception as e:  # noqa: BLE001
            print("ошибка", r["фонд"], r["дата"], str(e)[:80], flush=True)
            continue
        s = sum(float(x.get("value") or 0) for x in rows)
        ok = bool(tot) and abs(s - tot) <= 0.005 * tot
        ok_n += ok
        c.write_text(json.dumps({"файл": r["файл"], "модель": model, "позиции": rows, "сумма": s, "итог_02_07": tot, "сошлось": ok}, ensure_ascii=False))
        print(f"{r['фонд'][:40]:<40} {r['дата']} {len(rows):3d} поз. {'✓' if ok else '✗'} ({s:,.0f} / {tot:,.0f})", flush=True)
    print(f"{uk}: сошлось {ok_n} из {len(inv)}")


if __name__ == "__main__":
    if sys.argv[1:2] != ["batch"] or len(sys.argv) < 3:
        sys.exit(__doc__)
    batch(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "qwen/qwen3.8-27b", sys.argv[4] if len(sys.argv) > 4 else "local")
