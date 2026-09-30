"""Признаки бумаг в случаях продаж и исходное правило «без данных».
Случай продажи (sellrank.py build) — текст с описанием четырёх обезличенных бумаг на дату решения; отсюда признаки
FEAT. Исходное правило написала языковая модель (Qwen 3.8 27B, локально в LM Studio) только из знаний: без данных,
без исходов и без указания периода (prompt_prior); три ответа модели лежат в правила_prior.json, баллы складываются.
python3 rulegen.py prior [K] [файл]   # K ответов модели -> данные/составы_фондов/<файл> (по умолчанию правила_prior_новые.json)
"""
import json
import re
import sys

from robust import ROOT

FEAT = {"доля": "доля в портфеле, %", "p1": "цена за 1 мес., %", "p3": "цена за 3 мес., %", "p12": "цена за 12 мес., %",
        "yld": "дивидендная доходность за 12 мес., %", "pe": "P/E", "pb": "P/B", "ev_ebitda": "EV/EBITDA",
        "debt_ebitda": "Долг/EBITDA", "roe": "ROE, %", "fcf_neg": "FCF отрицательный (1/0)", "np_neg": "убыток (1/0)",
        "rev_g": "рост выручки за год, %", "np_g": "рост чистой прибыли за год, %"}


def num(x):
    x = (x or "").replace(" ", "").replace("%", "").replace("−", "-")
    return float(x) if re.fullmatch(r"-?\d+(\.\d+)?", x) else None


def feats(text):
    """Базовые признаки каждой бумаги из текста случая (ровно то, что видела бы модель)."""
    out = {}
    for L, body in re.findall(r"Бумага ([АБВГ]): (.*?)(?=\n\nБумага [АБВГ]:|\Z)", text, re.S):
        num_ = lambda s: None if s is None else float(s.replace("−", "-"))
        g = lambda pat: (m := re.search(pat, body)) and m.group(1)
        last = lambda name: (m := re.search(rf"^{name}[^:\n]*:([^\n]*)", body, re.M)) and [x.strip() for x in m.group(1).split("|")][-1]
        f = lambda s: num_(s) if s and re.fullmatch(r"-?[\d.]+", s) else None
        out[L] = {"доля": num_(g(r"доля в портфеле ([\d.]+)%")), "p12": num_(g(r"за 12 мес\. ([+-][\d.]+)%")),
                  "p3": num_(g(r"за 3 мес\. ([+-][\d.]+)%")), "yld": num_(g(r"доходность за 12 мес\. ([\d.]+)%")),
                  "pe": f(last("P/E")), "fcf": f(last("FCF ")), "np": f(last("Чистая прибыль"))}
    return out


def feats2(text):
    base = feats(text)
    out = {}
    for L, body in re.findall(r"Бумага ([АБВГ]): (.*?)(?=\n\nБумага [АБВГ]:|\Z)", text, re.S):
        def series(name):
            m = re.search(rf"^{re.escape(name)}\s*[,:][^\n]*?:([^\n]*)$", body, re.M) or re.search(rf"^{re.escape(name)}:([^\n]*)$", body, re.M)
            return [num(v.strip()) for v in m.group(1).split("|")] if m else []
        last = lambda n: next((v for v in reversed(series(n)) if v is not None), None)
        def growth(n):
            s = [v for v in series(n) if v is not None]
            return (s[-1] / s[-2] - 1) * 100 if len(s) >= 2 and s[-2] > 0 else None
        b = base[L]
        out[L] = {"доля": b["доля"], "p1": b.get("p1"), "p3": b["p3"], "p12": b["p12"], "yld": b["yld"], "pe": b["pe"],
                  "pb": last("P/B") if last("P/B") is not None else last("P/BV"), "ev_ebitda": last("EV/EBITDA"),
                  "debt_ebitda": last("Долг/EBITDA"), "roe": last("ROE"),
                  "fcf_neg": None if b["fcf"] is None else float(b["fcf"] < 0), "np_neg": None if b["np"] is None else float(b["np"] < 0),
                  "rev_g": growth("Выручка"), "np_g": growth("Чистая прибыль")}
        m = re.search(r"цена за 1 мес\. ([+-][\d.]+)%", body)
        out[L]["p1"] = float(m.group(1)) if m else None
    return out


def load(tags):
    rows = []
    for t in tags:
        for c in json.loads((ROOT / f"sellrank_cases{t}.json").read_text()):
            r = {b["буква"]: b["доходность_3м"] for b in c["бумаги"]}
            rows.append({"tag": t, "m": c["d1"][:7], "fe": feats2(c["текст"]), "r": r, "text": c["текст"],
                         "pm": next(b["буква"] for b in c["бумаги"] if b["продана"]),
                         "worst": min(r, key=r.get)})
    return rows


def prompt_prior():
    """Без данных: правило только из знаний модели (проверка «не подогнано ли под наши данные»)."""
    feats_desc = "\n".join(f"- {k}: {v}" for k, v in FEAT.items())
    return f"""Ты — количественный аналитик управляющей компании (российский рынок акций). В портфеле фонда 4 бумаги
(обезличены), нужно продать одну — ту, что, вероятно, покажет ХУДШУЮ полную доходность в следующие 3 месяца. Данных
об исходах у тебя нет — опирайся только на знания о том, какие характеристики компаний предвещают слабую доходность.
Признаки, доступные на дату решения:
{feats_desc}

Предложи ПРОЗРАЧНОЕ правило: не больше 6 условий, каждое — признак, знак (">" или "<"), порог и баллы (от 1 до 3).
Продаётся бумага с наибольшей суммой баллов. Сначала коротко (3–6 предложений) объясни логику, затем выдай правило
последним блоком строго в формате JSON:
[{{"признак": "...", "знак": ">", "порог": 0, "баллы": 1}}, ...]"""


def gen(k, out_name="правила_prior_новые.json"):
    """K ответов модели на prompt_prior. Модель — через окружение RG_MODEL, RG_PROV (local | openrouter)."""
    import os
    from llm_read import ask
    model, prov = os.environ.get("RG_MODEL", "qwen/qwen3.8-27b"), os.environ.get("RG_PROV", "local")
    out_f = ROOT / out_name
    if out_name == "правила_prior.json":
        sys.exit("правила_prior.json — исходное правило записки; новые ответы пишутся в другой файл")
    rules = json.loads(out_f.read_text()) if out_f.exists() else []
    for _ in range(k):
        out = ask(model, prompt_prior() + (" /no_think" if prov == "local" else ""), effort="none" if prov == "local" else "low", provider=prov) or ""
        m = re.findall(r"\[\s*\{.*?\}\s*\]", out, re.S)
        try:
            rule = [c for c in json.loads(m[-1]) if c.get("признак") in FEAT and c.get("знак") in (">", "<")]
        except Exception:  # noqa: BLE001
            print("не разобрал ответ")
            continue
        rules.append({"id": len(rules), "модель": model, "правило": rule, "логика": out[: out.rfind("[")].strip()[-1500:]})
        out_f.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
        print("правило", len(rules) - 1, rule, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] != ["prior"]:
        sys.exit(__doc__)
    gen(int(sys.argv[2]) if len(sys.argv) > 2 else 3, *(sys.argv[3:4]))
