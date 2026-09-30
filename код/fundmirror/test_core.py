"""Минимальные проверки ключевой логики (без сети и моделей): python3 test_core.py"""
import rule as A
from trades import ex_date


def test_picker():
    rules = [[{"признак": "fcf_neg", "знак": ">", "порог": 0, "баллы": 3}, {"признак": "debt_ebitda", "знак": ">", "порог": 3, "баллы": 2}]]
    fe = {L: {"fcf_neg": 0.0, "debt_ebitda": 1.0, "p12": 0.0} for L in "АБВГ"}
    fe["В"]["fcf_neg"] = 1.0
    fe["Г"]["debt_ebitda"] = 4.0
    pick = A.picker(rules)
    assert pick({"fe": fe}) == "В"                       # 3 балла больше 2
    fe["В"]["fcf_neg"] = None                            # нет данных — условие не срабатывает
    assert pick({"fe": fe}) == "Г"
    fe["Г"]["debt_ebitda"] = 1.0                         # у всех 0 баллов: продаётся бумага с наибольшим ростом за 12 мес.
    fe["Б"]["p12"] = 40.0
    assert pick({"fe": fe}) == "Б"
    assert A.points(rules + rules, {"fcf_neg": 1, "debt_ebitda": 5}) == 10   # баллы версий правила складываются


def test_block_boot():
    pairs = [(f"2024-{m:02d}", float(m)) for m in range(1, 13) for _ in range(3)]
    d, lo, hi, n = A.block_boot(pairs)
    assert n == 36 and d == 6.5 and lo < d < hi
    assert A.block_boot(pairs) == (d, lo, hi, n)         # одна величина — один интервал в любом скрипте


def test_ex_date():
    assert ex_date("2023-06-05") == "2023-06-02"         # T+2: реестр в понедельник -> цена падает в пятницу
    assert ex_date("2023-07-03") == "2023-06-30"         # реестр 3 июля, цена падает 30 июня — дивиденд в окне июня
    assert ex_date("2024-07-18") == "2024-07-18"         # T+1: цена падает в день реестра


def test_pit_year_labels():
    from sellrank import pit_block
    b = pit_block("SBER", "2025-03-31")                  # отчёт за 2024 г. вышел 27.02.2025 — последний столбец 2024
    assert b.startswith("год: 2022 | 2023 | 2024") and "1 582" in b.split("\n")[1]
    b = pit_block("SBER", "2026-07-31")                  # отчёт за 2025 г. вышел 26.02.2026 — должен попасть
    assert b.startswith("год: 2023 | 2024 | 2025") and "1 707" in b.split("\n")[1]
    b = pit_block("NLMK", "2024-06-28")                  # за 2022 г. отчитались 15.04.2024: подпись 2022, а не 2023
    assert b.startswith("год: 2021 | 2022 | 2023"), b.split("\n")[0]


def test_key_rate():
    from agent_rate import rate
    assert (rate("2023-07"), rate("2023-10"), rate("2024-10"), rate("2025-01"), rate("2026-04")) == (7.5, 13.0, 19.0, 21.0, 15.0)


def test_cycle_reasons():
    import sys
    argv, sys.argv = sys.argv, ["mirror_cycle.py", "2026-08"]
    from mirror_cycle import why
    sys.argv = argv
    rule = [{"признак": "fcf_neg", "знак": ">", "порог": 0, "баллы": 3}, {"признак": "yld", "знак": "<", "порог": 6.9, "баллы": 2},
            {"признак": "debt_ebitda", "знак": ">", "порог": 1.81, "баллы": 1}, {"признак": "roe", "знак": "<", "порог": 10.1, "баллы": 1}]
    assert why({"fcf_neg": 1, "yld": 0.0, "debt_ebitda": 3.8, "roe": 12}, [rule]) == "денежный поток < 0 · нет дивидендов · долг 3,8 EBITDA"
    assert why({"fcf_neg": 0, "roe": -300.5}, [rule]) == "капитал < 0"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
