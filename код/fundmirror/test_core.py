"""Минимальные проверки ключевой логики (без сети и моделей): python3 test_core.py"""
import permute as P
import rulegen as R
from parse_bonds import ISIN, num


def test_permute_roundtrip():
    text = "Вопрос\n\nБумага А: доля 1\n\nБумага Б: доля 2\n\nБумага В: доля 3\n\nБумага Г: доля 4"
    perm = [2, 0, 3, 1]
    t = P.permuted(text, perm)
    assert "Бумага А: доля 3" in t and "Бумага Г: доля 2" in t
    # модель ответила «А» в переставленном тексте -> это исходная «В»
    assert P.LET[perm[P.LET.index("А")]] == "В"


def test_apply_rule():
    rule = [{"признак": "fcf_neg", "знак": ">", "порог": 0, "баллы": 3}, {"признак": "debt_ebitda", "знак": ">", "порог": 3, "баллы": 2}]
    fe = {L: {"fcf_neg": 0.0, "debt_ebitda": 1.0, "p12": 0.0} for L in "АБВГ"}
    fe["В"]["fcf_neg"] = 1.0
    fe["Г"]["debt_ebitda"] = 4.0
    assert R.apply_rule(rule, fe) == "В"                 # 3 балла больше 2
    fe["В"]["fcf_neg"] = None                            # нет данных — условие не срабатывает
    assert R.apply_rule(rule, fe) == "Г"


def test_bond_parsing():
    assert num("9 296 550,00") == 9296550.0 and num("7750004150") is None      # ИНН — не количество
    assert ISIN.match("RU000A10BPF3") and ISIN.match("SU26238RMFS4") and not ISIN.match("RU000A10")


def test_pit_year_labels():
    from sellrank import pit_block
    b = pit_block("SBER", "2025-03-31")                  # отчёт за 2024 г. вышел 27.02.2025 — последний столбец 2024
    assert b.startswith("год: 2022 | 2023 | 2024") and "1 582" in b.split("\n")[1]
    b = pit_block("SBER", "2026-07-31")                  # отчёт за 2025 г. вышел 26.02.2026 — должен попасть (ревью К4)
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
