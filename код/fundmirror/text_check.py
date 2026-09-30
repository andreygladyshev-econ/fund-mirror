"""Проверка текста записки: знаки как в Word (с пробелами и без), слова, средняя длина предложения, индекс
удобочитаемости Флеша в адаптации Оборневой (FRE = 206,835 − 1,52·ASL − 65,14·ASW) и самые длинные предложения.
python3 text_check.py <файл.md>"""
import re
import sys

t = open(sys.argv[1], encoding="utf-8").read()
t = re.sub(r"!\[([^]|]*)(?:\|\d+)?\]\([^)]*\)", r"\1", t)                    # рисунки: подпись видна в документе и входит в объём
t = re.sub(r"^\|?[-| :]+\|?$", "", t, flags=re.M)                 # разделители таблиц
t = re.sub(r"[#*|>`_]", "", t)
lines = [l.strip() for l in t.splitlines() if l.strip()]
plain = " ".join(lines)
words = re.findall(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁё]+)?", plain)
sents = [s.strip() for s in re.split(r"(?<=[.!?:])\s+(?=[А-ЯA-Z0-9«])", plain) if len(s.split()) > 2]
syl = sum(len(re.findall(r"[аеёиоуыэюяaeiouy]", w.lower())) for w in words)
asl, asw = len(words) / len(sents), syl / len(words)
nospace = len(re.sub(r"\s", "", plain))
print(f"знаков с пробелами: {len(plain)}; без пробелов: {nospace}; слов: {len(words)}")
print(f"предложений: {len(sents)}; средняя длина: {asl:.1f} слова; слогов на слово: {asw:.2f}; "
      f"индекс Флеша–Оборневой: {206.835 - 1.52 * asl - 65.14 * asw:.0f} (выше — легче)")
print("самые длинные предложения:")
for s in sorted(sents, key=lambda s: -len(s.split()))[:6]:
    print(f"  [{len(s.split())}] {s[:140]}…")
