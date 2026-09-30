"""Тексты случаев продаж на исправленной отчётности (ревью 2, Н5; К4): те же бумаги, даты, буквы и исходы, заново
собранные описания (pit_block после исправления столбцов). Старые файлы не трогаются: пишется sellrank_cases<тег>_fix.json.
python3 rebuild_cases.py"""
import json

import sellrank as S
import trades as T
from robust import ROOT

hold, px, div = T.load()
for tag in ("_r1", "_r2", "_r3", "_2025h2", ""):
    cases = json.loads((ROOT / f"sellrank_cases{tag}.json").read_text())
    out, changed = [], 0
    for c in cases:
        h0 = hold.get((c["фонд"], c["d0"]), {})
        v0 = {x: q * (px[c["d0"]].get(x) or 0) for x, q in h0.items()}
        tot = sum(v0.values()) or 1
        blocks = [f"Бумага {b['буква']}: " + S.block(b["secid"], c["d0"], v0.get(b["secid"], 0) / tot, px, div) for b in c["бумаги"]]
        text = S.PROMPT.format(date=c["d0"], blocks="\n\n".join(blocks))
        changed += text != c["текст"]
        out.append(dict(c, текст=text))
    (ROOT / f"sellrank_cases{tag}_fix.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{tag or '_2026'}: случаев {len(out)}, текст изменился у {changed}")
