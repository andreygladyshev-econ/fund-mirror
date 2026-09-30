"""Сбор ежемесячных справок о стоимости чистых активов (форма 0420502) активных фондов акций с сайтов УК.
Опись: данные/составы_фондов/опись.csv (ук, фонд, дата, url, файл); файлы: данные/составы_фондов/raw/<ук>/<фонд>/<дата>.<ext>.
Повторный запуск докачивает только новое.

python3 fetch.py tcap     # Т-Капитал: список документов зашит в страницу /documents/mutual_funds/ (архив бессрочный)
python3 fetch.py sistema  # Система Капитал, «Доверительная – фонд Российские акции»: архив справок, Excel
python3 fetch.py vim      # ВИМ Инвестиции (wealthim.ru): фонды акций + «Индекс МосБиржи» (контроль), постранично
python3 fetch.py alfa     # Альфа-Капитал: /disclosure/pifs/<фонд>/monthly, список всех лет — во встроенных данных страницы
python3 fetch.py dohod    # ДОХОДЪ: активные «Первый эшелон», «Дивидендные акции», «Анализ акций» + индексные DIVD, GROD (контроль)
python3 fetch.py psb      # ПСБ (upravlyaem.ru): фонды акций, справки на странице фонда «Spravka-…-na-ДД.ММ.ГГГГ»
python3 fetch.py akbars   # АК Барс: /info/pif/<фонд>/, «Справка СЧА_ДД.ММ.ГГГГ-ДД.ММ.ГГГГ_…» (конец периода = дата)
python3 fetch.py rshb     # РСХБ: …/<фонд>/raskrytie-informatsii-<фонд>/, «…na-ДД.ММ.ГГГГ…» или «scha_xx_ММ_ГГГГ»
python3 fetch.py arsagera # Арсагера: «Справка СЧА (опубликована дд.мм.гггг)», ежемесячно с осени 2021, Excel
"""
import csv
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "данные" / "составы_фондов"
RAW = ROOT / "raw"
INV = ROOT / "опись.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128 Safari/537.36"}
CTX = ssl.create_default_context()                       # сертификат сайта проверяется
_INSECURE = ssl.create_default_context()
_INSECURE.check_hostname, _INSECURE.verify_mode = False, ssl.CERT_NONE
_UNVERIFIED = set()


def urlopen(req, timeout=120):
    """Запрос с проверкой сертификата. Часть сайтов УК отдаёт неполную цепочку (или сертификат российского УЦ, которого
    нет в системном хранилище): для такого сайта запрос повторяется без проверки, и это печатается."""
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=CTX)
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
        host = urllib.parse.urlsplit(req.full_url).hostname
        if host not in _UNVERIFIED:
            _UNVERIFIED.add(host)
            print(f"внимание: сертификат {host} не проверен ({e.reason.verify_message}); запрос без проверки", flush=True)
        return urllib.request.urlopen(req, timeout=timeout, context=_INSECURE)


def get(url, tries=3):
    for k in range(tries):
        try:
            return urlopen(urllib.request.Request(url, headers=UA)).read()
        except Exception:  # noqa: BLE001
            time.sleep(3 * (k + 1))
    return None


def date_of(name):
    m = re.search(r"(\d{2})[.\-](\d{2})[.\-](\d{4})", name) or re.search(r"(\d{2})(\d{2})(\d{4})", name)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def kind(data):
    """Тип файла по сигнатуре (у части сайтов в ссылке нет расширения)."""
    return "pdf" if data[:4] == b"%PDF" else "xlsx" if data[:2] == b"PK" else "xls" if data[:4] == b"\xd0\xcf\x11\xe0" else "bin"


def save(rows):
    """rows: (ук, фонд, дата, url). Скачивает недостающее и дописывает опись."""
    have = {r["url"] for r in csv.DictReader(open(INV, encoding="utf-8"))} if INV.exists() else set()
    new = INV.exists()
    with open(INV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if not new:
            w.writerow(["ук", "фонд", "дата", "url", "файл"])
        for uk, fund, d, url in rows:
            if url in have:
                continue
            data = get(url)
            if not data:
                print("не скачалось:", url, flush=True)
                continue
            path = RAW / uk / re.sub(r"[^\wа-яА-ЯёЁ\- ]", "", fund)[:60].strip() / f"{d}.{kind(data)}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            w.writerow([uk, fund, d, url, str(path.relative_to(ROOT))])
            fh.flush()
            print(uk, fund[:40], d, len(data), flush=True)
            time.sleep(0.5)


def tcap():
    import bisect
    t = get("https://t-capital-funds.ru/documents/mutual_funds/").decode("utf-8", "ignore")
    titles = [(m.start(), m.group(1)) for m in re.finditer(r'"(?:name|title)":"((?:ОПИФ|БПИФ|ИПИФ|ЗПИФ)[^"]{3,200})"', t)]
    pos = [p for p, _ in titles]
    want = ("Дивидендные акции", "Акции роста", "Пассивный Доход", "Трендовые акции", "второго эшелона", "Баланс")
    rows = []
    for m in re.finditer(r'"name":"([^"]{3,300})","link":"(https://cdn\.tbank\.ru/static/documents/[^"]+)"', t):
        k = bisect.bisect_right(pos, m.start()) - 1
        fund, name, url = titles[k][1] if k >= 0 else "", m.group(1), m.group(2)
        if not any(x in fund for x in want) or "Правил" in name or "правил" in name:
            continue
        if not re.search(r"Справка о стоимости (чистых )?активов|^СЧА ", name):
            continue
        d = date_of(name)
        if d:
            rows.append(("Т-Капитал", fund.replace("\\u0022", "").strip(), d, url))
    print("справок найдено:", len(rows))
    save(sorted(set(rows)))


def arsagera(pages=None):
    """Отчётная дата — конец месяца, предшествующего публикации (справку публикуют в середине следующего месяца)."""
    import datetime as dt
    pages = pages or {"Арсагера - фонд акций": "opifa_arsagera_-_fond_akcij/dokumenty_i_otchetnost_fonda",
             "Арсагера - акции 6.4": "ipifa_arsagera_-_akcii_64/dokumenty_i_otchetnost_fonda",
             "Арсагера - фонд смешанных инвестиций": "opifsi_arsagera_-_fond_smeshannyh_investicij/dokumenty_i_otchetnost_fonda"}
    rows = []
    for fund, path in pages.items():
        t = get(f"https://arsagera.ru/products/{path}/").decode("utf-8", "ignore")
        for href, pub in re.findall(r'href="(/library/download/\d+/?)"[^>]*>(?:\s*<[^>]+>)*\s*Справка СЧА\s*\(опубликована (\d\d\.\d\d\.\d{4})', t):
            d = dt.datetime.strptime(pub, "%d.%m.%Y").date().replace(day=1) - dt.timedelta(days=1)
            if d >= dt.date(2021, 9, 30):
                rows.append(("Арсагера", fund, d.isoformat(), "https://arsagera.ru" + href.rstrip("/") + "/"))
        print(fund, sum(r[1] == fund for r in rows), flush=True)
    save(sorted(set(rows)))


def sistema_date(name):
    """Дата справки из имени файла: 300425, 30-01-2026, na-31-03-21, 31032023g, 30.09.2021_29.10.2021 (период — берём конец),
    dekabr-2021."""
    n = name.rsplit("/", 1)[-1]
    if m := re.search(r"dekabr-(\d{4})", n):
        return f"{m.group(1)}-12-31"
    ms = list(re.finditer(r"(\d{2})[.\-]?(\d{2})[.\-]?(\d{4}|\d{2})(?!\d)", n))
    m = ms[-1] if ms else None
    if not m:
        return None
    y = m.group(3) if len(m.group(3)) == 4 else "20" + m.group(3)
    return f"{y}-{m.group(2)}-{m.group(1)}"


def sistema():
    t = get("https://sistema-capital.com/disclosure_old/opif-rynochnykh-finansovykh-instrumentov-sistema-kapital-rossiyskie-aktsii-arkhiv/").decode("utf-8", "ignore")
    t += get("https://sistema-capital.com/disclosure_old/opif-aktsiy-sistema-investitsii-fond-aktsiy/").decode("utf-8", "ignore")
    rows = []
    for l in set(re.findall(r'href="\s*([^"]*SCHA[^"]*\.xlsx?)\s*"', t, re.I)):
        if re.search(r"pravil|change|REZ-|_MA|-MA-", l, re.I):
            continue
        d = sistema_date(l)
        if d and d >= "2021-09-30":
            rows.append(("Система Капитал", "Доверительная – фонд Российские акции", d, urllib.parse.urljoin("https://sistema-capital.com/", l.strip())))
    print("справок:", len(rows))
    save(sorted(set(rows)))


VIM = {"wimfa": "ВИМ – Акции", "wimfeqr": "ВИМ – Акции. Российские эмитенты", "wimfns": "ВИМ – Нефтегазовый сектор",
       "wimfm": "ВИМ – Металлургия", "wimfeesg": "ВИМ – Перспективные размещения", "wimfe": "ВИМ – Инновационный",
       "wimfimb": "ВИМ – Индекс МосБиржи", "wimbalanced": "ВИМ – Сбалансированный. Российские эмитенты"}


def vim(funds=None):
    """Отчётность фонда — /documents/reports/ с фильтром дат и страницами PAGEN_1; дата справки — в имени файла
    (26_08_31_SCHA_…pdf = 31.08.2026)."""
    rows = []
    for slug, fund in (funds or VIM).items():
        seen = set()
        for page in range(1, 40):
            t = get(f"https://www.wealthim.ru/about/disclosure/pif/opif/{slug}/documents/reports/"
                    f"?date_from=01.01.2021&date_to=31.12.2026&PAGEN_1={page}")
            links = set(re.findall(r'href="(/upload/[^"]*?(\d{2})_(\d{2})_(\d{2})_SCHA[^"]*)"', (t or b"").decode("utf-8", "ignore"), re.I))
            if not links - seen:
                break
            seen |= links
        for href, y, m, d in seen:
            rows.append(("ВИМ", fund, f"20{y}-{m}-{d}", "https://www.wealthim.ru" + href))
        print(fund, len(seen), flush=True)
    save(sorted(set(rows)))


ALFA = {"opifa_akliq": "Альфа – Ликвидные акции", "opif_ak_growth": "Альфа – Акции компаний роста",
        "opifa_akn": "Альфа – Ресурсы", "opif_akstockip": "Альфа – Акции с выплатой дохода",
        "opif_ak_new_n": "Альфа – Новые имена", "opif_aks": "Альфа – Баланс"}
MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]


def alfa(funds=None):
    """Страница выводит только текущий год, но во встроенных данных (__SERVER_STATE__) — все документы с категорией
    monthly-<год>-month-<МЕСЯЦ>; отчётная дата — конец этого месяца."""
    import calendar
    rows = []
    for slug, fund in (funds or ALFA).items():
        t = (get(f"https://www.alfacapital.ru/disclosure/pifs/{slug}/monthly") or b"").decode("utf-8", "ignore")
        n = 0
        for fid, name, y, mon in re.findall(r'\{"id":(\d+),"name":"([^"]*)","isArchived":\w+,"path":"[^"]*","categoryId":"monthly-(\d{4})-month-([A-Z]+)"', t):
            if not name.startswith("Справка о стоимости чистых активов"):
                continue
            m = MONTHS.index(mon) + 1
            rows.append(("Альфа-Капитал", fund, f"{y}-{m:02d}-{calendar.monthrange(int(y), m)[1]:02d}",
                         f"https://www.alfacapital.ru/disclosure/file/{fid}"))
            n += 1
        print(fund, n, flush=True)
    save(sorted(set(rows)))


DOHOD = {"ra": "ДОХОДЪ – Российские акции. Первый эшелон", "di": "ДОХОДЪ – Дивидендные акции. Россия",
         "stocks": "ДОХОДЪ – Анализ акций", "divd": "ДОХОДЪ – Индекс дивидендных акций", "grod": "ДОХОДЪ – Индекс акций роста"}


def dohod():
    rows = []
    for slug, fund in DOHOD.items():
        t = (get(f"https://www.dohod.ru/information-disclosure/mutual-fund-reports/{slug}/monthly-reporting") or b"").decode("utf-8", "ignore")
        L = set(re.findall(r'href="(/assets/dist/upload/docs/scha[^"]*\.(?:pdf|xlsx?))"', t, re.I))
        for h in L:
            if d := date_of(h):
                rows.append(("ДОХОДЪ", fund, d, "https://www.dohod.ru" + h))
        print(fund, len(L), flush=True)
    save(sorted(set(rows)))


PSB = {"rossiyskie-aktsii": "ПСБ – Российские акции", "dividendnye-aktsii": "ПСБ – Дивидендные акции",
       "nedra-rossii": "ПСБ – Недра России", "oboronnyi": "ПСБ – Оборонный", "kapital-perspective": "ПСБ – Капитал перспектива",
       "promsvyaz-okno-vozmozhnostey": "ПСБ – Окно возможностей", "shares": "ПСБ – Акции", "balanced": "ПСБ – Сбалансированный"}


def psb():
    rows = []
    for slug, fund in PSB.items():
        t = (get(f"https://www.upravlyaem.ru/funds/{slug}/") or b"").decode("utf-8", "ignore")
        L = set(re.findall(r'href="(/upload/[^"]*Spravka-o-stoimosti-chistykh-aktivov[^"]*?(\d\d)\.(\d\d)\.(\d{4})[^"]*\.pdf)"', t))
        for h, d, m, y in L:
            rows.append(("ПСБ", fund, f"{y}-{m}-{d}", "https://www.upravlyaem.ru" + h))
        print(fund, len(L), flush=True)
    save(sorted(set(rows)))


AKB = {"stocks": "АК Барс – Акции", "maxwell": "АК Барс – Максвелл Капитал", "lale": "АК Барс – Лале",
       "index-mmvb": "АК Барс – Индекс МосБиржи"}


def akbars(funds=None):
    rows = []
    for slug, fund in (funds or AKB).items():
        t = (get(f"https://www.akbars-capital.ru/info/pif/{slug}/") or b"").decode("utf-8", "ignore")
        n = 0
        for h in set(re.findall(r'href="(/upload/[^"]+)"', t)):
            name = urllib.parse.unquote(h)
            m = re.search(r"Справка СЧА_\d\d\.\d\d\.\d{4}-(\d\d)\.(\d\d)\.(\d{4})", name)
            if m and f"{m.group(3)}-{m.group(2)}" >= "2021-09":
                rows.append(("АК Барс", fund, f"{m.group(3)}-{m.group(2)}-{m.group(1)}", "https://www.akbars-capital.ru" + urllib.parse.quote(name)))
                n += 1
        print(fund, n, flush=True)
    save(sorted(set(rows)))


RSHB = {"fond-aktsiy": "РСХБ – Фонд акций", "fond-kompaniy-maloy-i-sredney-kapitalizatsii": "РСХБ – Компании малой и средней капитализации",
        "luchshie-otrasli": "РСХБ – Лучшие отрасли", "novyy-impuls": "РСХБ – Новый импульс"}


def rshb():
    import calendar
    base = "https://www.rshb-am.ru/raskrytie-informacii/otchetnost-tarify-i-dokumenty-po-produktam/otkrytye-paevye-investitsionnye-fondy/"
    rows = []
    for slug, fund in RSHB.items():
        t = (get(f"{base}opif-rynochnykh-finansovykh-instrumentov-rskhb-{slug}/raskrytie-informatsii-rskhb-{slug}/") or b"").decode("utf-8", "ignore")
        n = 0
        for h in set(re.findall(r'href="(/upload/[^"]+\.(?:pdf|xlsx?))"', t, re.I)):
            name = urllib.parse.unquote(h).rsplit("/", 1)[-1]
            if re.search(r"prirost|priroste|pravil", name, re.I) or not re.search(r"scha|Spravka-o-stoimosti", name, re.I):
                continue
            if m := re.search(r"(\d\d)\.(\d\d)\.(\d{4})", name):
                d = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
            elif m := re.search(r"_(\d\d)_(\d{4})", name):
                y, mo = int(m.group(2)), int(m.group(1))
                d = f"{y}-{mo:02d}-{calendar.monthrange(y, mo)[1]:02d}"
            else:
                continue
            if d >= "2021-09-01":
                rows.append(("РСХБ", fund, d, "https://www.rshb-am.ru" + h))
                n += 1
        print(fund, n, flush=True)
    save(sorted(set(rows)))


BOND_ALFA = {"opif_akop": "Альфа – Облигации Плюс", "opif_akipo": "Альфа – Облигации с выплатой дохода",
             "opif_h-y-bonds": "Альфа – Высокодоходные облигации", "bpif-ctrlbonds": "Альфа – Управляемые облигации"}
BOND_VIM = {"wimfbr": "ВИМ – Облигации. Российские эмитенты", "wimor": "ВИМ – Облигации. Рантье",
            "wimfoesg": "ВИМ – Облигации. Ответственные инвестиции", "wimfk": "ВИМ – Казначейский",
            "wimfsi": "ВИМ – Сбалансированный", "wimsr": "ВИМ – Сбалансированный. Рантье"}
BOND_AKB = {"conservative": "АК Барс – Консервативный"}
BOND_ARS = {"Арсагера - фонд облигаций КР 1.55": "opifo_arsagera_fond_obligacij_kr_155/dokumenty_i_otchetnost"}


def bonds():
    """Облигационные фонды — в отдельную опись, чтобы не смешивать с конвейером акций."""
    global INV
    INV = ROOT / "опись_облигации.csv"
    alfa(BOND_ALFA)
    akbars(BOND_AKB)
    arsagera(BOND_ARS)
    vim(BOND_VIM)


if __name__ == "__main__":
    {"bonds": bonds, "tcap": tcap, "arsagera": arsagera, "sistema": sistema, "vim": vim, "alfa": alfa, "dohod": dohod, "psb": psb, "akbars": akbars, "rshb": rshb}[sys.argv[1]]()
