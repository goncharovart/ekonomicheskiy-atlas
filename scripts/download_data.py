"""Скачивает исходные данные в data/raw и пишет data/raw/SHA256SUMS.

Запуск:  .venv/Scripts/python.exe -X utf8 scripts/download_data.py
         ... --list-rosstat 32   — оглавление раздела БДМО (какие индикаторы есть)

Источники:
1. Архив хакатона СберИндекса (расходы, доступность рынков, связи МО) — sberbank.com.
2. Справочник границ МО СберИндекса (rar: xlsx + gpkg) и PDF с метаданными — sberbank.com / sberbank.ru.
3. API СберИндекса (parquet): расходы по МО и индекс мобильности — sberindex.ru.
4. Росстат БДМО из каталога «Если быть точным» — только нужные csv из многогигабайтных zip,
   читаются по HTTP Range (весь архив не качаем).

Сертификаты: сайты Сбера подписаны «Russian Trusted Root CA» (Минцифры). Корень и промежуточный
сертификат лежат в certs/russian_trusted_ca.pem (скачаны с gu-st.ru, отпечаток корня SHA-256
D2:6D:2D:02:31:B7:C3:9F:92:CC:73:85:12:BA:54:10:35:19:E4:40:5D:68:B5:BD:70:3E:97:88:CA:8E:CF:31);
проверка TLS не отключается, к списку certifi добавляется этот корень.
sberindex.ru отдаёт только свой сертификат без промежуточного (LiteSSL RSA CA 2025 от TrustAsia,
корень TrustAsia есть в certifi). Промежуточный взят по адресу из поля AIA сертификата
(http://ica.litessl.com/LiteSSLRSACA2025.crt) и лежит в certs/litessl_rsa_ca_2025.pem.
Недоступные адреса пишутся в data/raw/FAILED.txt, скрипт идёт дальше.
"""
import gzip
import hashlib
import io
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SBER = {
    "hackathonlicence.zip": "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip",
    "t_dict_municipal.rar": "https://www.sberbank.com/common/files/t_dict_municipal.rar",
    "metadata_municipal_dict_sberindex_2.pdf": "https://www.sberbank.ru/common/img/uploaded/files/pdf/sberindex/metadata_municipal_dict_sberindex_2.pdf",
}
API = "https://sberindex.ru/api/dataset/v1/download/{}/parquet"
API_SLUGS = [
    "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij",  # расходы по МО по категориям
    "indeks-mobilnosti",  # индекс покупательской мобильности
]
ROSSTAT_ZIP = "https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/by_indicator/data_section{}_112_v20250918.zip"
ROSSTAT = {  # индикатор: (раздел, смысл)
    "Y48112027": (31, "численность населения на 1 января"),
    "Y48423005": (32, "среднесписочная численность работников по ОКВЭД2 (без МП)"),
    "Y48423007": (32, "среднемесячная зарплата работников по ОКВЭД2 (без МП)"),
}
ROSSTAT_YEARS = {"2021", "2022", "2023", "2024"}
LOWER_LEVEL = "Муниципальное образование нижнего уровня"  # поселения не нужны: справочник Сбера — только верхний уровень

failed = []


def ca_bundle() -> str:
    out = ROOT / "certs" / "ca_bundle.pem"
    extra = [p.read_bytes() for p in sorted((ROOT / "certs").glob("*.pem")) if p != out]
    out.write_bytes(b"\n".join([Path(certifi.where()).read_bytes(), *extra]))
    return str(out)


S = requests.Session()
S.verify = ca_bundle()
S.headers["User-Agent"] = "Mozilla/5.0 (sberindex-atlas data prep)"


def fetch(url: str, dest: Path, tries: int = 4) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print("есть", dest.name)
        return True
    for i in range(tries):
        try:
            with S.get(url, stream=True, timeout=(20, 300)) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
            print("скачано", dest.name, f"{dest.stat().st_size / 1e6:.1f} МБ")
            return True
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            print("  попытка", i + 1, url, err)
            time.sleep(3 * (i + 1))
    failed.append(f"{url}\t{err}")
    return False


def bsdtar() -> str:
    # tar из Git Bash — GNU, rar не умеет; bsdtar из Windows (libarchive) умеет zip и rar.
    return shutil.which("bsdtar") or r"C:\Windows\System32\tar.exe"


def unpack(archive: Path, to: Path):
    to.mkdir(parents=True, exist_ok=True)
    if any(to.iterdir()):
        return
    subprocess.run([bsdtar(), "-xf", str(archive), "-C", str(to)], check=True)
    print("распаковано", archive.name, "->", to.relative_to(ROOT))


class HttpRangeFile(io.RawIOBase):
    """Файл по HTTP Range: zipfile читает оглавление и нужные записи, не скачивая архив целиком."""

    BLOCK = 8 << 20

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(requests.head(url, timeout=60).headers["Content-Length"])
        self.cache = {}

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, n):
        if n not in self.cache:
            a = n * self.BLOCK
            b = min(a + self.BLOCK, self.size) - 1
            for i in range(8):
                try:
                    r = requests.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
                    r.raise_for_status()
                    break
                except Exception:
                    if i == 7:
                        raise
                    time.sleep(3 * (i + 1))
            self.cache = {n: r.content}  # ponytail: кэш на один блок, zipfile читает последовательно
        return self.cache[n]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            blk = self._block(self.pos // self.BLOCK)
            off = self.pos % self.BLOCK
            piece = blk[off:off + n]
            out += piece
            self.pos += len(piece)
            n -= len(piece)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def rosstat_zip(section) -> zipfile.ZipFile:
    return zipfile.ZipFile(HttpRangeFile(ROSSTAT_ZIP.format(section)))


def rosstat(code, section):
    dest = RAW / "rosstat" / f"{code}.csv.gz"
    if dest.exists():
        print("есть", dest.name)
        return
    dest.parent.mkdir(exist_ok=True)
    try:
        zf = rosstat_zip(section)
        name = next(n for n in zf.namelist() if n.endswith(f"data_{code}_112_v20250918.csv"))
        kept = 0
        tmp = dest.with_suffix(".part")
        # mtime=0: без времени в заголовке gzip файл при повторном скачивании совпадает байт в байт (SHA256SUMS)
        with zf.open(name) as src, io.TextIOWrapper(gzip.GzipFile(tmp, "wb", mtime=0), encoding="utf-8",
                                                     newline="") as fo:
            header = None
            for raw in io.TextIOWrapper(src, encoding="utf-8", errors="replace", newline=""):
                if header is None:
                    header = raw.rstrip("\r\n").split(";")
                    yi, li = header.index("year"), header.index("mun_level")
                    fo.write(raw)
                    continue
                f = raw.split(";")
                if len(f) > yi and f[yi] in ROSSTAT_YEARS and f[li] != LOWER_LEVEL:
                    fo.write(raw)
                    kept += 1
        tmp.replace(dest)
        print("Росстат", code, "строк", kept)
    except Exception as e:
        failed.append(f"rosstat {code} (раздел {section})\t{type(e).__name__}: {e}"[:300])
        print("Росстат", code, "ошибка", e)


def sha256sums():
    lines = []
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.name not in ("SHA256SUMS", "FAILED.txt") and not p.name.endswith(".part"):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {p.relative_to(RAW).as_posix()}")
    (RAW / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SHA256SUMS:", len(lines), "файлов")


def main():
    if "--list-rosstat" in sys.argv:
        zf = rosstat_zip(int(sys.argv[sys.argv.index("--list-rosstat") + 1]))
        for i in zf.infolist():
            print(i.filename, i.file_size)
        return
    (RAW / "api").mkdir(parents=True, exist_ok=True)
    # Сервер Сбера отдаёт ~150 КБ/с на соединение, поэтому качаем все файлы параллельно.
    jobs = [(url, RAW / name) for name, url in SBER.items()]
    jobs += [(API.format(s), RAW / "api" / f"{s}.parquet") for s in API_SLUGS]
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(fetch, *j) for j in jobs]
        futs += [ex.submit(rosstat, code, sec) for code, (sec, _) in ROSSTAT.items()]
        for f in futs:
            f.result()
    if (RAW / "hackathonlicence.zip").exists():
        unpack(RAW / "hackathonlicence.zip", RAW / "hackathon")
    if (RAW / "t_dict_municipal.rar").exists():
        unpack(RAW / "t_dict_municipal.rar", RAW / "borders")
    (RAW / "FAILED.txt").write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    if failed:
        print("НЕ СКАЧАНО:\n" + "\n".join(failed))
    sha256sums()


if __name__ == "__main__":
    main()
