"""PDF отчёта и печатная версия лендинга через Chrome без окна (headless):

    .venv/Scripts/python.exe -X utf8 scripts/build_pdf.py [report|landing]   (без аргумента — оба)

  reports/ОТЧЁТ.md → reports/ОТЧЁТ.html (формулы — KaTeX) → reports/ОТЧЁТ.pdf, A4 с номерами страниц;
  site/index.html → site/landing.pdf (печатные стили — @media print в site/assets/style.css).
Нужен установленный Chrome или Chromium (путь можно задать переменной CHROME) и сеть:
KaTeX и шрифты берутся с CDN. Запускать после scripts/build_site.py.
"""
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/ОТЧЁТ.md"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/"
CSS = """
@page { size: A4; margin: 16mm 15mm 18mm;
  @bottom-center { content: counter(page) " / " counter(pages); font: 8.5pt 'IBM Plex Sans', sans-serif; color: #777; } }
body { font: 10.5pt/1.5 'PT Serif', Georgia, serif; color: #16171a; margin: 0; }
h1, h2, h3, h4 { font-family: 'IBM Plex Sans', Arial, sans-serif; font-weight: 600; line-height: 1.25;
  break-after: avoid; page-break-after: avoid; }
h1 { font-size: 19pt; margin: 0 0 8pt; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; padding-top: 6pt; border-top: 1.5pt solid #16171a; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; }
p, li { orphans: 3; widows: 3; }
a { color: #1d5fb4; text-decoration: none; }
hr { display: none; }
code { font: 8.6pt Consolas, 'DejaVu Sans Mono', monospace; background: #f1f0ec; padding: 0 2px; border-radius: 2px;
  overflow-wrap: anywhere; }
pre { background: #f1f0ec; padding: 7pt 9pt; border-radius: 3px; white-space: pre-wrap; break-inside: avoid;
  font-size: 8pt; line-height: 1.45; }
pre code { background: none; padding: 0; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font: 8.6pt/1.35 'IBM Plex Sans', Arial, sans-serif; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th { text-align: left; font-weight: 600; border-bottom: 1pt solid #16171a; padding: 3pt 5pt; vertical-align: bottom; }
td { border-bottom: 0.5pt solid #d6d4cc; padding: 3pt 5pt; vertical-align: top; }
blockquote { margin: 6pt 0; padding-left: 10pt; border-left: 2pt solid #c9c6bb; color: #444; }
.fn { font-size: 9pt; color: #444; }
.katex { font-size: 1.02em; }
.katex-display { margin: 6pt 0; overflow: hidden; }
"""


def md_to_html(text, title):
    """Markdown → HTML; формулы $…$ и $$…$$ прячутся от парсера и рендерятся KaTeX в браузере."""
    math = []

    def keep(m, display):
        math.append((m.group(1), display))
        return f"KTXMATH{len(math) - 1}KTX"

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: keep(m, True), text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$(?!\$)([^\n$]+?)(?<!\\)\$", lambda m: keep(m, False), text)
    notes = []  # сноски [^x]: номер по первой ссылке; текст сноски остаётся под своим абзацем

    def note(m):
        if m.group(1) not in notes:
            notes.append(m.group(1))
        return f"KTXNOTE{notes.index(m.group(1)) + 1}KTX"

    text = re.sub(r"\[\^([^\]]+)\]:?", note, text)
    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(text)
    body = re.sub(r"<p>KTXNOTE(\d+)KTX", r'<p class="fn"><sup>\1</sup>', body)
    body = re.sub(r"KTXNOTE(\d+)KTX", r"<sup>\1</sup>", body)
    body = re.sub(r"KTXMATH(\d+)KTX", lambda m: '<span class="m" data-d="{}">{}</span>'.format(
        int(math[int(m.group(1))][1]), html.escape(math[int(m.group(1))][0].strip())), body)
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=PT+Serif:ital,wght@0,400;0,700;1,400&display=swap">
<link rel="stylesheet" href="{KATEX}katex.min.css"><script src="{KATEX}katex.min.js"></script>
<style>{CSS}</style></head><body>{body}
<script>document.querySelectorAll('.m').forEach(e => katex.render(e.textContent, e,
  {{displayMode: e.dataset.d === '1', throwOnError: false}}));</script></body></html>"""


def chrome():
    for c in (os.environ.get("CHROME"), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")):
        if c and Path(c).exists():
            return c
    sys.exit("не найден Chrome: задайте путь в переменной CHROME")


def print_pdf(src, dst):
    with tempfile.TemporaryDirectory() as prof:  # свой профиль: не мешать открытому Chrome
        subprocess.run([chrome(), "--headless=new", "--disable-gpu", f"--user-data-dir={prof}",
                        "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
                        "--virtual-time-budget=20000", f"--print-to-pdf={dst}", src.as_uri()],
                       check=True, capture_output=True, timeout=180)
    print(f"{dst.relative_to(ROOT)}: {dst.stat().st_size / 1e6:.1f} МБ")


def main():
    what = sys.argv[1:] or ["report", "landing"]
    if "report" in what:
        page = REPORT.with_suffix(".html")
        text = REPORT.read_text(encoding="utf-8")
        page.write_text(md_to_html(text, text.splitlines()[0].lstrip("# ")), encoding="utf-8")
        print_pdf(page, REPORT.with_suffix(".pdf"))
    if "landing" in what:
        print_pdf(ROOT / "site/index.html", ROOT / "site/landing.pdf")


if __name__ == "__main__":
    main()
