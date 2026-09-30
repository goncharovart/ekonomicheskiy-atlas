"""Клонирует код методов жюри (KEFRiN, CANUS — Шалилех) в external/.

У репозиториев нет файла LICENSE, поэтому их код в наш репозиторий не копируем:
external/ стоит в .gitignore, src/jury_methods.py импортирует оттуда.
Запуск: .venv/Scripts/python.exe -X utf8 scripts/fetch_external.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external"

# имя -> (url, закреплённый коммит; None = последний, хэш печатается при клонировании)
REPOS = {
    "KEFRiN": ("https://github.com/Sorooshi/KEFRiN", "f9f96b1a778cb8d2e5ba85baae452dc813c4f110"),
    "CANUS": ("https://github.com/Sorooshi/CANUS", "754622af6eff9604a590e862a108c4c4212acf2d"),
}


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main():
    EXT.mkdir(exist_ok=True)
    for name, (url, commit) in REPOS.items():
        dst = EXT / name
        if not dst.exists():
            git("clone", "--quiet", url, str(dst))
        if commit:
            git("checkout", "--quiet", commit, cwd=dst)
        print(f"{name}: {git('rev-parse', 'HEAD', cwd=dst)} ({git('log', '-1', '--format=%cs', cwd=dst)})")


if __name__ == "__main__":
    main()
