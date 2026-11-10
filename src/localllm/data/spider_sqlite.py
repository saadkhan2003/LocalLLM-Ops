"""Download the official Spider SQLite zip and lay out database/<db_id>/*.sqlite.

The file is the release linked from https://yale-lily.github.io/spider
(Google Drive id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J, filename spider_data.zip).
It is about 200 MB. It stays under data/ and is not committed.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
DRIVE_PAGE = "https://drive.google.com/uc?export=download&id={file_id}"
DRIVE_DOWNLOAD = (
    "https://drive.usercontent.google.com/download"
    "?id={file_id}&export=download&confirm={confirm}&uuid={uuid}"
)
MIN_DATABASES = 100
USER_AGENT = "localllm-ops"


def is_real_sqlite(path: Path) -> bool:
    """Skip macOS AppleDouble sidecars that the Spider zip ships next to each DB."""
    if path.name.startswith("._") or "__MACOSX" in path.parts:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(16).startswith(b"SQLite format 3")
    except OSError:
        return False


def sqlite_count(database_dir: Path) -> int:
    if not database_dir.exists():
        return 0
    return sum(1 for path in database_dir.glob("*/*.sqlite") if is_real_sqlite(path))


def parse_drive_confirm(html: str) -> tuple[str, str]:
    """Read confirm and uuid from the Drive virus-scan form."""
    confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html)
    uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
    if not confirm or not uuid:
        raise RuntimeError("Google Drive did not return a download confirmation form")
    return confirm.group(1), uuid.group(1)


def ensure_spider_databases(database_dir: Path, drive_id: str = SPIDER_DRIVE_ID) -> Path:
    """Download and extract unless this folder already holds the public Spider DBs."""
    database_dir = Path(database_dir)
    _drop_appledouble(database_dir)
    if sqlite_count(database_dir) >= MIN_DATABASES:
        print(f"sqlite already present: {sqlite_count(database_dir)} files in {database_dir}")
        return database_dir
    database_dir.mkdir(parents=True, exist_ok=True)
    zip_path = database_dir.parent / "spider_data.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        _download_drive_zip(drive_id, zip_path)
    extracted = database_dir.parent / "_spider_zip"
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted)
    placed = place_sqlite_tree(extracted, database_dir)
    tables = next(extracted.rglob("tables.json"), None)
    if tables is not None:
        target = database_dir.parent / "tables.json"
        if not target.exists():
            shutil.copy2(tables, target)
    print(f"placed {placed} sqlite files in {database_dir}")
    if placed < MIN_DATABASES:
        raise RuntimeError(
            f"expected at least {MIN_DATABASES} Spider databases, found {placed} in {database_dir}"
        )
    return database_dir


def place_sqlite_tree(extract_root: Path, database_dir: Path) -> int:
    """Copy or link each database folder to ``database_dir/<db_id>/<file>.sqlite``."""
    database_dir.mkdir(parents=True, exist_ok=True)
    placed = 0
    for path in sorted(extract_root.rglob("*.sqlite")):
        if not is_real_sqlite(path):
            continue
        db_id = path.parent.name
        if db_id in {"", "database", "spider"}:
            db_id = path.stem
        target_dir = database_dir / db_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        if not target.exists():
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
        placed += 1
    return placed


def _drop_appledouble(database_dir: Path) -> None:
    if not database_dir.exists():
        return
    for path in database_dir.rglob("._*"):
        if path.is_file():
            path.unlink()


def _download_drive_zip(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    page = _get(DRIVE_PAGE.format(file_id=file_id)).decode("utf-8", "replace")
    confirm, uuid = parse_drive_confirm(page)
    url = DRIVE_DOWNLOAD.format(file_id=file_id, confirm=confirm, uuid=uuid)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    print(f"downloading official Spider zip to {dest}")
    with urlopen(request, timeout=120) as response:
        kind = response.headers.get("Content-Type", "")
        if "text/html" in kind:
            raise RuntimeError("Google Drive returned HTML instead of the zip. Re-run the download.")
        total = int(response.headers.get("Content-Length") or 0)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        written = 0
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if total and written // (20 * 1024 * 1024) != (written - len(chunk)) // (
                    20 * 1024 * 1024
                ):
                    print(f"  {written // (1024 * 1024)} / {total // (1024 * 1024)} MB")
        with tmp.open("rb") as handle:
            magic = handle.read(2)
        if written < 1_000_000 or magic != b"PK":
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"download was not a zip ({written} bytes)")
        tmp.replace(dest)
    print(f"downloaded {dest} ({dest.stat().st_size} bytes)")


def _get(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()
