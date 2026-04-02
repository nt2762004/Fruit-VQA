from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import gdown  # pyright: ignore[reportMissingImports]


EXPECTED_DATA_SPLITS = ("train", "valid", "test")
DEFAULT_DATA_ZIP_URL = "https://drive.google.com/file/d/1Y5H61uAQLOAltYrpXGmDLfDIYSVyqQoZ/view?usp=sharing"


def has_dataset_structure(data_dir: str | Path) -> bool:
    data_path = Path(data_dir)
    return data_path.exists() and all((data_path / split).is_dir() for split in EXPECTED_DATA_SPLITS)


def find_dataset_root(search_dir: str | Path) -> Path | None:
    search_path = Path(search_dir)

    if has_dataset_structure(search_path):
        return search_path

    for candidate in search_path.rglob("*"):
        if candidate.is_dir() and has_dataset_structure(candidate):
            return candidate

    return None


def build_drive_download_url(download_url: str | None = None, file_id: str | None = None) -> str | None:
    if download_url:
        return download_url.strip()

    if file_id:
        cleaned_file_id = file_id.strip()
        if cleaned_file_id:
            return f"https://drive.google.com/uc?id={cleaned_file_id}"

    return None


def ensure_dataset_dir(
    local_data_dir: str | Path,
    *,
    download_url: str | None = None,
    file_id: str | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    local_path = Path(local_data_dir)
    resolved_local = find_dataset_root(local_path)
    if resolved_local is not None:
        return resolved_local

    url = build_drive_download_url(download_url=download_url, file_id=file_id) or DEFAULT_DATA_ZIP_URL
    if url is None:
        raise RuntimeError(
            "Dataset not found locally. Set DATA_ZIP_URL or DATA_ZIP_FILE_ID in Streamlit secrets so the app can download the zip from Google Drive."
        )

    cache_path = Path(cache_root) if cache_root is not None else Path.home() / ".cache" / "fruits_vqa"
    cache_path.mkdir(parents=True, exist_ok=True)

    zip_path = cache_path / "dataset.zip"
    extract_root = cache_path / "dataset"

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    downloaded_path = gdown.download(url=url, output=zip_path.as_posix(), quiet=False, fuzzy=True)
    if not downloaded_path:
        raise RuntimeError("Failed to download the dataset zip from Google Drive.")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)

    resolved_extracted = find_dataset_root(extract_root)
    if resolved_extracted is None:
        raise RuntimeError("The downloaded zip does not contain a valid fruit dataset structure.")

    return resolved_extracted


def get_runtime_cache_root() -> Path:
    env_cache_root = os.getenv("FRUITS_VQA_CACHE_DIR")
    if env_cache_root:
        return Path(env_cache_root).expanduser()
    return Path.home() / ".cache" / "fruits_vqa"