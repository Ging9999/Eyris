"""Download the organizer's hourly dataset and clone the official starter kit."""
import io
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.data import DATA_URL, DEFAULT_PARQUET  # noqa: E402

KIT_REPO = "https://github.com/DeepIntoStreams/2026ICAIF_Trading_Agent_Competition"


def _remove_tree(path):
    """Cross-platform rm -rf (git marks pack files read-only on Windows)."""
    def make_writable(func, p, *_):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=make_writable)
        else:
            shutil.rmtree(path, onerror=make_writable)


def main():
    if "--kit-only" not in sys.argv and not DEFAULT_PARQUET.exists():
        print(f"downloading {DATA_URL}")
        with urllib.request.urlopen(DATA_URL, timeout=120) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DEFAULT_PARQUET.parent)
    print(f"data: {DEFAULT_PARQUET}")
    kit = ROOT / "starter-kit"
    tmp = ROOT / ".kit-clone"
    if not kit.exists():
        _remove_tree(tmp)
        subprocess.run(["git", "clone", "--depth", "1", KIT_REPO, str(tmp)], check=True)
        (tmp / "starter-kit").rename(kit)
    _remove_tree(tmp)  # also cleans up a leftover from an interrupted run
    print(f"starter kit: {kit}")


if __name__ == "__main__":
    main()
