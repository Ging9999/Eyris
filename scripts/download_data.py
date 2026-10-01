"""Download the organizer's hourly dataset and clone the official starter kit."""
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.data import DATA_URL, DEFAULT_PARQUET  # noqa: E402

KIT_REPO = "https://github.com/DeepIntoStreams/2026ICAIF_Trading_Agent_Competition"


def main():
    if not DEFAULT_PARQUET.exists():
        print(f"downloading {DATA_URL}")
        with urllib.request.urlopen(DATA_URL, timeout=120) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DEFAULT_PARQUET.parent)
    print(f"data: {DEFAULT_PARQUET}")
    kit = ROOT / "starter-kit"
    if not kit.exists():
        tmp = ROOT / ".kit-clone"
        subprocess.run(["git", "clone", "--depth", "1", KIT_REPO, str(tmp)], check=True)
        (tmp / "starter-kit").rename(kit)
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    print(f"starter kit: {kit}")


if __name__ == "__main__":
    main()
