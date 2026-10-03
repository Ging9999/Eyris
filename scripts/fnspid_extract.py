"""Download FNSPID headlines (Hugging Face, free) for our 30 stocks into a small parquet.

FNSPID's 2016-2023 news is in Stock_news/nasdaq_exteral_data.csv (23 GB, with full
article text). The file is sorted by stock symbol, so instead of downloading all of
it we binary-search byte offsets with HTTP Range requests, fetch only each symbol's
slice, parse it, and keep date, symbol and title. (All_external.csv ends in June
2020, before our period.) META traded as FB before June 2022.
Writes data/fnspid_headlines.parquet (gitignored with the rest of data/*.parquet).
"""
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.config import UNIVERSE  # noqa: E402

URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
OUT = ROOT / "data" / "fnspid_headlines.parquet"
SYMBOLS = sorted(set(UNIVERSE) | {"FB", "GOOG"})
ALIASES = {"FB": "META", "GOOG": "GOOGL"}
START = "2020-11-01"
COLUMNS = ["idx", "Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author", "Article",
           "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"]
ROW = re.compile(rb"\n\d+(?:\.0)?,(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d UTC),(?:\"(?:[^\"]|\"\")*\"|[^,\n]*),([A-Z][A-Z.]{0,5}),")
EMPTY = pd.DataFrame(columns=["Date", "Article_title", "Stock_symbol"])


def size():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def get(lo, hi):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={lo}-{hi}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


_probe_cache = {}


def probe(off, total):
    """(row_offset, symbol) of the first row starting at or after ``off``."""
    if off in _probe_cache:
        return _probe_cache[off]
    win = 1 << 17
    while True:
        blob = get(off, min(off + win, total - 1))
        m = ROW.search(blob)
        if m or off + win >= total - 1:
            res = (off + m.start(), m.group(2).decode()) if m else (total, "\x7f")
            _probe_cache[off] = res
            return res
        win *= 4


def lower_bound(sym, total):
    """A byte offset at or before the first row whose symbol is >= sym (within 64 KB)."""
    lo, hi = 0, total
    while hi - lo > (1 << 16):
        mid = (lo + hi) // 2
        if probe(mid, total)[1] < sym:
            lo = mid
        else:
            hi = mid
    return lo


def slice_rows(sym, total):
    """All rows of ``sym``: (frame, bytes fetched)."""
    a = lower_bound(sym, total)
    # lower_bound is exact only to 64 KB, so read past the end of the block and filter by symbol
    b = min(total, lower_bound(sym + "\x7f", total) + (1 << 18))
    blob = get(a, b - 1)
    starts = [m.start() for m in ROW.finditer(blob)]
    if len(starts) < 2:
        return EMPTY, len(blob)
    # start at the first complete row and stop before the last (possibly cut) one
    df = pd.read_csv(io.BytesIO(blob[starts[0] + 1:starts[-1] + 1]), header=None, names=COLUMNS,
                     usecols=["Date", "Article_title", "Stock_symbol"], dtype=str, on_bad_lines="skip")
    return df[df.Stock_symbol == sym], len(blob)


def main():
    t0 = time.time()
    total = size()
    parts, fetched = [], 0
    for sym in SYMBOLS:
        df, n = slice_rows(sym, total)
        fetched += n
        kept = df[df.Date >= START]
        parts.append(kept)
        print(f"{sym:6s} {len(df):6,} rows ({df.Date.min()[:10] if len(df) else '-'} .. "
              f"{df.Date.max()[:10] if len(df) else '-'}), {len(kept):6,} since {START[:7]}, "
              f"{n / 1e6:6.1f} MB, {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.columns = ["date", "title", "symbol"]
    df["symbol"] = df["symbol"].replace(ALIASES)
    df["date"] = pd.to_datetime(df["date"].str.replace(" UTC", "", regex=False), errors="coerce")
    df = df.dropna(subset=["date", "title"]).drop_duplicates(["symbol", "title", "date"])
    df.to_parquet(OUT)
    print(f"done: {len(df):,} headlines, {df.date.min()} .. {df.date.max()}, fetched {fetched / 1e9:.2f} GB, "
          f"{time.time() - t0:.0f}s")
    print(df.groupby("symbol").size().to_string())


if __name__ == "__main__":
    main()
