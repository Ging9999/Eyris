"""Download earnings announcement timestamps (public, Yahoo Finance) to data/earnings_dates.csv.

Run before each competition day as well: it also picks up newly confirmed dates.
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.config import UNIVERSE  # noqa: E402
from eyris.events import CALENDAR_FILE  # noqa: E402


def main():
    import yfinance as yf
    rows = []
    for sym in UNIVERSE:
        for attempt in range(4):
            try:
                d = yf.Ticker(sym).get_earnings_dates(limit=40)
                break
            except Exception as e:  # rate limits: back off and retry
                print(f"{sym}: {e}; retry")
                time.sleep(2 ** (attempt + 1))
        else:
            raise RuntimeError(f"could not fetch {sym}")
        for ts, row in d.iterrows():
            ts = pd.Timestamp(ts).tz_convert("America/New_York")
            rows.append({"symbol": sym, "announced_at": ts.tz_localize(None).isoformat(),
                         "eps_estimate": row.get("EPS Estimate"), "eps_reported": row.get("Reported EPS")})
        print(sym, len(d))
        time.sleep(0.5)
    df = pd.DataFrame(rows).sort_values(["symbol", "announced_at"])
    df.to_csv(CALENDAR_FILE, index=False)
    print(f"wrote {len(df)} rows to {CALENDAR_FILE}")


if __name__ == "__main__":
    main()
