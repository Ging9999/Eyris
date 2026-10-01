"""Competition constants and strategy parameters."""
from dataclasses import dataclass, field, asdict

# Order matches the official universe.json / decision.schema.json.
UNIVERSE = (
    "AAPL", "MSFT", "NVDA", "INTC", "CRM",
    "JPM", "BAC", "GS", "V", "PYPL",
    "LLY", "JNJ", "UNH", "PFE", "TMO",
    "AMZN", "TSLA", "WMT", "NKE", "KO",
    "CAT", "GE", "BA", "XOM", "CVX",
    "GOOGL", "META", "DIS", "T", "NEE",
)
N_ASSETS = len(UNIVERSE)

# Official rules (Codabench competition 99 + starter kit docs/rules.md).
INITIAL_NAV = 1_000_000.0
FEE_RATE = 0.001          # 0.1% of buy+sell notional
MAX_WEIGHT = 0.30         # per-stock cap
ROUNDS_PER_DAY = 7
WINDOW_DAYS = 15          # Official Competition length in trading days
ANNUALIZATION = 42.0      # sqrt(252 * 7)

# Historical bars are labelled by start time. The 09:30 bar spans 09:30-10:00,
# later bars are clock-hour bars (10:00-11:00, ..., 15:00-16:00).
BAR_SLOTS = ("09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00")
BAR_END = ("10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00")
EXEC_TIMES = ("09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30")
DEADLINES = ("09:10", "10:25", "11:25", "12:25", "13:25", "14:25", "15:25")

# Live 30m bars only reach ~59 calendar days back (~40 trading days), so every
# lookback must fit inside that.
MAX_LOOKBACK_DAYS = 40


@dataclass(frozen=True)
class Params:
    # risk.py
    risk_method: str = "invvol"      # "ew" | "invvol" | "minvar" | "blend"
    lookback_days: int = 30          # bar-return history used for vol / covariance
    stock_cap: float = 0.10          # internal cap, <= MAX_WEIGHT
    gross: float = 1.0               # stock exposure; 1 - gross stays in cash
    # alpha.py
    use_alpha: bool = False
    tilt: float = 0.0                # multiplicative tilt strength on risk weights
    # execution.py
    lam: float = 0.5                 # partial-rebalance speed
    band: float = 0.05               # skip the round if L1(target - current) < band
    min_trade: float = 0.002         # per-name trades smaller than this are dropped

    def __post_init__(self):
        if not 0 < self.stock_cap <= MAX_WEIGHT:
            raise ValueError("stock_cap must be in (0, 0.30]")
        if not 0 <= self.gross <= 1:
            raise ValueError("gross must be in [0, 1]")
        if self.gross > self.stock_cap * N_ASSETS + 1e-12:
            raise ValueError("gross exposure not reachable under stock_cap")
        if not 2 <= self.lookback_days <= MAX_LOOKBACK_DAYS:
            raise ValueError("lookback_days must be in [2, %d]" % MAX_LOOKBACK_DAYS)
        if not 0 < self.lam <= 1:
            raise ValueError("lam must be in (0, 1]")

    def to_dict(self):
        return asdict(self)
