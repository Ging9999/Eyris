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
# Organizer sectors (universe.json): the universe is ordered in blocks of five.
SECTORS = tuple(tuple(range(i, i + 5)) for i in range(0, N_ASSETS, 5))

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
    risk_method: str = "invvol"      # "ew" | "invvol" | "minvar" | "blend" | "hrp" | "erc" | "sector_eq" | "sector_invvol"
    lookback_days: int = 30          # bar-return history used for vol / covariance
    stock_cap: float = 0.10          # internal cap, <= MAX_WEIGHT
    gross: float = 1.0               # stock exposure (max exposure if gross_mode != "fixed")
    halflife_days: float = 0.0       # >0: EWMA vol/cov with this half-life (Man AHL style)
    vol_power: float = 1.0           # invvol weights ~ vol**-vol_power (2 = inverse variance)
    top_k: int = 0                   # >0: hold only the top_k lowest-vol names
    # exposure overlays, recomputed once per day from completed days
    gross_mode: str = "fixed"        # "fixed" | "voltarget" | "trend" | "voltarget_trend"
    vol_target: float = 0.10         # annualized portfolio vol target (voltarget modes)
    gross_min: float = 0.2           # exposure floor for overlays
    trend_days: int = 20             # EW-basket return lookback for the trend overlay
    trend_floor: float = 0.5         # exposure multiplier when the basket trend is negative
    # events.py: earnings risk control
    event_mode: str = "off"          # "off" | "trim" | "trim_restore"
    event_cut: float = 1.0           # fraction of the name's target removed before the jump
    event_min_move: float = 0.0      # only names whose typical earnings gap >= this (abs log)
    event_trim_round: int = 7        # trim from this round on the day before the jump
    event_restore_round: int = 1     # restore at this round on the jump day ("trim_restore")
    news_veto: bool = False          # live only: reduce-only LLM news veto (eyris/news.py)
    finbert_shadow: bool = False     # live only: log FinBERT headline sentiment after the round (never decides)
    # sentiment.py: contrarian VIX overlay (scale gross by vix_boost when fear is high)
    vix_mode: str = "off"            # "off" | "level" (VIX > threshold) | "z" (60-day z-score > threshold)
    vix_threshold: float = 25.0
    vix_boost: float = 1.4
    # alpha.py
    use_alpha: bool = False
    tilt: float = 0.0                # multiplicative tilt strength on risk weights
    # agent.py: bad-data circuit breaker. HOLD if the risk target moved more than this
    # since the previous completed day (2021-25 max: 0.038 L1, 0.016 per name). 0 disables.
    breaker_l1: float = 0.08
    breaker_name: float = 0.03
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
        if self.gross_mode not in ("fixed", "voltarget", "trend", "voltarget_trend"):
            raise ValueError("unknown gross_mode")
        if not 0 <= self.gross_min <= self.gross:
            raise ValueError("gross_min must be in [0, gross]")
        if not 1 <= self.trend_days <= MAX_LOOKBACK_DAYS:
            raise ValueError("trend_days must be in [1, %d]" % MAX_LOOKBACK_DAYS)
        if self.event_mode not in ("off", "trim", "trim_restore"):
            raise ValueError("unknown event_mode")
        if not 0 <= self.event_cut <= 1 or not 1 <= self.event_trim_round <= 7 \
                or not 1 <= self.event_restore_round <= 7:
            raise ValueError("bad event parameters")
        if not 0 < self.vol_power <= 4:
            raise ValueError("vol_power must be in (0, 4]")
        if not 0 <= self.top_k <= N_ASSETS or (self.top_k and self.gross > self.stock_cap * self.top_k + 1e-12):
            raise ValueError("top_k must be 0 or reach gross under stock_cap")
        if self.vix_mode not in ("off", "level", "z"):
            raise ValueError("unknown vix_mode")
        if not 0.5 <= self.vix_boost <= 2 or (self.vix_mode != "off" and self.gross * self.vix_boost > 1 + 1e-12):
            raise ValueError("vix_boost must be in [0.5, 2] and keep gross * boost <= 1")
        if not 0 <= self.trend_floor <= 1:
            raise ValueError("trend_floor must be in [0, 1]")

    def to_dict(self):
        return asdict(self)
