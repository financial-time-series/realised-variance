"""The two asset universes, and the honest account of how they were chosen.

Selection is hardcoded and dated rather than screened live, because a universe
that changes when you re-run it is not a universe -- it is a moving target that
silently rewrites your sample every time the campaign is repeated.

SURVIVORSHIP. The EQ equity leg is chosen from large-capitalisation US firms
that were listed before 2000 AND are still trading in 2026. That is a
survivorship-selected sample and no free data source fixes it: point-in-time
index membership needs CRSP or Compustat. The paper says so plainly rather than
implying otherwise (README, "Evaluation design traps").

What bounds the damage here is the target. Survivorship biases the *level* of
volatility downward -- firms that failed were violently volatile before they
did. It does not obviously bias the *ranking of forecasting models within an
asset*, which is what this study compares, because every model sees the same
series. The direction that would matter is if survivors' volatility were
systematically easier to forecast; the CX universe is the control for that,
since crypto pairs are neither survivorship-screened nor capitalisation-screened
and behave far less tamely.
"""

from __future__ import annotations

# --- EQ: US equities and ETFs, daily OHLCV from Yahoo ------------------------
# Grouped so the paper's sample table can report per-group counts, and so a
# group can be dropped wholesale in a robustness check.

EQ_INDEX_ETF = [
    "SPY",   # S&P 500                          (1993)
    "QQQ",   # Nasdaq-100                       (1999)
    "DIA",   # Dow 30                           (1998)
    "IWM",   # Russell 2000, small cap          (2000)
    "MDY",   # S&P MidCap 400                   (1995)
    "EFA",   # developed ex-US                  (2001)
    "EEM",   # emerging markets                 (2003)
]

EQ_SECTOR_ETF = [
    "XLF",   # financials     -- all nine Select Sector SPDRs launched Dec 1998,
    "XLE",   # energy            so the sector leg has one common start date and
    "XLK",   # technology        no staggered-entry artefact.
    "XLV",   # health care
    "XLI",   # industrials
    "XLP",   # consumer staples
    "XLU",   # utilities
    "XLY",   # consumer discretionary
    "XLB",   # materials
]

EQ_MACRO_ETF = [
    "TLT",   # 20y+ Treasuries                  (2002)
    "IEF",   # 7-10y Treasuries                 (2002)
    "LQD",   # investment-grade credit          (2002)
    "HYG",   # high-yield credit                (2007)
    "GLD",   # gold                             (2004)
    "SLV",   # silver                           (2006)
    "USO",   # crude oil                        (2006)
]

# Large-cap US equities listed before 2000 and still trading. Spread across
# sectors on purpose: an all-technology equity leg would make the 2000 and 2022
# drawdowns look like market-wide events when they are sector events.
EQ_STOCK = [
    "AAPL", "MSFT", "ORCL", "CSCO", "INTC", "IBM", "TXN", "AMD", "ADBE", "QCOM",
    "JPM", "BAC", "C", "WFC", "AXP",
    "JNJ", "PFE", "MRK", "UNH",
    "XOM", "CVX",
    "PG", "KO", "WMT", "MCD", "HD", "LOW", "TGT", "SBUX", "NKE", "DIS",
    "BA", "CAT", "MMM", "HON", "GE",
    "T", "VZ",
]

EQ_UNIVERSE: dict[str, list[str]] = {
    "index_etf": EQ_INDEX_ETF,
    "sector_etf": EQ_SECTOR_ETF,
    "macro_etf": EQ_MACRO_ETF,
    "stock": EQ_STOCK,
}

EQ_TICKERS = [t for g in EQ_UNIVERSE.values() for t in g]

EQ_START = "2000-01-01"
EQ_END = "2026-08-01"   # exclusive; the sample ends 2026-07-31

# --- CX: crypto pairs, 5-minute klines from Binance --------------------------
# The replication leg, and the answer to a referee who objects to the range
# proxy: here realised variance is computed from actual intraday returns.
#
# USDT quote throughout, so the pairs are commensurable. Chosen for continuous
# availability of 5-minute klines from 2018-01 onward, which is what the bulk
# archive supports; SOL and DOGE start later and their true first month is
# detected at download rather than assumed.
CX_TICKERS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT",
    "ADAUSDT", "EOSUSDT", "TRXUSDT", "ETCUSDT", "XLMUSDT",
    "DOGEUSDT", "SOLUSDT",
]

CX_START = "2018-01"    # first month of the bulk 5m archive to attempt
CX_END = "2026-07"      # last complete month at the time of writing

CX_INTERVAL = "5m"
CX_BARS_PER_DAY = 288   # 24h / 5min -- a complete UTC day


def eq_group(ticker: str) -> str:
    """Which EQ group a ticker belongs to. Used for per-group result tables."""
    for group, members in EQ_UNIVERSE.items():
        if ticker in members:
            return group
    raise KeyError(f"{ticker!r} is not in the EQ universe")
