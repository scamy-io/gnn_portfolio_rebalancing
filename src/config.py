from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPHS_DIR    = DATA_DIR / "graphs"
RESULTS_DIR   = PROJECT_ROOT / "results"

TICKERS = [
    "AAPL", "AMT", "AMZN", "APD", "BA", "BAC", "CAT", "COST", "CVX", "DUK",
    "GE", "GOOGL", "GS", "HD", "JNJ", "JPM", "KO", "LIN", "MSFT", "NEE",
    "NFLX", "NVDA", "PG", "PLD", "TMO", "TSLA", "UNH", "VLO", "XOM",
]

PRICE_START  = "2017-11-01"
PRICE_END    = "2024-01-31"
SAMPLE_START = "2018-01-01"
SAMPLE_END   = "2023-12-16"

LOOKBACK    = 20
HORIZON     = 5
EMA_HALFLIFE    = 3
AFTER_HOURS_HOUR = 16

CORR_THRESHOLD         = 0.6
SENT_CORR_THRESHOLD    = 0.5
SENT_MIN_OVERLAP_DAYS  = 10
TOP_K_NEIGHBORS        = 5
MAX_DEGREE             = 10

TRAIN_END = "2020-12-31"
VAL_END   = "2021-12-31"

N_TOP, N_BOTTOM = 5, 5


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, GRAPHS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


SECTORS = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech", "NFLX": "tech",
    "CVX": "energy", "XOM": "energy", "VLO": "energy",
    "BA": "industrials", "CAT": "industrials", "GE": "industrials",
    "GS": "financials", "BAC": "financials", "JPM": "financials",
    "KO": "consumer", "COST": "consumer", "HD": "consumer", "PG": "consumer",
    "TSLA": "consumer", "AMZN": "consumer",
    "JNJ": "healthcare", "UNH": "healthcare", "TMO": "healthcare",
    "LIN": "materials", "APD": "materials",
    "NEE": "utilities", "DUK": "utilities",
    "AMT": "reits", "PLD": "reits",
}
