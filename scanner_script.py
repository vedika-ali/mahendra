"""
SMART NSE STOCK SCANNER + DAILY PAPER TRADING ENGINE
"""

from _ _future_ _ import annotations
import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# STOCK UNIVERSE
# =========================

STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "LT", "TITAN",
    "AXISBANK", "KOTAKBANK", "BAJFINANCE", "MARUTI",
    "NTPC", "SUNPHARMA", "HINDALCO", "VEDL", "ADANIENT", "VBL",
]


# =========================
# SETTINGS
# =========================

SL_PCT = 1.50
TARGET_PCT = 3.00
MAX_HOLD_BARS = 20

RESULT_FILE = Path("MY_STOCK_SCANNER_RESULT.csv")
PAPER_TRADES_FILE = Path("paper_trades.csv")

TRADE_COLUMNS = [
    "Trade_ID",
    "Symbol",
    "Entry_Date",
    "Entry",
    "SL",
    "Target",
    "Status",
    "Exit_Date",
    "Exit_Price",
    "Bars_Held",
    "Result",
    "Return_%",
    "MFE_%",
    "MAE_%",
]


# =========================
# SYMBOL
# =========================

def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol is empty.")

    if symbol.endswith(".NS"):
        return symbol

    return symbol + ".NS"


# =========================
# DATA CLEANING
# =========================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:

    if isinstance(df.columns, pd.MultiIndex):

        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)

        else:
            df.columns = df.columns.get_level_values(-1)

    return df


# =========================
# DOWNLOAD NSE DATA
# =========================

def download_history(
    symbol: str,
    period: str = "1y"
) -> pd.DataFrame:

    df = yf.download(
        normalize_symbol(symbol),
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    df = clean_columns(df)

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    if df.empty:
        return pd.DataFrame()

    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(
        subset=list(required)
    ).copy()

    return df


# =========================
# RSI
# =========================

def calculate_rsi(
    close: pd.Series,
    period: int = 14
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    ).rolling(period).mean()

    loss = (
        -delta.clip(upper=0)
    ).rolling(period).mean()

    rs = gain / loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


# =========================
# STOCK ANALYSIS
# =========================

def analyze_stock(
    symbol: str
) -> dict | None:

    df = download_history(symbol)

    if df.empty or len(df) < 60:
        return None

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    high = pd.to_numeric(
        df["High"],
        errors="coerce"
    )

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    # EMA
    ema9 = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI
    rsi_series = calculate_rsi(close)

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    # Volume
    volume_avg20 = volume.rolling(
        20
    ).mean()

    price = float(close.iloc[-1])

    e9 = float(
        ema9.iloc[-1]
    )

    e20 = float(
        ema20.iloc[-1]
    )

    e50 = float(
        ema50.iloc[-1]
    )

    rsi_now = float(
        rsi_series.iloc[-1]
    )

    macd_now = float(
        macd.iloc[-1]
    )

    macd_signal_now = float(
        macd_signal.iloc[-1]
    )

    avg_volume = float(
        volume_avg20.iloc[-1]
    )

    if avg_volume > 0:
        volume_ratio = float(
            volume.iloc[-1] / avg_volume
        )
    else:
        volume_ratio = 0

    # 52 week
    high_52w = float(
        high.tail(252).max()
    )

    low_52w = float(
        close.tail(252).min()
    )

    distance_from_high = (
        price / high_52w - 1
    ) * 100

    # =========================
    # 7 POINT SCORE
    # =========================

    conditions = {

        "Price > EMA9":
            price > e9,

        "Price > EMA20":
            price > e20,

        "EMA20 > EMA50":
            e20 > e50,

        "RSI 50-70":
            50 < rsi_now < 70,

        "Volume > 1.5x":
            volume_ratio > 1.5,

        "MACD > Signal":
            macd_now > macd_signal_now,

        "Near 52W High":
            price >= high_52w * 0.98,
    }

    score = int(
        sum(conditions.values())
    )

    # Signal
    if score >= 6:

        signal = "STRONG BUY"

    elif score >= 4:

        signal = "BUY"

    elif score >= 2:

        signal = "WATCH"

    else:

        signal = "AVOID"

    # Trend
    if price > e20 and e20 > e50:

        trend = "Bullish"

    elif price < e20 and e20 < e50:

        trend = "Bearish"

    else:

        trend = "Neutral"

    return {

        "Stock":
            normalize_symbol(symbol).replace(
                ".NS",
                ""
            ),

        "Date":
            str(
                pd.Timestamp(
                    df.index[-1]
                ).date()
            ),

        "Price":
            round(price, 2),

        "EMA9":
            round(e9, 2),

        "EMA20":
            round(e20, 2),

        "EMA50":
            round(e50, 2),

        "RSI":
            round(rsi_now, 1),

        "Volume_Ratio":
            round(volume_ratio, 2),

        "MACD":
            round(macd_now, 3),

        "MACD_Signal":
            round(macd_signal_now, 3),

        "52W_High":
            round(high_52w, 2),

        "52W_Low":
            round(low_52w, 2),

        "Distance_From_52W_High_%":
            round(
                distance_from_high,
                2
            ),

        "Score":
            score,

        "Max_Score":
            7,

        "Trend":
            trend,

        "Signal":
            signal,
    }


# =========================
# MARKET SCANNER
# =========================

def run_scanner() -> pd.DataFrame:

    results = []

    print(
        f"Scanning {len(STOCKS)} stocks..."
    )

    for symbol in STOCKS:

        try:

            print(
                "Scanning:",
                symbol
            )

            result = analyze_stock(
                symbol
            )

            if result:
                results.append(
                    result
                )

        except Exception as exc:

            print(
                "Skipped",
                symbol,
                ":",
                exc
            )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(
        results
    )

    signal_order = {
        "STRONG BUY": 0,
        "BUY": 1,
        "WATCH": 2,
        "AVOID": 3,
    }

    df["_order"] = (
        df["Signal"]
        .map(signal_order)
        .fillna(99)
    )

    df = df.sort_values(
        [
            "_order",
            "Score",
            "RSI"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )

    df = df.drop(
        columns="_order"
    )

    df.to_csv(
        RESULT_FILE,
        index=False
    )

    print(
        "\nScanner result saved:",
        RESULT_FILE
    )

    return df


# =========================
# LOAD PAPER TRADES
# =========================

def load_trades() -> pd.DataFrame:

    if not PAPER_TRADES_FILE.exists():

        return pd.DataFrame(
            columns=TRADE_COLUMNS
        )

    try:

        df = pd.read_csv(
            PAPER_TRADES_FILE
        )

        for col in TRADE_COLUMNS:

            if col not in df.columns:
                df[col] = np.nan

        return df[
            TRADE_COLUMNS
        ]

    except Exception:

        return pd.DataFrame(
            columns=TRADE_COLUMNS
        )


# =========================
# SAVE PAPER TRADES
# =========================

def save_trades(
    df: pd.DataFrame
) -> None:

    temp_file = (
        PAPER_TRADES_FILE
        .with_suffix(".tmp")
    )

    df.to_csv(
        temp_file,
        index=False
    )

    os.replace(
        temp_file,
        PAPER_TRADES_FILE
    )


# =========================
# TRADE ID
# =========================

def make_trade_id(
    symbol: str,
    date: str
) -> str:

    return (
        f"{symbol}_{date}"
    )


# =========================
# OPEN NEW TRADES
# =========================

def open_new_trades(
    scanner: pd.DataFrame,
    trades: pd.DataFrame
) -> pd.DataFrame:

    if scanner.empty:
        return trades

    existing_ids = set(
        trades[
            "Trade_ID"
        ].astype(str)
    )

    new_trades = []

    for _, row in scanner.iterrows():

        if row["Signal"] not in {
            "STRONG BUY",
            "BUY"
        }:
            continue

        symbol = str(
            row["Stock"]
        )

        entry_date = str(
            row["Date"]
        )

        tid = make_trade_id(
            symbol,
            entry_date
        )

        if tid in existing_ids:
            continue

        entry = float(
            row["Price"]
        )

        sl = entry * (
            1 - SL_PCT / 100
        )

        target = entry * (
            1 + TARGET_PCT / 100
        )

        new_trades.append({

            "Trade_ID":
                tid,

            "Symbol":
                symbol,

            "Entry_Date":
                entry_date,

            "Entry":
                round(entry, 4),

            "SL":
                round(sl, 4),

            "Target":
                round(target, 4),

            "Status":
                "OPEN",

            "Exit_Date":
                "",

            "Exit_Price":
                np.nan,

            "Bars_Held":
                0,

            "Result":
                "",

            "Return_%":
                np.nan,

            "MFE_%":
                0.0,

            "MAE_%":
                0.0,
        })

        existing_ids.add(
            tid
        )

    if new_trades:

        trades = pd.concat(
            [
                trades,
                pd.DataFrame(
                    new_trades
                )
            ],
            ignore_index=True
        )

    return trades


# =========================
# UPDATE OPEN TRADES
# =========================

def update_open_trades(
    trades: pd.DataFrame
) -> pd.DataFrame:

    if trades.empty:
        return trades

    for idx, trade in trades.iterrows():

        if str(
            trade["Status"]
        ).upper() != "OPEN":

            continue

        try:

            hist = download_history(
                str(trade["Symbol"]),
                "3mo"
            )

            if hist.empty:
                continue

            entry_date = pd.Timestamp(
                trade["Entry_Date"]
            )

            future = hist.loc[
                hist.index > entry_date
            ].copy()

            if future.empty:
                continue

            entry = float(
                trade["Entry"]
            )

            sl = float(
                trade["SL"]
            )

            target = float(
                trade["Target"]
            )

            mfe = float(
                trade["MFE_%"]
            )

            mae = float(
                trade["MAE_%"]
            )

            result = None
            exit_price = None
            exit_date = None
            bars = 0

            window = future.head(
                MAX_HOLD_BARS
            )

            for bars, (
                date,
                row
            ) in enumerate(
                window.iterrows(),
                start=1
            ):

                high = float(
                    row["High"]
                )

                low = float(
                    row["Low"]
                )

                mfe = max(
                    mfe,
                    (
                        high / entry - 1
                    ) * 100
                )

                mae = min(
                    mae,
                    (
                        low / entry - 1
                    ) * 100
                )

                hit_sl = (
                    low <= sl
                )

                hit_target = (
                    high >= target
                )

                # Conservative rule:
                # If both happen on same candle,
                # assume SL first.

                if hit_sl:

                    result = "SL"
                    exit_price = sl
                    exit_date = date

                    break

                if hit_target:

                    result = "TARGET"
                    exit_price = target
                    exit_date = date

                    break

            # Time exit
            if (
                result is None
                and len(window) >= MAX_HOLD_BARS
            ):

                last = window.iloc[
                    -1
                ]

                exit_price = float(
                    last["Close"]
                )

                exit_date = (
                    window.index[-1]
                )

                result = "TIME_EXIT"
                bars = MAX_HOLD_BARS

            if result is not None:

                trades.at[
                    idx,
                    "Status"
                ] = "CLOSED"

                trades.at[
                    idx,
                    "Exit_Date"
                ] = str(
                    pd.Timestamp(
                        exit_date
                    ).date()
                )

                trades.at[
                    idx,
                    "Exit_Price"
                ] = round(
                    float(exit_price),
                    4
                )

                trades.at[
                    idx,
                    "Bars_Held"
                ] = int(bars)

                trades.at[
                    idx,
                    "Result"
                ] = result

                trades.at[
                    idx,
                    "Return_%"
                ] = round(
                    (
                        float(exit_price)
                        / entry
                        - 1
                    ) * 100,
                    4
                )

            trades.at[
                idx,
                "MFE_%"
            ] = round(
                mfe,
                4
            )

            trades.at[
                idx,
                "MAE_%"
            ] = round(
                mae,
                4
            )

        except Exception as exc:

            print(
                "Trade update failed for",
                trade["Symbol"],
                ":",
                exc
            )

    return trades


# =========================
# PERFORMANCE AUDIT
# =========================

def audit(
    trades: pd.DataFrame
) -> dict:

    if trades.empty:

        return {
            "Closed Trades": 0,
            "Open Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win Rate %": 0.0,
            "Total Return %": 0.0,
            "Profit Factor": np.nan,
            "Max Drawdown %": 0.0,
        }

    status = (
        trades["Status"]
        .astype(str)
        .str.upper()
    )

    closed = trades[
        status == "CLOSED"
    ].copy()

    open_count = int(
        (status == "OPEN").sum()
    )

    if closed.empty:

        return {
            "Closed Trades": 0,
            "Open Trades": open_count,
            "Wins": 0,
            "Losses": 0,
            "Win Rate %": 0.0,
            "Total Return %": 0.0,
            "Profit Factor": np.nan,
            "Max Drawdown %": 0.0,
        }

    returns = pd.to_numeric(
        closed["Return_%"],
        errors="coerce"
    ).dropna()

    if returns.empty:

        return {
            "Closed Trades": 0,
            "Open Trades": open_count,
            "Wins": 0,
            "Losses": 0,
            "Win Rate %": 0.0,
            "Total Return %": 0.0,
            "Profit Factor": np.nan,
            "Max Drawdown %": 0.0,
        }

    wins = int(
        (returns > 0).sum()
    )

    losses = int(
        (returns <= 0).sum()
    )

    gross_profit = float(
        returns[
            returns > 0
        ].sum()
    )

    gross_loss = float(
        abs(
            returns[
                returns < 0
            ].sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = math.inf

    equity = returns.cumsum()

    drawdown = (
        equity
        - equity.cummax()
    )

    return {
        "Closed Trades":
            len(returns),

        "Open Trades":
            open_count,

        "Wins":
            wins,

        "Losses":
            losses,

        "Win Rate %":
            round(
                wins / len(returns) * 100,
                2
            ),

        "Total Return %":
            round(
                float(
                    returns.sum()
                ),
                2
            ),

        "Profit Factor":
            round(
                profit_factor,
                2
            )
            if math.isfinite(
                profit_factor
            )
            else math.inf,

        "Max Drawdown %":
            round(
                float(
                    drawdown.min()
                ),
                2
            ),
    }


# =========================
# MAIN
# =========================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Smart NSE Stock Scanner + Paper Trading"
    )

    parser.add_argument(
        "--stock",
        help=
        "Analyze one NSE stock, example VBL"
    )

    parser.add_argument(
        "--scan",
        action="store_true",
        help=
        "Run stock scanner"
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        help=
        "Run scanner and paper trading"
    )

    args = parser.parse_args()

    # Single stock
    if args.stock:

        result = analyze_stock(
            args.stock
        )

        if result is None:

            print(
                "No data found."
            )

        else:

            print("\nSTOCK ANALYSIS")

            for key, value in result.items():

                print(
                    f"{key:<30}: {value}"
                )

        return

    # Scanner
    scanner = run_scanner()

    if scanner.empty:

        print(
            "No stock data returned."
        )

        return

    print(
        "\nTOP STOCKS"
    )

    print(
        scanner.head(10)
        .to_string(index=False)
    )

    # Paper trading
    if args.paper:

        trades = load_trades()

        # First update old OPEN trades
        trades = update_open_trades(
            trades
        )

        # Then create new trades
        trades = open_new_trades(
            scanner,
            trades
        )

        save_trades(
            trades
        )

        print(
            "\nPAPER TRADING AUDIT"
        )

        report = audit(
            trades
        )

        for key, value in report.items():

            print(
                f"{key:<25}: {value}"
            )


if _name_ == "_main_":

    main()
