"""
SMART NSE STOCK SCANNER + DAILY PAPER TRADING ENGINE  (fixed version)

Run examples:
    python scanner_script.py                  # sirf scan
    python scanner_script.py --paper          # scan + paper trading update
    python scanner_script.py --stock TCS      # ek stock analyze
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "LT", "TITAN",
    "AXISBANK", "KOTAKBANK", "BAJFINANCE", "MARUTI",
    "NTPC", "SUNPHARMA", "HINDALCO", "VEDL", "ADANIENT", "VBL",
]

SL_PCT = 1.50
TARGET_PCT = 3.00
MAX_HOLD_BARS = 20

# True karne par SL/Target ATR se banenge (volatile stocks ke liye better)
USE_ATR_STOPS = False
ATR_SL_MULT = 1.5
ATR_TARGET_MULT = 3.0

# NSE 15:30 IST par band hota hai; data finalize hone ke liye 15 min buffer
IST = "Asia/Kolkata"
DATA_READY_TIME = (15, 45)

RESULT_FILE = Path("MY_STOCK_SCANNER_RESULT.csv")
PAPER_TRADES_FILE = Path("paper_trades.csv")

TRADE_COLUMNS = [
    "Trade_ID", "Symbol", "Entry_Date", "Entry", "SL", "Target",
    "Status", "Exit_Date", "Exit_Price", "Bars_Held", "Result",
    "Return_%", "MFE_%", "MAE_%",
]
TEXT_COLUMNS = ["Trade_ID", "Symbol", "Entry_Date", "Status", "Exit_Date", "Result"]
NUM_COLUMNS = [c for c in TRADE_COLUMNS if c not in TEXT_COLUMNS]


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    return symbol if symbol.endswith(".NS") else symbol + ".NS"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


def drop_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Market chalu hai to aaj ki adhuri candle hata do."""
    if df.empty:
        return df

    now = pd.Timestamp.now(tz=IST)
    last_date = pd.Timestamp(df.index[-1]).date()
    day_finished = (now.hour, now.minute) >= DATA_READY_TIME

    if last_date == now.date() and not day_finished:
        return df.iloc[:-1].copy()
    return df


def download_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(
        normalize_symbol(symbol),
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    df = clean_columns(df)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if df.empty or not set(required).issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=required).copy()

    # timezone hatao, warna entry_date se compare karte time error aata hai
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return drop_incomplete_bar(df)


# --------------------------------------------------------------------------
# INDICATORS
# --------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (TradingView jaisa)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # loss = 0 aur gain > 0 ho to RSI 100 hai
    out = out.where(~((loss == 0) & (gain > 0)), 100.0)
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------
# SCANNER
# --------------------------------------------------------------------------
def analyze_stock(symbol: str) -> Optional[dict]:
    df = download_history(symbol)
    if df.empty or len(df) < 60:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    rsi_now = float(rsi(close).iloc[-1])
    atr_now = float(atr(high, low, close).iloc[-1])

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    # pichle 20 din ka average (aaj ka volume us average me shamil nahi)
    volume_avg20 = volume.shift(1).rolling(20).mean()

    price = float(close.iloc[-1])
    e9 = float(ema9.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])

    avg_vol = volume_avg20.iloc[-1]
    vol_ratio = float(volume.iloc[-1] / avg_vol) if pd.notna(avg_vol) and avg_vol > 0 else 0.0

    high_52w = float(high.tail(252).max())
    low_52w = float(low.tail(252).min())

    conditions = {
        "Price > EMA9": price > e9,
        "Price > EMA20": price > e20,
        "EMA20 > EMA50": e20 > e50,
        "RSI 50-70": 50 < rsi_now < 70,
        "Volume > 1.5x": vol_ratio > 1.5,
        "MACD > Signal": float(macd.iloc[-1]) > float(macd_signal.iloc[-1]),
        "Near 52W High": price >= high_52w * 0.98,
    }
    score = sum(conditions.values())

    bullish = price > e20 and e20 > e50

    if price > e20 and e20 > e50:
        trend = "Bullish"
    elif price < e20 and e20 < e50:
        trend = "Bearish"
    else:
        trend = "Neutral"

    # BUY tabhi jab trend bhi bullish ho (sirf score kaafi nahi)
    if bullish and score >= 6:
        signal = "STRONG BUY"
    elif bullish and score >= 4:
        signal = "BUY"
    elif score >= 2:
        signal = "WATCH"
    else:
        signal = "AVOID"

    return {
        "Stock": normalize_symbol(symbol).replace(".NS", ""),
        "Date": str(pd.Timestamp(df.index[-1]).date()),
        "Price": round(price, 2),
        "EMA9": round(e9, 2),
        "EMA20": round(e20, 2),
        "EMA50": round(e50, 2),
        "RSI": round(rsi_now, 1),
        "ATR": round(atr_now, 2),
        "Volume_Ratio": round(vol_ratio, 2),
        "52W_High": round(high_52w, 2),
        "52W_Low": round(low_52w, 2),
        "Distance_From_52W_High_%": round((price / high_52w - 1) * 100, 2),
        "Score": int(score),
        "Max_Score": 7,
        "Trend": trend,
        "Signal": signal,
    }


def run_scanner() -> pd.DataFrame:
    rows = []

    for symbol in STOCKS:
        try:
            print("Scanning:", symbol)
            result = analyze_stock(symbol)
            if result:
                rows.append(result)
        except Exception as exc:
            print("Skipped", symbol, ":", exc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    order = {"STRONG BUY": 0, "BUY": 1, "WATCH": 2, "AVOID": 3}
    df["_order"] = df["Signal"].map(order).fillna(99)
    df = df.sort_values(
        ["_order", "Score", "RSI"],
        ascending=[True, False, True],
    ).drop(columns="_order")

    df.to_csv(RESULT_FILE, index=False)
    return df


# --------------------------------------------------------------------------
# PAPER TRADES: LOAD / SAVE
# --------------------------------------------------------------------------
def coerce_trade_types(df: pd.DataFrame) -> pd.DataFrame:
    """Text columns object rahen, number columns numeric (pandas dtype warning se bachne ke liye)."""
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(object)
    for col in NUM_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_trades() -> pd.DataFrame:
    if not PAPER_TRADES_FILE.exists():
        return coerce_trade_types(pd.DataFrame(columns=TRADE_COLUMNS))

    try:
        df = pd.read_csv(PAPER_TRADES_FILE)
        for col in TRADE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return coerce_trade_types(df[TRADE_COLUMNS].copy())
    except Exception as exc:
        # File kharab ho to chupchap khali mat karo, warna history overwrite ho jaayegi
        backup = PAPER_TRADES_FILE.with_suffix(".corrupt.csv")
        os.replace(PAPER_TRADES_FILE, backup)
        print(f"paper_trades.csv padh nahi paaya ({exc}). Backup: {backup}")
        return coerce_trade_types(pd.DataFrame(columns=TRADE_COLUMNS))


def save_trades(df: pd.DataFrame) -> None:
    temp = PAPER_TRADES_FILE.with_suffix(".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, PAPER_TRADES_FILE)


def trade_id(symbol: str, date: str) -> str:
    return f"{symbol}_{date}"


# --------------------------------------------------------------------------
# PAPER TRADES: OPEN / UPDATE
# --------------------------------------------------------------------------
def open_new_trades(scanner: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if scanner.empty:
        return trades

    existing = set(trades["Trade_ID"].astype(str))
    open_symbols = set(
        trades.loc[trades["Status"].astype(str).str.upper() == "OPEN", "Symbol"].astype(str)
    )
    new_rows = []

    for _, row in scanner.iterrows():
        if row["Signal"] not in {"STRONG BUY", "BUY"}:
            continue

        symbol = str(row["Stock"])
        entry_date = str(row["Date"])
        tid = trade_id(symbol, entry_date)

        # ek symbol ka ek hi OPEN trade
        if symbol in open_symbols or tid in existing:
            continue

        entry = float(row["Price"])
        atr_val = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan

        if USE_ATR_STOPS and pd.notna(atr_val) and atr_val > 0:
            sl = entry - ATR_SL_MULT * atr_val
            target = entry + ATR_TARGET_MULT * atr_val
        else:
            sl = entry * (1 - SL_PCT / 100)
            target = entry * (1 + TARGET_PCT / 100)

        new_rows.append({
            "Trade_ID": tid,
            "Symbol": symbol,
            "Entry_Date": entry_date,
            "Entry": round(entry, 4),
            "SL": round(sl, 4),
            "Target": round(target, 4),
            "Status": "OPEN",
            "Exit_Date": "",
            "Exit_Price": np.nan,
            "Bars_Held": 0,
            "Result": "",
            "Return_%": np.nan,
            "MFE_%": 0.0,
            "MAE_%": 0.0,
        })
        existing.add(tid)
        open_symbols.add(symbol)

    if new_rows:
        new_df = coerce_trade_types(pd.DataFrame(new_rows, columns=TRADE_COLUMNS))
        trades = new_df if trades.empty else pd.concat([trades, new_df], ignore_index=True)

    return trades


def update_open_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    for idx, trade in trades.iterrows():
        if str(trade["Status"]).upper() != "OPEN":
            continue

        try:
            hist = download_history(str(trade["Symbol"]), "6mo")
            if hist.empty:
                continue

            entry_date = pd.Timestamp(trade["Entry_Date"])
            future = hist.loc[hist.index > entry_date].head(MAX_HOLD_BARS)

            if future.empty:
                continue

            entry = float(trade["Entry"])
            sl = float(trade["SL"])
            target = float(trade["Target"])

            mfe = float(trade["MFE_%"]) if pd.notna(trade["MFE_%"]) else 0.0
            mae = float(trade["MAE_%"]) if pd.notna(trade["MAE_%"]) else 0.0

            result = None
            exit_price = None
            exit_date = None
            bars = 0

            for bars, (date, row) in enumerate(future.iterrows(), start=1):
                o = float(row["Open"])
                h = float(row["High"])
                l = float(row["Low"])

                mfe = max(mfe, (h / entry - 1) * 100)
                mae = min(mae, (l / entry - 1) * 100)

                # Order important hai:
                # 1) gap-down SL ke neeche -> open par exit
                # 2) intraday SL hit -> SL par exit (same candle me target bhi ho to SL pehle)
                # 3) gap-up target ke upar -> open par exit
                # 4) intraday target hit -> target par exit
                if o <= sl:
                    result, exit_price = "SL", o
                elif l <= sl:
                    result, exit_price = "SL", sl
                elif o >= target:
                    result, exit_price = "TARGET", o
                elif h >= target:
                    result, exit_price = "TARGET", target

                if result is not None:
                    exit_date = date
                    break

            if result is None and len(future) >= MAX_HOLD_BARS:
                result = "TIME_EXIT"
                exit_price = float(future["Close"].iloc[-1])
                exit_date = future.index[-1]

            trades.at[idx, "Bars_Held"] = int(bars)
            trades.at[idx, "MFE_%"] = round(mfe, 4)
            trades.at[idx, "MAE_%"] = round(mae, 4)

            if result is not None:
                trades.at[idx, "Status"] = "CLOSED"
                trades.at[idx, "Exit_Date"] = str(pd.Timestamp(exit_date).date())
                trades.at[idx, "Exit_Price"] = round(float(exit_price), 4)
                trades.at[idx, "Result"] = result
                trades.at[idx, "Return_%"] = round((float(exit_price) / entry - 1) * 100, 4)

        except Exception as exc:
            print("Trade update failed for", trade["Symbol"], ":", exc)

    return trades


# --------------------------------------------------------------------------
# AUDIT
# --------------------------------------------------------------------------
def audit(trades: pd.DataFrame) -> dict:
    status = trades["Status"].astype(str).str.upper()
    open_count = int((status == "OPEN").sum())
    closed = trades[status == "CLOSED"].copy()

    empty = {
        "Closed Trades": 0,
        "Open Trades": open_count,
        "Wins": 0,
        "Losses": 0,
        "Win Rate %": 0.0,
        "Avg Return %": 0.0,
        "Total Return %": 0.0,
        "Profit Factor": np.nan,
        "Max Drawdown %": 0.0,
    }
    if closed.empty:
        return empty

    # equity curve exit date ke order me banni chahiye
    closed["_exit"] = pd.to_datetime(closed["Exit_Date"], errors="coerce")
    closed = closed.sort_values("_exit")
    returns = pd.to_numeric(closed["Return_%"], errors="coerce").dropna()
    if returns.empty:
        return empty

    wins = int((returns > 0).sum())
    losses = int((returns <= 0).sum())

    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(abs(returns[returns < 0].sum()))
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = math.inf
    else:
        pf = np.nan

    # 0 se shuru karo, taaki shuru ke losses bhi drawdown me ginein
    equity = np.concatenate([[0.0], returns.cumsum().to_numpy()])
    drawdown = equity - np.maximum.accumulate(equity)

    return {
        "Closed Trades": len(returns),
        "Open Trades": open_count,
        "Wins": wins,
        "Losses": losses,
        "Win Rate %": round(wins / len(returns) * 100, 2),
        "Avg Return %": round(float(returns.mean()), 2),
        "Total Return %": round(float(returns.sum()), 2),
        "Profit Factor": round(pf, 2) if math.isfinite(pf) else pf,
        "Max Drawdown %": round(float(drawdown.min()), 2),
    }


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", help="Analyze one NSE stock")
    parser.add_argument("--scan", action="store_true", help="Sirf scan chalao (default)")
    parser.add_argument("--paper", action="store_true", help="Scan + paper trading update")
    args = parser.parse_args()

    if args.stock:
        result = analyze_stock(args.stock)
        print(result if result else "No data")
        return

    scanner = run_scanner()

    if scanner.empty:
        print("No stock data returned.")
        return

    print("\nTOP STOCKS")
    print(scanner.head(10).to_string(index=False))

    if args.paper:
        trades = load_trades()
        trades = update_open_trades(trades)       # pehle purane trades close karo
        trades = open_new_trades(scanner, trades)  # phir naye kholo
        save_trades(trades)

        print("\nPAPER TRADING AUDIT")
        for key, value in audit(trades).items():
            print(f"{key}: {value}")


main()
