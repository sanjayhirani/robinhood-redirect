# robinhood_sell_covered_calls_with_risk.py

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
import sys
import subprocess

try:
    import yfinance
except ImportError:
    print("yfinance not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance

try:
    import lxml
except ImportError:
    print("lxml not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lxml"])
    import lxml

# ------------------ OTHER IMPORTS ------------------
import os
import requests
import robin_stocks.robinhood as r
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from math import log, sqrt
from scipy.stats import norm
import numpy as np
import pandas as pd

# ------------------ CONFIG ------------------
NUM_EXPIRATIONS = 3
NUM_CALLS = 2
PRICE_ADJUST = 0.01
RISK_FREE_RATE = 0.05
MIN_PRICE = 0.10
HV_PERIOD = 21
CANDLE_WIDTH = 0.6
LOW_DAYS = 14

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILITY FUNCTIONS ------------------
def black_scholes_call_delta(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0: return 1.0
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return norm.cdf(d1)

def risk_emoji(prob_otm):
    if prob_otm >= 0.8: return "✅"
    elif prob_otm >= 0.6: return "🟡"
    else: return "⚠️"

def historical_volatility(prices, period=21):
    prices = np.array(prices)
    log_returns = np.diff(np.log(prices))
    if len(log_returns) < period:
        return 0.3
    rolling_std = np.std(log_returns[-period:])
    return rolling_std * np.sqrt(252)

def send_telegram_photo(buf, caption):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo': buf},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    )

def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

def plot_candlestick(df, current_price, last_14_low, selected_strikes=None, exp_date=None):
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    for i in range(len(df)):
        color = 'lime' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(df.index[i])-CANDLE_WIDTH/2, min(df['open'].iloc[i], df['close'].iloc[i])),
            CANDLE_WIDTH,
            abs(df['close'].iloc[i]-df['open'].iloc[i]),
            color=color
        ))
        ax.plot([mdates.date2num(df.index[i]), mdates.date2num(df.index[i])],
                 [df['low'].iloc[i], df['high'].iloc[i]], color=color, linewidth=1)
    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=2, label=f'14-day Low: ${last_14_low:.2f}')
    if selected_strikes:
        for strike in selected_strikes:
            ax.axhline(strike, color='cyan', linestyle='--', linewidth=1.5, label=f'Strike: ${strike:.2f}')
    if exp_date is not None:
        exp_date_obj = pd.to_datetime(exp_date).tz_localize(None)
        if df.index.min() <= exp_date_obj <= df.index.max():
            ax.axvline(mdates.date2num(exp_date_obj), color='orange', linestyle='--', linewidth=2, label=f'Expiration: {exp_date_obj.strftime("%d-%m-%y")}')
    ax.set_ylabel('Price ($)', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
    buf.seek(0)
    plt.close()
    return buf

# ------------------ LOGIN ------------------
r.login(USERNAME, PASSWORD)
today = datetime.now().date()
cutoff = today + timedelta(days=30)

# ------------------ TEST MODE ------------------
TEST_MODE = True  # <-- Set to False for real Robinhood scan

if TEST_MODE:
    print("⚠️ TEST MODE ACTIVE: Mocking ownership of 100 shares for OPEN")
    owned_positions = [{'quantity': '100', 'instrument': 'https://fake_url'}]
    r.stocks.get_instrument_by_url = lambda url: {'symbol': 'OPEN'}
    safe_tickers = ["OPEN"]
    MIN_PRICE = 0.01  # allow all options for testing

# ------------------ GET TICKERS YOU OWN 100 SHARES ------------------
owned_positions = r.account.get_open_stock_positions()
tickers_owned_100 = []
for pos in owned_positions:
    quantity = float(pos['quantity'])
    if quantity == 100:
        instrument_data = r.stocks.get_instrument_by_url(pos['instrument'])
        tickers_owned_100.append(instrument_data['symbol'].upper())

# ------------------ PART 1: EARNINGS/DIVIDENDS RISK CHECK ------------------
safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

for ticker in tickers_owned_100:
    try:
        stock = yfinance.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False

        # Dividend
        try:
            if not stock.dividends.empty:
                div_date = stock.dividends.index[-1].date()
                if today <= div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except Exception as e:
            msg_parts.append(f"⚠️ Dividend check error: {e}")

        # Earnings
        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index.min().date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"⚠️ 📢 Earnings on {earnings_date.strftime('%d-%m-%y')}")
                    has_event = True
        except Exception as e:
            msg_parts.append(f"⚠️ Earnings check error: {e}")

        if has_event:
            risky_msgs.append(" | ".join(msg_parts))
            risky_count += 1
        else:
            safe_tickers.append(ticker)
            safe_count += 1

    except Exception as e:
        risky_msgs.append(f"⚠️ <b>{ticker}</b> error: {e}")
        risky_count += 1

summary_lines = []
if risky_msgs:
    summary_lines.append("⚠️ <b>Risky Tickers</b>\n" + "\n".join(risky_msgs) + "\n")
else:
    summary_lines.append("⚠️ <b>No risky tickers found 🎉</b>\n")

safe_tickers_sorted = sorted(safe_tickers)
safe_bold = [f"<b>{t}</b>" for t in safe_tickers_sorted]
safe_rows = [", ".join(safe_bold[i:i+4]) for i in range(0, len(safe_bold), 4)]
if safe_rows:
    summary_lines.append("✅ <b>Safe Tickers</b>\n" + "\n".join(safe_rows))

summary_lines.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary_lines))

# ------------------ PART 2: ROBINHOOD COVERED CALLS ------------------
all_options = []
for TICKER in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(TICKER)[0])
        rh_url = f"https://robinhood.com/stocks/{TICKER}"

        historicals = r.stocks.get_stock_historicals(TICKER, interval='day', span='month', bounds='regular')
        df = pd.DataFrame(historicals)
        df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
        df.set_index('begins_at', inplace=True)
        df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
        df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(all_days)
        df.index = df.index.tz_localize(None)
        df['close'] = df['close'].ffill()
        df['open'] = df['open'].fillna(df['close'])
        df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
        df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
        df['volume'] = df['volume'].fillna(0)

        last_14_low = df['low'][-LOW_DAYS:].min()

        # Best Covered Call Candlestick Plot & Telegram
        if all_options:
            buf = plot_candlestick(df, current_price, last_14_low, [opt['Strike Price'] for opt in all_options], exp_date=None)
            send_telegram_photo(buf, "🔥 <b>Best Covered Call Alert</b>")

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")
