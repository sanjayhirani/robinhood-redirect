# robinhood_options_manager.py

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
TICKERS = ["SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC", "HTZ", "RZLV" ,"NVTS"]
NUM_EXPIRATIONS = 3
NUM_OPTIONS = 2
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
def black_scholes_put_delta(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0: return -1.0
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return norm.cdf(d1) - 1

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

# ------------------ PART 1: EARNINGS/DIVIDENDS RISK CHECK ------------------
safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

for ticker in TICKERS:
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

# ------------------ OWNERSHIP CHECK ------------------
positions = r.account.build_holdings()
owned_tickers = [t for t, data in positions.items() if float(data.get('quantity', 0)) >= 100]

# ------------------ FUNCTIONS TO PROCESS PUTS AND CALLS ------------------
# ... [Include the process_puts(), process_calls(), and generate_best_chart() exactly as in previous answer] ...

# ------------------ MAIN LOGIC ------------------
if owned_tickers:
    process_calls(owned_tickers)

safe_tickers_no_shares = [t for t in safe_tickers if t not in owned_tickers]
if safe_tickers_no_shares:
    process_puts(safe_tickers_no_shares)
