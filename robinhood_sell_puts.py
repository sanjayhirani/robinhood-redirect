# robinhood_sell_puts.py

import os
import sys
import subprocess
import requests
import robin_stocks.robinhood as r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import yaml
from datetime import datetime, timedelta
import yfinance as yf
import re

# ------------------ AUTO-INSTALL OPTIONAL DEPENDENCIES ------------------

def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

for pkg in ["pandas","numpy","matplotlib","requests","robin_stocks","yfinance","PyYAML"]:
    ensure_package(pkg)

# ------------------ LOAD CONFIG ------------------

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ------------------ LOAD TICKERS ------------------

TICKERS_FILE = config.get("tickers_file", "tickers.txt")
if not os.path.exists(TICKERS_FILE):
    raise FileNotFoundError(f"{TICKERS_FILE} not found.")

TICKERS_RAW = [line.strip() for line in open(TICKERS_FILE, encoding="utf-8") if line.strip()]
# Clean tickers for API (alphanumeric only)
TICKERS = [re.sub(r'[^A-Z0-9.-]', '', t.upper()) for t in TICKERS_RAW]

# ------------------ SECRETS ------------------

USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ TELEGRAM UTILITIES ------------------

def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

def send_telegram_photo(buf, caption):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo': buf},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    )

# ------------------ PLOTTING ------------------

plot_cfg = config.get("plot", {})
CANDLE_WIDTH = plot_cfg.get("candle_width", 0.6)
FIGSIZE = plot_cfg.get("figsize", [12,6])
BG_COLOR = plot_cfg.get("background_color","black")
CURRENT_COLOR = plot_cfg.get("current_price_color","magenta")
LOW_COLOR = plot_cfg.get("low_price_color","yellow")
STRIKE_COLOR = plot_cfg.get("strike_color","cyan")
EXPIRY_COLOR = plot_cfg.get("expiry_color","orange")

def plot_candlestick(df, current_price, last_14_low, selected_strikes=None, exp_date=None, show_strikes=True):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for i in range(len(df)):
        color = 'lime' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(df.index[i]) - CANDLE_WIDTH/2, min(df['open'].iloc[i], df['close'].iloc[i])),
            CANDLE_WIDTH,
            abs(df['close'].iloc[i] - df['open'].iloc[i]),
            color=color
        ))
        ax.plot([mdates.date2num(df.index[i]), mdates.date2num(df.index[i])],
                [df['low'].iloc[i], df['high'].iloc[i]], color=color, linewidth=1)

    ax.axhline(current_price, color=CURRENT_COLOR, linestyle='--', linewidth=1.5)
    ax.axhline(last_14_low, color=LOW_COLOR, linestyle='--', linewidth=2)

    if show_strikes and selected_strikes:
        for strike in selected_strikes:
            ax.axhline(strike, color=STRIKE_COLOR, linestyle='--', linewidth=1.5)

    if exp_date:
        exp_date_obj = pd.to_datetime(exp_date).tz_localize(None)
        if df.index.min() <= exp_date_obj <= df.index.max():
            ax.axvline(mdates.date2num(exp_date_obj), color=EXPIRY_COLOR, linestyle='--', linewidth=2)

    ax.set_ylabel('Price ($)', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=BG_COLOR)
    buf.seek(0)
    plt.close()
    return buf

# ------------------ LOGIN ------------------

r.login(USERNAME, PASSWORD)
today = datetime.now().date()
expiry_limit_days = config.get("expiry_limit_days", 21)
cutoff = today + timedelta(days=expiry_limit_days)

# ------------------ SCAN TICKERS ------------------

safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

for ticker_raw, ticker_clean in zip(TICKERS_RAW, TICKERS):
    try:
        stock = yf.Ticker(ticker_clean)
        msg_parts = [f"{ticker_raw}"]
        has_event = False

        # Dividend check
        try:
            future_divs = stock.dividends[stock.dividends.index.date >= today]
            if not future_divs.empty:
                div_date = future_divs.index.min().date()
                if div_date <= cutoff:
                    msg_parts.append(f"{config['telegram_labels']['dividend_alert']} {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except:
            pass

        # Earnings check
        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index.min().date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"{config['telegram_labels']['earnings_alert']} {earnings_date.strftime('%d-%m-%y')}")
                    has_event = True
        except:
            pass

        if has_event:
            risky_msgs.append(" | ".join(msg_parts))
            risky_count += 1
        else:
            safe_tickers.append((ticker_raw, ticker_clean))
            safe_count += 1

    except Exception as e:
        risky_msgs.append(f"{ticker_raw} error: {e}")
        risky_count += 1

# Summary message
summary_lines = []
if risky_msgs:
    summary_lines.append(f"{config['telegram_labels']['risky_tickers']}\n" + "\n".join(risky_msgs))
if safe_tickers:
    safe_rows = [", ".join([t[0] for t in safe_tickers][i:i+4]) for i in range(0,len(safe_tickers),4)]
    summary_lines.append(f"{config['telegram_labels']['safe_tickers']}\n" + "\n".join(safe_rows))
summary_lines.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary_lines))

# ------------------ OPTIONS SCAN ------------------
all_options = []
candidate_scores = []

account_data = r.profiles.load_account_profile()
buying_power = float(account_data['cash_available_for_withdrawal'])

for ticker_raw, ticker_clean in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(ticker_clean)[0])
        historicals = r.stocks.get_stock_historicals(ticker_clean, interval='day', span='month', bounds='regular')
        df = pd.DataFrame(historicals)
        df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
        df.set_index('begins_at', inplace=True)
        df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
        df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        df = df.asfreq('B').ffill()
        last_14_low = df['low'][-config.get("low_days",14):].min()

        all_puts = r.options.find_tradable_options(ticker_clean, optionType="put")
        # ... Continue your existing option selection logic ...
        # Use ticker_clean for API, ticker_raw for Telegram messages

    except Exception as e:
        send_telegram_message(f"{ticker_raw} error: {e}")
