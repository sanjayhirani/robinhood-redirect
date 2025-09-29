# robinhood_sell_puts_async_prefilter.py

import os
import sys
import subprocess
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import robin_stocks.robinhood as r
import random

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
for pkg in ["yfinance", "lxml", "pandas", "numpy", "matplotlib", "robin_stocks", "aiohttp"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import yfinance

# ------------------ CONFIG ------------------
NUM_EXPIRATIONS = 3
NUM_PUTS = 3
MIN_PRICE = 0.10
CANDLE_WIDTH = 0.6
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21
NUM_TICKERS = 20
MAX_OTM_PCT = 0.20

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

today = datetime.now().date()
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

# ------------------ UTILITY FUNCTIONS ------------------
def send_telegram_message(msg):
    import requests
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

def send_telegram_photo(buf, caption):
    import requests
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo': buf},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    )

def plot_candlestick(df, current_price, last_14_low, selected_strikes=None, exp_date=None, show_strikes=True):
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
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
    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=2, label=f'14-day Low: ${last_14_low:.2f}')
    if show_strikes and selected_strikes:
        for strike in selected_strikes:
            ax.axhline(strike, color='cyan', linestyle='--', linewidth=1.5, label=f'Strike: ${strike:.2f}')
    if exp_date:
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

# ------------------ SELECT TICKERS ------------------
top_100 = r.stocks.get_top_100()
price_filtered = []
for t in top_100:
    try:
        price = float(t.get('last_trade_price') or 0.0)
        if 1 <= price <= 10:
            price_filtered.append(t['symbol'])
    except:
        continue

random.shuffle(price_filtered)
TICKERS = price_filtered[:NUM_TICKERS]

# ------------------ EARNINGS/DIVIDENDS CHECK ------------------
safe_tickers = []
risky_msgs = []
for ticker in TICKERS:
    try:
        stock = yfinance.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False
        try:
            if not stock.dividends.empty:
                div_date = stock.dividends.index[-1].date()
                if today <= div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except: pass
        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index.min().date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"⚠️ 📢 Earnings on {earnings_date.strftime('%d-%m-%y')}")
                    has_event = True
        except: pass
        if has_event:
            risky_msgs.append(" | ".join(msg_parts))
        else:
            safe_tickers.append(ticker)
    except:
        risky_msgs.append(f"⚠️ {ticker} error")

summary_msg = "⚠️ Risky:\n" + "\n".join(risky_msgs) if risky_msgs else "No risky tickers 🎉"
send_telegram_message(summary_msg)

# ------------------ ASYNC OPTION FETCHING ------------------
async def fetch_options_for_ticker(ticker):
    try:
        current_price = float(r.stocks.get_latest_price(ticker)[0])
        rh_url = f"https://robinhood.com/stocks/{ticker}"

        all_puts = r.options.find_tradable_options(ticker, optionType="put")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_puts]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:NUM_EXPIRATIONS]

        candidate_puts = []
        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date'] == exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            chosen_strikes = strikes_below[2:5]  # skip first 2

            for opt in puts_for_exp:
                strike = float(opt['strike_price'])
                if strike not in chosen_strikes:
                    continue
                if strike < current_price * (1 - MAX_OTM_PCT):
                    continue
                market_data = r.options.get_option_market_data_by_id(opt['id'])
                if not market_data: continue
                md = market_data[0]
                bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                if bid_price < MIN_PRICE: continue
                delta = float(md.get('delta') or 0.0)
                iv = float(md.get('implied_volatility') or 0.0)
                cop_short = float(md.get('chance_of_profit_short') or 0.0)

                candidate_puts.append({
                    "Ticker": ticker,
                    "Current Price": current_price,
                    "Expiration Date": exp_date,
                    "Strike Price": strike,
                    "Bid Price": bid_price,
                    "Delta": delta,
                    "IV": iv,
                    "COP Short": cop_short,
                    "URL": rh_url
                })
        return candidate_puts
    except:
        return []

async def main():
    tasks = [fetch_options_for_ticker(t) for t in safe_tickers]
    results = await asyncio.gather(*tasks)
    all_options = [opt for sublist in results for opt in sublist]

    # ------------------ BEST ALERT ------------------
    best = None
    best_score = -1
    for put in all_options:
        exp_date_obj = datetime.strptime(put['Expiration Date'], "%Y-%m-%d").date()
        days_to_exp = max((exp_date_obj - today).days, 1)
        weekly_premium = put['Bid Price'] / (days_to_exp / 7)
        score = weekly_premium * put['COP Short']
        if score > best_score:
            best_score = score
            best = put

    if best:
        msg_lines = [
            f"🔥 <b>Best Cash-Secured Put</b>:",
            f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}",
            f"✅ Expiration : {best['Expiration Date']}",
            f"💲 Strike    : ${best['Strike Price']}",
            f"💰 Bid Price : ${best['Bid Price']:.2f}",
            f"🔺 Delta     : {best['Delta']:.3f}",
            f"📈 IV       : {best['IV']*100:.2f}%",
            f"🎯 COP Short : {best['COP Short']*100:.1f}%"
        ]

        historicals = r.stocks.get_stock_historicals(best['Ticker'], interval='day', span='month', bounds='regular')
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
        buf = plot_candlestick(df, best['Current Price'], last_14_low, [best['Strike Price']], best['Expiration Date'])
        send_telegram_photo(buf, "\n".join(msg_lines))

# ------------------ RUN ------------------
asyncio.run(main())
