# robinhood_sell_puts_real_data.py

import sys, subprocess, os, requests, io
from datetime import datetime, timedelta
from math import log, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import robin_stocks.robinhood as r

# ------------------ CONFIG ------------------
TICKERS = ["TLRY", "PLUG", "BITF", "BBAI", "SPCE", "ONDS", "GRAB", "LUMN",
           "RIG", "BB", "HTZ", "RXRX", "CLOV", "RZLV" ,"NVTS" ,"RR"]
NUM_EXPIRATIONS = 3
MIN_PRICE = 0.15
HV_PERIOD = 21
CANDLE_WIDTH = 0.6
LOW_DAYS = 14
MAX_EXP_DAYS = 21

USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILITY ------------------
def risk_emoji(prob_otm):
    if prob_otm >= 0.8: return "✅"
    elif prob_otm >= 0.6: return "🟡"
    else: return "⚠️"

def historical_volatility(prices, period=21):
    log_returns = np.diff(np.log(prices))
    if len(log_returns) < period:
        return 0.3
    return np.std(log_returns[-period:]) * np.sqrt(252)

def send_telegram_photo(buf, caption):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                  files={'photo': buf},
                  data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"})

def send_telegram_message(msg):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                  data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})

def plot_candlestick(df, current_price, last_14_low, strike_price=None, exp_date=None):
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    for i in range(len(df)):
        color = 'lime' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(df.index[i])-CANDLE_WIDTH/2, min(df['open'].iloc[i], df['close'].iloc[i])),
            CANDLE_WIDTH, abs(df['close'].iloc[i]-df['open'].iloc[i]), color=color
        ))
        ax.plot([mdates.date2num(df.index[i]), mdates.date2num(df.index[i])],
                [df['low'].iloc[i], df['high'].iloc[i]], color=color, linewidth=1)
    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=2, label=f'14-day Low: ${last_14_low:.2f}')
    if strike_price is not None:
        ax.axhline(strike_price, color='cyan', linestyle='--', linewidth=1.5, label=f'Strike: ${strike_price:.2f}')
    if exp_date is not None:
        exp_dt = pd.to_datetime(exp_date).tz_localize(None)
        if df.index.min() <= exp_dt <= df.index.max():
            ax.axvline(mdates.date2num(exp_dt), color='orange', linestyle='--', linewidth=2, label=f'Expiration: {exp_dt.strftime("%d-%m-%y")}')
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

# ------------------ SCAN TICKERS ------------------
all_options = []
for ticker in TICKERS:
    try:
        current_price = float(r.stocks.get_latest_price(ticker)[0])
        rh_url = f"https://robinhood.com/stocks/{ticker}"
        # Historicals
        historicals = r.stocks.get_stock_historicals(ticker, interval='day', span='month', bounds='regular')
        df = pd.DataFrame(historicals)
        df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
        df.set_index('begins_at', inplace=True)
        df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
        df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        all_days = pd.date_range(df.index.min(), df.index.max(), freq='B')
        df = df.reindex(all_days)
        df.index = df.index.tz_localize(None)
        df['close'] = df['close'].ffill()
        df['open'] = df['open'].fillna(df['close'])
        df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
        df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
        df['volume'] = df['volume'].fillna(0)
        last_14_low = df['low'][-LOW_DAYS:].min()

        # Options
        all_puts = r.options.find_tradable_options(ticker, optionType="put")
        valid_puts = []
        for opt in all_puts:
            strike = float(opt['strike_price'])
            exp_date = datetime.strptime(opt['expiration_date'], "%Y-%m-%d").date()
            if strike >= current_price or (exp_date - today).days > MAX_EXP_DAYS:
                continue
            valid_puts.append((strike, opt))
        valid_puts.sort(key=lambda x: x[0], reverse=True)
        top_puts = valid_puts[:3]

        candidate_puts = []
        for strike, opt in top_puts:
            option_id = opt['id']
            market_data = r.options.get_option_market_data_by_id(option_id)
            if not market_data: continue
            last = float(market_data[0].get('last_trade_price') or 0.0)
            bid = float(market_data[0].get('bid_price') or 0.0)
            ask = float(market_data[0].get('ask_price') or 0.0)
            if last > 0: price = last
            elif bid > 0 and ask > 0: price = (bid + ask)/2
            else: continue
            delta = float(market_data[0].get('delta') or 0.0)
            prob_OTM = 1 - abs(delta)
            premium_risk = price / max(current_price - strike, 0.01)
            candidate_puts.append({
                "Ticker": ticker, "Strike Price": strike, "Option Price": price,
                "Delta": delta, "Prob OTM": prob_OTM, "Premium/Risk": premium_risk,
                "Expiration Date": opt['expiration_date'], "URL": rh_url, "Stock Price": current_price
            })

        candidate_puts.sort(key=lambda x: x['Prob OTM'], reverse=True)
        all_options.extend(candidate_puts)
        top2_alerts = candidate_puts[:2]

        # Telegram alert per ticker
        if top2_alerts:
            msg_lines = [f"📊 <a href='{rh_url}'>{ticker}</a> current: ${current_price:.2f}\n"]
            for opt in top2_alerts:
                exp_fmt = datetime.strptime(opt['Expiration Date'], "%Y-%m-%d").strftime("%d-%m-%y")
                msg_lines.append(f"{risk_emoji(opt['Prob OTM'])} 📅 Exp: {exp_fmt}")
                msg_lines.append(f"💲 Strike: ${opt['Strike Price']}")
                msg_lines.append(f"💰 Price : ${opt['Option Price']:.2f}")
                msg_lines.append(f"🔺 Delta : {opt['Delta']:.3f}")
                msg_lines.append(f"🎯 Prob  : {opt['Prob OTM']*100:.1f}%")
                msg_lines.append(f"💎 Premium/Risk: {opt['Premium/Risk']:.2f}\n")
            buf = plot_candlestick(df, current_price, last_14_low)
            send_telegram_photo(buf, "\n".join(msg_lines))

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {ticker}: {e}")

# Best overall option
if all_options:
    best = max(all_options, key=lambda x: x['Prob OTM'])
    exp_fmt = datetime.strptime(best['Expiration Date'], "%Y-%m-%d").strftime("%d-%m-%y")
    msg_lines = [
        "🔥 <b>Best Option to Sell</b>:",
        f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Stock Price']:.2f}",
        f"✅ Expiration : {exp_fmt}",
        f"💲 Strike    : ${best['Strike Price']}",
        f"💰 Price     : ${best['Option Price']:.2f}",
        f"🔺 Delta     : {best['Delta']:.3f}",
        f"🎯 Prob OTM  : {best['Prob OTM']*100:.1f}%",
        f"💎 Premium/Risk: {best['Premium/Risk']:.2f}"
    ]
    historicals = r.stocks.get_stock_historicals(best['Ticker'], interval='day', span='month', bounds='regular')
    df = pd.DataFrame(historicals)
    df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
    df.set_index('begins_at', inplace=True)
    df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
    all_days = pd.date_range(df.index.min(), df.index.max(), freq='B')
    df = df.reindex(all_days)
    df.index = df.index.tz_localize(None)
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
    df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
    df['volume'] = df['volume'].fillna(0)
    last_14_low = df['low'][-LOW_DAYS:].min()
    buf = plot_candlestick(df, best['Stock Price'], last_14_low, best['Strike Price'], best['Expiration Date'])
    send_telegram_photo(buf, "\n".join(msg_lines))
