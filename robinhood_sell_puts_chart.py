# robinhood_puts_full_final.py

import os, requests, io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import robin_stocks.robinhood as r
import yfinance as yf

# ------------------ CONFIG ------------------
TICKERS = ["TLRY", "PLUG", "BITF", "BBAI", "SPCE", "ONDS", "GRAB", "LUMN",
           "RIG", "BB", "HTZ", "RXRX", "CLOV", "RZLV", "NVTS", "RR"]
NUM_PUTS = 2
LOW_DAYS = 14
MAX_EXP_DAYS = 21
CANDLE_WIDTH = 0.6
MIN_PRICE = 0.05
BEST_MIN_PRICE = 0.1
BEST_MIN_PROB = 0.8

USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILITIES ------------------
def risk_emoji(prob_otm):
    if prob_otm >= 0.8: return "✅"
    elif prob_otm >= 0.6: return "🟡"
    else: return "⚠️"

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
cutoff = today + timedelta(days=30)

# ------------------ PART 1: EARNINGS/DIVIDENDS ------------------
safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False

        # Dividend
        try:
            if not stock.dividends.empty:
                div_date = stock.dividends.index[-1].date()
                if today <= div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except: pass

        # Earnings
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

# ------------------ PART 2: OPTIONS ------------------
all_options = []

for ticker in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(ticker)[0])
        rh_url = f"https://robinhood.com/stocks/{ticker}"

        # Historical data
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
            exp_date = pd.to_datetime(opt['expiration_date']).date()
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
            md = market_data[0]

            # Ask price
            try:
                price = float(md.get('ask_price') or 0.0)
                if price < MIN_PRICE: continue
            except: continue

            # Chance of Profit
            try:
                cop_str = md.get('chance_of_profit_long')
                cop_clean = str(cop_str).replace('%','').replace(',','').strip()
                prob_OTM = float(cop_clean)/100
            except:
                prob_OTM = 0.0

            # Delta
            try: delta = float(md.get('delta') or 0.0)
            except: delta = 0.0

            premium_risk = price / max(current_price - strike, 0.01)

            candidate_puts.append({
                "Ticker": ticker,
                "Strike Price": strike,
                "Option Price": price,
                "Delta": delta,
                "Prob OTM": prob_OTM,
                "Premium/Risk": premium_risk,
                "Expiration Date": opt['expiration_date'],
                "URL": rh_url,
                "Stock Price": current_price,
                "HistoricalDF": df,
                "Last14Low": last_14_low
            })

        candidate_puts.sort(key=lambda x: x['Prob OTM'], reverse=True)
        all_options.extend(candidate_puts)
        top2_alerts = candidate_puts[:2]

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
            buf = plot_candlestick(opt['HistoricalDF'], current_price, opt['Last14Low'])
            send_telegram_photo(buf, "\n".join(msg_lines))

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {ticker}: {e}")

# ------------------ BEST OPTION ALERT ------------------
valid_best = [opt for opt in all_options if opt['Option Price'] >= BEST_MIN_PRICE and opt['Prob OTM'] >= BEST_MIN_PROB]
if not valid_best and all_options:
    best = max(all_options, key=lambda x: x['Prob OTM'])
elif valid_best:
    best = max(valid_best, key=lambda x: x['Prob OTM'])
else:
    best = None

if best:
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
    buf = plot_candlestick(best['HistoricalDF'], best['Stock Price'], best['Last14Low'],
                            best['Strike Price'], best['Expiration Date'])
    send_telegram_photo(buf, "\n".join(msg_lines))
