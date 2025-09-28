# robinhood_sell_calls.py

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
import sys
import subprocess

try:
    import yfinance
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance

try:
    import lxml
except ImportError:
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

# ------------------ TEST MODE ------------------
TEST_MODE = True  # Set False for live Robinhood scan

# ------------------ UTILITY FUNCTIONS ------------------
def black_scholes_call_delta(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0: return 1.0
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return norm.cdf(d1)

def risk_emoji(prob_itm):
    if prob_itm <= 0.2: return "✅"
    elif prob_itm <= 0.4: return "🟡"
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

def plot_candlestick(df, current_price, last_30_high, selected_strikes=None, exp_date=None):
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
    ax.axhline(last_30_high, color='yellow', linestyle='--', linewidth=2, label=f'1M High: ${last_30_high:.2f}')
    
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

# ------------------ TEST MODE CONFIG ------------------
if TEST_MODE:
    tickers_owned_100 = ['OPEN']
    safe_tickers = ['OPEN']
    summary_lines = ["✅ <b>Safe Tickers</b>\n<b>OPEN</b>", "\n📊 Summary: ✅ Safe: 1 | ⚠️ Risky: 0"]
    send_telegram_message("\n".join(summary_lines))
else:
    owned_positions = r.account.get_open_stock_positions()
    tickers_owned_100 = []
    for pos in owned_positions:
        quantity = float(pos['quantity'])
        if quantity == 100:
            instrument_data = r.stocks.get_instrument_by_url(pos['instrument'])
            tickers_owned_100.append(instrument_data['symbol'].upper())
    safe_tickers = tickers_owned_100

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

        month_high = df['close'].max()
        last_30_high = df['high'][-LOW_DAYS:].max()
        distance_from_high = month_high - current_price
        distance_pct = distance_from_high / month_high
        proximity = "🔺 Closer to 1M High" if current_price >= month_high * 0.5 else "🔻 Far from 1M High"

        all_calls = r.options.find_tradable_options(TICKER, optionType="call")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_calls]))[:NUM_EXPIRATIONS]

        candidate_calls = []
        sigma = historical_volatility(df['close'].values, HV_PERIOD)

        # ------------------ Collect all candidate calls ------------------
        for exp_date in exp_dates:
            calls_for_exp = [opt for opt in all_calls if opt['expiration_date'] == exp_date]
            strikes_above = sorted([float(opt['strike_price']) for opt in calls_for_exp if float(opt['strike_price']) > current_price])
            closest_strikes = strikes_above[:4]  # top 4 strikes above price

            for opt in calls_for_exp:
                strike = float(opt['strike_price'])
                if strike not in closest_strikes:
                    continue

                option_id = opt['id']
                market_data = r.options.get_option_market_data_by_id(option_id)

                price, delta = 0.0, 1.0
                if market_data:
                    try: price = float(market_data[0].get('adjusted_mark_price') or market_data[0].get('mark_price') or 0.0)
                    except: price=0.0
                    try: delta = float(market_data[0].get('delta')) if market_data[0].get('delta') else None
                    except: delta=None
                    if delta is None or delta==0.0:
                        T = max((datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days / 365, 1/365)
                        delta = black_scholes_call_delta(current_price, strike, T, RISK_FREE_RATE, sigma)

                price = max(price - PRICE_ADJUST,0.0)
                if price < MIN_PRICE:
                    continue  # skip tiny premium

                # profit % relative to stock
                profit_pct = price / current_price
                # risk-adjusted score: favors higher strikes and lower delta
                risk_adjusted_score = profit_pct * (1 - delta)

                candidate_calls.append({
                    "Ticker": TICKER, "Current Price": current_price, "Expiration Date": exp_date,
                    "Strike Price": strike, "Option Price": price, "Delta": delta,
                    "Prob ITM": delta, "Score": risk_adjusted_score, "URL": rh_url,
                    "Month High": month_high, "Distance From High $": distance_from_high,
                    "Distance From High %": distance_pct
                })

        # ------------------ Select top 2 calls overall -----------------
        final_selected_calls = sorted(candidate_calls, key=lambda x: x['Score'], reverse=True)[:NUM_CALLS]
        all_options.extend(final_selected_calls)

        # ------------------ Telegram text alert (top 2 calls) -----------------
        msg_lines = [f"📊 <a href='{rh_url}'>{TICKER}</a> current: ${current_price:.2f}",
                     f"💹 1M High: ${month_high:.2f}", f"📌 Proximity: {proximity}\n"]
        for opt in final_selected_calls:
            msg_lines.append(f"{risk_emoji(opt['Prob ITM'])} 📅 Exp: {opt['Expiration Date']}")
            msg_lines.append(f"💲 Strike: {opt['Strike Price']}")
            msg_lines.append(f"💰 Price : ${opt['Option Price']:.2f}")
            msg_lines.append(f"🔺 Delta : {opt['Delta']:.3f}")
            msg_lines.append(f"🎯 Prob ITM  : {opt['Prob ITM']*100:.1f}%")
            msg_lines.append(f"💎 Premium / Prob ITM: {opt['Score']*100:.2f}%")
            msg_lines.append(f"📉 Dist. from 1M High: {opt['Distance From High %']*100:.1f}%\n")
        send_telegram_message("\n".join(msg_lines))

        # ------------------ Chart + alert ONLY for best call -----------------
        if final_selected_calls:
            best_call = max(final_selected_calls, key=lambda x: x['Score'])
            buf = plot_candlestick(df, best_call['Current Price'], last_30_high, [best_call['Strike Price']], best_call['Expiration Date'])
            best_msg = [
                "🔥 <b>Best Covered Call to Sell</b>:",
                f"📊 <a href='{best_call['URL']}'>{best_call['Ticker']}</a> current: ${best_call['Current Price']:.2f}",
                f"✅ Expiration : {best_call['Expiration Date']}",
                f"💲 Strike    : {best_call['Strike Price']}",
                f"💰 Price     : ${best_call['Option Price']:.2f}",
                f"🔺 Delta     : {best_call['Delta']:.3f}",
                f"🎯 Prob ITM  : {best_call['Prob ITM']*100:.1f}%",
                f"💎 Premium / Prob ITM: {best_call['Score']*100:.2f}%",
                f"📉 Dist. from 1M High: {best_call['Distance From High %']*100:.1f}%"
            ]
            send_telegram_photo(buf, "\n".join(best_msg))

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")
