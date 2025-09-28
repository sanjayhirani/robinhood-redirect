# robinhood_sell_calls.py

import os
import requests
import robin_stocks.robinhood as r
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from datetime import datetime, timedelta

# ------------------ CONFIG ------------------
TICKER = "OPEN"       # hard-coded to your owned stock
MIN_PRICE = 0.10
CANDLE_WIDTH = 0.6
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21
NUM_CALLS = 3  # how many strikes to alert

# ------------------ SECRETS ------------------
USERNAME = os.environ.get("RH_USERNAME")
PASSWORD = os.environ.get("RH_PASSWORD")
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ TELEGRAM ------------------
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

# ------------------ PLOTTING ------------------
def plot_candlestick(df, current_price, last_14_high, selected_strikes=None, exp_date=None):
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
    ax.axhline(last_14_high, color='yellow', linestyle='--', linewidth=2, label=f'14-day High: ${last_14_high:.2f}')

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

# ------------------ MAIN ------------------
r.login(USERNAME, PASSWORD)

today = datetime.now().date()
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

try:
    current_price = float(r.stocks.get_latest_price(TICKER)[0])
    rh_url = f"https://robinhood.com/stocks/{TICKER}"

    # get historical prices for chart
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
    last_14_high = df['high'][-LOW_DAYS:].max()
    distance_from_high = month_high - current_price
    distance_pct = distance_from_high / month_high

    # ---- risky alert ----
    if distance_pct < 0.05:  # within 5% of 1M high
        send_telegram_message(f"⚠️ Risky: {TICKER} is close to its 1M high ({distance_pct*100:.2f}%)")

    # ---- option chain ----
    all_calls = r.options.find_tradable_options(TICKER, optionType="call")

    exp_dates = sorted(set([opt['expiration_date'] for opt in all_calls]))
    exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]

    candidate_calls = []

    for exp_date in exp_dates:
        calls_for_exp = [opt for opt in all_calls if opt['expiration_date'] == exp_date]
        strikes_above = sorted([float(opt['strike_price']) for opt in calls_for_exp if float(opt['strike_price']) > current_price])

        # skip first 2 above, take next 3
        chosen_strikes = strikes_above[2:5]

        for opt in calls_for_exp:
            strike = float(opt['strike_price'])
            if strike not in chosen_strikes:
                continue

            option_id = opt['id']
            md = r.options.get_option_market_data_by_id(option_id)[0]

            bid_price = float(md.get("bid_price") or 0.0)
            delta = float(md.get("delta") or 0.0)
            cop_short = float(md.get("chance_of_profit_short") or 0.0)

            if bid_price >= MIN_PRICE:
                candidate_calls.append({
                    "Ticker": TICKER,
                    "Current Price": current_price,
                    "Expiration Date": exp_date,
                    "Strike Price": strike,
                    "Bid Price": bid_price,
                    "Delta": delta,
                    "COP Short": cop_short,
                    "Month High": month_high,
                    "Distance From High %": distance_pct,
                    "URL": rh_url
                })

    # pick top 3 across expirations
    selected_calls = sorted(candidate_calls, key=lambda x: x['COP Short'], reverse=True)[:NUM_CALLS]

    # individual alert with chart
    if selected_calls:
        strikes = [c['Strike Price'] for c in selected_calls]
        msg = [f"📊 <a href='{rh_url}'>{TICKER}</a> current: ${current_price:.2f}"]
        msg.append(f"💹 1M High: ${month_high:.2f}\n")
        for c in selected_calls:
            msg.append(f"📅 {c['Expiration Date']} | 💲 {c['Strike Price']} | 💰 {c['Bid Price']:.2f} | 🔺 {c['Delta']:.3f} | 🎯 {c['COP Short']*100:.2f}%")
        buf = plot_candlestick(df, current_price, last_14_high, strikes)
        send_telegram_photo(buf, "\n".join(msg))

    # best alert
    if selected_calls:
        best = max(selected_calls, key=lambda x: x['COP Short'])
        msg = [
            "🔥 <b>Best Covered Call</b>:",
            f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}",
            f"✅ Exp: {best['Expiration Date']} | 💲 Strike: {best['Strike Price']}",
            f"💰 Bid: {best['Bid Price']:.2f} | 🔺 Delta: {best['Delta']:.3f}",
            f"🎯 COP Short: {best['COP Short']*100:.2f}%",
            f"📉 Dist. from 1M High: {best['Distance From High %']*100:.2f}%"
        ]
        buf = plot_candlestick(df, best['Current Price'], last_14_high, [best['Strike Price']], best['Expiration Date'])
        send_telegram_photo(buf, "\n".join(msg))

except Exception as e:
    send_telegram_message(f"⚠️ Error: {e}")
