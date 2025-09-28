# robinhood_sell_calls_main_full.py

import os
import requests
import robin_stocks.robinhood as r
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from datetime import datetime, timedelta

# ------------------ CONFIG ------------------
TICKERS = ["OPEN"]  # tickers you own
TEST_MODE = True
MIN_PRICE = 0.10
CANDLE_WIDTH = 0.6
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21
NUM_CALLS = 3  # top strikes per ticker

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
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
def plot_candlestick(df, current_price, last_14_high, selected_strikes=None, exp_date=None, show_strikes=True):
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
    if show_strikes and selected_strikes:
        for strike in selected_strikes:
            ax.axhline(strike, color='cyan', linestyle='--', linewidth=1.5)
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
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

# ------------------ TEST MODE: MOCK OWNED POSITIONS ------------------
mock_owned = {t: 100 for t in TICKERS}

# ------------------ MAIN ------------------
safe_tickers = []
risky_msgs = []

for TICKER in TICKERS:
    try:
        shares_owned = mock_owned.get(TICKER, 0)
        if shares_owned < 100:
            risky_msgs.append(f"⚠️ {TICKER} has less than 100 shares.")
            continue
        safe_tickers.append(TICKER)
    except:
        risky_msgs.append(f"⚠️ Could not verify {TICKER}")

# ------------------ SEND RISK SUMMARY ------------------
safe_count = len(safe_tickers)
risky_count = len(risky_msgs)
summary_lines = []
if risky_msgs:
    summary_lines.append("⚠️ <b>Risky Tickers</b>\n" + "\n".join(risky_msgs))
else:
    summary_lines.append("⚠️ <b>No risky tickers found 🎉</b>")

if safe_tickers:
    safe_bold = [f"<b>{t}</b>" for t in safe_tickers]
    safe_rows = [", ".join(safe_bold[i:i+4]) for i in range(0, len(safe_bold), 4)]
    summary_lines.append("✅ <b>Safe Tickers</b>\n" + "\n".join(safe_rows))

summary_lines.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary_lines))

# ------------------ PROCESS EACH SAFE TICKER ------------------
for TICKER in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(TICKER)[0])
        rh_url = f"https://robinhood.com/stocks/{TICKER}"

        # ------------------ Historicals ------------------
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

        # ------------------ Options ------------------
        all_calls = r.options.find_tradable_options(TICKER, optionType="call")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_calls]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]

        candidate_calls = []

        for exp_date in exp_dates:
            calls_for_exp = [opt for opt in all_calls if opt['expiration_date'] == exp_date]
            strikes_above = sorted([float(opt['strike_price']) for opt in calls_for_exp if float(opt['strike_price']) > current_price])
            chosen_strikes = strikes_above[2:5]  # skip first 2, take next 3

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

        # ------------------ Individual alert (top 3) ------------------
        selected_calls = sorted(candidate_calls, key=lambda x: x['COP Short'], reverse=True)[:NUM_CALLS]
        if selected_calls:
            msg_lines = [f"📊 <a href='{rh_url}'>{TICKER}</a> current: ${current_price:.2f}",
                         f"💹 1M High: ${month_high:.2f}\n"]
            for c in selected_calls:
                msg_lines.append(
                    f"Expiration: {c['Expiration Date']}\n"
                    f"Strike: ${c['Strike Price']}\n"
                    f"Bid Price: ${c['Bid Price']:.2f}\n"
                    f"Delta: {c['Delta']:.3f}\n"
                    f"Chance of Profit (Short): {c['COP Short']*100:.2f}%\n"
                    f"Distance from 1M High: {c['Distance From High %']*100:.2f}%\n"
                    "-------------------"
                )
            buf = plot_candlestick(df, current_price, last_14_high, show_strikes=False)
            send_telegram_photo(buf, "\n".join(msg_lines))

        # ------------------ Best alert ------------------
        if selected_calls:
            best = max(selected_calls, key=lambda x: x['COP Short'])
            msg_lines = [
                "🔥 <b>Best Covered Call</b>:",
                f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}",
                f"✅ Exp: {best['Expiration Date']} | 💲 Strike: {best['Strike Price']}",
                f"💰 Bid: ${best['Bid Price']:.2f} | 🔺 Delta: {best['Delta']:.3f}",
                f"🎯 COP Short: {best['COP Short']*100:.2f}%",
                f"📉 Dist. from 1M High: {best['Distance From High %']*100:.2f}%"
            ]
            buf = plot_candlestick(df, best['Current Price'], last_14_high,
                                   [best['Strike Price']], best['Expiration Date'], show_strikes=True)
            send_telegram_photo(buf, "\n".join(msg_lines))

    except Exception as e:
        send_telegram_message(f"⚠️ Error on {TICKER}: {e}")
