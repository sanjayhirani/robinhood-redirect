# robinhood_sell_puts.py

import os
import requests
import robin_stocks.robinhood as r
import yfinance
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

# ------------------ CONFIG ------------------
TICKERS = ["SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC",
           "HTZ", "RZLV", "NVTS", "CLOV", "RIG", "LDI", "SPCE", "AMC", "LAZR"]  # extra tickers
NUM_EXPIRATIONS = 3
NUM_PUTS = 3
MIN_PRICE = 0.10
HV_PERIOD = 21
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ TELEGRAM UTILS ------------------
def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

# ------------------ DATA UTILS ------------------
def prepare_historicals(df):
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(all_days)
    df.index = df.index.tz_localize(None)
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
    df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
    df['volume'] = df['volume'].fillna(0)
    return df

# ------------------ LOGIN ------------------
r.login(USERNAME, PASSWORD)
today = datetime.now().date()
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

# ------------------ PART 1: RISK CHECK ------------------
safe_tickers = []
risky_msgs = []

for ticker in TICKERS:
    try:
        stock = yfinance.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False

        # Dividend check
        try:
            future_divs = stock.dividends[stock.dividends.index.date >= today]
            if not future_divs.empty:
                div_date = future_divs.index.min().date()
                if div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except:
            pass

        # Earnings check
        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index.min().date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"⚠️ 📢 Earnings on {earnings_date.strftime('%d-%m-%y')}")
                    has_event = True
        except:
            pass

        if has_event:
            risky_msgs.append(" | ".join(msg_parts))
        else:
            safe_tickers.append(ticker)

    except:
        risky_msgs.append(f"⚠️ <b>{ticker}</b> error")

summary_lines = []
if risky_msgs:
    summary_lines.append("⚠️ <b>Risky Tickers</b>\n" + "\n".join(risky_msgs) + "\n")
else:
    summary_lines.append("⚠️ <b>No risky tickers found 🎉</b>\n")

safe_bold = [f"<b>{t}</b>" for t in sorted(safe_tickers)]
safe_rows = [", ".join(safe_bold[i:i+4]) for i in range(0, len(safe_bold), 4)]
if safe_rows:
    summary_lines.append("✅ <b>Safe Tickers</b>\n" + "\n".join(safe_rows))

summary_lines.append(f"\n📊 Summary: ✅ Safe: {len(safe_tickers)} | ⚠️ Risky: {len(risky_msgs)}")
send_telegram_message("\n".join(summary_lines))

# ------------------ PART 2: OPTIONS SCAN ------------------
all_options = []
candidate_scores = []

# Buying power
account_data = r.profiles.load_account_profile()
buying_power = float(account_data['cash_available_for_withdrawal'])

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
        df = prepare_historicals(df)
        last_14_low = df['low'][-LOW_DAYS:].min()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        hv = df['returns'].rolling(HV_PERIOD).std().iloc[-1] * np.sqrt(252)

        all_puts = r.options.find_tradable_options(TICKER, optionType="put")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_puts]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:NUM_EXPIRATIONS]

        candidate_puts = []

        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date'] == exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            chosen_strikes = strikes_below[:NUM_PUTS]

            for opt in puts_for_exp:
                strike = float(opt['strike_price'])
                if strike not in chosen_strikes:
                    continue

                md = r.options.get_option_market_data_by_id(opt['id'])[0]
                bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                delta = float(md.get('delta') or 0.0)
                iv = float(md.get('implied_volatility') or 0.0)
                cop_short = float(md.get('chance_of_profit_short') or 0.0)
                theta = float(md.get('theta') or 0.0)
                open_interest = int(md.get('open_interest') or 0)
                volume = int(md.get('volume') or 0)
                dist_from_low = (strike - last_14_low)/last_14_low
                if dist_from_low < 0.03 or bid_price < MIN_PRICE:
                    continue

                candidate_puts.append({
                    "Ticker": TICKER,
                    "Current Price": current_price,
                    "Expiration Date": exp_date,
                    "Strike Price": strike,
                    "Bid Price": bid_price,
                    "Delta": delta,
                    "IV": iv,
                    "COP Short": cop_short,
                    "Theta": theta,
                    "Open Interest": open_interest,
                    "Volume": volume,
                    "URL": rh_url,
                    "HV": hv
                })

        # Send grouped individual ticker alerts (every 3 tickers)
        selected_puts = sorted(candidate_puts, key=lambda x: x['COP Short'], reverse=True)[:NUM_PUTS]
        if selected_puts:
            all_options.extend(selected_puts)

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")

# ------------------ GROUPED ALERTS (every 3 tickers) ------------------
grouped_msg = []
for idx, opt in enumerate(all_options, 1):
    max_contracts = max(1, int(buying_power // (opt['Strike Price']*100)))
    total_premium = opt['Bid Price'] * 100 * max_contracts
    grouped_msg.append(
        f"📊 <a href='{opt['URL']}'>{opt['Ticker']}</a> current: ${opt['Current Price']:.2f}\n"
        f"✅ Expiration : {opt['Expiration Date']}\n"
        f"💲 Strike    : {opt['Strike Price']}\n"
        f"💰 Bid Price : ${opt['Bid Price']:.2f}\n"
        f"🔺 Delta     : {opt['Delta']:.3f}\n"
        f"📈 IV       : {opt['IV']*100:.2f}%\n"
        f"🎯 COP Short : {opt['COP Short']*100:.1f}%\n"
        f"📝 Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}\n"
        "────────────────────────────"
    )
    if idx % 3 == 0 or idx == len(all_options):
        send_telegram_message("\n".join(grouped_msg))
        grouped_msg = []

# ------------------ CANDIDATE SCORE ALERT ------------------
candidate_scores = []
for opt in all_options:
    days_to_exp = (pd.to_datetime(opt['Expiration Date']).date() - today).days
    if days_to_exp <= 0:
        continue
    iv_hv_ratio = opt['IV']/opt['HV'] if opt['HV']>0 else 1.0
    liquidity_weight = 1 + 0.5*(opt['Volume'] + opt['Open Interest'])/1000
    max_contracts = max(1, int(buying_power // (opt['Strike Price']*100)))
    total_premium = opt['Bid Price'] * 100 * max_contracts * opt['COP Short']
    score = total_premium * iv_hv_ratio * liquidity_weight / (days_to_exp**1.0)
    candidate_scores.append((opt['Ticker'], opt['Strike Price'], opt['Expiration Date'], score))

candidate_scores.sort(key=lambda x: x[3], reverse=True)
score_msg = "📊 <b>Top 10 Candidate Put Scores</b>\n"
for t, strike, exp, score in candidate_scores[:10]:
    score_msg += f"{t} | Exp: {exp} | Strike: {strike} | Score: {score:.2f}\n"
send_telegram_message(score_msg)

# ------------------ BEST ALERT ------------------
def adjusted_score(opt):
    days_to_exp = (pd.to_datetime(opt['Expiration Date']).date() - today).days
    if days_to_exp <= 0:
        return 0
    iv_hv_ratio = opt['IV']/opt['HV'] if opt['HV']>0 else 1.0
    liquidity_weight = 1 + 0.5*(opt['Volume'] + opt['Open Interest'])/1000
    max_contracts = max(1, int(buying_power // (opt['Strike Price']*100)))
    total_premium = opt['Bid Price'] * 100 * max_contracts * opt['COP Short']
    return total_premium * iv_hv_ratio * liquidity_weight / (days_to_exp**1.0)

if all_options:
    best = max(all_options, key=adjusted_score)
    max_contracts = max(1, int(buying_power // (best['Strike Price']*100)))
    total_premium = best['Bid Price'] * 100 * max_contracts

    msg_lines = [
        "🔥 <b>Best Cash-Secured Put (Max Premium)</b>:",
        f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}",
        f"✅ Expiration : {best['Expiration Date']}",
        f"💲 Strike    : {best['Strike Price']}",
        f"💰 Bid Price : ${best['Bid Price']:.2f}",
        f"🔺 Delta     : {best['Delta']:.3f}",
        f"📈 IV       : {best['IV']*100:.2f}%",
        f"🎯 COP Short : {best['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}",
        f"📝 Adjusted Score: {adjusted_score(best):.2f}"
    ]

    historicals = r.stocks.get_stock_historicals(best['Ticker'], interval='day', span='month', bounds='regular')
    df = pd.DataFrame(historicals)
    df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
    df.set_index('begins_at', inplace=True)
    df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
    df = prepare_historicals(df)
    last_14_low = df['low'][-LOW_DAYS:].min()

    # Plot chart
    def plot_candlestick(df, current_price, last_14_low, selected_strikes):
        fig, ax = plt.subplots(figsize=(12,6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        CANDLE_WIDTH = 0.6
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
        ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5)
        ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=2)
        for strike in selected_strikes:
            ax.axhline(strike, color='cyan', linestyle='--', linewidth=1.5)
        ax.set_ylabel('Price ($)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, color='gray', linestyle='--', alpha=0.3)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        buf.seek(0)
        plt.close()
        return buf

    buf = plot_candlestick(df, best['Current Price'], last_14_low, [best['Strike Price']])
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo': buf},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": "\n".join(msg_lines), "parse_mode": "HTML"}
    )
