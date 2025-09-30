# robinhood_sell_puts_final_batch.py

import os
import sys
import subprocess
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import robin_stocks.robinhood as r

# ------------------ CONFIG ------------------
TICKERS = ["SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC",
           "HTZ", "RZLV", "NVTS", "CLOV", "RIG", "LDI", "SPCE", "AMC", "LAZR"]
NUM_EXPIRATIONS = 3
NUM_PUTS = 3
MIN_PRICE = 0.10
HV_PERIOD = 21
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21
CANDLE_WIDTH = 0.6
COP_THRESHOLD = 0.05  # 5% below top single contract COP

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILITIES ------------------
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
            (mdates.date2num(df.index[i]) - CANDLE_WIDTH/2, min(df['open'].iloc[i], df['close'].iloc[i])),
            CANDLE_WIDTH,
            abs(df['close'].iloc[i] - df['open'].iloc[i]),
            color=color
        ))
        ax.plot([mdates.date2num(df.index[i]), mdates.date2num(df.index[i])],
                 [df['low'].iloc[i], df['high'].iloc[i]], color=color, linewidth=1)

    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=2, label=f'14-day Low: ${last_14_low:.2f}')

    if selected_strikes:
        for strike in selected_strikes:
            ax.axhline(strike, color='cyan', linestyle='--', linewidth=1.5, label=f'Strike: ${strike:.2f}')

    if exp_date:
        exp_date_obj = pd.to_datetime(exp_date).tz_localize(None)
        if df.index.min() <= exp_date_obj <= df.index.max():
            ax.axvline(mdates.date2num(exp_date_obj), color='orange', linestyle='--', linewidth=2,
                       label=f'Expiration: {exp_date_obj.strftime("%d-%m-%y")}')

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

# ------------------ PART 1: RISKY CHECK ------------------
safe_tickers = []
risky_msgs = []

for ticker in TICKERS:
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False

        try:
            future_divs = stock.dividends[stock.dividends.index.date >= today]
            if not future_divs.empty:
                div_date = future_divs.index.min().date()
                if div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except:
            pass

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

    except Exception as e:
        risky_msgs.append(f"⚠️ <b>{ticker}</b> error: {e}")

safe_count = len(safe_tickers)
risky_count = len(risky_msgs)

summary_lines = []
if risky_msgs:
    summary_lines.append("⚠️ <b>Risky Tickers</b>\n" + "\n".join(risky_msgs) + "\n")
else:
    summary_lines.append("⚠️ <b>No risky tickers found 🎉</b>\n")

safe_bold = [f"<b>{t}</b>" for t in sorted(safe_tickers)]
safe_rows = [", ".join(safe_bold[i:i+4]) for i in range(0, len(safe_bold), 4)]
if safe_rows:
    summary_lines.append("✅ <b>Safe Tickers</b>\n" + "\n".join(safe_rows))

summary_lines.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary_lines))

# ------------------ PART 2: OPTIONS & ALERTS ------------------
all_options = []
candidate_scores = []
account_data = r.profiles.load_account_profile()
buying_power = float(account_data['cash_available_for_withdrawal'])

ticker_alerts_batch = []
for idx, TICKER in enumerate(safe_tickers, start=1):
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
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff][:NUM_EXPIRATIONS]

        ticker_options = []

        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date'] == exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            # Skip top 2
            chosen_strikes = strikes_below[2:2+NUM_PUTS]

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
                if dist_from_low < 0.03:
                    continue

                # Max contracts for candidate score
                max_contracts = max(1, int(buying_power // (strike * 100)))
                total_premium = bid_price * 100 * max_contracts if bid_price >= MIN_PRICE else 0.0

                ticker_options.append({
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
                    "Max Contracts": max_contracts,
                    "Total Premium": total_premium,
                    "Dist from Low": dist_from_low,
                    "URL": rh_url,
                    "HV": hv
                })

        # Batch individual ticker alerts every 3 tickers
        batch_msg_lines = []
        if ticker_options:
            batch_msg_lines.append(f"📊 <a href='{rh_url}'>{TICKER}</a> current: ${current_price:.2f}")
            for p in ticker_options:
                if p['Bid Price'] < MIN_PRICE:
                    batch_msg_lines.append(
                        f"✅ Expiration : {p['Expiration Date']}\n"
                        f"💲 Strike    : {p['Strike Price']}\n"
                        f"⚠️ Bid too low to trade\n"
                        "────────────────────────────"
                    )
                else:
                    batch_msg_lines.append(
                        f"✅ Expiration : {p['Expiration Date']}\n"
                        f"💲 Strike    : {p['Strike Price']}\n"
                        f"💰 Bid Price : ${p['Bid Price']:.2f}\n"
                        f"🔺 Delta     : {p['Delta']:.3f}\n"
                        f"📈 IV       : {p['IV']*100:.2f}%\n"
                        f"🎯 COP Short : {p['COP Short']*100:.1f}%\n"
                        f"📝 Max Contracts: {p['Max Contracts']} | Total Premium: ${p['Total Premium']:.2f}\n"
                        "────────────────────────────"
                    )

                # Candidate scoring
                days_to_exp = (pd.to_datetime(p['Expiration Date']).date() - today).days
                iv_hv_ratio = p['IV']/p['HV'] if p['HV']>0 else 1.0
                liquidity_weight = 1 + 0.5*(p['Volume'] + p['Open Interest'])/1000
                enhanced_score = p['Total Premium'] * iv_hv_ratio * liquidity_weight / (days_to_exp**1.0)
                candidate_scores.append({
                    "Ticker": p['Ticker'],
                    "Strike": p['Strike Price'],
                    "Expiration": p['Expiration Date'],
                    "Max Contracts": p['Max Contracts'],
                    "Total Premium": p['Total Premium'],
                    "Score": enhanced_score
                })
        else:
            batch_msg_lines.append(f"📊 <b>{TICKER}</b> - No valid options available or bid too low.")

        ticker_alerts_batch.append("\n".join(batch_msg_lines))

        if idx % 3 == 0 or idx == len(safe_tickers):
            send_telegram_message("\n\n".join(ticker_alerts_batch))
            ticker_alerts_batch = []

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")

# ------------------ Candidate Score Alert ------------------
if candidate_scores:
    candidate_scores_sorted = sorted(candidate_scores, key=lambda x: x['Score'], reverse=True)[:10]
    score_msg = "📊 <b>Top 10 Candidate Scores</b>\n"
    for c in candidate_scores_sorted:
        score_msg += (f"{c['Ticker']} | Exp: {c['Expiration']} | Strike: {c['Strike']} | "
                      f"Score: {c['Score']:.2f}\n"
                      "────────────────────────────\n")
    send_telegram_message(score_msg)

# ------------------ Best Option Alert ------------------
if candidate_scores_sorted:
    best = max(candidate_scores_sorted, key=lambda x: x['Score'])
    buf = io.BytesIO()
    try:
        hist = r.stocks.get_stock_historicals(best['Ticker'], interval='day', span='month', bounds='regular')
        df_best = pd.DataFrame(hist)
        df_best['begins_at'] = pd.to_datetime(df_best['begins_at']).dt.tz_localize(None)
        df_best.set_index('begins_at', inplace=True)
        df_best = df_best[['open_price','close_price','high_price','low_price','volume']].astype(float)
        df_best.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        df_best = prepare_historicals(df_best)
        last_14_low = df_best['low'][-LOW_DAYS:].min()
        buf = plot_candlestick(df_best, best['Total Premium']/best['Max Contracts']/100, last_14_low,
                               [best['Strike']], best['Expiration'])
    except:
        pass

    # Use max available investment for best alert
    max_contracts = max(1, int(buying_power // (best['Strike'] * 100)))
    total_premium = best['Strike'] * 100 * max_contracts  # Placeholder for actual bid price calculation if needed

    best_msg = [
        f"🔥 <b>Best Cash-Secured Put</b>",
        f"📊 {best['Ticker']}",
        f"✅ Expiration : {best['Expiration']}",
        f"💲 Strike    : {best['Strike']}",
        f"📝 Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}",
        f"📝 Score: {best['Score']:.2f}"
    ]
    if buf.getbuffer().nbytes > 0:
        send_telegram_photo(buf, "\n".join(best_msg))
    else:
        send_telegram_message("\n".join(best_msg))
