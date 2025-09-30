# robinhood_sell_puts_main_fixed.py

import os
import requests
import robin_stocks.robinhood as r
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ------------------ CONFIG ------------------
TICKERS = ["SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC", "HTZ", "RZLV",
           "NVTS", "CLOV", "RIG", "LDI", "SPCE", "AMC", "LAZR"]
NUM_EXPIRATIONS = 3
NUM_PUTS = 3
MIN_PRICE = 0.10
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILITY FUNCTIONS ------------------
def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

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

# ------------------ PART 1: EARNINGS/DIVIDENDS RISK CHECK ------------------
safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

import yfinance as yf

for ticker in TICKERS:
    try:
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

# ------------------ PART 2: ROBINHOOD OPTIONS ------------------
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

        all_puts = r.options.find_tradable_options(TICKER, optionType="put")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_puts]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:NUM_EXPIRATIONS]

        ticker_options = []

        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date'] == exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            chosen_strikes = strikes_below[2:2+NUM_PUTS]  # skip top 2 strikes under current price

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

                if bid_price >= MIN_PRICE:
                    max_contracts = int(buying_power // (strike * 100))
                    max_contracts = max(1, max_contracts)
                    total_premium = bid_price * 100 * max_contracts

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
                        "URL": rh_url
                    })

        # Send grouped ticker alerts (no charts)
        msg_lines = [f"📊 <a href='{rh_url}'>{TICKER}</a> current: ${current_price:.2f}"]
        for idx, p in enumerate(ticker_options, start=1):
            msg_lines.append(
                f"✅ Expiration : {p['Expiration Date']}\n"
                f"💲 Strike    : {p['Strike Price']}\n"
                f"💰 Bid Price : ${p['Bid Price']:.2f}\n"
                f"🔺 Delta     : {p['Delta']:.3f}\n"
                f"📈 IV       : {p['IV']*100:.2f}%\n"
                f"🎯 COP Short : {p['COP Short']*100:.1f}%\n"
                f"📝 Max Contracts: {p['Max Contracts']} | Total Premium: ${p['Total Premium']:.2f}\n"
                "────────────────────────────"
            )
            # store candidate scores
            score = p['Total Premium'] * p['COP Short']
            candidate_scores.append((p['Ticker'], p['Strike Price'], p['Expiration Date'], p['Max Contracts'], p['Total Premium'], score))

        if ticker_options:
            send_telegram_message("\n".join(msg_lines))
            all_options.extend(ticker_options)

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")

# ------------------ CANDIDATE SCORE ALERT ------------------
if candidate_scores:
    candidate_scores.sort(key=lambda x: x[5], reverse=True)  # sort by score
    top_10 = candidate_scores[:10]
    score_msg = "📊 <b>Top 10 Candidate Scores</b>\n"
    for t, strike, exp, max_ct, prem, score in top_10:
        score_msg += f"{t} | Exp: {exp} | Strike: {strike} | Max Contracts: {max_ct} | Total Premium: ${prem:.2f} | Score: {score:.2f}\n"
    send_telegram_message(score_msg)

# ------------------ BEST OPTION ALERT ------------------
if all_options:
    def adjusted_score(opt):
        return opt['Total Premium'] * opt['COP Short']

    best = max(all_options, key=adjusted_score)
    msg_lines = [
        "🔥 <b>Best Cash-Secured Put (Max Premium)</b>:",
        f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}",
        f"✅ Expiration : {best['Expiration Date']}",
        f"💲 Strike    : {best['Strike Price']}",
        f"💰 Bid Price : ${best['Bid Price']:.2f}",
        f"🔺 Delta     : {best['Delta']:.3f}",
        f"📈 IV       : {best['IV']*100:.2f}%",
        f"🎯 COP Short : {best['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {best['Max Contracts']} | Total Premium: ${best['Total Premium']:.2f}",
        f"📝 Score: {adjusted_score(best):.2f}"
    ]
    send_telegram_message("\n".join(msg_lines))
