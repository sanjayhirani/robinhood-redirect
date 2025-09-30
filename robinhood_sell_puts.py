# robinhood_sell_puts_main_final_grouped.py

import os
import requests
import robin_stocks.robinhood as r
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# ------------------ CONFIG ------------------
TICKERS = ["SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC", "HTZ",
           "RZLV", "NVTS", "CLOV", "RIG", "LDI", "SPCE", "AMC", "LAZR"]
NUM_EXPIRATIONS = 3
NUM_PUTS = 3
MIN_PRICE = 0.10
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21
GROUP_SIZE = 3  # group 3 tickers per Telegram message

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ TELEGRAM ------------------
def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

# ------------------ LOGIN ------------------
r.login(USERNAME, PASSWORD)
today = datetime.now().date()
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

# ------------------ EARNINGS/DIVIDENDS RISK CHECK ------------------
safe_tickers = []
risky_msgs = []
for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
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
    except Exception as e:
        risky_msgs.append(f"⚠️ <b>{ticker}</b> error: {e}")

summary_lines = []
if risky_msgs:
    summary_lines.append("⚠️ <b>Risky Tickers</b>\n" + "\n".join(risky_msgs))
else:
    summary_lines.append("⚠️ <b>No risky tickers found 🎉</b>")
safe_bold = [f"<b>{t}</b>" for t in sorted(safe_tickers)]
summary_lines.append("✅ <b>Safe Tickers</b>\n" + ", ".join(safe_bold))
send_telegram_message("\n".join(summary_lines))

# ------------------ OPTIONS SCAN ------------------
all_options = []
candidate_scores = []

# Get buying power
account_data = r.profiles.load_account_profile()
buying_power = float(account_data['cash_available_for_withdrawal'])

group_msgs = []
current_group = []

for TICKER in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(TICKER)[0])
        rh_url = f"https://robinhood.com/stocks/{TICKER}"

        # Get all puts
        all_puts = r.options.find_tradable_options(TICKER, optionType="put")
        exp_dates = sorted({opt['expiration_date'] for opt in all_puts})
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:NUM_EXPIRATIONS]

        ticker_options = []
        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date']==exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price'])<current_price], reverse=True)
            chosen_strikes = strikes_below[2:2+NUM_PUTS]  # skip top 2

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

                max_contracts = max(1, int(buying_power // (strike*100)))
                total_premium = bid_price*100*max_contracts

                ticker_options.append({
                    "Ticker": TICKER, "Expiration Date": exp_date, "Strike Price": strike,
                    "Bid Price": bid_price, "Delta": delta, "IV": iv, "COP Short": cop_short,
                    "Theta": theta, "Open Interest": open_interest, "Volume": volume,
                    "Max Contracts": max_contracts, "Total Premium": total_premium,
                    "URL": rh_url
                })

                # Candidate score with realistic max contracts and COP weighting
                candidate_scores.append({
                    "Ticker": TICKER,
                    "Strike": strike,
                    "Expiration": exp_date,
                    "Max Contracts": max_contracts,
                    "Total Premium": total_premium,
                    "Score": total_premium*cop_short,
                    "COP Short": cop_short,
                    "URL": rh_url
                })

        # Individual alert message
        msg_lines = [f"📊 <a href='{rh_url}'>{TICKER}</a> current: ${current_price:.2f}"]
        if ticker_options:
            for p in ticker_options:
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
        else:
            msg_lines.append("⚠️ No valid put options found for this ticker (all bids below MIN_PRICE or top 2 strikes)")

        current_group.append("\n".join(msg_lines))
        if len(current_group) >= GROUP_SIZE:
            send_telegram_message("\n\n".join(current_group))
            current_group = []

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")

if current_group:
    send_telegram_message("\n\n".join(current_group))

# ------------------ CANDIDATE SCORE ALERT ------------------
if candidate_scores:
    top_candidates = sorted(candidate_scores, key=lambda x: x['Score'], reverse=True)[:10]
    msg_lines = ["📊 <b>Top 10 Candidate Scores</b>"]
    for c in top_candidates:
        msg_lines.append(
            f"{c['Ticker']} | Exp: {c['Expiration']} | Strike: {c['Strike']} | "
            f"Max Contracts: {c['Max Contracts']} | Total Premium: ${c['Total Premium']:.2f} | "
            f"Score: {c['Score']:.2f}\n" +
            "────────────────────────────"
        )
    send_telegram_message("\n".join(msg_lines))

# ------------------ BEST ALERT ------------------
if top_candidates:
    best_opt = top_candidates[0]
    next_best_cop = top_candidates[1]['COP Short'] if len(top_candidates) > 1 else None
    # Adjusted score with multiple contract logic if COP within 5%
    adj_score = best_opt['Score']
    if next_best_cop and abs(best_opt['COP Short'] - next_best_cop) <= 0.05:
        if best_opt['Max Contracts'] > 1:
            adj_score *= 1 + (best_opt['Max Contracts'] - 1) * 0.5

    msg_lines = [
        "🔥 <b>Best Cash-Secured Put (Adjusted Score)</b>",
        f"📊 <a href='{best_opt['URL']}'>{best_opt['Ticker']}</a>",
        f"💲 Strike : {best_opt['Strike']}",
        f"✅ Expiration: {best_opt['Expiration']}",
        f"🎯 COP Short: {best_opt['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {best_opt['Max Contracts']} | Total Premium: ${best_opt['Total Premium']:.2f}",
        f"📝 Adjusted Score: {adj_score:.2f}"
    ]
    send_telegram_message("\n".join(msg_lines))
