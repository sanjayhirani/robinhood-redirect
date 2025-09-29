# robinhood_sell_puts_main_optimized.py

# ------------------ AUTO-INSTALL DEPENDENCIES (optional) ------------------
import sys
import subprocess

try:
    import yfinance
except ImportError:
    print("yfinance not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance

try:
    import lxml
except ImportError:
    print("lxml not found. Installing...")
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
import numpy as np
import pandas as pd

# ------------------ CONFIG ------------------
TICKERS = ["SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC", "HTZ", "RZLV", "NVTS"]
NUM_EXPIRATIONS = 3
NUM_PUTS = 3
MIN_PRICE = 0.10
HV_PERIOD = 21
CANDLE_WIDTH = 0.6
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILITY FUNCTIONS ------------------
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

for ticker in TICKERS:
    try:
        stock = yfinance.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False

        try:
            future_divs = stock.dividends[stock.dividends.index.date >= today]
            if not future_divs.empty:
                div_date = future_divs.index.min().date()
                if div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except Exception as e:
            msg_parts.append(f"⚠️ Dividend check error: {e}")

        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index.min().date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"⚠️ 📢 Earnings on {earnings_date.strftime('%d-%m-%y')}")
                    has_event = True
        except Exception as e:
            msg_parts.append(f"⚠️ Earnings check error: {e}")

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

# Get buying power
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

                if bid_price >= MIN_PRICE:
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
                        "Dist from Low": dist_from_low,
                        "URL": rh_url,
                        "HV": hv
                    })

        selected_puts = sorted(candidate_puts, key=lambda x: x['COP Short'], reverse=True)[:NUM_PUTS]
        all_options.extend(selected_puts)

        # Send individual ticker alert with line separators
        msg_lines = [f"📊 <a href='{rh_url}'>{TICKER}</a> current: ${current_price:.2f}"]
        for idx, p in enumerate(selected_puts, start=1):
            # max contracts for candidate score
            max_contracts = max(1, int(buying_power // (p['Strike Price'] * 100)))
            total_premium = p['Bid Price'] * 100 * max_contracts

            msg_lines.append(
                f"<b>Option {idx}:</b>\n"
                f"Exp: {p['Expiration Date']} | Strike: ${p['Strike Price']} | Bid: ${p['Bid Price']:.2f}\n"
                f"Delta: {p['Delta']:.3f} | IV: {p['IV']*100:.2f}% | COP: {p['COP Short']*100:.2f}%\n"
                f"Theta: {p['Theta']:.2f} | OI: {p['Open Interest']} | Vol: {p['Volume']}\n"
                "────────────────────────────"
            )

            # store candidate score info
            days_to_exp = (pd.to_datetime(p['Expiration Date']).date() - today).days
            iv_hv_ratio = p['IV']/p['HV'] if p['HV']>0 else 1.0
            liquidity_weight = 1 + 0.5*(p['Volume']+p['Open Interest'])/1000
            enhanced_score = total_premium * iv_hv_ratio * liquidity_weight / (days_to_exp**1.0)
            candidate_scores.append((p['Ticker'], p['Strike Price'], p['Expiration Date'], max_contracts, total_premium, enhanced_score))

        if selected_puts:
            buf = plot_candlestick(df, current_price, last_14_low, [p['Strike Price'] for p in selected_puts])
            send_telegram_photo(buf, "\n".join(msg_lines))

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")

# ------------------ SEND CANDIDATE SCORES MESSAGE ------------------
if candidate_scores:
    candidate_scores.sort(key=lambda x: x[5], reverse=True)  # sort by enhanced score
    score_msg = "<b>📊 Candidate Put Scores (Max Contracts & Premium)</b>\n"
    for t, strike, exp, max_ct, prem, score in candidate_scores:
        score_msg += f"{t} | Strike: ${strike} | Exp: {exp} | Max Contracts: {max_ct} | Total Premium: ${prem:.2f} | Score: {score:.2f}\n"
    send_telegram_message(score_msg)

# ------------------ BEST OPTION ALERT ------------------
def adjusted_score(opt):
    days_to_exp = (pd.to_datetime(opt['Expiration Date']).date() - today).days
    if days_to_exp <= 0:
        return 0
    iv_hv_ratio = opt['IV']/opt['HV'] if opt['HV']>0 else 1.0
    liquidity_weight = 1 + 0.5*(opt['Volume']+opt['Open Interest'])/1000
    max_contracts = max(1, int(buying_power // (opt['Strike Price'] * 100)))
    total_premium = opt['Bid Price'] * 100 * max_contracts * opt['COP Short']
    return total_premium * iv_hv_ratio * liquidity_weight / (days_to_exp**1.0)

if all_options:
    best = max(all_options, key=adjusted_score)
    max_contracts = max(1, int(buying_power // (best['Strike Price'] * 100)))
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
    buf = plot_candlestick(df, best['Current Price'], last_14_low, [best['Strike Price']], best['Expiration Date'])
    send_telegram_photo(buf, "\n".join(msg_lines))
