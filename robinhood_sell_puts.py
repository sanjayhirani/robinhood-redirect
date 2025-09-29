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
TICKERS = ["SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC", "HTZ", "RZLV", "NVTS", "CLOV", "RIG", "LDI", "SPCE", "AMC", "LAZR"]
NUM_EXPIRATIONS = 3
NUM_PUTS = 3
PRICE_ADJUST = 0.01
RISK_FREE_RATE = 0.05
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

def plot_candlestick(df, current_price, last_14_low, selected_strikes=None, exp_date=None, annotations=None, show_strikes=True):
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

    if annotations:
        for text, price in annotations:
            ax.text(df.index[-1], price, text, color='white', fontsize=10, verticalalignment='bottom')

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

# ------------------ SCAN TICKERS ------------------
safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

for ticker in TICKERS:
    try:
        stock = yfinance.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False
        # Dividends
        try:
            future_divs = stock.dividends[stock.dividends.index.date >= today]
            if not future_divs.empty:
                div_date = future_divs.index.min().date()
                if div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except:
            pass
        # Earnings
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

# ------------------ GET ALL OPTIONS ------------------
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
        df = prepare_historicals(df)
        last_14_low = df['low'][-LOW_DAYS:].min()

        all_puts = r.options.find_tradable_options(TICKER, optionType="put")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_puts]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:NUM_EXPIRATIONS]

        candidate_puts = []
        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date'] == exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            chosen_strikes = strikes_below[2:5]  # skip 2 closest below
            for opt in puts_for_exp:
                strike = float(opt['strike_price'])
                if strike not in chosen_strikes:
                    continue
                option_id = opt['id']
                market_data = r.options.get_option_market_data_by_id(option_id)
                if not market_data:
                    continue
                md = market_data[0]
                bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                delta = float(md.get('delta') or 0.0)
                iv = float(md.get('implied_volatility') or 0.0)
                cop_short = float(md.get('chance_of_profit_short') or 0.0)
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
                        "URL": rh_url
                    })
        selected_puts = sorted(candidate_puts, key=lambda x: x['COP Short'], reverse=True)[:NUM_PUTS]
        all_options.extend(selected_puts)
    except:
        continue

# ------------------ CALCULATE SCORES ------------------
candidate_scores = []
buying_power = float(r.profiles.load_account_profile(info='buying_power'))
for opt in all_options:
    days_to_exp = (pd.to_datetime(opt['Expiration Date']).date() - today).days
    max_contracts = max(1, int(buying_power // (opt['Strike Price']*100)))
    total_premium = opt['Bid Price']*100*max_contracts
    score = total_premium / (days_to_exp or 1)
    candidate_scores.append((opt['Ticker'], opt['Strike Price'], opt['Expiration Date'], max_contracts, total_premium, score))

# ------------------ SEND TOP 5 INDIVIDUAL ALERTS ------------------
top_per_ticker = {}
for t, strike, exp, max_ct, prem, score in candidate_scores:
    if t not in top_per_ticker or score > top_per_ticker[t][5]:
        top_per_ticker[t] = (t, strike, exp, max_ct, prem, score)
sorted_tickers_by_score = sorted(top_per_ticker.values(), key=lambda x: x[5], reverse=True)
top_5_tickers = [t[0] for t in sorted_tickers_by_score[:5]]

for TICKER in top_5_tickers:
    selected_puts = [p for p in all_options if p['Ticker'] == TICKER]
    if not selected_puts:
        continue
    current_price = selected_puts[0]['Current Price']
    historicals = r.stocks.get_stock_historicals(TICKER, interval='day', span='month', bounds='regular')
    df = pd.DataFrame(historicals)
    df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
    df.set_index('begins_at', inplace=True)
    df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    df.rename(columns={'open':'open','close':'close','high':'high','low':'low'}, inplace=True)
    df = prepare_historicals(df)
    last_14_low = df['low'][-LOW_DAYS:].min()

    msg_lines = [f"📊 <b>{TICKER}</b> current: ${current_price:.2f}"]
    for p in selected_puts:
        msg_lines.append(
            f"▪️ 📅 Exp: {p['Expiration Date']} | 💲 Strike: ${p['Strike Price']} | 💰 Bid: ${p['Bid Price']:.2f}\n"
            f"   🔺 Delta: {p['Delta']:.3f} | 📈 IV: {p['IV']*100:.2f}% | 🎯 COP: {p['COP Short']*100:.2f}%"
        )

    buf = plot_candlestick(df, current_price, last_14_low, [p['Strike Price'] for p in selected_puts])
    send_telegram_photo(buf, "\n".join(msg_lines))

# ------------------ SEND CANDIDATE SCORES ------------------
score_lines = ["📊 <b>Candidate Scores</b> (all tickers)"]
for t, strike, exp, max_ct, prem, score in sorted(candidate_scores, key=lambda x: x[5], reverse=True):
    score_lines.append(f"▪️ <b>{t}</b>\n"
                       f"   📅 Exp: {exp}\n"
                       f"   💲 Strike: ${strike}\n"
                       f"   📝 Max Contracts: {max_ct}\n"
                       f"   💰 Premium: ${prem:.2f}\n"
                       f"   ⭐ Score: {score:.2f}")

send_telegram_message("\n".join(score_lines))

# ------------------ SEND BEST OPTION ALERT ------------------
if all_options:
    best = max(all_options, key=lambda o: (o['Bid Price']*100*max(1,int(buying_power//(o['Strike Price']*100))) / max((pd.to_datetime(o['Expiration Date']).date()-today).days,1)))
    max_contracts = max(1, int(buying_power // (best['Strike Price']*100)))
    total_premium = best['Bid Price']*100*max_contracts

    msg_lines = [
        "🔥 <b>Best Cash-Secured Put (Max Premium)</b>:",
        f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}",
        f"✅ Expiration : {best['Expiration Date']}",
        f"💲 Strike    : {best['Strike Price']}",
        f"💰 Bid Price : ${best['Bid Price']:.2f}",
        f"🔺 Delta     : {best['Delta']:.3f}",
        f"📈 IV       : {best['IV']*100:.2f}%",
        f"🎯 COP Short : {best['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}"
    ]

    df = pd.DataFrame(r.stocks.get_stock_historicals(best['Ticker'], interval='day', span='month', bounds='regular'))
    df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
    df.set_index('begins_at', inplace=True)
    df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    df.rename(columns={'open':'open','close':'close','high':'high','low':'low'}, inplace=True)
    df = prepare_historicals(df)
    last_14_low = df['low'][-LOW_DAYS:].min()

    annotations = [
        (f"Max Contracts: {max_contracts}", best['Strike Price']*1.002),
        (f"Total Premium: ${total_premium:.2f}", best['Strike Price']*1.005)
    ]

    buf = plot_candlestick(df, best['Current Price'], last_14_low, [best['Strike Price']], best['Expiration Date'], annotations=annotations)
    send_telegram_photo(buf, "\n".join(msg_lines))
