# ------------------ AUTO-INSTALL DEPENDENCIES (optional) ------------------
import sys
import subprocess
import numpy as np

def install_package(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Only install missing packages
install_package("yfinance")
install_package("lxml")
install_package("robin_stocks")
install_package("matplotlib")
install_package("pandas")
install_package("requests")

# ------------------ OTHER IMPORTS ------------------
import os
import requests
import robin_stocks.robinhood as r
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
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
COP_THRESHOLD = 0.95  # 95% COP threshold for multi-contract adjustment

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
    for col in ['open','close','high','low','volume']:
        if col not in df.columns:
            df[col] = np.nan
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
import yfinance as yf

safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

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
        except: pass
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

# ------------------ GET OPTIONS & SCORE ------------------
all_options = []
for TICKER in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(TICKER)[0])
        rh_url = f"https://robinhood.com/stocks/{TICKER}"

        historicals = pd.DataFrame(r.stocks.get_stock_historicals(TICKER, interval='day', span='month', bounds='regular'))
        df = prepare_historicals(historicals.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'})) if not historicals.empty else pd.DataFrame()
        last_14_low = df['low'][-LOW_DAYS:].min() if not df.empty else current_price

        all_puts = r.options.find_tradable_options(TICKER, optionType="put")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_puts]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:NUM_EXPIRATIONS]

        candidate_puts = []
        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date'] == exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            chosen_strikes = strikes_below[2:5]
            for opt in puts_for_exp:
                strike = float(opt['strike_price'])
                if strike not in chosen_strikes:
                    continue
                md = r.options.get_option_market_data_by_id(opt['id'])
                if not md: continue
                md = md[0]
                bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                delta = float(md.get('delta') or 0.0)
                iv = float(md.get('implied_volatility') or 0.0)
                cop_short = float(md.get('chance_of_profit_short') or 0.0)
                if bid_price >= MIN_PRICE:
                    candidate_puts.append({"Ticker":TICKER,"Current Price":current_price,"Expiration Date":exp_date,"Strike Price":strike,"Bid Price":bid_price,"Delta":delta,"IV":iv,"COP Short":cop_short,"URL":rh_url})
        all_options.extend(candidate_puts)
    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")

for opt in all_options:
    days_to_exp = max((pd.to_datetime(opt['Expiration Date']).date() - today).days, 1)
    opt['Score'] = (opt['Bid Price'] * 100 * opt['COP Short']) / days_to_exp

sorted_options = sorted(all_options, key=lambda x: x['Score'], reverse=True)

# ------------------ TOP 5 INDIVIDUAL ALERTS ------------------
for opt in sorted_options[:5]:
    msg_lines = [f"📊 <b>{opt['Ticker']}</b> current: ${opt['Current Price']:.2f}"]
    msg_lines.append(f"📅 Exp: {opt['Expiration Date']} | 💲 Strike: ${opt['Strike Price']} | 💰 Bid: ${opt['Bid Price']:.2f} | 🔺 Delta: {opt['Delta']:.3f} | 📈 IV: {opt['IV']*100:.2f}% | 🎯 COP: {opt['COP Short']*100:.2f}%")
    msg_lines.append("—" * 20)  # horizontal separator between tickers
    send_telegram_message("\n".join(msg_lines))

# ------------------ CANDIDATE SCORES ALERT ------------------
score_lines = ["📊 Candidate Scores (all tickers)"]
for i, opt in enumerate(sorted_options):
    prefix = "🔥" if i == 0 else "⚡" if i <= 2 else "✅" if i <= 4 else "▫️"
    score_lines.append(f"{prefix} {opt['Ticker']} | Exp: {opt['Expiration Date']} | Strike: ${opt['Strike Price']} | Score: {opt['Score']:.2f}")
send_telegram_message("\n".join(score_lines))

# ------------------ BEST OPTION ALERT WITH MULTI-CONTRACT LOGIC ------------------
buying_power = float(r.profiles.load_account_profile(info='buying_power') or 0.0)
best_option = None
max_adjusted_score = -1

for i, opt in enumerate(sorted_options):
    max_contracts = int(buying_power // (opt['Strike Price'] * 100))
    total_premium = max_contracts * opt['Bid Price'] * 100
    multi_contract_score = opt['Score'] * max_contracts
    if max_contracts > 1:
        if i+1 < len(sorted_options) and opt['COP Short'] < 0.95*sorted_options[i+1]['COP Short']:
            multi_contract_score = opt['Score']  # don't boost if COP drops too much
    if multi_contract_score > max_adjusted_score:
        best_option = opt.copy()
        best_option['Max Contracts'] = max_contracts
        best_option['Total Premium'] = total_premium
        max_adjusted_score = multi_contract_score

if best_option:
    msg_lines = [
        "🔥 <b>Best Cash-Secured Put</b>:",
        f"📊 <a href='{best_option['URL']}'>{best_option['Ticker']}</a> current: ${best_option['Current Price']:.2f}",
        f"✅ Expiration : {best_option['Expiration Date']}",
        f"💲 Strike    : ${best_option['Strike Price']}",
        f"💰 Bid Price : ${best_option['Bid Price']:.2f}",
        f"🔺 Delta     : {best_option['Delta']:.3f}",
        f"📈 IV       : {best_option['IV']*100:.2f}%",
        f"🎯 COP Short : {best_option['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {best_option['Max Contracts']} | Premium: ${best_option['Total Premium']:.2f}"
    ]
    if not df.empty:
        buf = plot_candlestick(df, best_option['Current Price'], df['low'][-LOW_DAYS:].min(), [best_option['Strike Price']], best_option['Expiration Date'])
        send_telegram_photo(buf, "\n".join(msg_lines))
    else:
        send_telegram_message("\n".join(msg_lines))
