# robinhood_sell_puts.py (refactored)

import sys, subprocess, os, io
from datetime import datetime, timedelta

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
def ensure_package(pkg):
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in ["yaml", "yfinance", "pandas", "numpy", "matplotlib", "robin_stocks", "requests", "lxml"]:
    ensure_package(pkg)

import yaml, pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates, requests
import robin_stocks.robinhood as r
import yfinance as yf

# ------------------ LOAD CONFIG ------------------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

TICKERS_FILE = cfg["tickers_file"]
NUM_EXPIRATIONS = cfg["num_expirations"]
MIN_PRICE = cfg["min_price"]
HV_PERIOD = cfg["hv_period"]
CANDLE_WIDTH = cfg["candle_width"]
LOW_DAYS = cfg["low_days"]
EXPIRY_LIMIT_DAYS = cfg["expiry_limit_days"]
OPTION_FILTERS = cfg.get("option_filters", {})

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILITIES ------------------
def send_telegram(msg, photo=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
    if photo:
        requests.post(url+"sendPhoto", files={"photo": photo}, data={"chat_id": TELEGRAM_CHAT_ID, "caption": msg, "parse_mode": "HTML"})
    else:
        requests.post(url+"sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})

def prepare_historicals(df):
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="B"))
    df.index = df.index.tz_localize(None)
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
    df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
    df['volume'] = df['volume'].fillna(0)
    return df

def get_prepared_historicals(ticker, interval="day", span="month"):
    data = r.stocks.get_stock_historicals(ticker, interval=interval, span=span, bounds='regular')
    df = pd.DataFrame(data)
    df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
    df.set_index('begins_at', inplace=True)
    df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
    return prepare_historicals(df)

def plot_candlestick(df, current_price, last_14_low, strikes=None, exp_date=None):
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    for i in range(len(df)):
        color = 'lime' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
        ax.add_patch(plt.Rectangle((mdates.date2num(df.index[i])-CANDLE_WIDTH/2,
                                   min(df['open'].iloc[i], df['close'].iloc[i])),
                                   CANDLE_WIDTH, abs(df['close'].iloc[i]-df['open'].iloc[i]), color=color))
        ax.plot([mdates.date2num(df.index[i])]*2, [df['low'].iloc[i], df['high'].iloc[i]], color=color, linewidth=1)
    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=2, label=f'14-day Low: ${last_14_low:.2f}')
    if strikes:
        for s in strikes: ax.axhline(s, color='cyan', linestyle='--', linewidth=1.5, label=f'Strike: ${s:.2f}')
    if exp_date:
        exp_dt = pd.to_datetime(exp_date).tz_localize(None)
        if df.index.min() <= exp_dt <= df.index.max():
            ax.axvline(mdates.date2num(exp_dt), color='orange', linestyle='--', linewidth=2,
                       label=f'Expiration: {exp_dt.strftime("%d-%m-%y")}')
    ax.set_ylabel('Price ($)', color='white'); ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', alpha=0.3); ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45); plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black'); buf.seek(0); plt.close()
    return buf

def calculate_score(opt, buying_power):
    days = max((pd.to_datetime(opt['Expiration Date']).date() - datetime.now().date()).days, 1)
    hv_val = max(opt.get('HV', 0.05), 0.05)
    iv_val = opt.get('IV', 1.0) or 1.0
    iv_hv_ratio = iv_val / hv_val
    liquidity_weight = 1 + 0.5*(opt['Volume']+opt['Open Interest'])/1000
    max_contracts = max(1, int(buying_power//(opt['Strike Price']*100)))
    total_premium = opt['Bid Price']*100*max_contracts*opt['COP Short']
    return total_premium*iv_hv_ratio*liquidity_weight/days

# ------------------ MAIN SCRIPT ------------------
if not os.path.exists(TICKERS_FILE): raise FileNotFoundError(f"{TICKERS_FILE} not found.")
with open(TICKERS_FILE) as f: TICKERS = [line.strip().upper() for line in f if line.strip()]

r.login(USERNAME, PASSWORD)
today, cutoff = datetime.now().date(), datetime.now().date() + timedelta(days=EXPIRY_LIMIT_DAYS)

# ---------- Earnings/Dividend Risk Check ----------
safe_tickers, risky_msgs = [], []
for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        msg_parts, has_event = [f"📊 <b>{ticker}</b>"], False
        # Dividends
        try:
            future_divs = stock.dividends[stock.dividends.index.date >= today]
            if not future_divs.empty and future_divs.index.min().date() <= cutoff:
                msg_parts.append(f"⚠️ 💰 Dividend on {future_divs.index.min().date().strftime('%d-%m-%y')}")
                has_event = True
        except: pass
        # Earnings
        try:
            ed = stock.get_earnings_dates(limit=2)
            if not ed.empty and today <= ed.index.min().date() <= cutoff:
                msg_parts.append(f"⚠️ 📢 Earnings on {ed.index.min().date().strftime('%d-%m-%y')}")
                has_event = True
        except: pass
        if has_event: risky_msgs.append(" | ".join(msg_parts))
        else: safe_tickers.append(ticker)
    except Exception as e: risky_msgs.append(f"⚠️ <b>{ticker}</b> error: {e}")

summary = []
if risky_msgs: summary.append("⚠️ <b>Risky Tickers</b>\n"+ "\n".join(risky_msgs))
else: summary.append("✅ <b>No risky tickers found 🎉</b>")
if safe_tickers:
    safe_rows = [", ".join([f"<b>{t}</b>" for t in safe_tickers[i:i+4]]) for i in range(0,len(safe_tickers),4)]
    summary.append("✅ <b>Safe Tickers</b>\n"+ "\n".join(safe_rows))
summary.append(f"\n📊 Summary: ✅ Safe: {len(safe_tickers)} | ⚠️ Risky: {len(risky_msgs)}")
send_telegram("\n".join(summary))

# ---------- Robinhood Options ----------
account_data = r.profiles.load_account_profile()
buying_power = float(account_data['cash_available_for_withdrawal'])
all_options, candidate_scores = [], []

for TICKER in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(TICKER)[0])
        df = get_prepared_historicals(TICKER)
        last_14_low = df['low'][-LOW_DAYS:].min()
        df['returns'] = np.log(df['close']/df['close'].shift(1))
        hv = df['returns'].rolling(HV_PERIOD).std().iloc[-1]*np.sqrt(252)

        all_puts = r.options.find_tradable_options(TICKER, optionType="put")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_puts]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d,"%Y-%m-%d").date() <= cutoff][:NUM_EXPIRATIONS]

        candidate_puts = []
        for exp_date in exp_dates:
            puts = [opt for opt in all_puts if opt['expiration_date']==exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts if float(opt['strike_price']) < current_price], reverse=True)
            chosen_strikes = strikes_below[1:4] if len(strikes_below)>1 else strikes_below
            for opt in puts:
                strike = float(opt['strike_price'])
                if strike not in chosen_strikes: continue
                md = r.options.get_option_market_data_by_id(opt['id'])[0]
                bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                if bid_price < OPTION_FILTERS.get("min_bid_price", MIN_PRICE): continue
                delta = float(md.get('delta') or 0.0)
                cop = float(md.get('chance_of_profit_short') or 0.0)
                vol, oi = int(md.get('volume') or 0), int(md.get('open_interest') or 0)
                if (strike - last_14_low)/last_14_low < OPTION_FILTERS.get("min_distance_from_low", 0.01): continue
                candidate_puts.append({"Ticker":TICKER,"Current Price":current_price,"Expiration Date":exp_date,
                                       "Strike Price":strike,"Bid Price":bid_price,"Delta":delta,"COP Short":cop,
                                       "Open Interest":oi,"Volume":vol,"HV":hv})

        selected_puts = sorted(candidate_puts, key=lambda x:x['COP Short'], reverse=True)[:3]
        all_options.extend(selected_puts)

        if selected_puts:
            msg_lines = [f"📊 {TICKER} current: ${current_price:.2f}"]
            for idx, p in enumerate(selected_puts, start=1):
                msg_lines.append(f"<b>Option {idx}</b> | Exp: {p['Expiration Date']} | Strike: ${p['Strike Price']} | Bid: ${p['Bid Price']:.2f} | Delta: {p['Delta']:.3f} | COP: {p['COP Short']*100:.1f}%")
            send_telegram("\n".join(msg_lines))
            candidate_scores.append((max(selected_puts,key=lambda x:x['COP Short']), calculate_score(max(selected_puts,key=lambda x:x['COP Short']), buying_power)))
        else: send_telegram(f"⚠️ No valid options found for {TICKER}")
    except Exception as e:
        send_telegram(f"⚠️ Error processing {TICKER}: {e}")

# ---------- Candidate Scores ----------
if candidate_scores:
    candidate_scores.sort(key=lambda x:x[1], reverse=True)
    msg = f"<b>📊 Candidate Put Scores</b>\n💰 Available: ${buying_power:,.2f}\n" + "─"*30 + "\n"
    for put, score in candidate_scores:
        max_contracts = max(1, int(buying_power // (put['Strike Price']*100)))
        total_premium = put['Bid Price']*100*max_contracts
        msg += f"{put['Ticker']} | Exp: {put['Expiration Date']} | Strike: ${put['Strike Price']} | Max Contracts: {max_contracts} | Premium: ${total_premium:.2f} | Score: {score:.2f}\n"
    send_telegram(msg)

# ---------- Best Overall ----------
if all_options:
    best = max(all_options, key=lambda o: calculate_score(o, buying_power))
    max_contracts = max(1, int(buying_power//(best['Strike Price']*100)))
    total_premium = best['Bid Price']*100*max_contracts
    msg_lines = [
        "🔥 <b>Best Cash-Secured Put (Max Premium)</b>:",
        f"📊 {best['Ticker']} current: ${best['Current Price']:.2f}",
        f"✅ Expiration : {best['Expiration Date']}",
        f"💲 Strike    : {best['Strike Price']}",
        f"💰 Bid Price : ${best['Bid Price']:.2f}",
        f"🔺 Delta     : {best['Delta']:.3f}",
        f"🎯 COP Short : {best['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}"
    ]
    df = get_prepared_historicals(best['Ticker'])
    last_14_low = df['low'][-LOW_DAYS:].min()
    buf = plot_candlestick(df, best['Current Price'], last_14_low, [best['Strike Price']], best['Expiration Date'])
    send_telegram("\n".join(msg_lines), photo=buf)
