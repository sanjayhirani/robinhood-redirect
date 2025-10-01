# robinhood_leaps.py

import os
import subprocess
import requests
import robin_stocks.robinhood as r
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from datetime import datetime, timedelta
import yfinance as yf
import pickle
import time
import functools

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", pkg_name])

for pkg in ["pandas","matplotlib","requests","robin_stocks","yfinance"]:
    ensure_package(pkg)

# ------------------ CONFIG ------------------
LEAPS_TOP_N = 20           # scan top 20
TOP_ALERTS = 5
MIN_PRICE = 5
FIGSIZE = [12,6]
BG_COLOR = "black"
CURRENT_COLOR = "magenta"
STRIKE_COLOR = "cyan"
CACHE_DIR = ".cache"
OPTIONABLE_CACHE_FILE = os.path.join(CACHE_DIR, "optionable_tickers.pkl")
os.makedirs(CACHE_DIR, exist_ok=True)
DELAY_BETWEEN_RH_CALLS = 2.0  # seconds

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ TELEGRAM UTILITIES ------------------
def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

def send_telegram_photo(buf, caption):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo': buf},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    )

# ------------------ PLOTTING ------------------
def plot_stock_with_strike(df, current_price, strike_price):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.plot(df.index, df['Close'], color='lime', label='Close')
    ax.axhline(current_price, color=CURRENT_COLOR, linestyle='--', linewidth=1.5, label='Current Price')
    ax.axhline(strike_price, color=STRIKE_COLOR, linestyle='--', linewidth=1.5, label='LEAPS Strike')
    ax.set_ylabel('Price ($)', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=BG_COLOR)
    buf.seek(0)
    plt.close()
    return buf

# ------------------ RETRY DECORATOR FOR RATE LIMIT ------------------
def retry_on_rate_limit(max_retries=3, wait_sec=5):
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "Too many requests" in str(e):
                        retries += 1
                        time.sleep(wait_sec)
                    else:
                        raise
            raise Exception("Max retries exceeded for rate-limited request")
        return wrapper
    return decorator

@retry_on_rate_limit()
def get_option_market_data(opt_id):
    return r.options.get_option_market_data_by_id(opt_id)

# ------------------ FETCH NASDAQ + NYSE ------------------
def fetch_all_tickers():
    nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    nyse_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"

    nasdaq_df = pd.read_csv(nasdaq_url, sep='|', dtype=str)
    nyse_df = pd.read_csv(nyse_url, sep='|', dtype=str)

    nasdaq_tickers = nasdaq_df['Symbol'].astype(str).tolist()
    nyse_tickers = nyse_df['ACT Symbol'].astype(str).tolist()
    all_tickers = list(set(nasdaq_tickers + nyse_tickers))
    all_tickers = [str(t).strip().upper() for t in all_tickers if str(t).strip().upper() not in ("", "NAN")]
    return all_tickers

# ------------------ DETERMINE OPTIONABLE TICKERS ------------------
def get_optionable_tickers():
    if os.path.exists(OPTIONABLE_CACHE_FILE):
        with open(OPTIONABLE_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    tickers = fetch_all_tickers()
    optionable = []
    print("Checking which tickers have options (this may take a few minutes)...")
    for t in tickers:
        try:
            yf_ticker = yf.Ticker(t)
            if len(yf_ticker.options) > 0:
                optionable.append(t)
        except:
            continue
    with open(OPTIONABLE_CACHE_FILE, "wb") as f:
        pickle.dump(optionable, f)
    return optionable

# ------------------ SCORE CANDIDATES ------------------
def score_ticker(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")['Close']
        if hist.empty or len(hist) < 200:
            return None
        current_price = hist.iloc[-1]
        if current_price < MIN_PRICE:
            return None
        sma50 = hist.rolling(50).mean().iloc[-1]
        sma200 = hist.rolling(200).mean().iloc[-1]
        reasons = []
        score = 0
        if current_price > sma200:
            reasons.append("✅ Above 200-day SMA")
            score += 2
        if current_price > sma50:
            reasons.append("✅ Above 50-day SMA")
            score += 1
        earnings_growth = t.info.get('earningsQuarterlyGrowth') or 0
        if earnings_growth > 0:
            reasons.append(f"✅ Positive earnings growth: {earnings_growth*100:.1f}%")
            score += 2
        avg_vol = hist.iloc[-63:].mean() if len(hist) >= 63 else hist.mean()
        score += min(avg_vol/1e6,2)
        if score > 0:
            return {"Ticker": ticker, "Score": score, "Reasons": reasons}
        return None
    except:
        return None

# ------------------ MAIN ------------------
print("Fetching optionable tickers...")
optionable_tickers = get_optionable_tickers()
print(f"Total optionable tickers: {len(optionable_tickers)}")

candidates = []
for t in optionable_tickers:
    res = score_ticker(t)
    if res:
        candidates.append(res)

candidates.sort(key=lambda x: x['Score'], reverse=True)
top_candidates = candidates[:LEAPS_TOP_N]

# Save top N to file
with open("leapstickers.txt", "w") as f:
    for c in top_candidates:
        f.write(c["Ticker"] + "\n")

# ------------------ TELEGRAM ALERTS ------------------
alerts_candidates = top_candidates[:TOP_ALERTS]

r.login(USERNAME, PASSWORD)
summary = f"📊 LEAPS Scan Complete\nTotal optionable tickers scanned: {len(candidates)}\nTop candidates alerted: {len(alerts_candidates)}"
send_telegram_message(summary)

today = datetime.now().date()

for candidate in alerts_candidates:
    ticker = candidate['Ticker']
    try:
        stock = yf.Ticker(ticker)
        current_price = float(stock.history(period='1d')['Close'].iloc[-1])
        all_options = r.options.find_tradable_options(ticker, optionType="call")
        leaps_options = [opt for opt in all_options if datetime.strptime(opt['expiration_date'], "%Y-%m-%d").date() >= today + timedelta(days=365)]
        if not leaps_options:
            continue
        best_option = None
        best_score = -1
        for opt in leaps_options:
            strike = float(opt['strike_price'])
            md = get_option_market_data(opt['id'])[0]
            bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
            open_interest = int(md.get('open_interest') or 0)
            volume = int(md.get('volume') or 0)
            if bid_price < 1:
                continue
            proximity_score = max(0, 1 - abs(strike - current_price)/current_price)
            liquidity_score = (open_interest + volume)/1000
            score = len(candidate['Reasons'])*2 + proximity_score*2 + liquidity_score + (bid_price/current_price)
            if score > best_score:
                best_score = score
                best_option = {"Ticker": ticker, "Strike": strike, "Expiry": opt['expiration_date'], "Bid": bid_price, "Reasons": candidate['Reasons']}
        if best_option:
            hist6mo = stock.history(period='6mo')
            buf = plot_stock_with_strike(hist6mo, current_price, best_option['Strike'])
            msg_lines = [
                f"🔥 <b>LEAPS Candidate</b>: {best_option['Ticker']}",
                f"Current Price: ${current_price:.2f}",
                "Reasons:",
                *best_option['Reasons'],
                "Suggested LEAPS:",
                f"📅 Expiry: {best_option['Expiry']}",
                f"💲 Strike: {best_option['Strike']}",
                f"💰 Premium: ${best_option['Bid']:.2f}"
            ]
            send_telegram_photo(buf, "\n".join(msg_lines))
    except Exception as e:
        send_telegram_message(f"{ticker} error: {e}")

    # Avoid rate limit
    time.sleep(DELAY_BETWEEN_RH_CALLS)
