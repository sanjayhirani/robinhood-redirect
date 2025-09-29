# robinhood_finnhub_prefsample.py
"""
Fetch candidate tickers using Finnhub index constituents (broad universe),
filter by price ($1-$10) via Finnhub quote endpoint (minimizing Robinhood calls),
then use Robinhood for option chains and market data (async fetch), keeping
Telegram photo sending synchronous to avoid image mixups.
"""

import os
import time
import random
import asyncio
import aiohttp
import requests
import robin_stocks.robinhood as r
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

# ---------------- CONFIG ----------------
NUM_TICKERS = 20               # target sample size (not minimum)
PRICE_MIN = 1.0
PRICE_MAX = 10.0
NUM_EXPIRATIONS = 3
NUM_PUTS = 3
MIN_BID = 0.05                 # pre-filter tiny bids
MAX_OTM_PCT = 0.20             # skip strikes more than 20% below spot
EXPIRY_LIMIT_DAYS = 21
LOW_DAYS = 14
CANDLE_WIDTH = 0.6

# SECRETS via env (set in GH Actions secrets)
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")
USERNAME = os.environ.get("RH_USERNAME")
PASSWORD = os.environ.get("RH_PASSWORD")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not FINNHUB_KEY:
    raise SystemExit("Missing FINNHUB_API_KEY in environment. Add it to GitHub Secrets.")

today = datetime.now().date()
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

# ---------------- Utility: Telegram ----------------
def send_telegram_message(txt):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": txt, "parse_mode": "HTML"}
    )

def send_telegram_photo(buf, caption):
    # buf must be a BytesIO opened at position 0
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={"photo": ("chart.png", buf)},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    )

# ---------------- Utility: Candlestick plot ----------------
def plot_candlestick(df, current_price, last_14_low, selected_strikes=None, exp_date=None):
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    for i in range(len(df)):
        o = df['open'].iloc[i]; c = df['close'].iloc[i]; h = df['high'].iloc[i]; l = df['low'].iloc[i]
        color = 'lime' if c >= o else 'red'
        ax.add_patch(plt.Rectangle((mdates.date2num(df.index[i]) - CANDLE_WIDTH/2, min(o,c)),
                                   CANDLE_WIDTH, abs(c - o), color=color))
        ax.plot([mdates.date2num(df.index[i]), mdates.date2num(df.index[i])], [l, h], color=color, linewidth=1)
    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.2, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=1.5, label=f'14-day Low: ${last_14_low:.2f}')
    if selected_strikes:
        for s in selected_strikes:
            ax.axhline(s, color='cyan', linestyle='--', linewidth=1, label=f'Strike: ${s:.2f}')
    if exp_date:
        try:
            exp_obj = pd.to_datetime(exp_date).tz_localize(None)
            if df.index.min() <= exp_obj <= df.index.max():
                ax.axvline(mdates.date2num(exp_obj), color='orange', linestyle='--', linewidth=1.5, label=f'Expiration: {exp_obj.strftime("%d-%m-%y")}')
        except Exception:
            pass
    ax.set_ylabel('Price ($)', color='white'); ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45); plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black'); buf.seek(0); plt.close()
    return buf

# ---------------- Finnhub helpers ----------------
FINNHUB_BASE = "https://finnhub.io/api/v1"

def finnhub_get(path, params=None):
    if params is None:
        params = {}
    params['token'] = FINNHUB_KEY
    url = f"{FINNHUB_BASE}/{path}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def get_index_constituents(index_symbol):
    # index_symbol examples: ^GSPC (S&P 500), ^NDX (Nasdaq 100), ^RUT (Russell 2000), ^DJI (Dow)
    try:
        data = finnhub_get("index/constituents", {"symbol": index_symbol})
        # returns {"constituents": ["AAPL","MSFT",...], "symbol": "^GSPC"}
        if isinstance(data, dict):
            return data.get("constituents") or data.get("constituents", [])
        return []
    except Exception:
        return []

def quote_price(symbol):
    try:
        q = finnhub_get("quote", {"symbol": symbol})
        # q has 'c' current price (float) or 0
        return float(q.get("c") or 0.0)
    except Exception:
        return 0.0

# ---------------- Build universe via indices ----------------
def build_finnhub_universe():
    indices = ["^GSPC", "^NDX", "^RUT", "^DJI"]  # S&P500, Nasdaq100, Russell2000, Dow
    symbols = []
    for idx in indices:
        try:
            part = get_index_constituents(idx)
            if part:
                symbols.extend(part)
        except Exception:
            pass
        time.sleep(0.15)  # be polite
    # dedupe and uppercase
    symbols = sorted({s.upper() for s in symbols if isinstance(s, str)})
    return symbols

# ---------------- Use Finnhub to price-filter and pick tickers ----------------
def pick_price_filtered_tickers(universe, target=NUM_TICKERS):
    candidates = []
    # Randomize universe so we don't always pick same subset of index constituents
    random.shuffle(universe)
    for s in universe:
        try:
            p = quote_price(s)
            if PRICE_MIN <= p <= PRICE_MAX:
                candidates.append(s)
            if len(candidates) >= target * 3:  # fetch a few extra to give randomness
                break
        except Exception:
            pass
        time.sleep(0.05)
    random.shuffle(candidates)
    return candidates[:target]

# ------------------ Robinhood login ------------------
r.login(USERNAME, PASSWORD)

# ------------------ Earnings/Dividend check using yfinance (events only) ------------------
import yfinance as yf
def earnings_dividend_filter(tickers):
    safe = []
    risky_msgs = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            has_event = False
            parts = [f"📊 <b>{t}</b>"]
            try:
                if not stock.dividends.empty:
                    div_date = stock.dividends.index[-1].date()
                    if today <= div_date <= cutoff:
                        parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                        has_event = True
            except Exception:
                pass
            try:
                ed = stock.get_earnings_dates(limit=2)
                if hasattr(ed, "empty"):
                    if not ed.empty:
                        edate = ed.index.min().date()
                        if today <= edate <= cutoff:
                            parts.append(f"⚠️ 📢 Earnings on {edate.strftime('%d-%m-%y')}")
                            has_event = True
                else:
                    # catch other return shapes
                    pass
            except Exception:
                pass

            if has_event:
                risky_msgs.append(" | ".join(parts))
            else:
                safe.append(t)
        except Exception:
            risky_msgs.append(f"⚠️ {t} error")
    # send summary
    summary = []
    if risky_msgs:
        summary.append("⚠️ <b>Risky Tickers</b>\n" + "\n".join(risky_msgs))
    else:
        summary.append("✅ <b>No risky tickers found</b>")
    if safe:
        rows = [", ".join([f"<b>{s}</b>" for s in safe[i:i+4]]) for i in range(0, len(safe), 4)]
        summary.append("✅ <b>Safe Tickers</b>\n" + "\n".join(rows))
    send_telegram_message("\n\n".join(summary))
    return safe

# ------------------ Async Robinhood option fetch for selected tickers ------------------
async def fetch_options_for_ticker(session, ticker):
    """
    This function runs synchronously inside asyncio but uses robin_stocks for option details.
    We wrap it in async to run concurrently (calls are blocking though — still gives concurrency
    benefits if some network operations release the GIL). robin_stocks functions are HTTP calls.
    """
    try:
        current_price = float(r.stocks.get_latest_price(ticker)[0])
    except Exception:
        return []

    # get historical for plotting (1 month)
    try:
        historicals = r.stocks.get_stock_historicals(ticker, interval='day', span='month', bounds='regular')
        df = pd.DataFrame(historicals)
        df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
        df.set_index('begins_at', inplace=True)
        df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
        df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(all_days)
        df['close'] = df['close'].ffill()
        df['open'] = df['open'].fillna(df['close'])
        df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
        df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
        df['volume'] = df['volume'].fillna(0)
        last_14_low = df['low'][-LOW_DAYS:].min()
    except Exception:
        df = None
        last_14_low = None

    try:
        all_puts = r.options.find_tradable_options(ticker, optionType="put")
    except Exception:
        return []

    # collect expirations of interest
    exp_dates = sorted({opt['expiration_date'] for opt in all_puts})
    exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff][:NUM_EXPIRATIONS]

    candidate_puts = []
    for exp_date in exp_dates:
        puts_for_exp = [opt for opt in all_puts if opt['expiration_date'] == exp_date]
        strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
        chosen_strikes = strikes_below[2:5]  # skip first two below spot
        for opt in puts_for_exp:
            strike = float(opt['strike_price'])
            if strike not in chosen_strikes:
                continue
            # prefilter deep OTM
            if strike < current_price * (1 - MAX_OTM_PCT):
                continue
            try:
                md = r.options.get_option_market_data_by_id(opt['id'])
                if not md:
                    continue
                md = md[0]
            except Exception:
                continue

            bid = float(md.get('bid_price') or md.get('mark_price') or 0.0)
            if bid < MIN_BID:
                continue

            delta = float(md.get('delta') or 0.0)
            iv = float(md.get('implied_volatility') or 0.0)
            cop_short = float(md.get('chance_of_profit_short') or 0.0)

            candidate_puts.append({
                "Ticker": ticker,
                "Current Price": current_price,
                "Expiration Date": exp_date,
                "Strike Price": strike,
                "Bid Price": bid,
                "Delta": delta,
                "IV": iv,
                "COP Short": cop_short,
                "URL": f"https://robinhood.com/stocks/{ticker}",
                "DFL": current_price - (df['low'].min() if df is not None else 0.0),
                "df": df,
                "last_14_low": last_14_low
            })

    await asyncio.sleep(0)  # yield control
    return candidate_puts

async def gather_options(tickers):
    tasks = []
    # create a single aiohttp session for respectful rate-limited tasks (we mostly use robin_stocks)
    async with aiohttp.ClientSession() as sess:
        for t in tickers:
            tasks.append(fetch_options_for_ticker(sess, t))
        results = await asyncio.gather(*tasks)
    # flatten
    all_opts = [item for sub in results for item in sub]
    return all_opts

# ------------------ Best-option scoring: weekly-premium * COP Short ----------------
def compute_best_option(all_options):
    best = None
    best_score = -1.0
    for opt in all_options:
        try:
            exp_date_obj = datetime.strptime(opt['Expiration Date'], "%Y-%m-%d").date()
            days_to_exp = max((exp_date_obj - today).days, 1)
            weekly_premium = opt['Bid Price'] / (days_to_exp / 7.0)
            score = weekly_premium * opt['COP Short']
            if score > best_score:
                best_score = score
                best = dict(opt, weekly_premium=weekly_premium, score=score)
        except Exception:
            continue
    return best

# ------------------ MAIN flow ----------------
def main():
    # 1) build universe from Finnhub indices
    universe = build_finnhub_universe()
    if not universe:
        # fallback: sample S&P500 from robin_stocks instruments? But we'll error out for clarity.
        send_telegram_message("⚠️ Unable to build universe from Finnhub indices. Check FINNHUB_API_KEY and network.")
        return

    # 2) price filter via Finnhub quotes
    price_filtered = pick_price_filtered_tickers(universe, NUM_TICKERS * 2)  # get extra
    if not price_filtered:
        send_telegram_message("⚠️ No tickers found in price range $1-$10 from Finnhub indices.")
        return

    # limit to first N candidates to reduce load
    sample = price_filtered[:NUM_TICKERS * 2]
    random.shuffle(sample)
    sample = sample[:NUM_TICKERS]

    # 3) events check with yfinance (dividends/earnings)
    safe = earnings_dividend_filter(sample)
    if not safe:
        send_telegram_message("⚠️ No safe tickers after earnings/dividend filtering.")
        return

    # 4) async fetch options data from Robinhood (fast), prefiltered earlier
    all_options = asyncio.run(gather_options(safe))

    if not all_options:
        send_telegram_message("⚠️ No candidate options found after scanning safe tickers.")
        return

    # 5) compute best
    best = compute_best_option(all_options)
    if not best:
        send_telegram_message("⚠️ No best option found after scoring.")
        return

    # Prepare and send best alert (synchronous photo sending)
    df = best.get("df")
    buf = None
    if df is not None:
        buf = plot_candlestick(df, best["Current Price"], best.get("last_14_low") or 0.0, selected_strikes=[best["Strike Price"]], exp_date=best["Expiration Date"])

    caption = (
        f"🔥 <b>Best Cash-Secured Put</b>\n"
        f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}\n"
        f"✅ Expiration: {best['Expiration Date']}\n"
        f"💲 Strike: ${best['Strike Price']:.2f}\n"
        f"💰 Bid: ${best['Bid Price']:.2f}\n"
        f"📈 IV: {best['IV']*100:.1f}% | 🎯 COP Short: {best['COP Short']*100:.1f}%\n"
        f"📅 Weekly-premium equiv: ${best.get('weekly_premium',0):.2f} | Score: {best.get('score',0):.4f}"
    )

    if buf:
        send_telegram_photo(buf, caption)
    else:
        send_telegram_message(caption)

if __name__ == "__main__":
    main()
