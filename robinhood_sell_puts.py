# robinhood_sell_puts_main_optimized.py

# ------------------ AUTO-INSTALL DEPENDENCIES (optional, minimal) ------------------
import sys
import subprocess
import importlib
import pkgutil

def ensure_package(pkg_name, import_name=None):
    import_name = import_name or pkg_name
    if not pkgutil.find_loader(import_name):
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

# Only ensure the packages you used
ensure_package("yfinance")
ensure_package("lxml")
ensure_package("robin_stocks")
ensure_package("matplotlib")
ensure_package("pandas")
ensure_package("requests")

# Optionally upgrade pip only if too old
try:
    import pip
    from packaging import version
    if version.parse(pip.__version__) < version.parse("21.0"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
except Exception:
    # If packaging or pip check fails, continue (not fatal)
    pass

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
NUM_PUTS = 3                # choose top 3 puts per ticker
MIN_PRICE = 0.10
HV_PERIOD = 21
CANDLE_WIDTH = 0.6
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21
COP_THRESHOLD = 0.95       # require COP >= 95% of max COP to allow multi-contract boost

# ------------------ SECRETS / ENV ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ TELEGRAM HELPERS ------------------
def send_telegram_photo(buf, caption):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo': ('image.png', buf.getvalue())},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    )

def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

# ------------------ PLOTTING ------------------
def plot_candlestick(df, current_price, last_14_low, selected_strikes=None, exp_date=None, annotations=None, show_strikes=True):
    # df: index is datetime (no tz)
    fig, ax = plt.subplots(figsize=(10,6))  # slightly narrower for mobile-friendly images
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # ensure numeric types
    df = df.copy()
    df[['open','close','high','low']] = df[['open','close','high','low']].astype(float)

    # draw candles manually (keeps your original styling)
    for i in range(len(df)):
        xi = mdates.date2num(df.index[i])
        o = df['open'].iloc[i]
        c = df['close'].iloc[i]
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]
        color = 'lime' if c >= o else 'red'
        ax.add_patch(plt.Rectangle((xi - CANDLE_WIDTH/2, min(o,c)), CANDLE_WIDTH, abs(c - o), color=color))
        ax.plot([xi, xi], [l, h], color=color, linewidth=1)

    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=1.5, label=f'14-day Low: ${last_14_low:.2f}')

    if show_strikes and selected_strikes:
        for strike in selected_strikes:
            ax.axhline(strike, color='cyan', linestyle='--', linewidth=1.25, label=f'Strike: ${strike:.2f}')

    if exp_date is not None:
        try:
            exp_date_obj = pd.to_datetime(exp_date).tz_localize(None)
            if df.index.min() <= exp_date_obj <= df.index.max():
                ax.axvline(mdates.date2num(exp_date_obj), color='orange', linestyle='--', linewidth=1.5, label=f'Expiration: {exp_date_obj.strftime("%d-%m-%y")}')
        except Exception:
            pass

    if annotations:
        for text, price in annotations:
            ax.text(df.index[-1], price, text, color='cyan', fontsize=10, ha='right', va='bottom', weight='bold')

    ax.set_ylabel('Price ($)', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', alpha=0.25)
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize='small')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
    buf.seek(0)
    plt.close(fig)
    return buf

# ------------------ HISTORICAL PREP (robust) ------------------
def prepare_historicals(df):
    # ensure index datetime and columns exist
    if df.empty:
        return df
    df = df.copy()
    # If index isn't datetime, try to find datetime column
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to use 'begins_at' column if present
        if 'begins_at' in df.columns:
            df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
            df.set_index('begins_at', inplace=True)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)

    # rename check already done in caller; ensure required cols present
    for col in ['open','close','high','low','volume']:
        if col not in df.columns:
            df[col] = np.nan

    # calendar business days index
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(all_days)
    df.index = df.index.tz_localize(None)

    # forward/back fill close, then fill other columns with safe logic
    # use infer_objects to avoid FutureWarning downcast behavior
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
    df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
    df['volume'] = df['volume'].fillna(0)
    return df.infer_objects(copy=False)

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

        # Dividend (future only)
        try:
            future_divs = stock.dividends[stock.dividends.index.date >= today]
            if not future_divs.empty:
                div_date = future_divs.index.min().date()
                if div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except Exception:
            pass

        # Earnings
        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index.min().date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"⚠️ 📢 Earnings on {earnings_date.strftime('%d-%m-%y')}")
                    has_event = True
        except Exception:
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
all_options = []          # list of option dicts
candidate_scores = []     # list of dicts used for candidate score message

# Get buying power (cash available for withdrawals)
account_data = r.profiles.load_account_profile()
buying_power = float(account_data.get('cash_available_for_withdrawal') or 0.0)

for TICKER in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(TICKER)[0])
        rh_url = f"https://robinhood.com/stocks/{TICKER}"

        # fetch historicals and prepare safely
        historicals = r.stocks.get_stock_historicals(TICKER, interval='day', span='month', bounds='regular')
        if historicals:
            df_hist = pd.DataFrame(historicals)
            # ensure begins_at -> index
            if 'begins_at' in df_hist.columns:
                df_hist['begins_at'] = pd.to_datetime(df_hist['begins_at']).dt.tz_localize(None)
                df_hist.set_index('begins_at', inplace=True)
            df_hist = df_hist.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'})
            # ensure numeric cast if possible
            for c in ['open','close','high','low','volume']:
                if c in df_hist.columns:
                    df_hist[c] = pd.to_numeric(df_hist[c], errors='coerce')
            df = prepare_historicals(df_hist)
            if not df.empty:
                last_14_low = float(df['low'][-LOW_DAYS:].min())
            else:
                last_14_low = current_price
        else:
            df = pd.DataFrame()
            last_14_low = current_price

        # historical volatility (if df present)
        try:
            if not df.empty:
                df['returns'] = np.log(df['close'] / df['close'].shift(1))
                hv = float(df['returns'].rolling(HV_PERIOD).std().iloc[-1] * np.sqrt(252))
            else:
                hv = 0.0
        except Exception:
            hv = 0.0

        # tradable puts
        all_puts = r.options.find_tradable_options(TICKER, optionType="put")
        exp_dates = sorted(set([opt.get('expiration_date') for opt in all_puts if opt.get('expiration_date')]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:NUM_EXPIRATIONS]

        candidate_puts = []
        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt.get('expiration_date') == exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            # skip closest 2 below and take next up to NUM_PUTS
            chosen_strikes = strikes_below[2:2+NUM_PUTS] if len(strikes_below) > 2 else strikes_below[:NUM_PUTS]

            for opt in puts_for_exp:
                try:
                    strike = float(opt.get('strike_price'))
                except Exception:
                    continue
                if strike not in chosen_strikes:
                    continue

                md_list = r.options.get_option_market_data_by_id(opt.get('id'))
                if not md_list:
                    continue
                md = md_list[0]

                # safe conversions
                def safe_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return 0.0
                def safe_int(x):
                    try:
                        return int(x)
                    except Exception:
                        return 0

                bid_price = safe_float(md.get('bid_price') or md.get('mark_price') or 0.0)
                delta = safe_float(md.get('delta') or 0.0)
                iv = safe_float(md.get('implied_volatility') or 0.0)
                cop_short = safe_float(md.get('chance_of_profit_short') or 0.0)
                theta = safe_float(md.get('theta') or 0.0)
                open_interest = safe_int(md.get('open_interest') or 0)
                volume = safe_int(md.get('volume') or 0)

                dist_from_low = (strike - last_14_low)/last_14_low if last_14_low != 0 else 0
                if dist_from_low < 0.03:
                    continue

                if bid_price >= MIN_PRICE:
                    opt_dict = {
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
                    }
                    candidate_puts.append(opt_dict)

        # choose top NUM_PUTS by COP Short for messaging and for scoring aggregation
        selected_puts = sorted(candidate_puts, key=lambda x: x['COP Short'], reverse=True)[:NUM_PUTS]
        all_options.extend(selected_puts)

        # For each selected put compute enhanced score and collect candidate list (for candidate_scores message)
        for p in selected_puts:
            # compute max contracts based on buying_power (conservative: at least 1)
            max_contracts = max(1, int(buying_power // (p['Strike Price'] * 100))) if p['Strike Price'] > 0 else 1
            total_premium = p['Bid Price'] * 100 * max_contracts
            days_to_exp = max((pd.to_datetime(p['Expiration Date']).date() - today).days, 1)
            iv_hv_ratio = (p['IV'] / p['HV']) if (p.get('HV') and p.get('HV') > 0) else 1.0
            liquidity_weight = 1 + 0.5 * (p['Volume'] + p['Open Interest']) / 1000.0
            enhanced_score = total_premium * iv_hv_ratio * liquidity_weight / (days_to_exp ** 1.0)

            candidate_scores.append({
                'Ticker': p['Ticker'],
                'Strike': p['Strike Price'],
                'Exp': p['Expiration Date'],
                'Max Contracts': max_contracts,
                'Premium': total_premium,
                'Score': enhanced_score,
                'Bid': p['Bid Price'],
                'Delta': p['Delta'],
                'IV': p['IV'],
                'COP': p['COP Short'],
                'Theta': p['Theta'],
                'OI': p['Open Interest'],
                'Vol': p['Volume'],
                'URL': p['URL']
            })

    except Exception as e:
        send_telegram_message(f"⚠️ Error processing {TICKER}: {e}")

# ------------------ CANDIDATE SCORES MESSAGE (portrait-friendly single message) ------------------
if candidate_scores:
    # sort by score desc
    candidate_scores = sorted(candidate_scores, key=lambda x: x['Score'], reverse=True)

    lines = ["📊 <b>Candidate Put Scores (all tickers)</b>\n"]
    for opt in candidate_scores:
        # two-line compact block; exp show MM-DD for mobile brevity
        try:
            exp_short = opt['Exp'][5:]  # 'YYYY-MM-DD' -> 'MM-DD'
        except Exception:
            exp_short = opt['Exp']
        lines.append(f"▫️ <b>{opt['Ticker']}</b> | Exp: {exp_short} | Strike: ${opt['Strike']}")
        lines.append(f"💰 ${opt['Premium']:.2f} | 📊 {opt['Score']:.2f} | 📦 {opt['Max Contracts']}x\n")

    send_telegram_message("\n".join(lines))

# ------------------ TOP 5 INDIVIDUAL ALERTS (unique tickers only, with charts) ------------------
# group candidate_scores by ticker and pick best scoring option per ticker
best_per_ticker = {}
for opt in candidate_scores:
    t = opt['Ticker']
    if t not in best_per_ticker or opt['Score'] > best_per_ticker[t]['Score']:
        best_per_ticker[t] = opt

# sort tickers by their best scores
sorted_tickers_by_score = sorted(best_per_ticker.values(), key=lambda x: x['Score'], reverse=True)
top5 = sorted_tickers_by_score[:5]

for opt in top5:
    ticker = opt['Ticker']
    # fetch historicals again for the chart
    try:
        historicals = r.stocks.get_stock_historicals(ticker, interval='day', span='month', bounds='regular')
        if historicals:
            df_hist = pd.DataFrame(historicals)
            if 'begins_at' in df_hist.columns:
                df_hist['begins_at'] = pd.to_datetime(df_hist['begins_at']).dt.tz_localize(None)
                df_hist.set_index('begins_at', inplace=True)
            df_hist = df_hist.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'})
            for c in ['open','close','high','low','volume']:
                if c in df_hist.columns:
                    df_hist[c] = pd.to_numeric(df_hist[c], errors='coerce')
            df_plot = prepare_historicals(df_hist)
            last_14_low = float(df_plot['low'][-LOW_DAYS:].min()) if not df_plot.empty else opt['Bid']  # fallback
        else:
            df_plot = pd.DataFrame()
            last_14_low = opt['Bid']
    except Exception:
        df_plot = pd.DataFrame()
        last_14_low = opt['Bid']

    # create chart if we have data, else send just text
    caption_lines = [
        f"📊 <a href='{opt['URL']}'>{ticker}</a> current: ${opt.get('Bid', 0.0):.2f}",
        f"📅 Exp: {opt['Exp']} | 💲 Strike: ${opt['Strike']} | 💰 Premium: ${opt['Premium']:.2f}",
        f"📊 Score: {opt['Score']:.2f} | 📦 Max Contracts: {opt['Max Contracts']} | 🎯 COP: {opt['COP']:.2f}%"
    ]
    caption = "\n".join(caption_lines)

    if not df_plot.empty:
        # attempt to annotate with strike and premium/contracts
        annotations = [
            (f"Max Contracts: {opt['Max Contracts']}", opt['Strike'] * 1.002),
            (f"Premium: ${opt['Premium']:.2f}", opt['Strike'] * 1.005)
        ]
        buf = plot_candlestick(df_plot, df_plot['close'].iloc[-1], last_14_low, [opt['Strike']], opt['Exp'], annotations=annotations)
        send_telegram_photo(buf, caption)
    else:
        send_telegram_message(caption)

# ------------------ BEST OPTION ALERT (multi-contract logic with COP tolerance) ------------------
if candidate_scores:
    # compute base_score (we already have 'Score' as enhanced_score). For best selection, compute base per-contract metric
    # Find max COP across options (for COP threshold comparison)
    max_COP = max(opt['COP'] for opt in candidate_scores) if candidate_scores else 0.0

    best_option = None
    best_adjusted = -1.0
    for opt in candidate_scores:
        days_to_exp = max((pd.to_datetime(opt['Exp']).date() - today).days, 1)
        base_score = (opt['Bid'] * 100 * opt['COP']) / days_to_exp if days_to_exp > 0 else 0.0

        max_contracts = opt['Max Contracts']
        # only boost by number of contracts if COP is within COP_THRESHOLD of max_COP
        if max_contracts > 1 and opt['COP'] >= COP_THRESHOLD * max_COP:
            adjusted = base_score * max_contracts
        else:
            adjusted = base_score

        if adjusted > best_adjusted:
            best_adjusted = adjusted
            best_option = opt.copy()
            best_option['Adjusted Score'] = adjusted
            best_option['Max Contracts'] = max_contracts
            best_option['Total Premium'] = opt['Premium']

    # send best alert with chart if possible
    if best_option:
        ticker = best_option['Ticker']
        try:
            historicals = r.stocks.get_stock_historicals(ticker, interval='day', span='month', bounds='regular')
            if historicals:
                df_hist = pd.DataFrame(historicals)
                if 'begins_at' in df_hist.columns:
                    df_hist['begins_at'] = pd.to_datetime(df_hist['begins_at']).dt.tz_localize(None)
                    df_hist.set_index('begins_at', inplace=True)
                df_hist = df_hist.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'})
                for c in ['open','close','high','low','volume']:
                    if c in df_hist.columns:
                        df_hist[c] = pd.to_numeric(df_hist[c], errors='coerce')
                df_plot = prepare_historicals(df_hist)
                last_14_low = float(df_plot['low'][-LOW_DAYS:].min()) if not df_plot.empty else best_option['Bid']
            else:
                df_plot = pd.DataFrame()
                last_14_low = best_option['Bid']
        except Exception:
            df_plot = pd.DataFrame()
            last_14_low = best_option['Bid']

        caption_lines = [
            "🔥 <b>Best Cash-Secured Put (Smart Choice)</b>:",
            f"📊 <a href='{best_option['URL']}'>{ticker}</a> current: ${best_option.get('Bid',0.0):.2f}",
            f"✅ Exp: {best_option['Exp']} | 💲 Strike: ${best_option['Strike']}",
            f"💰 Bid Price: ${best_option['Bid']:.2f} | 📦 Max Contracts: {best_option['Max Contracts']} | Total Premium: ${best_option['Total Premium']:.2f}",
            f"🔺 COP: {best_option['COP']*100:.1f}% | 🔢 Adjusted Score: {best_option['Adjusted Score']:.2f}"
        ]
        caption = "\n".join(caption_lines)

        if not df_plot.empty:
            annotations = [
                (f"Max Contracts: {best_option['Max Contracts']}", best_option['Strike'] * 1.002),
                (f"Total Premium: ${best_option['Total Premium']:.2f}", best_option['Strike'] * 1.005)
            ]
            buf = plot_candlestick(df_plot, df_plot['close'].iloc[-1], last_14_low, [best_option['Strike']], best_option['Exp'], annotations=annotations)
            send_telegram_photo(buf, caption)
        else:
            send_telegram_message(caption)
