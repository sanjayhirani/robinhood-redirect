# robinhood_sell_puts_final.py
"""
Full corrected script:
- Only uses tickers from the TICKERS list
- Gets top 2 puts per ticker
- Groups alerts: every 3 tickers -> 1 Telegram message (no charts)
- Candidate scores: each contract (top2 per ticker) gets a score; send top 10 overall
- Best alert: best contract (by same score) shown last with candlestick chart
- Robust checks to avoid list-index errors, missing market data, missing historicals
- Proper max_contracts calculation: int(buying_power // (strike * 100))
"""

import sys
import subprocess
import importlib
import pkgutil

# ---------------------------
# Ensure minimal dependencies
# ---------------------------
def ensure(pkg):
    if pkgutil.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in ("yfinance", "lxml", "robin_stocks", "matplotlib", "pandas", "numpy", "requests"):
    ensure(p)

# ---------------------------
# Imports
# ---------------------------
import os
import requests
import robin_stocks.robinhood as r
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------
# Config - edit this list if you want different tickers
# ---------------------------
TICKERS = ["SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC", "HTZ", "RZLV", "NVTS"]
NUM_EXPIRATIONS = 3
TOP_PUTS_PER_TICKER = 2      # top 2 puts per ticker
MIN_PRICE = 0.10
HV_PERIOD = 21
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21
GROUP_SIZE = 3                # grouped alerts: every 3 tickers -> 1 telegram message
CANDIDATE_TOP_N = 10         # top 10 contracts to include in candidate message
COP_TOLERANCE = 0.95         # multi-contract allowed if COP >= 95% of next best COP

# ---------------------------
# Secrets (from env)
# ---------------------------
USERNAME = os.environ.get("RH_USERNAME")
PASSWORD = os.environ.get("RH_PASSWORD")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not (USERNAME and PASSWORD and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
    raise RuntimeError("Missing one of RH_USERNAME, RH_PASSWORD, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID in env")

# ---------------------------
# Telegram helpers
# ---------------------------
def send_telegram_message(text):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    )

def send_telegram_photo(buf, caption):
    # buf is BytesIO
    buf.seek(0)
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo': ('image.png', buf.read())},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    )

# ---------------------------
# Helpers: plotting & historical prepping
# ---------------------------
def prepare_historicals(df):
    """Take a DataFrame indexed by datetime with columns open/close/high/low/volume.
       Make business-day index, forward-fill close and fill others safely.
       Returns possibly-empty df.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'begins_at' in df.columns:
            df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
            df.set_index('begins_at', inplace=True)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
    # ensure columns exist
    for c in ['open','close','high','low','volume']:
        if c not in df.columns:
            df[c] = np.nan
    # business days reindex
    try:
        all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(all_days)
    except Exception:
        # fallback: return as-is
        pass
    # tz-drop if present
    try:
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass
    # safe fills
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
    df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
    df['volume'] = df['volume'].fillna(0)
    # ensure numeric where possible to avoid warning downcasts
    df = df.infer_objects(copy=False)
    return df

def plot_candlestick(df, current_price, last_14_low, selected_strikes=None, exp_date=None):
    """Return BytesIO PNG of chart. If df empty, return None."""
    if df is None or df.empty:
        return None
    df = df.copy()
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    # ensure numeric and no NaNs
    df[['open','close','high','low']] = df[['open','close','high','low']].astype(float)
    width = 0.6
    for i in range(len(df)):
        xi = mdates.date2num(df.index[i])
        o = df['open'].iloc[i]
        c = df['close'].iloc[i]
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]
        color = 'lime' if c >= o else 'red'
        ax.add_patch(plt.Rectangle((xi - width/2, min(o,c)), width, abs(c-o), color=color))
        ax.plot([xi, xi], [l, h], color=color, linewidth=1)
    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.25, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=1.25, label=f'14-day Low: ${last_14_low:.2f}')
    if selected_strikes:
        for s in selected_strikes:
            ax.axhline(s, color='cyan', linestyle='--', linewidth=1.0)
    if exp_date:
        try:
            exp_dt = pd.to_datetime(exp_date).tz_localize(None)
            if df.index.min() <= exp_dt <= df.index.max():
                ax.axvline(mdates.date2num(exp_dt), color='orange', linestyle='--', linewidth=1.25)
        except Exception:
            pass
    ax.tick_params(colors='white')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, color='gray', linestyle='--', alpha=0.25)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------------------
# Scoring function (same logic used for candidate ranking and best selection)
# ---------------------------
def compute_adjusted_score(contract, buying_power, today_date):
    # contract is dict with required fields
    try:
        days_to_exp = max((pd.to_datetime(contract['Expiration Date']).date() - today_date).days, 1)
    except Exception:
        return 0.0
    hv = contract.get('HV', 0.0) or 0.0
    iv = contract.get('IV', 0.0) or 0.0
    iv_hv_ratio = (iv / hv) if hv > 0 else 1.0
    liquidity_weight = 1.0 + 0.5 * (contract.get('Volume', 0) + contract.get('Open Interest', 0)) / 1000.0
    # compute how many contracts we could sell with full buying_power
    strike = contract.get('Strike Price', 0.0) or 0.0
    max_contracts = int(buying_power // (strike * 100)) if strike > 0 else 0
    # total premium if we sold that many
    total_prem = contract.get('Bid Price', 0.0) * 100 * (max_contracts if max_contracts > 0 else 1)
    # incorporate COP Short (chance_of_profit_short)
    cop = contract.get('COP Short', 0.0) or 0.0
    score = total_prem * cop * iv_hv_ratio * liquidity_weight / (days_to_exp ** 1.0)
    return float(score)

# ---------------------------
# Safe getters
# ---------------------------
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

def get_market_data(option_id):
    """Return first market data entry or None (safe)."""
    try:
        md_list = r.options.get_option_market_data_by_id(option_id)
        if md_list and isinstance(md_list, (list, tuple)) and len(md_list) > 0:
            return md_list[0]
    except Exception:
        return None
    return None

# ---------------------------
# Login & account
# ---------------------------
r.login(USERNAME, PASSWORD)
today = datetime.now().date()
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

acct = r.profiles.load_account_profile() or {}
# try a few keys for buying power/cash (robust)
for k in ('cash_available_for_withdrawal', 'buying_power', 'cash'):
    if acct.get(k) not in (None, '', []):
        try:
            buying_power = float(acct.get(k))
            break
        except Exception:
            continue
else:
    buying_power = 0.0

# ---------------------------
# PART 1: earnings/dividend risk check (ONLY your TICKERS)
# ---------------------------
safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False

        # dividend
        try:
            if not stock.dividends.empty:
                future_divs = stock.dividends[stock.dividends.index.date >= today]
                if not future_divs.empty:
                    div_date = future_divs.index.min().date()
                    if div_date <= cutoff:
                        msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                        has_event = True
        except Exception:
            pass

        # earnings
        try:
            ed = stock.get_earnings_dates(limit=2)
            if (ed is not None) and (len(ed) > 0):
                try:
                    # ed is a DataFrame-like
                    if hasattr(ed, 'index'):
                        ed_date = ed.index.min().date()
                        if today <= ed_date <= cutoff:
                            msg_parts.append(f"⚠️ 📢 Earnings on {ed_date.strftime('%d-%m-%y')}")
                            has_event = True
                except Exception:
                    pass
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

# send risky/safe summary
summary = []
if risky_msgs:
    summary.append("⚠️ <b>Risky Tickers</b>\n" + "\n".join(risky_msgs) + "\n")
else:
    summary.append("✅ <b>No risky tickers found</b>\n")

if safe_tickers:
    rows = []
    safe_bold = [f"<b>{t}</b>" for t in safe_tickers]
    for i in range(0, len(safe_bold), 4):
        rows.append(", ".join(safe_bold[i:i+4]))
    summary.append("✅ <b>Safe Tickers</b>\n" + "\n".join(rows))

summary.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary))

# ---------------------------
# PART 2: collect top-2 puts per safe ticker
# ---------------------------
all_contracts = []     # every contract (top2 per ticker), with computed Score
group_messages = []    # accumulation of text blocks per ticker; will be sent grouped every GROUP_SIZE tickers
ticker_blocks = []     # block texts for current group

for idx, T in enumerate(safe_tickers):
    try:
        # get latest price
        latest = r.stocks.get_latest_price(T)
        if not latest:
            # no price -> skip
            continue
        current_price = safe_float(latest[0])
        rh_url = f"https://robinhood.com/stocks/{T}"

        # historicals (may be empty)
        historicals = r.stocks.get_stock_historicals(T, interval='day', span='month', bounds='regular') or []
        if historicals:
            df_hist = pd.DataFrame(historicals)
            if 'begins_at' in df_hist.columns:
                df_hist['begins_at'] = pd.to_datetime(df_hist['begins_at']).dt.tz_localize(None)
                df_hist.set_index('begins_at', inplace=True)
            # rename when present
            for a,b in [('open_price','open'), ('close_price','close'), ('high_price','high'), ('low_price','low')]:
                if a in df_hist.columns:
                    df_hist = df_hist.rename(columns={a:b})
            # coerce numeric
            for col in ['open','close','high','low','volume']:
                if col in df_hist.columns:
                    df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')
            df_prepped = prepare_historicals(df_hist)
            last_14_low = float(df_prepped['low'][-LOW_DAYS:].min()) if (not df_prepped.empty) else current_price
            # hv
            try:
                df_prepped['returns'] = np.log(df_prepped['close'] / df_prepped['close'].shift(1))
                hv = float(df_prepped['returns'].rolling(HV_PERIOD).std().iloc[-1] * np.sqrt(252))
            except Exception:
                hv = 0.0
        else:
            df_prepped = pd.DataFrame()
            last_14_low = current_price
            hv = 0.0

        # find tradable puts for this ticker
        puts = r.options.find_tradable_options(T, optionType="put") or []
        exp_dates = sorted(set([opt.get('expiration_date') for opt in puts if opt.get('expiration_date')]))
        # filter expiries within cutoff
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:NUM_EXPIRATIONS]

        ticker_candidate_puts = []
        for exp in exp_dates:
            puts_for_exp = [o for o in puts if o.get('expiration_date') == exp]
            strikes_below = sorted({float(o.get('strike_price')) for o in puts_for_exp if safe_float(o.get('strike_price')) < current_price}, reverse=True)
            # choose up to TOP_PUTS_PER_TICKER nearest strikes below; if fewer available take what exists
            chosen_strikes = strikes_below[:TOP_PUTS_PER_TICKER]

            if not chosen_strikes:
                continue

            for o in puts_for_exp:
                try:
                    strike = safe_float(o.get('strike_price'))
                except Exception:
                    continue
                if strike not in chosen_strikes:
                    continue
                md = get_market_data(o.get('id'))
                if not md:
                    continue
                bid = safe_float(md.get('bid_price') or md.get('mark_price') or 0.0)
                if bid < MIN_PRICE:
                    continue
                delta = safe_float(md.get('delta'))
                iv = safe_float(md.get('implied_volatility'))
                cop_short = safe_float(md.get('chance_of_profit_short'))
                theta = safe_float(md.get('theta'))
                oi = safe_int(md.get('open_interest'))
                vol = safe_int(md.get('volume'))
                dist_from_low = (strike - last_14_low)/last_14_low if last_14_low != 0 else 0.0
                # preserve your previous guard: skip if too close to 14-day low
                if dist_from_low < 0.03:
                    continue

                contract = {
                    "Ticker": T,
                    "Current Price": current_price,
                    "Expiration Date": exp,
                    "Strike Price": strike,
                    "Bid Price": bid,
                    "Delta": delta,
                    "IV": iv,
                    "COP Short": cop_short,
                    "Theta": theta,
                    "Open Interest": oi,
                    "Volume": vol,
                    "URL": rh_url,
                    "HV": hv
                }
                # compute score using buying power & today
                contract['Score'] = compute_adjusted_score(contract, buying_power, today)
                ticker_candidate_puts.append(contract)

        # pick the top 2 puts for this ticker by Score (if any)
        chosen_for_alert = sorted(ticker_candidate_puts, key=lambda x: x['Score'], reverse=True)[:TOP_PUTS_PER_TICKER]

        # append them to full list for later candidate ranking
        for c in chosen_for_alert:
            all_contracts.append(c)

        # build the block text for this ticker (top2)
        if chosen_for_alert:
            block_lines = [f"📊 <a href='{rh_url}'>{T}</a> current: ${current_price:.2f}"]
            for i, c in enumerate(chosen_for_alert, start=1):
                # compute max contracts (allow 0 if not affordable)
                strike = c['Strike Price']
                max_contracts = int(buying_power // (strike * 100)) if strike > 0 else 0
                total_prem = c['Bid Price'] * 100 * max_contracts
                block_lines.extend([
                    "────────────────────────────",
                    f"🔢 Option {i}: Exp: {c['Expiration Date']} | Strike: ${c['Strike Price']}",
                    f"💰 Bid: ${c['Bid Price']:.2f} | Δ: {c['Delta']:.3f} | IV: {c['IV']*100:.1f}%",
                    f"🎯 COP: {c['COP Short']*100:.1f}% | 📝 Max Contracts: {max_contracts} | Prem: ${total_prem:.2f}",
                    f"📊 Score: {c['Score']:.2f}"
                ])
            ticker_blocks.append("\n".join(block_lines))

        # Every GROUP_SIZE tickers, send a grouped Telegram message (or on final iteration)
        if (len(ticker_blocks) >= GROUP_SIZE) or (idx == len(safe_tickers)-1 and ticker_blocks):
            # join blocks with a blank line
            send_telegram_message("\n\n".join(ticker_blocks))
            ticker_blocks = []

    except Exception as exc:
        # Risky to send full exception trace to Telegram; send compact message
        send_telegram_message(f"⚠️ Error processing {T}: {str(exc)}")
        # continue to next ticker

# ---------------------------
# Candidate Scores alert: top N overall (from all_contracts)
# ---------------------------
if all_contracts:
    top_candidates = sorted(all_contracts, key=lambda x: x['Score'], reverse=True)[:CANDIDATE_TOP_N]
    lines = ["<b>📊 Top Candidate Contracts (best 10)</b>\n"]
    for rank, c in enumerate(top_candidates, start=1):
        exp_short = c['Expiration Date'][5:] if isinstance(c['Expiration Date'], str) and len(c['Expiration Date'])>=7 else c['Expiration Date']
        lines.append(f"{rank}. {c['Ticker']} | Exp: {exp_short} | Strike: ${c['Strike Price']} | Score: {c['Score']:.1f}")
    send_telegram_message("\n".join(lines))

# ---------------------------
# Best Option alert (with chart)
# ---------------------------
if all_contracts:
    sorted_all = sorted(all_contracts, key=lambda x: x['Score'], reverse=True)
    best = sorted_all[0]
    next_best = sorted_all[1] if len(sorted_all) > 1 else None

    # compute max_contracts allowed by buying_power
    strike = best['Strike Price'] if best.get('Strike Price') else 0.0
    max_contracts = int(buying_power // (strike * 100)) if strike > 0 else 0

    # enforce COP multi-contract rule: only allow multi-contract if best COP >= COP_TOLERANCE * next_best.COP
    if max_contracts > 1 and next_best is not None:
        if best.get('COP Short', 0.0) < COP_TOLERANCE * (next_best.get('COP Short', 0.0) or 0.0):
            # don't allow multi-contract boost
            max_contracts = 1 if (buying_power >= strike * 100) else 0

    total_premium = best['Bid Price'] * 100 * max_contracts if max_contracts > 0 else 0.0

    caption_lines = [
        "🔥 <b>Best Cash-Secured Put</b>",
        f"📊 <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}",
        f"✅ Expiration : {best['Expiration Date']}",
        f"💲 Strike    : ${best['Strike Price']}",
        f"💰 Bid Price : ${best['Bid Price']:.2f}",
        f"🔺 Delta     : {best['Delta']:.3f}",
        f"📈 IV       : {best['IV']*100:.2f}%",
        f"🎯 COP Short : {best['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}",
        f"📝 Score: {best['Score']:.2f}"
    ]
    caption = "\n".join(caption_lines)

    # build chart (if we can fetch historicals)
    try:
        historicals = r.stocks.get_stock_historicals(best['Ticker'], interval='day', span='month', bounds='regular') or []
        if historicals:
            dfh = pd.DataFrame(historicals)
            if 'begins_at' in dfh.columns:
                dfh['begins_at'] = pd.to_datetime(dfh['begins_at']).dt.tz_localize(None)
                dfh.set_index('begins_at', inplace=True)
            for a,b in [('open_price','open'),('close_price','close'),('high_price','high'),('low_price','low')]:
                if a in dfh.columns:
                    dfh = dfh.rename(columns={a:b})
            for col in ['open','close','high','low','volume']:
                if col in dfh.columns:
                    dfh[col] = pd.to_numeric(dfh[col], errors='coerce')
            df_plot = prepare_historicals(dfh)
            last_14_low = float(df_plot['low'][-LOW_DAYS:].min()) if not df_plot.empty else best['Current Price']
            buf = plot_candlestick(df_plot, best['Current Price'], last_14_low, [best['Strike Price']], best['Expiration Date'])
            if buf:
                send_telegram_photo(buf, caption)
            else:
                send_telegram_message(caption)
        else:
            send_telegram_message(caption)
    except Exception as e:
        send_telegram_message(caption)

# End
