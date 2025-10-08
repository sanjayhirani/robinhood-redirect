# robinhood_sell_puts.py

import os
import subprocess
import requests
import robin_stocks.robinhood as r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import yaml
from datetime import datetime, timedelta
import yfinance as yf
import re

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------

def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", pkg_name])

for pkg in ["pandas","numpy","matplotlib","requests","robin_stocks","yfinance","PyYAML"]:
    ensure_package(pkg)

# ------------------ LOAD CONFIG ------------------

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ------------------ LOAD TICKERS ------------------

TICKERS_FILE = config.get("tickers_file", "tickers.txt")
if not os.path.exists(TICKERS_FILE):
    raise FileNotFoundError(f"{TICKERS_FILE} not found.")

TICKERS_RAW = [line.strip() for line in open(TICKERS_FILE, encoding="utf-8") if line.strip()]
TICKERS = [re.sub(r'[^A-Z0-9.-]', '', t.upper()) for t in TICKERS_RAW]

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

plot_cfg = config.get("plot", {})
CANDLE_WIDTH = plot_cfg.get("candle_width", 0.6)
FIGSIZE = plot_cfg.get("figsize", [12,6])
BG_COLOR = plot_cfg.get("background_color","black")
CURRENT_COLOR = plot_cfg.get("current_price_color","magenta")
LOW_COLOR = plot_cfg.get("low_price_color","yellow")
STRIKE_COLOR = plot_cfg.get("strike_color","cyan")
EXPIRY_COLOR = plot_cfg.get("expiry_color","orange")

def plot_candlestick(df, current_price, last_14_low, selected_strikes=None, exp_date=None, show_strikes=True):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
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

    ax.axhline(current_price, color=CURRENT_COLOR, linestyle='--', linewidth=1.5)
    ax.axhline(last_14_low, color=LOW_COLOR, linestyle='--', linewidth=2)

    if show_strikes and selected_strikes:
        for strike in selected_strikes:
            ax.axhline(strike, color=STRIKE_COLOR, linestyle='--', linewidth=1.5)

    if exp_date:
        exp_date_obj = pd.to_datetime(exp_date).tz_localize(None)
        if df.index.min() <= exp_date_obj <= df.index.max():
            ax.axvline(mdates.date2num(exp_date_obj), color=EXPIRY_COLOR, linestyle='--', linewidth=2)

    ax.set_ylabel('Price ($)', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=BG_COLOR)
    buf.seek(0)
    plt.close()
    return buf

# ------------------ LOGIN ------------------

r.login(USERNAME, PASSWORD)
today = datetime.now().date()
cutoff = today + timedelta(days=config.get("expiry_limit_days", 21))

# ------------------ EARNINGS / DIVIDENDS CHECK ------------------

safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

for ticker_raw, ticker_clean in zip(TICKERS_RAW, TICKERS):
    try:
        stock = yf.Ticker(ticker_clean)
        msg_parts = [f"{ticker_raw}"]
        has_event = False

        try:
            future_divs = stock.dividends[stock.dividends.index.date >= today]
            if not future_divs.empty:
                div_date = future_divs.index.min().date()
                if div_date <= cutoff:
                    msg_parts.append(f"{config['telegram_labels']['dividend_alert']} {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except:
            pass

        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index.min().date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"{config['telegram_labels']['earnings_alert']} {earnings_date.strftime('%d-%m-%y')}")
                    has_event = True
        except:
            pass

        if has_event:
            risky_msgs.append(" | ".join(msg_parts))
            risky_count += 1
        else:
            safe_tickers.append((ticker_raw, ticker_clean))
            safe_count += 1

    except Exception as e:
        risky_msgs.append(f"{ticker_raw} error: {e}")
        risky_count += 1

summary_lines = []
if risky_msgs:
    summary_lines.append(f"{config['telegram_labels']['risky_tickers']}\n" + "\n".join(risky_msgs))
if safe_tickers:
    safe_rows = [", ".join([t[0] for t in safe_tickers][i:i+4]) for i in range(0,len(safe_tickers),4)]
    summary_lines.append(f"{config['telegram_labels']['safe_tickers']}\n" + "\n".join(safe_rows))
summary_lines.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary_lines))

# ------------------ OPTIONS SCAN ------------------

all_options = []

account_data = r.profiles.load_account_profile()
buying_power = float(account_data.get('buying_power', 0.0))

for ticker_raw, ticker_clean in safe_tickers:
    try:
        current_price = float(r.stocks.get_latest_price(ticker_clean)[0])
        historicals = r.stocks.get_stock_historicals(ticker_clean, interval='day', span='month', bounds='regular')
        df = pd.DataFrame(historicals)
        df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
        df.set_index('begins_at', inplace=True)
        df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
        df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        df = df.asfreq('B').ffill()
        last_14_low = df['low'][-config.get("low_days",14):].min()

        # Find tradable puts
        all_puts = r.options.find_tradable_options(ticker_clean, optionType="put")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_puts]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:config.get("num_expirations",3)]

        candidate_puts = []
        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date']==exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            chosen_strikes = strikes_below[1:4] if len(strikes_below)>1 else strikes_below
            for opt in puts_for_exp:
                strike = float(opt['strike_price'])
                if strike not in chosen_strikes:
                    continue
                md = r.options.get_option_market_data_by_id(opt['id'])[0]
                bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                if bid_price < config.get("min_price",0.10):
                    continue
                delta = float(md.get('delta') or 0.0)
                cop_short = float(md.get('chance_of_profit_short') or 0.0)
                open_interest = int(md.get('open_interest') or 0)
                volume = int(md.get('volume') or 0)
                dist_from_low = (strike - last_14_low) / last_14_low
                if dist_from_low < 0.01:
                    continue
                candidate_puts.append({
                    "Ticker": ticker_raw,
                    "TickerClean": ticker_clean,
                    "Current Price": current_price,
                    "Expiration Date": exp_date,
                    "Strike Price": strike,
                    "Bid Price": bid_price,
                    "Delta": delta,
                    "COP Short": cop_short,
                    "Open Interest": open_interest,
                    "Volume": volume
                })

        selected_puts = sorted(candidate_puts, key=lambda x:x['COP Short'], reverse=True)[:3]
        if selected_puts:
            all_options.extend(selected_puts)

    except Exception as e:
        send_telegram_message(f"{ticker_raw} error: {e}")

# ------------------ FILTER & SEND TOP 10 TICKERS ------------------

if all_options:
    def score(opt):
        days_to_exp = (datetime.strptime(opt['Expiration Date'], "%Y-%m-%d").date() - today).days
        if days_to_exp <= 0:
            return 0
        liquidity = 1 + 0.5 * (opt['Volume'] + opt['Open Interest']) / 1000
        max_contracts = max(1, int(buying_power // (opt['Strike Price'] * 100)))
        return opt['Bid Price'] * 100 * max_contracts * opt['COP Short'] * liquidity / days_to_exp

    # Compute best score per ticker
    ticker_best = {}
    for opt in all_options:
        t = opt['Ticker']
        sc = score(opt)
        if t not in ticker_best or sc > ticker_best[t]['score'] or (
            abs(sc - ticker_best[t]['score']) < 1e-6 and opt['COP Short'] > ticker_best[t]['COP Short']
        ):
            ticker_best[t] = {'score': sc, **opt}

    # Sort and keep top 10 tickers
    top_tickers = sorted(
        ticker_best.values(),
        key=lambda x: (x['score'], x['COP Short']),
        reverse=True
    )[:10]
    top_ticker_names = {t['Ticker'] for t in top_tickers}

    # Send Telegram messages only for top 10 tickers
    for t in top_ticker_names:
        puts_for_ticker = [opt for opt in all_options if opt['Ticker'] == t]
        top3 = sorted(puts_for_ticker, key=lambda x: x['COP Short'], reverse=True)[:3]
        if not top3:
            continue
        msg_lines = [f"📊 {t} current: ${top3[0]['Current Price']:.2f}"]
        for idx, p in enumerate(top3, 1):
            max_contracts = max(1, int(buying_power // (p['Strike Price'] * 100)))
            total_premium = p['Bid Price'] * 100 * max_contracts
            msg_lines.append(
                f"<b>Option {idx}:</b>\nExp: {p['Expiration Date']} | Strike: ${p['Strike Price']} | "
                f"Bid: ${p['Bid Price']:.2f}\nDelta: {p['Delta']:.3f} | COP: {p['COP Short']*100:.1f}%\n"
                f"Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}\n"
                "────────────────────────────"
            )
        send_telegram_message("\n".join(msg_lines))

# ------------------ TOP 10 SUMMARY ALERT ------------------

if all_options:
    summary_rows = []
    top10_best_options = []

    # Build the data rows first
    for t in top_ticker_names:
        puts_for_ticker = [opt for opt in all_options if opt['Ticker'] == t]
        if not puts_for_ticker:
            continue

        best_for_ticker = max(puts_for_ticker, key=lambda x: x['COP Short'])
        max_contracts = max(1, int(buying_power // (best_for_ticker['Strike Price'] * 100)))
        total_premium = best_for_ticker['Bid Price'] * 100 * max_contracts

        best_for_ticker['Max Contracts'] = max_contracts
        best_for_ticker['Total Premium'] = total_premium
        top10_best_options.append(best_for_ticker)

    # Sort by total premium descending
    top10_best_options = sorted(top10_best_options, key=lambda x: x['Total Premium'], reverse=True)

    # Fixed-width formatting for data rows
    for opt in top10_best_options:
        exp_md = opt['Expiration Date'][5:]  # MM-DD
        summary_rows.append(
            f"{opt['Ticker']:<5}|{exp_md:<5}|{opt['Strike Price']:<6.2f}|"
            f"{opt['Bid Price']:<4.2f}|{opt['COP Short']*100:<5.1f}%|"
            f"{opt['Max Contracts']:<2}|${opt['Total Premium']:<5.0f}"
        )

    # Header row left-aligned to match data
    header = "<b>📋 Top 10 Summary — Best Option per Ticker</b>\n"
    table_header = f"{'Tkr':<5}|{'Exp':<5}|{'Strk':<6}|{'Bid':<4}|{'COP%':<5}|{'Ct':<2}|{'Prem':<5}\n" + "-"*40

    table_body = "\n".join(summary_rows)
    send_telegram_message(header + "\n<pre>" + table_header + "\n" + table_body + "</pre>")

# ------------------ CURRENT OPEN POSITIONS ALERT ------------------
try:
    positions = r.options.get_open_option_positions()
    if positions:
        msg_lines = ["📋 <b>Current Open Positions</b>\n"]

        for pos in positions:
            qty_raw = float(pos.get("quantity", 0))
            if qty_raw == 0:
                continue

            contracts = abs(int(qty_raw))
            is_short = qty_raw < 0

            instrument = r.helper.request_get(pos.get("option"))
            ticker = instrument.get("chain_symbol")
            inst_type = instrument.get("type", "").lower()
            strike = float(instrument.get("strike_price"))
            exp_date = pd.to_datetime(instrument.get("expiration_date")).strftime("%Y-%m-%d")

            # Human-readable label
            if inst_type == "put":
                opt_label = "📉 Sell Put" if is_short else "📉 Buy Put"
            elif inst_type == "call":
                opt_label = "📈 Sell Call" if is_short else "📈 Buy Call"
            else:
                opt_label = inst_type.capitalize() or "Option"

            # PnL calculations using md_mark_price
            avg_price_raw = float(pos.get("average_price") or 0.0)
            contracts = abs(int(qty_raw))
            is_short = qty_raw < 0
            
            # Live mark price from market data
            md_mark_price = float(md.get("mark_price") or 0.0)
            mark_per_contract = md_mark_price * 100  # convert per-share to per-contract
            
            if is_short:
                orig_pnl = abs(avg_price_raw) * contracts
                pnl_now = orig_pnl - (mark_per_contract * contracts)
            else:
                orig_pnl = -abs(avg_price_raw) * contracts
                pnl_now = (mark_per_contract * contracts) + orig_pnl

            pnl_emoji = "🟢" if pnl_now >= 0 else "🔴"

            # Build message lines (per-contract, no Avg column)
            msg_lines.append(
                f"📌 <b>{ticker}</b> | {opt_label}\n"
                f"Strike: ${strike:.2f} | Exp: {exp_date} | Qty: {contracts}\n"
                f"Current Price: ${float(r.stocks.get_latest_price(ticker)[0]):.2f}\n"
                f"Mark (per contract): ${market_value/contracts:.2f}\n"
                f"OrigPnL: ${abs(orig_pnl):.2f} | PnLNow: {pnl_emoji} ${abs(pnl_now):.2f}\n"
                "────────────────────"
            )

        send_telegram_message("\n".join(msg_lines))

except Exception as e:
    send_telegram_message(f"Error generating current positions alert: {e}")

# ------------------ BEST PUT ALERT ------------------
if top10_best_options:
    # Use top10_best_options from summary directly
    best = max(top10_best_options, key=score)
    max_contracts = max(1, int(buying_power // (best['Strike Price']*100)))
    total_premium = best['Bid Price']*100*max_contracts
    orig_pnl = total_premium
    pnl_now = total_premium

    buf = plot_candlestick(df, best['Current Price'], last_14_low, [best['Strike Price']], best['Expiration Date'])

    msg_lines = [
        "🔥 <b>Best Cash-Secured Put</b>",
        f"📊 {best['Ticker']} current: ${best['Current Price']:.2f}",
        f"✅ Expiration: {best['Expiration Date']}",
        f"💲 Strike: ${best['Strike Price']:.2f}",
        f"💰 Bid: ${best['Bid Price']:.2f}",
        f"🔺 Delta: {best['Delta']:.3f} | COP: {best['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}",
        f"💵 Buying Power: ${buying_power:,.2f}",
        f"💹 OrigPnL: ${orig_pnl:.2f} | PnLNow: ${pnl_now:.2f}"
    ]
    send_telegram_photo(buf, "\n".join(msg_lines))

