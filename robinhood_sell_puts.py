import os
import subprocess
import requests
import robin_stocks.robinhood as r
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import yfinance as yf
import re
import io

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------

def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", pkg_name])

for pkg in ["pandas","numpy","requests","robin_stocks","yfinance","PyYAML"]:
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

# ------------------ LOGIN ------------------

r.login(USERNAME, PASSWORD)
today = datetime.now().date()
cutoff = today + timedelta(days=config.get("expiry_limit_days", 30))

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

# Add header for the check
summary_lines.append("<b>📋 Earnings/Dividends Check</b>\n")

if risky_msgs:
    summary_lines.append(f"{config['telegram_labels']['risky_tickers']}\n" + "\n".join(risky_msgs))

if safe_tickers:
    # Format safe tickers in rows of 4 per line
    safe_rows = [", ".join([t[0] for t in safe_tickers][i:i+4]) for i in range(0, len(safe_tickers), 4)]
    summary_lines.append(f"{config['telegram_labels']['safe_tickers']}\n" + "\n".join(safe_rows))

summary_lines.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary_lines))

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ------------------ CURRENT OPEN POSITIONS ALERT (Sell Calls & Puts) ------------------
try:
    positions = r.options.get_open_option_positions()
    if positions:
        msg_lines = ["📋 <b>Current Open Positions</b>\n"]

        for pos in positions:
            qty_raw = float(pos.get("quantity") or 0)
            if qty_raw == 0:
                continue

            contracts = abs(int(qty_raw))

            # Fetch instrument data
            instrument = r.helper.request_get(pos.get("option"))
            ticker = instrument.get("chain_symbol")
            strike = float(instrument.get("strike_price"))
            exp_date = pd.to_datetime(instrument.get("expiration_date")).strftime("%Y-%m-%d")

            # Determine option type
            opt_type = instrument.get("type") or instrument.get("option_type") or "put"
            opt_label = "Sell Put" if opt_type.lower() == "put" else "Sell Call"

            # Average price & market data
            avg_price_raw = float(pos.get("average_price") or 0.0)
            md_list = r.options.get_option_market_data_by_id(instrument.get("id"))
            if md_list:
                md = md_list[0] if isinstance(md_list, list) else md_list
            else:
                md = {"mark_price": 0.0}
            md_mark_price = float(md.get("mark_price") or 0.0)
            mark_per_contract = md_mark_price * 100

            # PnL calculation (per-contract)
            orig_pnl = abs(avg_price_raw) * contracts  # total premium received
            pnl_now = orig_pnl - (mark_per_contract * contracts)  # current value of position
            pnl_emoji = "🟢" if pnl_now >= 0.7 * orig_pnl else "🔴"

            # Latest stock price
            latest_price_list = r.stocks.get_latest_price(ticker)
            current_price = float(latest_price_list[0]) if latest_price_list and latest_price_list[0] else 0.0

            # Format message
            msg_lines.extend([
                f"📌 <b>{ticker}</b> | {opt_label}",
                f"💲 Strike: ${strike:.2f}",
                f"✅ Exp: {exp_date}",
                f"📦 Qty: {contracts}",
                f"📊 Current Price: ${current_price:.2f}",
                f"💰 Full Premium: ${orig_pnl:.2f}",
                f"💵 Current Profit: {pnl_emoji} ${pnl_now:.2f}",
                "────────────────────"
            ])

        send_telegram_message("\n".join(msg_lines))

except Exception as e:
    send_telegram_message(f"Error generating current positions alert: {e}")

# ------------------ OPTIONS SCAN (PARALLELIZED WITH THROTTLE) ------------------
all_options = []

account_data = r.profiles.load_account_profile()
buying_power = float(account_data.get('buying_power', 0.0))

def scan_ticker(ticker_raw, ticker_clean):
    """
    Robust per-ticker scanner:
    - handles missing historicals
    - checks shapes returned by Robinhood wrapper
    - uses last 30 business days low
    - includes far OTM strikes up to 50% below current price
    - skips the first strike below current price
    """
    ticker_results = []
    try:
        latest_price = r.stocks.get_latest_price(ticker_clean)
        if not latest_price or latest_price[0] is None:
            return []
        current_price = float(latest_price[0])

        # historicals
        historicals = r.stocks.get_stock_historicals(
            ticker_clean, interval='day', span='month', bounds='regular'
        )
        if not historicals:
            return []

        df = pd.DataFrame(historicals)
        if df.empty or 'begins_at' not in df.columns:
            return []

        df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
        df.set_index('begins_at', inplace=True)

        expected_cols = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']
        if not all(c in df.columns for c in expected_cols):
            return []

        df = df[expected_cols].astype(float)
        df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        df = df.asfreq('B').ffill()

        low_days = int(config.get("low_days", 30))
        last_low = df['low'][-low_days:].min()
        if pd.isna(last_low) or last_low <= 0:
            return []

        # get tradable puts
        all_puts = r.options.find_tradable_options(ticker_clean, optionType="put")
        if not all_puts:
            return []

        if isinstance(all_puts, dict):
            maybe = all_puts.get('results') or all_puts.get('options') or None
            if isinstance(maybe, list):
                all_puts = maybe
            else:
                found = False
                for v in all_puts.values():
                    if isinstance(v, list):
                        all_puts = v
                        found = True
                        break
                if not found:
                    return []
        if not isinstance(all_puts, list):
            try:
                all_puts = list(all_puts)
            except Exception:
                return []

        min_days = config.get("expiry_window_days", {}).get("min", 15)
        max_days = config.get("expiry_window_days", {}).get("max", 35)
        exp_dates = sorted({opt.get('expiration_date') for opt in all_puts if opt.get('expiration_date')})
        exp_dates = [d for d in exp_dates if min_days <= (datetime.strptime(d, "%Y-%m-%d").date() - today).days <= max_days]
        exp_dates = exp_dates[:config.get("num_expirations", 4)]

        candidate_puts = []

        max_otm_fraction = 0.5  # include puts up to 50% below current price
        max_otm_strike = current_price * (1 - max_otm_fraction)

        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt.get('expiration_date') == exp_date]

            # ----------------- CHANGED: skip first strike below current price -----------------
            strikes_below = sorted(
                [float(opt.get('strike_price')) for opt in puts_for_exp
                 if opt.get('strike_price') and float(opt.get('strike_price')) < current_price
                 and float(opt.get('strike_price')) >= max_otm_strike],
                reverse=True
            )
            chosen_strikes = strikes_below[1:] if len(strikes_below) > 1 else []  # skip first

            option_ids = [
                opt.get('id') for opt in puts_for_exp
                if opt.get('strike_price') and float(opt.get('strike_price')) in chosen_strikes and opt.get('id')
            ]
            if not option_ids:
                continue

            market_data_list = []
            for oid in option_ids:
                try:
                    md_resp = r.options.get_option_market_data_by_id(oid)
                    if md_resp:
                        market_data_list.append(md_resp[0] if isinstance(md_resp, list) else md_resp)
                except Exception:
                    pass
                time.sleep(0.05)

            if isinstance(market_data_list, dict):
                market_data_list = [market_data_list]
            elif isinstance(market_data_list, list):
                flat = []
                for item in market_data_list:
                    if isinstance(item, list):
                        flat.extend(item)
                    else:
                        flat.append(item)
                market_data_list = flat

            if not market_data_list:
                continue

            opts_selected = [opt for opt in puts_for_exp if opt.get('strike_price') and float(opt.get('strike_price')) in chosen_strikes]

            pairs = []
            if len(market_data_list) == len(option_ids) and len(opts_selected) == len(option_ids):
                pairs = list(zip(opts_selected, market_data_list))
            else:
                md_map = {}
                for md in market_data_list:
                    key = None
                    for possible_key in ('option', 'option_id', 'id'):
                        if possible_key in md and md.get(possible_key):
                            key = str(md.get(possible_key))
                            break
                    if key:
                        md_map[key] = md
                for opt in opts_selected:
                    oid = opt.get('id') or opt.get('option_id')
                    if oid and (str(oid) in md_map):
                        pairs.append((opt, md_map[str(oid)]))
                if not pairs:
                    pairs = list(zip(opts_selected, market_data_list))

            for opt, md in pairs:
                try:
                    bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                except Exception:
                    bid_price = 0.0
                if bid_price < config.get("min_price", 0.10):
                    continue

                delta = float(md.get('delta') or 0.0)
                cop_short = float(md.get('chance_of_profit_short') or 0.0)
                open_interest = int(md.get('open_interest') or 0)
                volume = int(md.get('volume') or 0)

                try:
                    strike_price = float(opt.get('strike_price'))
                except Exception:
                    continue

                candidate_puts.append({
                    "Ticker": ticker_raw,
                    "TickerClean": ticker_clean,
                    "Current Price": current_price,
                    "Expiration Date": exp_date,
                    "Strike Price": strike_price,
                    "Bid Price": bid_price,
                    "Delta": delta,
                    "COP Short": cop_short,
                    "Open Interest": open_interest,
                    "Volume": volume
                })

        return candidate_puts

    except Exception as e:
        send_telegram_message(f"{ticker_raw} error: {e}")
        return []

# ------------------ RUN PARALLELIZED SCAN ------------------
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(scan_ticker, t_raw, t_clean) for t_raw, t_clean in safe_tickers]
    for f in as_completed(futures):
        all_options.extend(f.result())
        time.sleep(0.15)  # small throttle delay to avoid API limits

# ------------------ FILTER OPTIONS GLOBALLY BY ABS(DELTA) AND COP ------------------
all_options = [
    opt for opt in all_options
    if abs(opt.get('Delta', 1)) <= 0.3 and opt.get('COP Short', 0) >= 0.7
]

# ------------------ TOP OPTIONS SCORING & SELECTION ------------------

if all_options:
    def score(opt):
        days_to_exp = (datetime.strptime(opt['Expiration Date'], "%Y-%m-%d").date() - today).days
        if days_to_exp <= 0:
            return 0
        liquidity = 1 + 0.5 * (opt['Volume'] + opt['Open Interest']) / 1000
        max_contracts = max(1, int(buying_power // (opt['Strike Price'] * 100)))
        return opt['Bid Price'] * 100 * max_contracts * opt['COP Short'] * liquidity / days_to_exp

    ticker_best = {}
    for opt in all_options:
        t = opt['Ticker']
        sc = score(opt)
        if t not in ticker_best or sc > ticker_best[t]['score'] or (
            abs(sc - ticker_best[t]['score']) < 1e-6 and opt['COP Short'] > ticker_best[t]['COP Short']
        ):
            ticker_best[t] = {'score': sc, **opt}

    top_tickers = sorted(
        ticker_best.values(),
        key=lambda x: (x['score'], x['COP Short']),
        reverse=True
    )[:10]
    top_ticker_names = {t['Ticker'] for t in top_tickers}

# ------------------ ALL PUT OPTIONS SUMMARY (INCLUDE ALL STRIKES) ------------------

if all_options:
    summary_rows = []

    # Use all candidate puts, do NOT filter by Delta or COP
    all_display_options = []
    for opt in all_options:
        max_contracts = max(1, int(buying_power // (opt['Strike Price'] * 100)))
        total_premium = opt['Bid Price'] * 100 * max_contracts
        opt['Max Contracts'] = max_contracts
        opt['Total Premium'] = total_premium
        all_display_options.append(opt)

    # Keep only top 5 options per ticker
    from collections import defaultdict
    
    top5_per_ticker = []
    options_by_ticker = defaultdict(list)
    
    for opt in all_display_options:
        options_by_ticker[opt['Ticker']].append(opt)
    
    for ticker, opts in options_by_ticker.items():
        # Sort each ticker's options by Total Premium descending
        sorted_opts = sorted(opts, key=lambda x: x['Total Premium'], reverse=True)
        top5_per_ticker.extend(sorted_opts[:5])  # keep only top 5
    
    # Use this list for building Telegram messages
    all_display_options = top5_per_ticker

    # Format each row
    for opt in all_display_options:
        exp_md = opt['Expiration Date'][5:]  # MM-DD
        summary_rows.append(
            f"{opt['Ticker']:<5}|{exp_md:<5}|{opt['Strike Price']:<6.2f}|"
            f"{opt['Bid Price']:<4.2f}|{abs(opt['Delta']):<5.2f}|{opt['COP Short']*100:<5.1f}%|"
            f"{opt['Max Contracts']:<2}|${opt['Total Premium']:<5.0f}"
        )

    # Header
    header = "<b>📋 All Put Options Summary — All Strikes Included</b>\n"
    table_header = f"{'Tkr':<5}|{'Exp':<5}|{'Strk':<6}|{'Bid':<4}|{'Δ':<5}|{'COP%':<5}|{'Ct':<2}|{'Prem':<5}\n" + "-"*50

    # Split into chunks of 30 rows for Telegram
    chunk_size = 30
    for i in range(0, len(summary_rows), chunk_size):
        chunk = summary_rows[i:i+chunk_size]
        chunk_body = "\n".join(chunk)
        msg = header + "\n<pre>" + table_header + "\n" + chunk_body + "</pre>"
        send_telegram_message(msg)

# Prepare list for best alert based on all options
top10_best_options = sorted(all_options, key=lambda x: x['Total Premium'], reverse=True)[:10]

# ------------------ BEST PUT ALERT (STRICT FILTER) ------------------
# Filter all options by COP ≥ 0.73 and abs(Delta) ≤ 0.25
eligible_options = [
    opt for opt in all_options
    if opt['COP Short'] >= 0.73 and abs(opt['Delta']) <= 0.25
]

if eligible_options:
    # Pick the option with the highest total premium
    best = max(eligible_options, key=lambda x: x['Total Premium'])

    max_contracts = max(1, int(buying_power // (best['Strike Price']*100)))
    total_premium = best['Bid Price']*100*max_contracts

    msg_lines = [
        "🔥 <b>Best Cash-Secured Put</b>",
        "",
        f"📊 {best['Ticker']} current: ${best['Current Price']:.2f}",
        f"✅ Expiration: {best['Expiration Date']}",
        f"💲 Strike: ${best['Strike Price']:.2f}",
        f"💰 Bid: ${best['Bid Price']:.2f}",
        f"🔺 Delta: {abs(best['Delta']):.3f} | COP: {best['COP Short']*100:.1f}%",
        f"📝 Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}",
        f"💵 Buying Power: ${buying_power:,.2f}"
    ]
    send_telegram_message("\n".join(msg_lines))

else:
    # No eligible option found
    send_telegram_message("⚠️ No option meets COP ≥ 73% and Δ ≤ 0.25")

# ------------------ OWNED TICKERS SELL CALLS SUMMARY (SORTED BY TOTAL PREMIUM) ------------------
try:
    holdings = r.build_holdings()
    owned_tickers = [(symbol.upper(), symbol.upper()) for symbol, data in holdings.items() if float(data.get("quantity", 0)) >= 100]

    if not owned_tickers:
        send_telegram_message("⚠️ No owned tickers with ≥100 shares for sell calls.")
    else:
        all_calls = []
        buying_power = float(r.profiles.load_account_profile().get('buying_power', 0))

        for ticker_raw, ticker_clean in owned_tickers:
            try:
                calls = r.options.find_tradable_options(ticker_clean, optionType="call")
                if not calls:
                    continue

                if isinstance(calls, dict):
                    maybe = calls.get('results') or calls.get('options') or None
                    if isinstance(maybe, list):
                        calls = maybe
                    else:
                        for v in calls.values():
                            if isinstance(v, list):
                                calls = v
                                break
                if not isinstance(calls, list):
                    continue

                for opt in calls:
                    exp_date = opt.get("expiration_date")
                    if not exp_date:
                        continue
                    exp_dt = datetime.strptime(exp_date, "%Y-%m-%d").date()
                    if not (today <= exp_dt <= cutoff):
                        continue

                    oid = opt.get("id")
                    if not oid:
                        continue

                    md_list = r.options.get_option_market_data_by_id(oid)
                    if not md_list:
                        continue
                    md = md_list[0] if isinstance(md_list, list) else md_list

                    bid_price = float(md.get("bid_price") or md.get("mark_price") or 0)
                    if bid_price < 0.10:
                        continue  # Filter out low bids

                    delta = float(md.get("delta") or 0)
                    cop_short = float(md.get("chance_of_profit_short") or 0)
                    strike = float(opt.get("strike_price") or 0)
                    max_contracts = max(1, int(buying_power // (strike*100)))
                    total_premium = bid_price * 100 * max_contracts

                    all_calls.append({
                        "Ticker": ticker_raw,
                        "Expiration": exp_date,
                        "Strike": strike,
                        "Bid": bid_price,
                        "Delta": delta,
                        "COP Short": cop_short,
                        "Max Contracts": max_contracts,
                        "Total Premium": total_premium
                    })

            except Exception as e:
                send_telegram_message(f"{ticker_raw} error fetching calls: {e}")

        # Filter by criteria: delta ≤ 0.3 & COP ≥ 0.7
        eligible_calls = [c for c in all_calls if abs(c["Delta"]) <= 0.3 and c["COP Short"] >= 0.7]

        if eligible_calls:
            # Sort descending by Total Premium
            eligible_calls = sorted(eligible_calls, key=lambda x: x['Total Premium'], reverse=True)

            # Format all calls summary (matching sell puts table)
            summary_rows = []
            for c in eligible_calls:
                exp_md = c['Expiration'][5:]  # MM-DD
                summary_rows.append(
                    f"{c['Ticker']:<5}|{exp_md:<5}|{c['Strike']:<6.2f}|"
                    f"{c['Bid']:<4.2f}|{abs(c['Delta']):<5.2f}|{c['COP Short']*100:<5.1f}%|"
                    f"{c['Max Contracts']:<2}|${c['Total Premium']:<5.0f}"
                )

            header = "<b>📋 All Sell Calls Summary — Owned Tickers</b>\n"
            table_header = f"{'Tkr':<5}|{'Exp':<5}|{'Strk':<6}|{'Bid':<4}|{'Δ':<5}|{'COP%':<5}|{'Ct':<2}|{'Prem':<5}\n" + "-"*50

            chunk_size = 30
            for i in range(0, len(summary_rows), chunk_size):
                chunk = summary_rows[i:i+chunk_size]
                chunk_body = "\n".join(chunk)
                msg = header + "\n<pre>" + table_header + "\n" + chunk_body + "</pre>"
                send_telegram_message(msg)

            # ------------------ BEST SELL CALL ALERT ------------------
            best_call = eligible_calls[0]  # top after sorting
            msg_lines = [
                "🔥 <b>Best Sell Call</b>",
                "",
                f"📊 {best_call['Ticker']} Strike: ${best_call['Strike']:.2f}",
                f"✅ Expiration: {best_call['Expiration']}",
                f"💰 Bid: ${best_call['Bid']:.2f}",
                f"🔺 Delta: {abs(best_call['Delta']):.3f} | COP: {best_call['COP Short']*100:.1f}%",
                f"📝 Max Contracts: {best_call['Max Contracts']} | Total Premium: ${best_call['Total Premium']:.2f}",
                f"💵 Buying Power: ${buying_power:,.2f}"
            ]
            send_telegram_message("\n".join(msg_lines))

        else:
            send_telegram_message("⚠️ No qualifying call options found for owned stocks (≥100 shares).")

except Exception as e:
    send_telegram_message(f"Error generating sell calls summary: {e}")

# ------------------ 30-DAY STATS ALERT WITH RSI30, TREND EMOJI, AND ALPHABETICAL SORT ------------------

table_data = []

for ticker_raw, ticker_clean in zip(TICKERS_RAW, TICKERS):
    try:
        stock = yf.Ticker(ticker_clean)
        hist = stock.history(period="30d", interval="1d")
        if hist.empty or not all(col in hist.columns for col in ['Low', 'High', 'Close']):
            continue

        low_30 = hist['Low'].min()
        high_30 = hist['High'].max()
        close_now = hist['Close'][-1]

        # Calculate RSI (30-day Wilder's method)
        delta = hist['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        period = 30
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # First value: simple average
        avg_gain.iloc[period-1] = gain.iloc[:period].mean()
        avg_loss.iloc[period-1] = loss.iloc[:period].mean()

        # Wilder's smoothing for subsequent values
        for i in range(period, len(avg_gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_latest = rsi[-1] if not pd.isna(rsi[-1]) else None

        # Determine dynamic RSI emoji with extreme levels
        if isinstance(rsi_latest, float):
            if rsi_latest <= 20:
                rsi_emoji = "🔴"  # Extremely oversold
            elif rsi_latest >= 80:
                rsi_emoji = "🟢"  # Extremely overbought
            else:
                rsi_min = rsi.min()
                rsi_max = rsi.max()
                if rsi_latest <= rsi_min + 0.33*(rsi_max - rsi_min):
                    rsi_emoji = "📉"  # Relatively low
                elif rsi_latest >= rsi_min + 0.66*(rsi_max - rsi_min):
                    rsi_emoji = "📈"  # Relatively high
                else:
                    rsi_emoji = "⚪"  # Neutral
        else:
            rsi_emoji = "❓"

        table_data.append({
            "Ticker": ticker_raw,
            "Current": close_now,
            "30D Low": low_30,
            "30D High": high_30,
            "RSI30": rsi_latest,
            "RSI Emoji": rsi_emoji
        })

    except Exception as e:
        table_data.append({
            "Ticker": ticker_raw,
            "Current": "Err",
            "30D Low": "Err",
            "30D High": "Err",
            "RSI30": "Err",
            "RSI Emoji": "❓"
        })

# Sort tickers alphabetically
table_data = sorted(table_data, key=lambda x: x['Ticker'])

# Format table in fixed-width style for Telegram
header = "<b>📊 30-Day Ticker Summary</b>\n<pre>"
table_header = f"{'Tkr':<6}|{'Cur':<7}|{'30L':<7}|{'30H':<7}|{'RSI':<6}|{'Trnd':<5}\n" + "-"*50

table_lines = [header + table_header]
for row in table_data:
    cur = f"${row['Current']:.2f}" if isinstance(row['Current'], float) else row['Current']
    low = f"${row['30D Low']:.2f}" if isinstance(row['30D Low'], float) else row['30D Low']
    high = f"${row['30D High']:.2f}" if isinstance(row['30D High'], float) else row['30D High']
    rsi_val = f"{row['RSI30']:.1f}" if isinstance(row['RSI30'], float) else row['RSI30']
    rsi_emoji = row['RSI Emoji']

    table_lines.append(f"{row['Ticker']:<6}|{cur:<7}|{low:<7}|{high:<7}|{rsi_val:<6}|{rsi_emoji:<5}")

table_lines.append("</pre>")

# Send Telegram alert
send_telegram_message("\n".join(table_lines))

