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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Only process tickers currently in tickers.txt
for ticker_raw, ticker_clean in zip(TICKERS_RAW, TICKERS):
    if ticker_clean not in TICKERS:
        # Skip removed tickers entirely
        continue

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


# ------------------ OPTIONS SCAN (PARALLELIZED WITH THROTTLE) ------------------

all_options = []
account_data = r.profiles.load_account_profile()
buying_power = float(account_data.get('buying_power', 0.0))

def scan_ticker(ticker_raw, ticker_clean):
    ticker_results = []
    try:
        latest_price = r.stocks.get_latest_price(ticker_clean)
        if not latest_price or latest_price[0] is None:
            return []
        current_price = float(latest_price[0])

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
        df.rename(columns={'open_price': 'open','close_price': 'close','high_price': 'high','low_price': 'low'}, inplace=True)
        df = df.asfreq('B').ffill()

        low_days = int(config.get("low_days", 30))
        last_low = df['low'][-low_days:].min()
        if pd.isna(last_low) or last_low <= 0:
            return []

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

        exp_dates = sorted({opt.get('expiration_date') for opt in all_puts if opt.get('expiration_date')})
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:config.get("num_expirations", 3)]

        candidate_puts = []
        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt.get('expiration_date') == exp_date]
            strikes_below = sorted(
                [float(opt.get('strike_price')) for opt in puts_for_exp if opt.get('strike_price') and float(opt.get('strike_price')) < current_price],
                reverse=True
            )
            chosen_strikes = strikes_below[1:4] if len(strikes_below) > 1 else strikes_below

            option_ids = [opt.get('id') for opt in puts_for_exp if opt.get('strike_price') and float(opt.get('strike_price')) in chosen_strikes and opt.get('id')]
            if not option_ids:
                continue

            market_data_list = None
            try:
                market_data_list = r.options.get_option_market_data_by_id(option_ids)
            except Exception:
                market_data_list = None

            if not market_data_list:
                market_data_list = []
                for oid in option_ids:
                    try:
                        md_resp = r.options.get_option_market_data_by_id(oid)
                        if md_resp:
                            if isinstance(md_resp, list):
                                market_data_list.append(md_resp[0])
                            else:
                                market_data_list.append(md_resp)
                    except Exception:
                        pass
                    time.sleep(0.05)
            else:
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
                if last_low == 0:
                    continue
                dist_from_low = (strike_price - last_low) / last_low
                if dist_from_low < 0.01:
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
        time.sleep(0.15)

# ------------------ FILTER OPTIONS GLOBALLY BY ABS(DELTA) AND COP ------------------

all_options = [opt for opt in all_options if abs(opt.get('Delta', 1)) <= 0.3 and opt.get('COP Short', 0) >= 0.7]

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

    top_tickers = sorted(ticker_best.values(), key=lambda x: (x['score'], x['COP Short']), reverse=True)[:10]
    top_ticker_names = {t['Ticker'] for t in top_tickers}

# ------------------ ALL OPTIONS SUMMARY ------------------

if all_options:
    summary_rows = []
    all_display_options = []
    for opt in all_options:
        max_contracts = max(1, int(buying_power // (opt['Strike Price'] * 100)))
        total_premium = opt['Bid Price'] * 100 * max_contracts
        opt['Max Contracts'] = max_contracts
        opt['Total Premium'] = total_premium
        all_display_options.append(opt)

    all_display_options = sorted(all_display_options, key=lambda x: x['Total Premium'], reverse=True)
    for opt in all_display_options:
        exp_md = opt['Expiration Date'][5:]
        summary_rows.append(
            f"{opt['Ticker']:<5}|{exp_md:<5}|{opt['Strike Price']:<6.2f}|"
            f"{opt['Bid Price']:<4.2f}|{abs(opt['Delta']):<5.2f}|{opt['COP Short']*100:<5.1f}%|"
            f"{opt['Max Contracts']:<2}|${opt['Total Premium']:<5.0f}"
        )

    header = "<b>📋 All Options Summary — Across All Tickers</b>\n"
    table_header = f"{'Tkr':<5}|{'Exp':<5}|{'Strk':<6}|{'Bid':<4}|{'Δ':<5}|{'COP%':<5}|{'Ct':<2}|{'Prem':<5}\n" + "-"*45

    chunk_size = 30
    for i in range(0, len(summary_rows), chunk_size):
        chunk = summary_rows[i:i+chunk_size]
        chunk_body = "\n".join(chunk)
        msg = header + "\n<pre>" + table_header + "\n" + chunk_body + "</pre>"
        send_telegram_message(msg)

# ------------------ CURRENT OPEN POSITIONS ALERT ------------------

try:
    positions = r.options.get_open_option_positions()
    if positions:
        msg_lines = ["📋 <b>Current Open Positions</b>\n"]
        for pos in positions:
            qty_raw = float(pos.get("quantity") or 0)
            if qty_raw == 0:
                continue
            contracts = abs(int(qty_raw))
            instrument = r.helper.request_get(pos.get("option"))
            ticker = instrument.get("chain_symbol")
            strike = float(instrument.get("strike_price"))
            exp_date = pd.to_datetime(instrument.get("expiration_date")).strftime("%Y-%m-%d")
            avg_price_raw = float(pos.get("average_price") or 0.0)
            md = r.options.get_option_market_data_by_id(instrument.get("id"))[0]
            md_mark_price = float(md.get("mark_price") or 0.0)
            mark_per_contract = md_mark_price * 100
            orig_pnl = abs(avg_price_raw) * contracts
            pnl_now = orig_pnl - (mark_per_contract * contracts)
            pnl_emoji = "🟢" if pnl_now >= 0.7 * orig_pnl else "🔴"
            msg_lines.extend([
                f"📌 <b>{ticker}</b> | 📉 Sell Put",
                f"💲 Strike: ${strike:.2f}",
                f"✅ Exp: {exp_date}",
                f"📦 Qty: {contracts}",
                f"📊 Current Price: ${float(r.stocks.get_latest_price(ticker)[0]):.2f}",
                f"💰 Full Premium: ${orig_pnl:.2f}",
                f"💵 Current Profit: {pnl_emoji} ${pnl_now:.2f}",
                "────────────────────"
            ])
        send_telegram_message("\n".join(msg_lines))
except Exception as e:
    send_telegram_message(f"Error generating current positions alert: {e}")

# ------------------ BEST PUT ALERT ------------------

top10_best_options = sorted(all_options, key=lambda x: x['Total Premium'], reverse=True)[:10]

if top10_best_options:
    eligible_options = [opt for opt in top10_best_options if opt['COP Short'] >= 0.7]
    if eligible_options:
        best = max(eligible_options, key=lambda x: x['Total Premium'])
    else:
        best = max(top10_best_options, key=lambda x: x['Total Premium'])

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
