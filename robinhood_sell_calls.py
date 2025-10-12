import os
import subprocess
import requests
import robin_stocks.robinhood as r
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", pkg_name])

for pkg in ["pandas", "numpy", "requests", "robin_stocks", "PyYAML"]:
    ensure_package(pkg)

# ------------------ LOAD CONFIG ------------------
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MIN_PRICE = float(config.get("min_price", 0.10))
LOW_DAYS = int(config.get("low_days", 30))
EXPIRY_LIMIT_DAYS = int(config.get("expiry_limit_days", 30))
NUM_CALLS = int(config.get("num_calls", 3))
MAX_WORKERS = int(config.get("max_workers", 5))

# ------------------ LOAD TICKERS ------------------
TICKERS_FILE = "calls_tickers.txt"
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
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

# ------------------ OWNERSHIP CHECK ------------------
safe_tickers = []
risky_msgs = []
for ticker_raw, ticker_clean in zip(TICKERS_RAW, TICKERS):
    try:
        positions = r.positions.get_open_stock_positions()
        shares_owned = 0
        for pos in positions:
            if pos.get("instrument") and pos.get("quantity"):
                instr = r.helper.request_get(pos["instrument"])
                if instr.get("symbol") == ticker_clean:
                    shares_owned = float(pos.get("quantity") or 0)
                    break
        if shares_owned < 100:
            risky_msgs.append(f"⚠️ {ticker_raw} has less than 100 shares.")
        else:
            safe_tickers.append((ticker_raw, ticker_clean))
    except Exception as e:
        risky_msgs.append(f"{ticker_raw} error: {e}")

# ------------------ SEND RISK SUMMARY ------------------
safe_count = len(safe_tickers)
risky_count = len(risky_msgs)
summary_lines = []

summary_lines.append("<b>📋 Tickers Ownership Check</b>\n")
if risky_msgs:
    summary_lines.append(f"{config['telegram_labels']['risky_tickers']}\n" + "\n".join(risky_msgs))

if safe_tickers:
    safe_rows = [", ".join([t[0] for t in safe_tickers][i:i+4]) for i in range(0, len(safe_tickers), 4)]
    summary_lines.append(f"{config['telegram_labels']['safe_tickers']}\n" + "\n".join(safe_rows))

summary_lines.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary_lines))

# ------------------ SCAN TICKERS FOR COVERED CALLS ------------------
all_calls = []

def scan_ticker(ticker_raw, ticker_clean):
    try:
        latest_price = r.stocks.get_latest_price(ticker_clean)
        if not latest_price or latest_price[0] is None:
            return []
        current_price = float(latest_price[0])

        # Historical low for distance filter
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
        df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        df = df.asfreq('B').ffill()
        last_low = df['low'][-LOW_DAYS:].min()
        if pd.isna(last_low) or last_low <= 0:
            return []

        # Tradable calls
        all_options = r.options.find_tradable_options(ticker_clean, optionType="call")
        if not all_options:
            return []

        # Normalize list
        if isinstance(all_options, dict):
            maybe = all_options.get('results') or all_options.get('options') or None
            if isinstance(maybe, list):
                all_options = maybe
            else:
                found = False
                for v in all_options.values():
                    if isinstance(v, list):
                        all_options = v
                        found = True
                        break
                if not found:
                    return []
        if not isinstance(all_options, list):
            try: all_options = list(all_options)
            except Exception: return []

        # Expiration filter
        exp_dates = sorted({opt.get('expiration_date') for opt in all_options if opt.get('expiration_date')})
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        exp_dates = exp_dates[:config.get("num_expirations", 3)]

        candidate_calls = []
        for exp_date in exp_dates:
            calls_for_exp = [opt for opt in all_options if opt.get('expiration_date') == exp_date]
            strikes_above = sorted([float(opt.get('strike_price')) for opt in calls_for_exp if opt.get('strike_price') and float(opt['strike_price']) > current_price])
            chosen_strikes = strikes_above[2:5] if len(strikes_above) > 2 else strikes_above

            option_ids = [opt.get('id') for opt in calls_for_exp if opt.get('strike_price') and float(opt['strike_price']) in chosen_strikes]
            if not option_ids:
                continue

            market_data_list = []
            for oid in option_ids:
                try:
                    md_resp = r.options.get_option_market_data_by_id(oid)
                    if isinstance(md_resp, list):
                        market_data_list.append(md_resp[0])
                    else:
                        market_data_list.append(md_resp)
                except:
                    continue
                time.sleep(0.05)

            for opt, md in zip(calls_for_exp, market_data_list):
                try:
                    strike_price = float(opt.get('strike_price'))
                    bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                    delta = float(md.get('delta') or 0.0)
                    cop_short = float(md.get('chance_of_profit_short') or 0.0)
                    if bid_price < MIN_PRICE:
                        continue
                    candidate_calls.append({
                        "Ticker": ticker_raw,
                        "TickerClean": ticker_clean,
                        "Current Price": current_price,
                        "Expiration Date": exp_date,
                        "Strike Price": strike_price,
                        "Bid Price": bid_price,
                        "Delta": delta,
                        "COP Short": cop_short,
                        "Distance From Low": (strike_price - last_low) / last_low
                    })
                except:
                    continue
        return candidate_calls
    except Exception as e:
        send_telegram_message(f"{ticker_raw} error: {e}")
        return []

# ------------------ PARALLEL SCAN ------------------
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(scan_ticker, t_raw, t_clean) for t_raw, t_clean in safe_tickers]
    for f in as_completed(futures):
        all_calls.extend(f.result())
        time.sleep(0.1)

# ------------------ FILTER CALLS ------------------
all_calls = [c for c in all_calls if abs(c.get('Delta', 1)) <= 0.3 and c.get('COP Short', 0) >= 0.7]

# ------------------ TOP CALLS SUMMARY ------------------
if all_calls:
    def score(opt):
        days_to_exp = (datetime.strptime(opt['Expiration Date'], "%Y-%m-%d").date() - today).days
        if days_to_exp <= 0:
            return 0
        return opt['Bid Price'] * 100 * opt['COP Short'] / days_to_exp

    ticker_best = {}
    for opt in all_calls:
        t = opt['Ticker']
        sc = score(opt)
        if t not in ticker_best or sc > ticker_best[t]['score'] or (
            abs(sc - ticker_best[t]['score']) < 1e-6 and opt['COP Short'] > ticker_best[t]['COP Short']
        ):
            ticker_best[t] = {'score': sc, **opt}

    top_tickers = sorted(ticker_best.values(), key=lambda x: (x['score'], x['COP Short']), reverse=True)[:10]

    # Format text-only summary
    summary_rows = []
    for opt in top_tickers:
        summary_rows.append(
            f"{opt['Ticker']} | Exp: {opt['Expiration Date']} | Strike: {opt['Strike Price']} | "
            f"Bid: {opt['Bid Price']:.2f} | Δ: {abs(opt['Delta']):.2f} | COP: {opt['COP Short']*100:.1f}%"
        )

    header = "<b>📋 Top Covered Calls</b>\n"
    send_telegram_message(header + "\n" + "\n".join(summary_rows))

    # ------------------ TEXT TABLE FOR ALL ELIGIBLE CALLS ------------------
    eligible_calls = [c for c in all_calls if c['COP Short'] >= 0.73 and abs(c['Delta']) <= 0.25]
    if eligible_calls:
        best_call = max(eligible_calls, key=lambda x: x['COP Short'])

        table_lines = ["<b>📋 Eligible Covered Calls</b>\n"]
        table_lines.append(f"{'Tkr':<6}|{'Exp':<10}|{'Strike':<7}|{'Bid':<6}|{'Δ':<6}|{'COP%':<6}")
        table_lines.append("-"*45)

        for c in sorted(eligible_calls, key=lambda x: x['COP Short'], reverse=True):
            tkr = c['Ticker']
            exp = c['Expiration Date']
            strike = f"{c['Strike Price']:.2f}"
            bid = f"{c['Bid Price']:.2f}"
            delta = f"{abs(c['Delta']):.2f}"
            cop = f"{c['COP Short']*100:.1f}%"

            line = f"{tkr:<6}|{exp:<10}|{strike:<7}|{bid:<6}|{delta:<6}|{cop:<6}"

            # Highlight best call in bold
            if c['Ticker'] == best_call['Ticker'] and c['Strike Price'] == best_call['Strike Price'] and c['Expiration Date'] == best_call['Expiration Date']:
                line = f"<b>{line}</b>"

            table_lines.append(line)

        send_telegram_message("\n".join(table_lines))
else:
    send_telegram_message("⚠️ No covered calls meet Δ ≤ 0.3 and COP ≥ 70%")
# ------------------ CURRENT OPEN CALL POSITIONS ------------------
try:
    positions = r.options.get_open_option_positions()
    if positions:
        msg_lines = ["📋 <b>Current Open Call Positions</b>\n"]
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

            # PnL calculation for sell calls (short calls)
            orig_pnl = abs(avg_price_raw) * contracts
            pnl_now = orig_pnl - (mark_per_contract * contracts)

            pnl_emoji = "🟢" if pnl_now >= 0.7 * orig_pnl else "🔴"

            msg_lines.extend([
                f"📌 <b>{ticker}</b> | 📈 Sell Call",
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
    send_telegram_message(f"⚠️ Error generating current call positions alert: {e}")
