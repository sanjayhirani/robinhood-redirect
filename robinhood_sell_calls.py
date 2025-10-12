import os
import requests
import robin_stocks.robinhood as r
import pandas as pd
import yaml
from datetime import datetime, timedelta

# ------------------ LOAD CONFIG ------------------
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MIN_PRICE = float(config.get("min_price", 0.10))
LOW_DAYS = int(config.get("low_days", 14))
EXPIRY_LIMIT_DAYS = int(config.get("expiry_limit_days", 21))
NUM_CALLS = int(config.get("num_calls", 3))

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ TELEGRAM ------------------
def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

# ------------------ LOGIN ------------------
r.login(USERNAME, PASSWORD)
today = datetime.now().date()
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

# ------------------ GET TICKERS WITH ≥100 SHARES ------------------
owned_positions = r.positions.get_open_stock_positions()
tickers = []
ticker_data = {}  # store current price & instrument id

for pos in owned_positions:
    if pos.get("instrument") and pos.get("quantity"):
        qty = float(pos["quantity"] or 0)
        if qty >= 100:
            instr = r.helper.request_get(pos["instrument"])
            symbol = instr.get("symbol")
            if symbol:
                tickers.append(symbol)
                ticker_data[symbol] = {"quantity": qty, "instrument_id": instr.get("id")}

if not tickers:
    send_telegram_message("⚠️ No tickers with ≥100 shares found.")
    exit()

# ------------------ SCAN EACH TICKER FOR COVERED CALLS ------------------
all_calls = []

for ticker in tickers:
    try:
        current_price = float(r.stocks.get_latest_price(ticker)[0])
        historicals = r.stocks.get_stock_historicals(
            ticker, interval='day', span='month', bounds='regular'
        )
        df = pd.DataFrame(historicals)
        if df.empty or 'begins_at' not in df.columns:
            continue
        df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
        df.set_index('begins_at', inplace=True)
        df.rename(columns={'open_price':'open','close_price':'close',
                           'high_price':'high','low_price':'low'}, inplace=True)
        df = df.asfreq('B').ffill()
        last_low = df['low'][-LOW_DAYS:].min()
        if last_low <= 0:
            continue

        # Find tradable calls
        all_options = r.options.find_tradable_options(ticker, optionType="call")
        if not all_options:
            continue
        if isinstance(all_options, dict):
            all_options = all_options.get('results', [])

        # Expiration filter
        exp_dates = sorted({opt['expiration_date'] for opt in all_options if opt.get('expiration_date')})
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]

        candidate_calls = []

        for exp_date in exp_dates:
            calls_for_exp = [opt for opt in all_options if opt['expiration_date'] == exp_date]
            strikes_above = sorted([float(opt['strike_price']) for opt in calls_for_exp if float(opt['strike_price']) > current_price])
            chosen_strikes = strikes_above[2:5] if len(strikes_above) > 2 else strikes_above

            for opt in calls_for_exp:
                strike = float(opt['strike_price'])
                if strike not in chosen_strikes:
                    continue
                option_id = opt['id']
                md = r.options.get_option_market_data_by_id(option_id)[0]
                bid_price = float(md.get("bid_price") or 0.0)
                delta = float(md.get("delta") or 0.0)
                cop_short = float(md.get("chance_of_profit_short") or 0.0)
                if bid_price >= MIN_PRICE:
                    candidate_calls.append({
                        "Ticker": ticker,
                        "Current Price": current_price,
                        "Expiration Date": exp_date,
                        "Strike Price": strike,
                        "Bid Price": bid_price,
                        "Delta": delta,
                        "COP Short": cop_short,
                        "Distance From Low": (strike - last_low)/last_low
                    })
        all_calls.extend(candidate_calls)

    except Exception as e:
        send_telegram_message(f"⚠️ Error on {ticker}: {e}")

# ------------------ FILTER CALLS ------------------
filtered_calls = [c for c in all_calls if abs(c['Delta']) <= 0.3 and c['COP Short'] >= 0.7]

# ------------------ SUMMARY ------------------
if filtered_calls:
    summary_rows = []
    for c in sorted(filtered_calls, key=lambda x: x['COP Short'], reverse=True):
        summary_rows.append(
            f"{c['Ticker']} | Exp: {c['Expiration Date']} | Strike: {c['Strike Price']} | "
            f"Bid: {c['Bid Price']:.2f} | Δ: {abs(c['Delta']):.2f} | COP: {c['COP Short']*100:.1f}%"
        )
    header = "<b>📋 Top Covered Calls</b>\n"
    send_telegram_message(header + "\n" + "\n".join(summary_rows))
else:
    send_telegram_message("⚠️ No covered calls meet Δ ≤ 0.3 and COP ≥ 70%")

# ------------------ CURRENT OPEN CALL POSITIONS ------------------
try:
    positions = r.options.get_open_option_positions()
    call_positions = []
    for pos in positions:
        qty = float(pos.get("quantity") or 0)
        if qty == 0:
            continue
        instrument = r.helper.request_get(pos.get("option"))
        if instrument.get("type") != "call":
            continue
        ticker = instrument.get("chain_symbol")
        strike = float(instrument.get("strike_price"))
        exp_date = instrument.get("expiration_date")
        avg_price = float(pos.get("average_price") or 0)
        md = r.options.get_option_market_data_by_id(instrument.get("id"))[0]
        mark_price = float(md.get("mark_price") or 0.0)
        pnl_now = (mark_price - avg_price) * abs(int(qty)) * 100
        call_positions.append(
            f"📌 <b>{ticker}</b> | Exp: {exp_date} | Strike: {strike:.2f} | Qty: {int(abs(qty))} | Current PnL: ${pnl_now:.2f}"
        )
    if call_positions:
        send_telegram_message("<b>📋 Current Open Call Positions</b>\n" + "\n".join(call_positions))
except Exception as e:
    send_telegram_message(f"⚠️ Error generating current call positions alert: {e}")

# ------------------ BEST COVERED CALL ALERT ------------------
eligible_calls = [c for c in filtered_calls if abs(c['Delta']) <= 0.25 and c['COP Short'] >= 0.73]
if eligible_calls:
    best = max(eligible_calls, key=lambda x: x['Bid Price']*100*x['COP Short'])
    send_telegram_message(
        f"🔥 <b>Best Covered Call</b>\n"
        f"{best['Ticker']} | Exp: {best['Expiration Date']} | Strike: {best['Strike Price']:.2f} | "
        f"Bid: {best['Bid Price']:.2f} | Δ: {abs(best['Delta']):.2f} | COP: {best['COP Short']*100:.1f}%"
    )
