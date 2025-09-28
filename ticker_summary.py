import os
import sys
import subprocess
import requests
import yfinance as yf
from datetime import datetime, timedelta

# --- Ensure lxml is installed ---
try:
    import lxml  # noqa
except ImportError:
    print("lxml not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lxml"])
    import lxml  # noqa

# --- Config ---
TICKERS = [
    "TLRY", "PLUG", "BITF", "BBAI", "SPCE", "ONDS", "GRAB",
    "LUMN", "RIG", "BB", "HTZ", "RXRX", "CLOV", "RZLV", "NVTS", "RR"
]

# --- Secrets ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment.")

# --- Build summary ---
today = datetime.now().date()
cutoff = today + timedelta(days=30)

safe_tickers = []
risky_msgs = []
safe_count = 0
risky_count = 0

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        msg_parts = [f"📊 <b>{ticker}</b>"]
        has_event = False

        # --- Dividend check ---
        try:
            if not stock.dividends.empty:
                div_date = stock.dividends.index[-1].date()
                if today <= div_date <= cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event = True
        except Exception as e:
            msg_parts.append(f"⚠️ Dividend check error: {e}")

        # --- Earnings check ---
        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index.min().date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"⚠️ 📢 Earnings on {earnings_date.strftime('%d-%m-%y')}")
                    has_event = True
        except Exception as e:
            msg_parts.append(f"⚠️ Earnings check error: {e}")

        # --- Sort into safe/risky ---
        if has_event:
            risky_msgs.append(" | ".join(msg_parts))
            risky_count += 1
        else:
            safe_tickers.append(ticker)
            safe_count += 1

    except Exception as e:
        risky_msgs.append(f"⚠️ <b>{ticker}</b> error: {e}")
        risky_count += 1

# --- Build header ---
header = "📅 <b>Upcoming Dividends & Earnings (Next 30d)</b>\n\n"

# --- Build body with separate sections ---
body = ""

if risky_msgs:
    body += "⚠️ <b>Risky Tickers</b>\n" + "\n".join(risky_msgs) + "\n\n"
else:
    body += "⚠️ <b>No risky tickers found 🎉</b>\n\n"

if safe_tickers:
    safe_tickers_sorted = sorted(safe_tickers)
    safe_bold = [f"<b>{t}</b>" for t in safe_tickers_sorted]

    # group into rows of 4
    safe_rows = [
        ", ".join(safe_bold[i:i+4]) for i in range(0, len(safe_bold), 4)
    ]
    body += "✅ <b>Safe Tickers</b>\n" + "\n".join(safe_rows)

# --- Add summary footer ---
footer = f"\n\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}"

# --- Final text ---
text = header + body + footer

# --- Send Telegram message (handle 4096 char limit) ---
MAX_LEN = 4000
if len(text) > MAX_LEN:
    for i in range(0, len(text), MAX_LEN):
        chunk = text[i:i + MAX_LEN]
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"}
        )
        print("Telegram response:", resp.text)
else:
    resp = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    )
    print("Telegram response:", resp.text)
