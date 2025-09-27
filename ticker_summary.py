import os
import requests
import yfinance as yf
from datetime import datetime, timedelta

# --- Config ---
TICKERS = ["TLRY", "PLUG", "BITF", "BBAI", "SPCE", "ONDS", "GRAB", "LUMN", "RIG", "BB", "HTZ", "RXRX", "CLOV", "RZLV", "NVTS", "RR"]

# Secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Build summary ---
messages = []
today = datetime.now().date()
cutoff = today + timedelta(days=30)  # look ahead 30 days

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        info = stock.calendar

        div_date = info.get("Ex-Dividend Date")
        earnings_date = info.get("Earnings Date")

        msg_parts = [f"📊 {ticker}"]

        if div_date and isinstance(div_date, (list, tuple)):
            div_date = div_date[0].date()
        if earnings_date and isinstance(earnings_date, (list, tuple)):
            earnings_date = earnings_date[0].date()

        if div_date and today <= div_date <= cutoff:
            msg_parts.append(f"💰 Dividend on {div_date}")
        if earnings_date and today <= earnings_date <= cutoff:
            msg_parts.append(f"📢 Earnings on {earnings_date}")

        if len(msg_parts) > 1:
            messages.append(" | ".join(msg_parts))

    except Exception as e:
        messages.append(f"⚠️ {ticker} error: {e}")

# --- Send to Telegram ---
if messages:
    text = "📅 <b>Upcoming Dividends & Earnings (Next 30d)</b>\n\n" + "\n".join(messages)
else:
    text = "✅ No dividends or earnings in the next 30 days."

requests.post(
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
    data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
)
