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

        # Get earnings dates (next 2 upcoming events)
        earnings_dates = stock.get_earnings_dates(limit=2)
        earnings_date = None
        if not earnings_dates.empty:
            earnings_date = earnings_dates.index[0].date()

        # Get dividend history (last ex-dividend date + estimate next if available)
        dividends = stock.dividends
        div_date = None
        if not dividends.empty:
            div_date = dividends.index[-1].date()  # last ex-dividend date

        msg_parts = [f"📊 {ticker}"]

        if div_date and today <= div_date <= cutoff:
            msg_parts.append(f"💰 Dividend on {div_date}")
        if earnings_date and today <= earnings_date <= cutoff:
            msg_parts.append(f"📢 Earnings on {earnings_date}")

        if len(msg_parts) > 1:
            messages.append(" | ".join(msg_parts))

    except Exception as e:
        messages.append(f"⚠️ {ticker} error: {e}")

# --- Always send message ---
if messages:
    text = "📅 <b>Upcoming Dividends & Earnings (Next 30d)</b>\n\n" + "\n".join(messages)
else:
    text = "✅ No dividends or earnings in the next 30 days."

resp = requests.post(
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
    data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
)

print("Telegram response:", resp.text)
