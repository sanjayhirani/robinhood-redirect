import os
import requests
import yfinance as yf
from datetime import datetime, timedelta

# --- Config ---
TICKERS = ["TLRY", "PLUG", "BITF", "BBAI", "SPCE", "ONDS", "GRAB",
           "LUMN", "RIG", "BB", "HTZ", "RXRX", "CLOV", "RZLV", "NVTS", "RR"]

# Secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Build summary ---
today = datetime.now().date()
cutoff = today + timedelta(days=30)  # look ahead 30 days

messages = []
safe_count = 0
risky_count = 0
earnings_skipped = False  # flag if earnings retrieval fails globally

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        msg_parts = [f"📊 {ticker}"]
        has_event = False

        # --- Dividend check ---
        div_date = None
        if not stock.dividends.empty:
            div_date = stock.dividends.index[-1].date()
            if today <= div_date <= cutoff:
                msg_parts.append(f"⚠️ 💰 Dividend on {div_date}")
                has_event = True

        # --- Earnings check (safe wrapper) ---
        earnings_date = None
        try:
            earnings_dates = stock.get_earnings_dates(limit=2)
            if not earnings_dates.empty:
                earnings_date = earnings_dates.index[0].date()
                if today <= earnings_date <= cutoff:
                    msg_parts.append(f"⚠️ 📢 Earnings on {earnings_date}")
                    has_event = True
        except ImportError:
            earnings_skipped = True
        except Exception:
            earnings_skipped = True

        # --- Mark safe tickers ---
        if not has_event:
            msg_parts.append("✅ No dividend/earnings in next 30d")
            safe_count += 1
        else:
            risky_count += 1

        messages.append(" | ".join(msg_parts))

    except Exception as e:
        messages.append(f"⚠️ {ticker} error: {e}")
        risky_count += 1

# --- Build header ---
header = "📅 <b>Upcoming Dividends & Earnings (Next 30d)</b>\n\n"
if earnings_skipped:
    header = "⚠️ Earnings data may be incomplete (lxml missing)\n\n" + header

# --- Add summary footer ---
footer = f"\n\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}"

# --- Send Telegram message ---
text = header + "\n".join(messages) + footer

resp = requests.post(
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
    data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
)

print("Telegram response:", resp.text)
