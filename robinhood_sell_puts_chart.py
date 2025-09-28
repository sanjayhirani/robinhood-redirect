# robinhood_puts_playwright_full.py

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
import sys, subprocess

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in ["yfinance", "pandas", "matplotlib", "numpy", "requests", "playwright"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Install Playwright browsers
subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])

# ------------------ IMPORTS ------------------
import os, io, asyncio
from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import yfinance
from playwright.async_api import async_playwright

# ------------------ CONFIG ------------------
TICKERS = ["TLRY","PLUG","BITF","BBAI","SPCE","ONDS","GRAB","LUMN",
           "RIG","BB","HTZ","RXRX","CLOV","RZLV","NVTS","RR"]
NUM_PUTS = 2
LOW_DAYS = 14
MAX_EXP_DAYS = 21
MIN_PRICE = 0.05
BEST_MIN_PRICE = 0.1
CANDLE_WIDTH = 0.6

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILS ------------------
def risk_emoji(prob_otm):
    if prob_otm >= 80: return "✅"
    elif prob_otm >= 60: return "🟡"
    else: return "⚠️"

def send_telegram_photo(buf, caption):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                  files={'photo': buf},
                  data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"})

def send_telegram_message(msg):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                  data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})

def plot_candlestick(df, current_price, last_14_low, strike_price=None, exp_date=None):
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    for i in range(len(df)):
        color = 'lime' if df['Close'].iloc[i]>=df['Open'].iloc[i] else 'red'
        ax.add_patch(plt.Rectangle((mdates.date2num(df.index[i])-CANDLE_WIDTH/2,
                                    min(df['Open'].iloc[i],df['Close'].iloc[i])),
                                   CANDLE_WIDTH, abs(df['Close'].iloc[i]-df['Open'].iloc[i]),
                                   color=color))
        ax.plot([mdates.date2num(df.index[i]),mdates.date2num(df.index[i])],
                [df['Low'].iloc[i], df['High'].iloc[i]], color=color, linewidth=1)
    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=2, label=f'14-day Low: ${last_14_low:.2f}')
    if strike_price is not None:
        ax.axhline(strike_price, color='cyan', linestyle='--', linewidth=1.5, label=f'Strike: ${strike_price:.2f}')
    if exp_date is not None:
        exp_dt = pd.to_datetime(exp_date).tz_localize(None)
        if df.index.min() <= exp_dt <= df.index.max():
            ax.axvline(mdates.date2num(exp_dt), color='orange', linestyle='--', linewidth=2, label=f'Expiration: {exp_dt.strftime("%d-%m-%y")}')
    ax.set_ylabel('Price ($)', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
    buf.seek(0)
    plt.close()
    return buf

# ------------------ SCRAPER ------------------
async def scrape_option_data(page, ticker):
    await page.goto(f"https://robinhood.com/stocks/{ticker}/options")
    await page.wait_for_timeout(5000)  # wait page load
    rows = await page.query_selector_all("div[class*='optionsRow']")
    results = []
    for row in rows:
        try:
            strike_text = await row.query_selector("span[class*='strikePrice']")
            ask_text = await row.query_selector("span[class*='askPrice']")
            delta_text = await row.query_selector("span[class*='delta']")
            cop_text = await row.query_selector("span[class*='chanceOfProfit']")
            exp_text = await row.query_selector("span[class*='expirationDate']")
            strike = float(await strike_text.inner_text() if strike_text else 0)
            ask = float(await ask_text.inner_text() if ask_text else 0)
            delta = float(await delta_text.inner_text() if delta_text else 0)
            prob_otm = int(float((await cop_text.inner_text()).replace('%','')) if cop_text else 0)
            exp_date = datetime.strptime(await exp_text.inner_text(), "%m/%d/%y").date() if exp_text else None
            results.append({
                "Strike": strike, "Ask": ask, "Delta": delta, "Prob OTM": prob_otm, "Expiration": exp_date
            })
        except:
            continue
    return results

# ------------------ MAIN ------------------
async def main():
    today = datetime.now().date()
    cutoff = today + timedelta(days=30)
    safe_tickers, risky_msgs, all_options = [], [], []

    # Earnings/Dividend alerts
    for ticker in TICKERS:
        try:
            stock = yfinance.Ticker(ticker)
            msg_parts = [f"📊 <b>{ticker}</b>"]
            has_event = False
            try:
                if not stock.dividends.empty:
                    div_date = stock.dividends.index[-1].date()
                    if today <= div_date <= cutoff:
                        msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                        has_event = True
            except: pass
            try:
                earnings_dates = stock.get_earnings_dates(limit=2)
                if not earnings_dates.empty:
                    earnings_date = earnings_dates.index.min().date()
                    if today <= earnings_date <= cutoff:
                        msg_parts.append(f"⚠️ 📢 Earnings on {earnings_date.strftime('%d-%m-%y')}")
                        has_event = True
            except: pass
            if has_event:
                risky_msgs.append(" | ".join(msg_parts))
            else:
                safe_tickers.append(ticker)
        except:
            risky_msgs.append(f"⚠️ <b>{ticker}</b> error")

    # Send summary
    summary_lines = []
    summary_lines.append("⚠️ <b>Risky Tickers</b>\n"+"\n".join(risky_msgs) if risky_msgs else "⚠️ <b>No risky tickers found 🎉</b>")
    safe_bold = [f"<b>{t}</b>" for t in sorted(safe_tickers)]
    safe_rows = [", ".join(safe_bold[i:i+4]) for i in range(0,len(safe_bold),4)]
    if safe_rows:
        summary_lines.append("✅ <b>Safe Tickers</b>\n"+"\n".join(safe_rows))
    send_telegram_message("\n".join(summary_lines))

    # Launch Playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://robinhood.com/login")
        await page.fill("input[name='username']", USERNAME)
        await page.fill("input[name='password']", PASSWORD)
        await page.click("button[type='submit']")
        await page.wait_for_timeout(10000)  # handle 2FA manually if needed

        # Individual ticker alerts
        for ticker in safe_tickers:
            options = await scrape_option_data(page, ticker)
            hist = yfinance.Ticker(ticker).history(period="1mo")
            current_price = float(hist['Close'][-1])
            last_14_low = hist['Low'][-LOW_DAYS:].min()
            valid_puts = [opt for opt in options if opt["Strike"] < current_price and 0 <= (opt["Expiration"]-today).days <= MAX_EXP_DAYS]
            valid_puts.sort(key=lambda x: x["Strike"], reverse=True)
            top_puts = valid_puts[:3]
            # Telegram messages: top 2 options
            buf = plot_candlestick(hist, current_price, last_14_low)
            msg_lines = [f"📊 <a href='https://robinhood.com/stocks/{ticker}'>{ticker}</a> current: ${current_price:.2f}"]
            for opt in top_puts[:2]:
                msg_lines.append(f"{risk_emoji(opt['Prob OTM'])} 📅 Exp: {opt['Expiration'].strftime('%d-%m-%y')}")
                msg_lines.append(f"💲 Strike: ${opt['Strike']}")
                msg_lines.append(f"💰 Price: ${opt['Ask']:.2f}")
                msg_lines.append(f"🔺 Delta: {opt['Delta']:.3f}")
                msg_lines.append(f"🎯 Prob: {opt['Prob OTM']}%\n")
                all_options.append(opt)
            send_telegram_photo(buf, "\n".join(msg_lines))

        # Best option alert
        best_options = [opt for opt in all_options if opt["Ask"] >= BEST_MIN_PRICE and opt["Prob OTM"]>=80]
        if not best_options and all_options:
            best_options = all_options  # fallback if none meet criteria
        if best_options:
            best = max(best_options, key=lambda x: x["Prob OTM"])
            hist = yfinance.Ticker(ticker).history(period="1mo")
            buf = plot_candlestick(hist, best['Strike'], last_14_low, best['Strike'], best['Expiration'])
            msg_lines = [
                "🔥 <b>Best Option to Sell</b>:",
                f"📊 <a href='https://robinhood.com/stocks/{ticker}'>{ticker}</a> current: ${current_price:.2f}",
                f"✅ Expiration: {best['Expiration'].strftime('%d-%m-%y')}",
                f"💲 Strike: ${best['Strike']}",
                f"💰 Price: ${best['Ask']:.2f}",
                f"🔺 Delta: {best['Delta']:.3f}",
                f"🎯 Prob OTM: {best['Prob OTM']}%"
            ]
            send_telegram_photo(buf, "\n".join(msg_lines))
        await browser.close()

# Run
asyncio.run(main())
