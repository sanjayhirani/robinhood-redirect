# test_robinhood_playwright_top3.py

# ------------------ AUTO-INSTALL PLAYWRIGHT ------------------
import sys, subprocess

try:
    import playwright
except ImportError:
    print("Playwright not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    import playwright
    print("Installing Chromium browser for Playwright...")
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])

# ------------------ IMPORTS ------------------
import asyncio, os
from playwright.async_api import async_playwright

# ------------------ CONFIG ------------------
TICKER = "RXRX"  # test ticker
RH_USERNAME = os.environ["RH_USERNAME"]
RH_PASSWORD = os.environ["RH_PASSWORD"]

# ------------------ SCRAPER ------------------
async def scrape_option_data(page, ticker):
    await page.goto(f"https://robinhood.com/stocks/{ticker}/options")
    await page.wait_for_timeout(5000)  # wait for options to load

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
            exp_date = await exp_text.inner_text() if exp_text else None

            results.append({
                "Ticker": ticker,
                "Strike": strike,
                "Ask": ask,
                "Delta": delta,
                "Prob OTM": prob_otm,
                "Expiration": exp_date
            })
        except Exception as e:
            print(f"Error parsing row: {e}")
    return results

# ------------------ MAIN ------------------
async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)  # headless for CI / GitHub Actions
        page = await browser.new_page()

        # Login
        await page.goto("https://robinhood.com/login")
        await page.fill("input[name='username']", RH_USERNAME)
        await page.fill("input[name='password']", RH_PASSWORD)
        await page.click("button[type='submit']")
        await page.wait_for_timeout(10000)  # handle 2FA manually if needed

        # Scrape options
        options = await scrape_option_data(page, TICKER)

        # Get current stock price
        await page.goto(f"https://robinhood.com/stocks/{TICKER}")
        await page.wait_for_timeout(3000)
        current_price_el = await page.query_selector("span[data-testid='StockPrice']")
        current_price = float(await current_price_el.inner_text()) if current_price_el else 0

        # Filter puts below current price and get top 3
        puts_below = [opt for opt in options if opt["Strike"] < current_price]
        top3_puts = sorted(puts_below, key=lambda x: x["Strike"], reverse=True)[:3]

        print(f"Current price: ${current_price}")
        print(f"Top 3 puts below current price for {TICKER}:")
        for opt in top3_puts:
            print(opt)

        await browser.close()

# Run
asyncio.run(main())
