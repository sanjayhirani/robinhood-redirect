# test_robinhood_playwright_one_ticker.py

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

    # Update selectors if needed based on Robinhood live DOM
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
        browser = await pw.chromium.launch(headless=False)  # headless=True for headless
        page = await browser.new_page()

        # Login using username/password
        await page.goto("https://robinhood.com/login")
        await page.fill("input[name='username']", RH_USERNAME)
        await page.fill("input[name='password']", RH_PASSWORD)
        await page.click("button[type='submit']")
        await page.wait_for_timeout(10000)  # handle 2FA manually once

        # Scrape options
        options = await scrape_option_data(page, TICKER)
        print(f"Found {len(options)} options for {TICKER}")
        for opt in options:
            print(opt)

        await browser.close()

# Run
asyncio.run(main())
