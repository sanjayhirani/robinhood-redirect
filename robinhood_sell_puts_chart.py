# test_robinhood_playwright_full.py

import sys
import subprocess
import os
import asyncio

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import playwright
except ImportError:
    install_package("playwright")
    import playwright

# Ensure Chromium browser is installed
subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])

# ------------------ CONFIG ------------------
TICKER = "RXRX"  # test ticker
RH_USERNAME = os.environ.get("RH_USERNAME")
RH_PASSWORD = os.environ.get("RH_PASSWORD")

if not RH_USERNAME or not RH_PASSWORD:
    raise ValueError("RH_USERNAME and RH_PASSWORD environment variables must be set.")

# ------------------ SCRAPER ------------------
async def scrape_option_data(page, ticker):
    await page.goto(f"https://robinhood.com/stocks/{ticker}/options")
    print(f"Navigated to {page.url}")

    try:
        await page.wait_for_selector("span[class*='strikePrice']", timeout=10000)
        print("Option data appears to be loaded.")
    except:
        print("Timeout: Option data did not load.")
        return []

    rows = await page.query_selector_all("div[class*='optionsRow']")
    print(f"Found {len(rows)} option rows")

    results = []
    for row in rows:
        try:
            strike_text = await row.query_selector("span[class*='strikePrice']")
            ask_text = await row.query_selector("span[class*='askPrice']")
            delta_text = await row.query_selector("span[class*='delta']")
            cop_text = await row.query_selector("span[class*='chanceOfProfit']")
            exp_text = await row.query_selector("span[class*='expirationDate']")

            strike = float((await strike_text.inner_text()).replace('$','').replace(',','')) if strike_text else 0
            ask = float((await ask_text.inner_text()).replace('$','').replace(',','')) if ask_text else 0
            delta = float(await delta_text.inner_text()) if delta_text else 0
            prob_otm = int(float((await cop_text.inner_text()).replace('%',''))) if cop_text else 0
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
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        # Login
        await page.goto("https://robinhood.com/login")
        print(f"Login page loaded: {page.url}")

        await page.fill("input[name='username']", RH_USERNAME)
        await page.fill("input[name='password']", RH_PASSWORD)
        await page.click("button[type='submit']")
        print("Login submitted, waiting for possible 2FA or redirect...")

        # Wait for redirect after login
        try:
            await page.wait_for_url("**/stocks/**", timeout=15000)
            print(f"Login successful, current page: {page.url}")
        except:
            print("Login may have failed or requires 2FA. Current page:", page.url)

        # Scrape options
        options = await scrape_option_data(page, TICKER)

        if not options:
            print("No options were scraped. Check selectors, login, or page loading.")
            await browser.close()
            return

        # Get current stock price
        await page.goto(f"https://robinhood.com/stocks/{TICKER}")
        try:
            await page.wait_for_selector("span[data-testid='StockPrice']", timeout=10000)
            current_price_el = await page.query_selector("span[data-testid='StockPrice']")
            current_price = float((await current_price_el.inner_text()).replace('$','').replace(',','')) if current_price_el else 0
            print(f"Current price: ${current_price}")
        except:
            print("Failed to get current price")
            current_price = 0

        # Filter puts below current price and get top 3
        puts_below = [opt for opt in options if opt["Strike"] < current_price]
        top3_puts = sorted(puts_below, key=lambda x: x["Strike"], reverse=True)[:3]

        print(f"Top 3 puts below current price for {TICKER}:")
        if top3_puts:
            for opt in top3_puts:
                print(f"{opt['Expiration']} | Strike: {opt['Strike']} | Ask: {opt['Ask']} | Delta: {opt['Delta']} | Prob OTM: {opt['Prob OTM']}%")
        else:
            print("No puts below current price found.")

        await browser.close()

# ------------------ RUN ------------------
if __name__ == "__main__":
    asyncio.run(main())
