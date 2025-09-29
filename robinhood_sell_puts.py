import random
import robin_stocks.robinhood as r

# ------------------ LOGIN ------------------
r.login(USERNAME, PASSWORD)

# ------------------ FETCH TOP 100 ------------------
top_100 = r.stocks.get_top_100()  # Single API call
print(f"Fetched {len(top_100)} top tickers from Robinhood")

# ------------------ FILTER BY PRICE ($1-$10) ------------------
price_filtered = []
for ticker_info in top_100:
    try:
        price = float(ticker_info.get('last_trade_price') or 0.0)
        if 1 <= price <= 10:
            price_filtered.append(ticker_info['symbol'])
    except:
        continue

print(f"{len(price_filtered)} tickers in $1-$10 range")

# ------------------ RANDOMLY SELECT UP TO 20 ------------------
random.shuffle(price_filtered)
TICKERS = price_filtered[:20]  # fewer if less than 20
print("Selected tickers:", TICKERS)

# ------------------ OPTIONS DATA FETCH ------------------
all_options = []
for ticker in TICKERS:
    try:
        options = r.options.find_tradable_options(ticker, optionType="put")
        # Filter options or process as needed
        all_options.append({ "ticker": ticker, "options": options })
    except Exception as e:
        print(f"Error fetching options for {ticker}: {e}")
