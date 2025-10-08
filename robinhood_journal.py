# ------------------- robinhood_journal.py -------------------

import robin_stocks.robinhood as r
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os

# ------------------- CONFIG -------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
SHEET_NAME = os.environ.get("SHEET_NAME", "Robinhood Option Log")
SHEET_TAB_NAME = os.environ.get("SHEET_TAB_NAME", "Sheet1")

# ------------------- LOGIN -------------------
r.login(USERNAME, PASSWORD)

# ------------------- FETCH OPEN POSITIONS -------------------
positions = r.options.get_open_option_positions()
if not positions:
    print("No open positions found.")
    exit()

log_rows = []
for pos in positions:
    try:
        qty = int(float(pos.get('quantity', 0)))
        if qty == 0:
            continue

        instrument_url = pos.get('option')
        instrument = r.helper.request_get(instrument_url)
        chain_symbol = instrument['chain_symbol']
        option_type = instrument['type'].upper()
        strike_price = float(instrument['strike_price'])
        expiration_date = instrument['expiration_date']

        avg_price = float(pos.get('average_price', 0.0))
        mark_price = float(pos.get('mark_price', 0.0))
        profit_now = (avg_price - mark_price) * 100 * qty
        orig_profit = avg_price * 100 * qty
        current_price = float(r.stocks.get_latest_price(chain_symbol)[0])

        md = r.options.get_option_market_data_by_id(pos.get('option'))[0]
        delta = float(md.get('delta') or 0.0)
        theta = float(md.get('theta') or 0.0)
        vega = float(md.get('vega') or 0.0)

        log_rows.append([
            chain_symbol, option_type, strike_price, expiration_date, qty,
            current_price, avg_price, mark_price, orig_profit, profit_now,
            delta, theta, vega, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
    except Exception as e:
        print("Error parsing position:", e)

# ------------------- CONNECT TO GOOGLE SHEETS -------------------
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open(SHEET_NAME).worksheet(SHEET_TAB_NAME)

# ------------------- APPEND ROWS -------------------
for row in log_rows:
    sheet.append_row(row)

print(f"Logged {len(log_rows)} positions to Google Sheets.")
