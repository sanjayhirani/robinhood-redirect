import os
import json
import gspread
import robin_stocks.robinhood as r
import pandas as pd
from google.oauth2.service_account import Credentials
from datetime import datetime

# ---------------- GOOGLE AUTH ----------------
google_creds_json = os.environ["GOOGLE_CREDENTIALS_JSON"]
creds_dict = json.loads(google_creds_json)
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"]
)
gc = gspread.authorize(creds)

# ---------------- SHEET SETUP ----------------
SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Robinhood Options Tracker")
sh = None
try:
    sh = gc.open(SHEET_NAME)
except gspread.SpreadsheetNotFound:
    sh = gc.create(SHEET_NAME)
    sh.share(creds_dict["client_email"], perm_type='user', role='writer')

worksheet = sh.sheet1
worksheet.update_title("Options Positions")

# ---------------- ROBINHOOD LOGIN ----------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]

r.login(USERNAME, PASSWORD)

# ---------------- FETCH OPTIONS ----------------
open_positions = r.options.get_open_option_positions()
closed_positions = r.options.get_all_option_positions()  # includes closed ones if available

def parse_positions(positions, status):
    records = []
    for pos in positions:
        qty = float(pos.get("quantity") or 0)
        if qty == 0 and status == "Open":
            continue
        instrument = r.helper.request_get(pos.get("option"))
        market_data = r.options.get_option_market_data_by_id(instrument.get("id"))[0]

        ticker = instrument.get("chain_symbol")
        opt_type = instrument.get("type", "put").capitalize()
        exp = instrument.get("expiration_date")
        strike = float(instrument.get("strike_price"))
        avg_price = float(pos.get("average_price") or 0)
        mark = float(market_data.get("mark_price") or 0)
        delta = float(market_data.get("delta") or 0)
        cop = float(market_data.get("chance_of_profit_short") or 0)
        open_date = pos.get("created_at", "")[:10]
        close_date = pos.get("updated_at", "")[:10]

        total_premium = avg_price * 100 * abs(qty)
        current_value = mark * 100 * abs(qty)
        pnl = total_premium - current_value if status == "Open" else total_premium
        pnl_pct = (pnl / total_premium * 100) if total_premium else 0

        records.append({
            "Ticker": ticker,
            "Option Type": opt_type,
            "Side": "Short" if qty < 0 else "Long",
            "Expiration": exp,
            "Strike": strike,
            "Quantity": abs(int(qty)),
            "Avg Price": avg_price,
            "Mark Price": mark,
            "Total Premium": total_premium,
            "Current Value": current_value,
            "PnL ($)": round(pnl, 2),
            "PnL (%)": round(pnl_pct, 2),
            "Chance of Profit": round(cop * 100, 1),
            "Delta": round(delta, 3),
            "Open Date": open_date,
            "Close Date": close_date if status == "Closed" else "",
            "Status": status,
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    return records

open_data = parse_positions(open_positions, "Open")
closed_data = parse_positions(closed_positions, "Closed")

all_data = open_data + closed_data
df = pd.DataFrame(all_data)

# ---------------- WRITE TO GOOGLE SHEET ----------------
if not df.empty:
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())
else:
    worksheet.clear()
    worksheet.update([[f"No option positions found as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]])
