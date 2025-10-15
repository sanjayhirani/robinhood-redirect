import subprocess
import sys

# ---------------- AUTO-INSTALL DEPENDENCIES ----------------
required_packages = ["robin_stocks", "gspread", "google-auth", "pandas"]
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"⚙️ Installing missing dependency: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ---------------- IMPORTS ----------------
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
SHEET_ID = "1ANhHY6M_BT0SeKjRXtBs9Iw6mjtJHlcHaDNtM58BvQU"

try:
    sh = gc.open_by_key(SHEET_ID)
except gspread.SpreadsheetNotFound:
    raise Exception(
        f"Spreadsheet with ID '{SHEET_ID}' not found. "
        "Ensure the service account has Editor access."
    )

# Use the first worksheet
worksheet = sh.sheet1
worksheet.update_title("Options Positions")
print(f"Successfully opened sheet: {sh.title}")

# ---------------- ROBINHOOD LOGIN ----------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]

r.login(USERNAME, PASSWORD)

# ---------------- FETCH OPTIONS (with error handling) ----------------
open_positions = r.options.get_open_option_positions()
closed_positions = r.options.get_all_option_positions()  # includes closed ones if available

def parse_positions(positions, status):
    records = []
    for pos in positions:
        qty = float(pos.get("quantity") or 0)
        if qty == 0:  # Skip positions with zero quantity
            continue

        instrument = r.helper.request_get(pos.get("option"))

        market_data_list = r.options.get_option_market_data_by_id(instrument.get("id"))
        if not market_data_list:
            print(f"⚠️ Market data list is empty for {instrument.get('chain_symbol')}")
            mark = delta = cop = 0
        else:
            market_data = market_data_list[0]
            mark = float(market_data.get("mark_price") or 0)
            delta = float(market_data.get("delta") or 0)
            cop = float(market_data.get("chance_of_profit_short") or 0)

        ticker = instrument.get("chain_symbol")
        opt_type = instrument.get("type", "put").capitalize()
        exp = instrument.get("expiration_date")
        strike = float(instrument.get("strike_price"))
        avg_price = float(pos.get("average_price") or 0)
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

# ---------------- WRITE TO GOOGLE SHEET ----------------
if not df.empty:
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())
    print(f"✅ Sheet updated successfully with {len(df)} positions.")
else:
    worksheet.clear()
    worksheet.update([[f"No option positions found as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]])
    print("⚠️ No option positions found.")
