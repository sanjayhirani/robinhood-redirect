import subprocess
import sys

# ---------------- AUTO-INSTALL DEPENDENCIES ----------------
required_packages = ["robin_stocks", "gspread", "google-auth", "pandas", "gspread-formatting"]
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
from gspread_formatting import *

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

# Ensure sheets exist
if "Options Positions" not in [ws.title for ws in sh.worksheets()]:
    options_ws = sh.add_worksheet(title="Options Positions", rows="100", cols="20")
else:
    options_ws = sh.worksheet("Options Positions")

if "Dashboard" not in [ws.title for ws in sh.worksheets()]:
    dashboard_ws = sh.add_worksheet(title="Dashboard", rows="50", cols="20")
else:
    dashboard_ws = sh.worksheet("Dashboard")

print(f"✅ Sheets ready: {options_ws.title}, {dashboard_ws.title}")

# ---------------- ROBINHOOD LOGIN ----------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
r.login(USERNAME, PASSWORD)
account_profile = r.profiles.load_account_profile()
rh_balance = float(account_profile.get("cash_available") or 0)
print(f"✅ Logged in. Robinhood cash available: ${rh_balance:,.2f}")

# ---------------- FETCH OPTIONS ----------------
open_positions = r.options.get_open_option_positions()
closed_positions = r.options.get_all_option_positions()

def parse_positions(positions, status):
    records = []
    for pos in positions:
        qty = float(pos.get("quantity") or 0)
        if qty == 0:
            continue

        instrument = r.helper.request_get(pos.get("option"))
        market_data_list = r.options.get_option_market_data_by_id(instrument.get("id"))

        if not market_data_list:
            print(f"⚠️ Market data missing for {instrument.get('chain_symbol')}")
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

        # Determine Buy/Sell for Call/Put
        if opt_type == "Call":
            side = "Buy Call" if qty > 0 else "Sell Call"
        else:
            side = "Buy Put" if qty > 0 else "Sell Put"

        total_premium = avg_price * 100 * abs(qty)
        current_value = mark * 100 * abs(qty)
        pnl = total_premium - current_value if status == "Open" else total_premium
        pnl_pct = (pnl / total_premium * 100) if total_premium else 0

        # Correct Delta for display: puts as positive
        if opt_type == "Put":
            delta = abs(delta)

        records.append({
            "Ticker": ticker,
            "Option Type": opt_type,
            "Buy/Sell": side,
            "Expiration": exp,
            "Strike": strike,
            "Quantity": abs(int(qty)),
            "Avg Price": abs(avg_price),
            "Mark Price": mark,
            "Total Premium": total_premium / 100,
            "Current Value": current_value / 100,
            "PnL ($)": pnl / 100,
            "PnL (%)": round(pnl_pct, 2),
            "Chance of Profit": round(cop * 100, 1),
            "Delta": round(delta, 3),
            "Open Date": open_date,
            "Close Date": close_date if status == "Closed" else "",
            "Status": status,
            "Instrument ID": instrument.get("id"),
            "Last Updated": datetime.now().strftime("%d/%m/%y %H:%M:%S"),
            "RH Balance ($)": rh_balance
        })
    return records

# ---------------- PARSE DATA ----------------
open_data = parse_positions(open_positions, "Open")
closed_data = parse_positions(closed_positions, "Closed")
all_data = open_data + closed_data
df = pd.DataFrame(all_data)

# ---------------- WRITE TO OPTIONS POSITIONS SHEET ----------------
if not df.empty:
    # Format dates to UK style
    for col in ["Expiration", "Open Date", "Close Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%d/%m/%y')

    options_ws.clear()
    options_ws.update([df.columns.values.tolist()] + df.values.tolist())
    worksheet.freeze(rows=1)
    set_frozen(options_ws, rows=1)
    set_basic_filter(options_ws)

    # Conditional formatting for PnL ($)
    pnl_col_index = df.columns.get_loc("PnL ($)") + 1
    green_rule = CellFormat(backgroundColor=color(0.8,1,0.8))
    red_rule = CellFormat(backgroundColor=color(1,0.8,0.8))
    for i, pnl in enumerate(df["PnL ($)"], start=2):
        cell_range = f"{gspread.utils.rowcol_to_a1(i, pnl_col_index)}"
        if pnl > 0:
            format_cell_range(options_ws, cell_range, green_rule)
        elif pnl < 0:
            format_cell_range(options_ws, cell_range, red_rule)

    # Hide Instrument ID column
    col_index = df.columns.get_loc("Instrument ID") + 1
    options_ws.hide_columns(col_index)

    # Auto resize
    options_ws.resize(cols=len(df.columns))
    print(f"✅ Options Positions sheet updated with {len(df)} positions.")
else:
    options_ws.clear()
    options_ws.update([[f"No option positions found as of {datetime.now().strftime('%d/%m/%y %H:%M:%S')}"]])
    print("⚠️ No option positions found.")

# ---------------- DASHBOARD ----------------
dashboard_ws.clear()
summary_values = [
    ["Robinhood Total Balance ($)", rh_balance],
    ["Total Positions", len(df)],
    ["Total PnL ($)", df["PnL ($)"].sum() if not df.empty else 0],
    ["Total PnL (%)", df["PnL (%)"].mean() if not df.empty else 0]
]
dashboard_ws.update(summary_values)

# Top 5 Gainers
if not df.empty:
    top5 = df.sort_values("PnL ($)", ascending=False).head(5)
    dashboard_ws.update([["Top 5 Gainers"]])
    dashboard_ws.update([top5.columns.values.tolist()] + top5.values.tolist(), value_input_option="RAW")

# Top 5 Losers
if not df.empty:
    bottom5 = df.sort_values("PnL ($)", ascending=True).head(5)
    dashboard_ws.update([["Top 5 Losers"]], value_input_option="RAW")
    dashboard_ws.update([bottom5.columns.values.tolist()] + bottom5.values.tolist(), value_input_option="RAW")

print("✅ Dashboard sheet updated.")
