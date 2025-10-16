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

# ---------------- WORKSHEET CREATION ----------------
try:
    dashboard_ws = sh.worksheet("Dashboard")
except gspread.WorksheetNotFound:
    dashboard_ws = sh.add_worksheet(title="Dashboard", rows=20, cols=10)

try:
    options_ws = sh.worksheet("Options Positions")
except gspread.WorksheetNotFound:
    options_ws = sh.add_worksheet(title="Options Positions", rows=100, cols=20)

print("✅ Sheets ready: Options Positions, Dashboard")

# ---------------- ROBINHOOD LOGIN ----------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
r.login(USERNAME, PASSWORD)

# Get cash balance
try:
    rh_account = r.profiles.load_account_profile()
    cash_balance = float(rh_account.get("cash", 0))
    print(f"✅ Logged in. Robinhood cash available: ${cash_balance:.2f}")
except Exception as e:
    cash_balance = 0
    print("⚠️ Could not fetch Robinhood cash balance:", e)

# ---------------- FETCH OPTIONS ----------------
open_positions = r.options.get_open_option_positions()
closed_positions = r.options.get_all_option_positions()  # includes closed

def parse_positions(positions, status):
    records = []
    for pos in positions:
        qty = float(pos.get("quantity") or 0)
        if qty == 0:
            continue

        instrument = r.helper.request_get(pos.get("option")) or {}
        ticker = instrument.get("chain_symbol", "")
        opt_type = instrument.get("type", "put").capitalize()
        exp = instrument.get("expiration_date")
        strike = float(instrument.get("strike_price") or 0)
        avg_price = float(pos.get("average_price") or 0)
        avg_price = abs(avg_price)
        open_date = pos.get("created_at", "")[:10]
        close_date = pos.get("updated_at", "")[:10]

        market_data_list = r.options.get_option_market_data_by_id(instrument.get("id") or "")
        if market_data_list:
            market_data = market_data_list[0]
            mark = float(market_data.get("mark_price") or 0)
            delta = float(market_data.get("delta") or 0)
            cop = float(market_data.get("chance_of_profit_short") or 0)
        else:
            mark = delta = cop = 0

        # Correct delta signs
        if opt_type.lower() == "put":
            delta = -abs(delta)  # puts are always negative
        else:
            delta = -abs(delta)   # calls always positive

        # ---------------- CALCULATIONS ----------------
        total_premium = abs(avg_price * qty)
        total_premium_display = total_premium / 100      # divide by 100, positive

        current_value = mark * qty
        pnl = current_value - avg_price * qty
        pnl_display = pnl / 100                           # divide by 100

        pnl_pct = round((pnl / total_premium * 100) if total_premium else 0, 2)

        action = f"{'Buy' if qty > 0 else 'Sell'} {opt_type}"

        records.append({
            "Ticker": ticker,
            "Option Type": opt_type,
            "Action": action,
            "Expiration": exp,
            "Strike": strike,
            "Quantity": int(abs(qty)),
            "Avg Price": avg_price,
            "Mark Price": mark,
            "Total Premium": total_premium_display,
            "Current Value": current_value,
            "PnL ($)": pnl_display,
            "PnL (%)": pnl_pct,
            "Chance of Profit": round(cop * 100, 1),
            "Delta": round(delta, 3),
            "Open Date": open_date,
            "Close Date": close_date if status == "Closed" else "",
            "Status": status,
            "Instrument ID": instrument.get("id"),
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Robinhood Cash": cash_balance
        })
    return records

open_data = parse_positions(open_positions, "Open")
closed_data = parse_positions(closed_positions, "Closed")
all_data = open_data + closed_data
df = pd.DataFrame(all_data)

# ---------------- CLEAN DATA (JSON-safe + UK dates) ----------------
df.replace([pd.NA, None, float('inf'), float('-inf')], '', inplace=True)

for col in ["Expiration", "Open Date", "Close Date", "Last Updated"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%d/%m/%y')
        df[col] = df[col].fillna('')

# ---------------- MOVE "Open Date" TO FIRST COLUMN AND RENAME ----------------
if "Open Date" in df.columns:
    df.rename(columns={"Open Date": "Date Opened"}, inplace=True)
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("Date Opened")))
    df = df[cols]

# ---------------- WRITE OPTIONS POSITIONS (JSON-safe) ----------------
options_ws.clear()

# Prepare data for sheet
sheet_values = [df.columns.tolist()]
for row in df.values.tolist():
    sheet_values.append([str(cell) if cell is not None else '' for cell in row])

# Update sheet
options_ws.update(sheet_values)

# Hide Instrument ID column
if "Instrument ID" in df.columns:
    hid_col = df.columns.get_loc("Instrument ID") + 1
    options_ws.hide_columns(hid_col, hid_col)

# Freeze header row
options_ws.freeze(1)
set_frozen(options_ws, rows=1)

# Conditional formatting for PnL ($)
if "PnL ($)" in df.columns:
    pnl_col = df.columns.get_loc("PnL ($)") + 1
    for i, pnl in enumerate(df["PnL ($)"], start=2):  # start=2 to skip header
        cell = f"{gspread.utils.rowcol_to_a1(i, pnl_col)}"
        if pnl == '' or pnl is None:
            continue
        if float(pnl) > 0:
            format_cell_range(options_ws, cell, CellFormat(backgroundColor=color(0.8,1,0.8)))
        elif float(pnl) < 0:
            format_cell_range(options_ws, cell, CellFormat(backgroundColor=color(1,0.8,0.8)))

print(f"✅ Options Positions sheet updated with {len(df)} positions.")

# ---------------- DASHBOARD (JSON-safe) ----------------
dashboard_ws.clear()
summary = {
    "Total Open Positions": len(open_data),
    "Total Closed Positions": len(closed_data),
    "Total Premiums": df["Total Premium"].sum(),
    "Total PnL": df["PnL ($)"].sum(),
    "Robinhood Cash": cash_balance
}

dashboard_values = [["Metric", "Value"]]
for k, v in summary.items():
    dashboard_values.append([str(k), str(v)])

dashboard_ws.update(dashboard_values)
print("✅ Dashboard sheet updated.")
