import sys
import subprocess

# ------------------ CONDITIONAL DEPENDENCY INSTALL ------------------
def install_if_missing(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_if_missing("yfinance")
install_if_missing("lxml")

# ------------------ OTHER IMPORTS ------------------
import os
import requests
import robin_stocks.robinhood as r
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import numpy as np
import pandas as pd

# ------------------ CONFIG ------------------
TICKERS = [
    "SNAP", "ACHR", "OPEN", "BBAI", "PTON", "ONDS", "GRAB", "LAC", "HTZ", "RZLV", "NVTS",
    "SOFI", "CHPT", "FUBO", "RIOT", "MARA", "DKNG", "PLTR", "DNA", "NOK", "IAG"
]
NUM_EXPIRATIONS = 3
NUM_PUTS = 2
MIN_PRICE = 0.10
HV_PERIOD = 21
LOW_DAYS = 14
EXPIRY_LIMIT_DAYS = 21

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ TELEGRAM ------------------
def send_telegram_message(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )

def send_telegram_photo(buf, caption):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo': buf},
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    )

# ------------------ UTILS ------------------
def prepare_historicals(df):
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(all_days)
    df.index = df.index.tz_localize(None)
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df[['open','close']].max(axis=1))
    df['low'] = df['low'].fillna(df[['open','close']].min(axis=1))
    df['volume'] = df['volume'].fillna(0)
    return df

def plot_candlestick(df, current_price, last_14_low, strikes, exp_date):
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    for i in range(len(df)):
        color = 'lime' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(df.index[i]) - 0.6/2, min(df['open'].iloc[i], df['close'].iloc[i])),
            0.6,
            abs(df['close'].iloc[i] - df['open'].iloc[i]),
            color=color
        ))
        ax.plot([mdates.date2num(df.index[i]), mdates.date2num(df.index[i])],
                 [df['low'].iloc[i], df['high'].iloc[i]], color=color, linewidth=1)
    ax.axhline(current_price, color='magenta', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14_low, color='yellow', linestyle='--', linewidth=2, label=f'14d Low: ${last_14_low:.2f}')
    for strike in strikes:
        ax.axhline(strike, color='cyan', linestyle='--', linewidth=1.5, label=f'Strike: ${strike:.2f}')
    exp_date_obj = pd.to_datetime(exp_date).tz_localize(None)
    ax.axvline(mdates.date2num(exp_date_obj), color='orange', linestyle='--', linewidth=2, label=f'Exp: {exp_date_obj.strftime("%d-%m")}')
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
    buf.seek(0)
    plt.close()
    return buf

def adjusted_score(opt, buying_power, today):
    days_to_exp = (pd.to_datetime(opt['Expiration Date']).date() - today).days
    if days_to_exp <= 0:
        return 0
    iv_hv_ratio = opt['IV']/opt['HV'] if opt['HV'] > 0 else 1.0
    liquidity_weight = 1 + 0.5*(opt['Volume']+opt['Open Interest'])/1000
    max_contracts = max(1, int(buying_power // (opt['Strike Price'] * 100)))
    total_premium = opt['Bid Price'] * 100 * max_contracts * opt['COP Short']
    return total_premium * iv_hv_ratio * liquidity_weight / days_to_exp

# ------------------ LOGIN ------------------
r.login(USERNAME, PASSWORD)
today = datetime.now().date()
cutoff = today + timedelta(days=EXPIRY_LIMIT_DAYS)

# ------------------ GET BUYING POWER ------------------
account_data = r.profiles.load_account_profile()
buying_power = float(account_data['cash_available_for_withdrawal'])

# ------------------ PROCESS TICKERS ------------------
all_contracts = []
group_msgs = []

for i, ticker in enumerate(TICKERS):
    try:
        current_price = float(r.stocks.get_latest_price(ticker)[0])
        rh_url = f"https://robinhood.com/stocks/{ticker}"
        historicals = r.stocks.get_stock_historicals(ticker, interval='day', span='month', bounds='regular')
        df = pd.DataFrame(historicals)
        df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
        df.set_index('begins_at', inplace=True)
        df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
        df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
        df = prepare_historicals(df)
        last_14_low = df['low'][-LOW_DAYS:].min()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        hv = df['returns'].rolling(HV_PERIOD).std().iloc[-1] * np.sqrt(252)

        all_puts = r.options.find_tradable_options(ticker, optionType="put")
        exp_dates = sorted(set([opt['expiration_date'] for opt in all_puts]))
        exp_dates = [d for d in exp_dates if today <= datetime.strptime(d, "%Y-%m-%d").date() <= cutoff][:NUM_EXPIRATIONS]

        candidate_puts = []
        for exp_date in exp_dates:
            puts_for_exp = [opt for opt in all_puts if opt['expiration_date'] == exp_date]
            strikes_below = sorted([float(opt['strike_price']) for opt in puts_for_exp if float(opt['strike_price']) < current_price], reverse=True)
            chosen_strikes = strikes_below[:NUM_PUTS]
            for opt in puts_for_exp:
                strike = float(opt['strike_price'])
                if strike not in chosen_strikes:
                    continue
                md = r.options.get_option_market_data_by_id(opt['id'])[0]
                bid_price = float(md.get('bid_price') or md.get('mark_price') or 0.0)
                if bid_price < MIN_PRICE:
                    continue
                contract = {
                    "Ticker": ticker,
                    "Current Price": current_price,
                    "Expiration Date": exp_date,
                    "Strike Price": strike,
                    "Bid Price": bid_price,
                    "Delta": float(md.get('delta') or 0.0),
                    "IV": float(md.get('implied_volatility') or 0.0),
                    "COP Short": float(md.get('chance_of_profit_short') or 0.0),
                    "Theta": float(md.get('theta') or 0.0),
                    "Open Interest": int(md.get('open_interest') or 0),
                    "Volume": int(md.get('volume') or 0),
                    "URL": rh_url,
                    "HV": hv
                }
                contract["Score"] = adjusted_score(contract, buying_power, today)
                all_contracts.append(contract)
                candidate_puts.append(contract)

        if candidate_puts:
            msg = [f"ðŸ“Š <a href='{rh_url}'>{ticker}</a> current: ${current_price:.2f}"]
            for c in candidate_puts:
                max_contracts = max(1, int(buying_power // (c['Strike Price'] * 100)))
                total_premium = c['Bid Price'] * 100 * max_contracts
                msg.extend([
                    "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€",
                    f"âœ… Expiration : {c['Expiration Date']}",
                    f"ðŸ’² Strike    : {c['Strike Price']}",
                    f"ðŸ’° Bid Price : ${c['Bid Price']:.2f}",
                    f"ðŸ”º Delta     : {c['Delta']:.3f}",
                    f"ðŸ“ˆ IV       : {c['IV']*100:.2f}%",
                    f"ðŸŽ¯ COP Short : {c['COP Short']*100:.1f}%",
                    f"ðŸ“ Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}",
                    f"ðŸ“ Score: {c['Score']:.2f}"
                ])
            group_msgs.append("\n".join(msg))

        if (i+1) % 3 == 0 or i == len(TICKERS)-1:
            if group_msgs:
                send_telegram_message("\n\n".join(group_msgs))
                group_msgs = []

    except Exception as e:
        send_telegram_message(f"âš ï¸ Error processing {ticker}: {e}")

# ------------------ CANDIDATE SCORE ALERT ------------------
if all_contracts:
    top10 = sorted(all_contracts, key=lambda x: x['Score'], reverse=True)[:10]
    msg = ["<b>ðŸ“Š Top 10 Candidate Scores</b>"]
    for c in top10:
        msg.append(f"{c['Ticker']} | Exp: {c['Expiration Date']} | Strike: {c['Strike Price']} | Score: {c['Score']:.2f}")
    send_telegram_message("\n".join(msg))

# ------------------ BEST ALERT ------------------
if all_contracts:
    best = max(all_contracts, key=lambda x: x['Score'])
    max_contracts = max(1, int(buying_power // (best['Strike Price'] * 100)))
    total_premium = best['Bid Price'] * 100 * max_contracts
    msg = [
        "ðŸ”¥ <b>Best Cash-Secured Put</b>",
        f"ðŸ“Š <a href='{best['URL']}'>{best['Ticker']}</a> current: ${best['Current Price']:.2f}",
        f"âœ… Expiration : {best['Expiration Date']}",
        f"ðŸ’² Strike    : {best['Strike Price']}",
        f"ðŸ’° Bid Price : ${best['Bid Price']:.2f}",
        f"ðŸ”º Delta     : {best['Delta']:.3f}",
        f"ðŸ“ˆ IV       : {best['IV']*100:.2f}%",
        f"ðŸŽ¯ COP Short : {best['COP Short']*100:.1f}%",
        f"ðŸ“ Max Contracts: {max_contracts} | Total Premium: ${total_premium:.2f}",
        f"ðŸ“ Score: {best['Score']:.2f}"
    ]
    historicals = r.stocks.get_stock_historicals(best['Ticker'], interval='day', span='month', bounds='regular')
    df = pd.DataFrame(historicals)
    df['begins_at'] = pd.to_datetime(df['begins_at']).dt.tz_localize(None)
    df.set_index('begins_at', inplace=True)
    df = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'}, inplace=True)
    df = prepare_historicals(df)
    last_14_low = df['low'][-LOW_DAYS:].min()
    buf = plot_candlestick(df, best['Current Price'], last_14_low, [best['Strike Price']], best['Expiration Date'])
    send_telegram_photo(buf, "\n".join(msg))
