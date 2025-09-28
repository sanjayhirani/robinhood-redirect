# robinhood_options_manager.py

# ------------------ AUTO-INSTALL DEPENDENCIES ------------------
import sys
import subprocess
for pkg in ['yfinance','lxml','scipy','numpy','pandas','matplotlib','requests','robin_stocks']:
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ------------------ IMPORTS ------------------
import os, io, requests
import robin_stocks.robinhood as r
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import log, sqrt
from scipy.stats import norm
import numpy as np
import pandas as pd
import yfinance

# ------------------ CONFIG ------------------
TICKERS = ["SNAP","ACHR","OPEN","BBAI","PTON","ONDS","GRAB","LAC","HTZ","RZLV","NVTS"]
NUM_EXPIRATIONS = 3
NUM_OPTIONS = 2
PRICE_ADJUST = 0.01
RISK_FREE_RATE = 0.05
MIN_PRICE = 0.10
HV_PERIOD = 21
CANDLE_WIDTH = 0.6
LOW_DAYS = 14

# ------------------ SECRETS ------------------
USERNAME = os.environ["RH_USERNAME"]
PASSWORD = os.environ["RH_PASSWORD"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ------------------ UTILITIES ------------------
def black_scholes_put_delta(S,K,T,r,sigma):
    if sigma<=0 or T<=0: return -1.0
    d1=(log(S/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    return norm.cdf(d1)-1

def black_scholes_call_delta(S,K,T,r,sigma):
    if sigma<=0 or T<=0: return 1.0
    d1=(log(S/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    return norm.cdf(d1)

def risk_emoji(prob_otm):
    return "✅" if prob_otm>=0.8 else "🟡" if prob_otm>=0.6 else "⚠️"

def historical_volatility(prices, period=21):
    prices=np.array(prices)
    lr=np.diff(np.log(prices))
    if len(lr)<period: return 0.3
    return np.std(lr[-period:])*np.sqrt(252)

def send_telegram_photo(buf,caption):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        files={'photo':buf}, data={"chat_id":TELEGRAM_CHAT_ID,"caption":caption,"parse_mode":"HTML"})

def send_telegram_message(msg):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"HTML"})

def plot_candlestick(df,current_price,last_14,strikes=None,exp_date=None):
    fig,ax=plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    for i in range(len(df)):
        color='lime' if df['close'].iloc[i]>=df['open'].iloc[i] else 'red'
        ax.add_patch(plt.Rectangle((mdates.date2num(df.index[i])-CANDLE_WIDTH/2,min(df['open'].iloc[i],df['close'].iloc[i])),
                                   CANDLE_WIDTH,abs(df['close'].iloc[i]-df['open'].iloc[i]),color=color))
        ax.plot([mdates.date2num(df.index[i]),mdates.date2num(df.index[i])],[df['low'].iloc[i],df['high'].iloc[i]],color=color,linewidth=1)
    ax.axhline(current_price,color='magenta',linestyle='--',linewidth=1.5,label=f'Current: ${current_price:.2f}')
    ax.axhline(last_14,color='yellow',linestyle='--',linewidth=2,label=f'14-day Low/High: ${last_14:.2f}')
    if strikes:
        for s in strikes: ax.axhline(s,color='cyan',linestyle='--',linewidth=1.5,label=f'Strike: ${s:.2f}')
    if exp_date:
        ed=pd.to_datetime(exp_date).tz_localize(None)
        if df.index.min()<=ed<=df.index.max(): ax.axvline(mdates.date2num(ed),color='orange',linestyle='--',linewidth=2,label=f'Exp: {ed.strftime("%d-%m-%y")}')
    ax.set_ylabel('Price ($)',color='white')
    ax.tick_params(colors='white')
    ax.grid(True,color='gray',linestyle='--',alpha=0.3)
    ax.legend(facecolor='black',edgecolor='white',labelcolor='white')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    buf=io.BytesIO()
    plt.savefig(buf,format='png',bbox_inches='tight',facecolor='black')
    buf.seek(0)
    plt.close()
    return buf

def generate_chart(best,is_put=True):
    hist=r.stocks.get_stock_historicals(best['Ticker'],interval='day',span='month',bounds='regular')
    df=pd.DataFrame(hist)
    df['begins_at']=pd.to_datetime(df['begins_at']).dt.tz_localize(None)
    df.set_index('begins_at',inplace=True)
    df=df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'},inplace=True)
    all_days=pd.date_range(df.index.min(),df.index.max(),freq='B')
    df=df.reindex(all_days)
    df.index=df.index.tz_localize(None)
    df['close']=df['close'].ffill()
    df['open']=df['open'].fillna(df['close'])
    df['high']=df['high'].fillna(df[['open','close']].max(axis=1))
    df['low']=df['low'].fillna(df[['open','close']].min(axis=1))
    df['volume']=df['volume'].fillna(0)
    last_14=df['low'][-LOW_DAYS:].min() if is_put else df['high'][-LOW_DAYS:].max()
    return plot_candlestick(df,best['Current Price'],last_14,[best['Strike Price']],best['Expiration Date'])

# ------------------ LOGIN ------------------
r.login(USERNAME,PASSWORD)
today=datetime.now().date()
cutoff=today+timedelta(days=30)

# ------------------ EARNINGS/DIVIDENDS ------------------
safe_tickers=[]
risky_msgs=[]
safe_count=0
risky_count=0
for ticker in TICKERS:
    try:
        stock=yfinance.Ticker(ticker)
        msg_parts=[f"📊 <b>{ticker}</b>"]
        has_event=False
        try:
            if not stock.dividends.empty:
                div_date=stock.dividends.index[-1].date()
                if today<=div_date<=cutoff:
                    msg_parts.append(f"⚠️ 💰 Dividend on {div_date.strftime('%d-%m-%y')}")
                    has_event=True
        except: pass
        try:
            ed=stock.get_earnings_dates(limit=2)
            if not ed.empty:
                edate=ed.index.min().date()
                if today<=edate<=cutoff:
                    msg_parts.append(f"⚠️ 📢 Earnings on {edate.strftime('%d-%m-%y')}")
                    has_event=True
        except: pass
        if has_event: risky_msgs.append(" | ".join(msg_parts)); risky_count+=1
        else: safe_tickers.append(ticker); safe_count+=1
    except: risky_msgs.append(f"⚠️ {ticker} error"); risky_count+=1

summary=[]
if risky_msgs: summary.append("⚠️ <b>Risky Tickers</b>\n"+ "\n".join(risky_msgs)+"\n")
else: summary.append("⚠️ <b>No risky tickers 🎉</b>\n")
safe_rows=[", ".join([f"<b>{t}</b>" for t in safe_tickers[i:i+4]]) for i in range(0,len(safe_tickers),4)]
if safe_rows: summary.append("✅ <b>Safe Tickers</b>\n"+ "\n".join(safe_rows))
summary.append(f"\n📊 Summary: ✅ Safe: {safe_count} | ⚠️ Risky: {risky_count}")
send_telegram_message("\n".join(summary))

# ------------------ OWNERSHIP ------------------
positions=r.account.build_holdings()
owned=[t for t,d in positions.items() if float(d.get('quantity',0))>=100]
safe_no_shares=[t for t in safe_tickers if t not in owned]

# ------------------ PUTS ------------------
def process_puts(tickers):
    all_options=[]
    for TICKER in tickers:
        try:
            cp=float(r.stocks.get_latest_price(TICKER)[0])
            rh_url=f"https://robinhood.com/stocks/{TICKER}"
            hist=r.stocks.get_stock_historicals(TICKER,interval='day',span='month',bounds='regular')
            df=pd.DataFrame(hist)
            df['begins_at']=pd.to_datetime(df['begins_at']).dt.tz_localize(None)
            df.set_index('begins_at',inplace=True)
            df=df[['open_price','close_price','high_price','low_price','volume']].astype(float)
            df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'},inplace=True)
            all_days=pd.date_range(df.index.min(),df.index.max(),freq='B')
            df=df.reindex(all_days); df.index=df.index.tz_localize(None)
            df['close']=df['close'].ffill(); df['open']=df['open'].fillna(df['close'])
            df['high']=df['high'].fillna(df[['open','close']].max(axis=1))
            df['low']=df['low'].fillna(df[['open','close']].min(axis=1))
            df['volume']=df['volume'].fillna(0)
            last_14=df['low'][-LOW_DAYS:].min()
            all_puts=r.options.find_tradable_options(TICKER,optionType="put")
            exp_dates=sorted(set([o['expiration_date'] for o in all_puts]))[:NUM_EXPIRATIONS]
            sigma=historical_volatility(df['close'].values,HV_PERIOD)
            candidates=[]
            for ed in exp_dates:
                T=max((datetime.strptime(ed,"%Y-%m-%d").date()-today).days/365,1/365)
                puts=[o for o in all_puts if o['expiration_date']==ed]
                strikes=sorted([float(o['strike_price']) for o in puts if float(o['strike_price'])<cp],reverse=True)[:3]
                for o in puts:
                    s=float(o['strike_price']); 
                    if s not in strikes: continue
                    md=r.options.get_option_market_data_by_id(o['id'])
                    price=0.0; delta=-1.0
                    if md:
                        try: price=float(md[0].get('adjusted_mark_price') or md[0].get('mark_price') or 0.0)
                        except: price=0.0
                        try: delta=float(md[0].get('delta')) if md[0].get('delta') else None
                        except: delta=None
                        if delta is None or delta==0.0: delta=black_scholes_put_delta(cp,s,T,RISK_FREE_RATE,sigma)
                    price=max(price-PRICE_ADJUST,0.0)
                    if price>=MIN_PRICE:
                        prob_OTM=1-abs(delta); risk=max(cp-s,0.01); pr=price/risk
                        candidates.append({"Ticker":TICKER,"Current Price":cp,"Expiration Date":ed,"Strike Price":s,
                                           "Option Price":price,"Delta":delta,"Prob OTM":prob_OTM,"Profit/Risk":pr,"URL":rh_url})
            selected=sorted(candidates,key=lambda x:x['Profit/Risk'],reverse=True)[:NUM_OPTIONS]
            all_options.extend(selected)
            if selected:
                msg=[f"📊 <a href='{rh_url}'>{TICKER}</a> Current: ${cp:.2f}"]
                for o in selected: msg.append(f"{risk_emoji(o['Prob OTM'])} Exp: {o['Expiration Date']} Strike: {o['Strike Price']} Price: ${o['Option Price']:.2f} Profit/Risk: {o['Profit/Risk']:.2f}")
                buf=generate_chart(selected[0],is_put=True)
                send_telegram_photo(buf,"\n".join(msg))
        except Exception as e: send_telegram_message(f"⚠️ {TICKER} put error: {e}")
    if all_options:
        best=max(all_options,key=lambda x:x['Profit/Risk'])
        buf=generate_chart(best,is_put=True)
        msg=[f"🔥 <b>Best Put to Sell</b>","📊 <a href='{best['URL']}'>{best['Ticker']}</a> Current: ${best['Current Price']:.2f}",
             f"✅ Exp: {best['Expiration Date']}","💲 Strike: {best['Strike Price']}",
             f"💰 Price: ${best['Option Price']:.2f}","🎯 Prob OTM: {best['Prob OTM']*100:.1f}%",
             f"💎 Profit/Risk: {best['Profit/Risk']:.2f}"]
        send_telegram_photo(buf,"\n".join(msg))

# ------------------ CALLS ------------------
def process_calls(tickers):
    all_options=[]
    for TICKER in tickers:
        try:
            cp=float(r.stocks.get_latest_price(TICKER)[0])
            rh_url=f"https://robinhood.com/stocks/{TICKER}"
            hist=r.stocks.get_stock_historicals(TICKER,interval='day',span='month',bounds='regular')
            df=pd.DataFrame(hist)
            df['begins_at']=pd.to_datetime(df['begins_at']).dt.tz_localize(None)
            df.set_index('begins_at',inplace=True)
            df=df[['open_price','close_price','high_price','low_price','volume']].astype(float)
            df.rename(columns={'open_price':'open','close_price':'close','high_price':'high','low_price':'low'},inplace=True)
            all_days=pd.date_range(df.index.min(),df.index.max(),freq='B')
            df=df.reindex(all_days); df.index=df.index.tz_localize(None)
            df['close']=df['close'].ffill(); df['open']=df['open'].fillna(df['close'])
            df['high']=df['high'].fillna(df[['open','close']].max(axis=1))
            df['low']=df['low'].fillna(df[['open','close']].min(axis=1))
            df['volume']=df['volume'].fillna(0)
            last_14=df['high'][-LOW_DAYS:].max()
            all_calls=r.options.find_tradable_options(TICKER,optionType="call")
            exp_dates=sorted(set([o['expiration_date'] for o in all_calls]))[:NUM_EXPIRATIONS]
            sigma=historical_volatility(df['close'].values,HV_PERIOD)
            candidates=[]
            for ed in exp_dates:
                T=max((datetime.strptime(ed,"%Y-%m-%d").date()-today).days/365,1/365)
                calls=[o for o in all_calls if o['expiration_date']==ed]
                strikes=sorted([float(o['strike_price']) for o in calls if float(o['strike_price'])>cp])[:3]
                for o in calls:
                    s=float(o['strike_price']); 
                    if s not in strikes: continue
                    md=r.options.get_option_market_data_by_id(o['id'])
                    price=0.0; delta=1.0
                    if md:
                        try: price=float(md[0].get('adjusted_mark_price') or md[0].get('mark_price') or 0.0)
                        except: price=0.0
                        try: delta=float(md[0].get('delta')) if md[0].get('delta') else None
                        except: delta=None
                        if delta is None or delta==0.0: delta=black_scholes_call_delta(cp,s,T,RISK_FREE_RATE,sigma)
                    price=max(price-PRICE_ADJUST,0.0)
                    if price>=MIN_PRICE:
                        prob_OTM=1-delta; risk=max(s-cp,0.01); pr=price/risk
                        candidates.append({"Ticker":TICKER,"Current Price":cp,"Expiration Date":ed,"Strike Price":s,
                                           "Option Price":price,"Delta":delta,"Prob OTM":prob_OTM,"Profit/Risk":pr,"URL":rh_url})
            selected=sorted(candidates,key=lambda x:x['Profit/Risk'],reverse=True)[:NUM_OPTIONS]
            all_options.extend(selected)
            if selected:
                msg=[f"📊 <a href='{rh_url}'>{TICKER}</a> Current: ${cp:.2f}"]
                for o in selected: msg.append(f"{risk_emoji(o['Prob OTM'])} Exp: {o['Expiration Date']} Strike: {o['Strike Price']} Price: ${o['Option Price']:.2f} Profit/Risk: {o['Profit/Risk']:.2f}")
                buf=generate_chart(selected[0],is_put=False)
                send_telegram_photo(buf,"\n".join(msg))
        except Exception as e: send_telegram_message(f"⚠️ {TICKER} call error: {e}")
    if all_options:
        best=max(all_options,key=lambda x:x['Profit/Risk'])
        buf=generate_chart(best,is_put=False)
        msg=[f"🔥 <b>Best Covered Call to Sell</b>","📊 <a href='{best['URL']}'>{best['Ticker']}</a> Current: ${best['Current Price']:.2f}",
             f"✅ Exp: {best['Expiration Date']}","💲 Strike: {best['Strike Price']}",
             f"💰 Price: ${best['Option Price']:.2f}","🎯 Prob OTM: {best['Prob OTM']*100:.1f}%",
             f"💎 Profit/Risk: {best['Profit/Risk']:.2f}"]
        send_telegram_photo(buf,"\n".join(msg))

# ------------------ EXECUTE ------------------
process_puts(safe_no_shares)
process_calls(owned)
