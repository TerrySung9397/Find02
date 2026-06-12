import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from io import StringIO
from datetime import datetime

# Headers for scraping TWSE/TPEx websites
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Telegram Push Configurations (已填入您的金鑰 - 僅供本機測試)
TELEGRAM_TOKEN = '8812419373:AAE3E5f7dBH40JmPbn7h91JzsxJfZv2tdgw'
TELEGRAM_CHAT_ID = '-5179213819'

def send_telegram_message(text):
    """Send HTML-formatted broadcast message via Telegram Bot API."""
    if not TELEGRAM_TOKEN:
        print("❌ 錯誤：找不到 TELEGRAM_TOKEN。")
        return
        
    print("Sending Telegram broadcast...")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            print("Telegram push successful!")
        else:
            print(f"Telegram push failed: {response.status_code} - {response.text}")
    except Exception as e:
        print("Telegram push error:", e)

def fetch_twse_tickers():
    """Scrape TWSE (上市) common stock tickers."""
    print("Scraping TWSE (上市) stocks...")
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.encoding = 'big5'
        dfs = pd.read_html(StringIO(response.text))
        df = dfs[0]
        df.columns = df.iloc[0]
        df = df[1:]
        df = df[df["市場別"] == "上市"]
        df[["code", "name"]] = df["有價證券代號及名稱"].str.split(n=1, expand=True)
        # Filter for 4-digit common stock codes
        df = df[df["code"].str.len() == 4]
        # Clean up ticker codes and names
        stocks = []
        for _, row in df.iterrows():
            stocks.append({
                "ticker": f"{row['code']}.TW",
                "code": row['code'],
                "name": row['name'].strip() if row['name'] else "",
                "market": "上市",
                "industry": row['產業別'].strip() if pd.notna(row['產業別']) else "未知"
            })
        print(f"Successfully scraped {len(stocks)} TWSE stocks.")
        return stocks
    except Exception as e:
        print(f"Error scraping TWSE tickers: {e}")
        return []

def fetch_tpex_tickers():
    """Scrape TPEx (上櫃) common stock tickers."""
    print("Scraping TPEx (上櫃) stocks...")
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.encoding = 'big5'
        dfs = pd.read_html(StringIO(response.text))
        df = dfs[0]
        df.columns = df.iloc[0]
        df = df[1:]
        df = df[df["市場別"] == "上櫃"]
        df[["code", "name"]] = df["有價證券代號及名稱"].str.split(n=1, expand=True)
        # Filter for 4-digit common stock codes
        df = df[df["code"].str.len() == 4]
        stocks = []
        for _, row in df.iterrows():
            stocks.append({
                "ticker": f"{row['code']}.TWO",
                "code": row['code'],
                "name": row['name'].strip() if row['name'] else "",
                "market": "上櫃",
                "industry": row['產業別'].strip() if pd.notna(row['產業別']) else "未知"
            })
        print(f"Successfully scraped {len(stocks)} TPEx stocks.")
        return stocks
    except Exception as e:
        print(f"Error scraping TPEx tickers: {e}")
        return []

def calculate_kdj(df, n=9, m1=3, m2=3):
    """Calculate KDJ (9, 3, 3) indicator."""
    low_n = df['Low'].rolling(window=n).min()
    high_n = df['High'].rolling(window=n).max()
    
    # Avoid division by zero
    denom = high_n - low_n
    rsv = pd.Series(50.0, index=df.index)
    rsv[denom > 0] = (df['Close'] - low_n) / denom * 100
    
    k = [50.0] * len(df)
    d = [50.0] * len(df)
    
    for i in range(1, len(df)):
        k[i] = (2/3) * k[i-1] + (1/3) * rsv.iloc[i]
        d[i] = (2/3) * d[i-1] + (1/3) * k[i]
        
    k_ser = pd.Series(k, index=df.index)
    d_ser = pd.Series(d, index=df.index)
    j_ser = 3 * k_ser - 2 * d_ser
    
    return k_ser, d_ser, j_ser

def calculate_macd(df, short=6, long=13, signal=9):
    """Calculate MACD (12, 26, 9) indicator."""
    ema_short = df['Close'].ewm(span=short, adjust=False).mean()
    ema_long = df['Close'].ewm(span=long, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist

def check_conditions(df):
    """
    Check if the stock data matches the three conditions.
    Returns: (is_match, details_dict)
    """
    if len(df) < 60:
        return False, None
        
    # Latest index
    t = -1
    t_1 = -2
    t_2 = -3
    t_3 = -4
    
    # Get values at t, t-1, t-2
    close = df['Close'].iloc[t]
    close_prev = df['Close'].iloc[t_1]
    open_val = df['Open'].iloc[t]
    volume = df['Volume'].iloc[t]
    volume_prev = df['Volume'].iloc[t_1]
    
    # 1. MACD Check
    # DIF, DEA, Hist
    dif, dea, macd_hist = calculate_macd(df)
    
    hist_t = macd_hist.iloc[t]
    hist_t1 = macd_hist.iloc[t_1]
    hist_t2 = macd_hist.iloc[t_2]
    
    dif_t = dif.iloc[t]
    dif_t1 = dif.iloc[t_1]
    
    # Condition: Green column (negative macd hist) continuously shortening (getting less negative)
    # OR green column has just turned red (DIF crossed above DEA) within the last 1-2 days (even stronger rebound!)
    macd_hist_shortening = (
        # Option A: Green columns shortening
        ((hist_t < 0) and (hist_t > hist_t1) and (hist_t1 > hist_t2)) or
        # Option B: Reversal from green to red column today (Hist crosses 0 today)
        ((hist_t >= 0) and (hist_t1 < 0)) or
        # Option C: Reversal from green to red column yesterday and remains positive today
        ((hist_t >= 0) and (hist_t1 >= 0) and (hist_t2 < 0))
    )
    # DIF line showing flattening or upward trend: dif_t >= dif_t1
    dif_upward = dif_t >= dif_t1
    
    macd_match = macd_hist_shortening and dif_upward
    
    # 2. KDJ Check
    k, d, j = calculate_kdj(df)
    k_t = k.iloc[t]
    d_t = d.iloc[t]
    j_t = j.iloc[t]
    k_t1 = k.iloc[t_1]
    d_t1 = d.iloc[t_1]
    k_t2 = k.iloc[t_2]
    d_t2 = d.iloc[t_2]
    
    # Low-level area: D line or K line below 35
    low_level = (d_t < 35) or (k_t < 35) or (d_t1 < 35) or (k_t1 < 35)
    
    # Golden cross today or yesterday
    golden_cross_today = (k_t1 <= d_t1) and (k_t > d_t)
    golden_cross_yesterday = (k_t2 <= d_t2) and (k_t1 > d_t1) and (k_t > d_t) # cross yesterday and remains crossed
    
    kdj_match = low_level and (golden_cross_today or golden_cross_yesterday)
    
    # 3. Support Level & Volume Check
    # Calculate MAs
    ma20 = df['Close'].rolling(window=20).mean().iloc[t]
    ma60 = df['Close'].rolling(window=60).mean().iloc[t]
    ma120 = df['Close'].rolling(window=120).mean().iloc[t]
    
    # Bollinger Bands
    std20 = df['Close'].rolling(window=20).std().iloc[t]
    bb_lower = ma20 - 2 * std20
    
    # Lows
    low_20 = df['Low'].rolling(window=20).min().iloc[t]
    low_60 = df['Low'].rolling(window=60).min().iloc[t]
    
    # Support Level Match (within 2.5% of any support level)
    near_low_20 = (close >= low_20) and (close <= low_20 * 1.025)
    near_low_60 = (close >= low_60) and (close <= low_60 * 1.025)
    near_ma60 = (abs(close - ma60) / ma60) <= 0.025
    near_ma120 = (abs(close - ma120) / ma120) <= 0.025
    near_bb_lower = (close <= bb_lower * 1.015) and (close >= bb_lower * 0.985)
    
    support_match = near_low_20 or near_low_60 or near_ma60 or near_ma120 or near_bb_lower
    
    # Identify active support
    active_supports = []
    if near_low_20: active_supports.append("近20日低點支撐")
    if near_low_60: active_supports.append("近60日低點支撐")
    if near_ma60: active_supports.append("季線 MA60 支撐")
    if near_ma120: active_supports.append("半年線 MA120 支撐")
    if near_bb_lower: active_supports.append("布林通道下軌支撐")
    support_desc = " + ".join(active_supports) if active_supports else "無明顯支撐"
    
    # Volume Check
    ma_vol_5 = df['Volume'].rolling(window=5).mean().iloc[t_1]
    ma_vol_20 = df['Volume'].rolling(window=20).mean().iloc[t_1]
    
    # Selling pressure decreases in previous consolidation:
    # 3-day average volume before today is less than 20-day average volume
    prev_vol_avg = df['Volume'].iloc[t_3:t].mean()
    selling_pressure_low = prev_vol_avg < ma_vol_20
    
    # Rebound: close > close_prev or close > open_val
    rebound = (close > close_prev) or (close > open_val)
    # Today's volume is larger than yesterday's volume: volume > volume_prev
    volume_expanding = volume > volume_prev
    # Volume moderately expands compared to 5-day MA volume: volume > 1.05 * ma_vol_5 and volume < 2.5 * ma_vol_5
    # Let's relax this slightly for high quality rebounds: volume > 0.95 * ma_vol_5
    volume_moderate = (volume >= 1.02 * ma_vol_5) and (volume <= 2.8 * ma_vol_5)
    
    volume_match = rebound and volume_expanding and volume_moderate
    
    # Overall Match
    # Wait, we want to match: MACD, KDJ, and Support. Volume check is highly recommended but sometimes volume expands today
    # or expanded yesterday. We'll require MACD and KDJ, and at least (Support near OR Volume match).
    # To be strictly aligned with the user's chart:
    # - MACD Match
    # - KDJ Match
    # - Support Match
    # - Volume Match (or close to it)
    is_match = macd_match and kdj_match and support_match and (volume_match or (volume > volume_prev and rebound))
    
    details = {
        "macd_match": bool(macd_match),
        "macd_hist_t": float(hist_t),
        "macd_hist_t1": float(hist_t1),
        "macd_hist_t2": float(hist_t2),
        "dif_t": float(dif_t),
        "dif_t1": float(dif_t1),
        
        "kdj_match": bool(kdj_match),
        "k_t": float(k_t),
        "d_t": float(d_t),
        "j_t": float(j_t),
        "kdj_cross_type": "今日金叉" if golden_cross_today else "昨日金叉",
        
        "support_match": bool(support_match),
        "support_desc": support_desc,
        "close": float(close),
        "low_20": float(low_20),
        "low_60": float(low_60),
        "ma60": float(ma60),
        "bb_lower": float(bb_lower),
        
        "volume_match": bool(volume_match),
        "volume": int(volume),
        "volume_prev": int(volume_prev),
        "volume_ratio_5": float(volume / ma_vol_5) if ma_vol_5 > 0 else 1.0
    }
    
    return is_match, details

def main():
    print(f"=== Starting Taiwan Stock Screener: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # 1. Fetch Tickers
    twse_stocks = fetch_twse_tickers()
    tpex_stocks = fetch_tpex_tickers()
    all_stocks = twse_stocks + tpex_stocks
    
    if not all_stocks:
        print("Failed to fetch tickers. Exiting...")
        return
        
    print(f"Total tickers to scan: {len(all_stocks)}")
    
    # 2. Extract tickers list for batch download
    tickers = [s["ticker"] for s in all_stocks]
    
    # To be friendly to memory and network, let's filter out very illiquid or inactive stocks first if possible.
    # Since we can't easily check volume without downloading, let's download daily K line data in chunks.
    # Group into chunks of 150 for downloading
    chunk_size = 150
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    matched_results = []
    processed_count = 0
    
    print("Downloading historical stock data from Yahoo Finance...")
    
    for idx, chunk in enumerate(chunks):
        print(f"\nProcessing batch {idx + 1}/{len(chunks)} ({len(chunk)} tickers)...")
        try:
            # Download past 6 months of
