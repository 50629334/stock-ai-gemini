import streamlit as st
import akshare as ak
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import google.generativeai as genai  # 替换 ollama
import json
import re
import time
import datetime
import calendar
import pytz
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 操盘手 (Gemini云端版)", layout="wide", page_icon="☁️")

# CSS 样式 (保持不变)
st.markdown("""
<style>
    .main { background-color: #fdfdfd; }
    .range-buy { background: linear-gradient(to right, #f0fff4, #c6f6d5); border-left: 5px solid #2f855a; border-radius: 8px; padding: 15px; color: #22543d; }
    .range-sell { background: linear-gradient(to right, #fff5f5, #fed7d7); border-left: 5px solid #c53030; border-radius: 8px; padding: 15px; color: #742a2a; }
    .range-val { font-size: 1.5em; font-weight: 800; margin: 5px 0; }
    .news-bull { color: #d20000; background-color:#fff5f5; padding:5px; border-left:4px solid #d20000; margin-bottom:5px; }
    .news-bear { color: #008000; background-color:#f0fff0; padding:5px; border-left:4px solid #008000; margin-bottom:5px; }
    .news-neu { color: #555; padding:5px; border-left:4px solid #ccc; margin-bottom:5px; }
    .live-price { font-size: 3em; font-weight: 900; color: #d93025; }
    .alert-box { border: 2px solid red; background-color: #ffebeb; color: red; padding: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. Gemini API 配置 ---
# 尝试从 Streamlit Secrets 获取 API Key
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    GEMINI_AVAILABLE = True
except:
    st.error("未检测到 GEMINI_API_KEY，请在 Streamlit Cloud 的 Secrets 中配置。")
    GEMINI_AVAILABLE = False

# --- 3. 核心工具函数 ---

def get_market_status():
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    if now.weekday() >= 5: return False, "休市(周末)"
    t = now.time()
    is_trade = (datetime.time(9,15)<=t<=datetime.time(11,30)) or (datetime.time(13,0)<=t<=datetime.time(15,0))
    return is_trade, "交易中" if is_trade else "休市"

def clean_num(n):
    try: return 0.0 if (pd.isna(n) or n==float('inf')) else round(float(n), 2)
    except: return 0.0

# --- 4. 数据获取 (AkShare) ---
def get_static_data(code):
    """双源获取静态数据: AkShare (首选) -> Yfinance (备选)"""
    code = str(code).strip()
    
    # --- 尝试 1: AkShare (国内源) ---
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        df.rename(columns={'日期':'Date','开盘':'Open','收盘':'Close','最高':'High','最低':'Low','成交量':'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 获取基本面
        name = code
        try:
            spot = ak.stock_zh_a_spot_em()
            row = spot[spot['代码']==code]
            if not row.empty: name = row.iloc[0]['名称']
        except: pass
        
        return df.tail(300), name
        
    except Exception:
        pass # AkShare 失败，静默进入方案 2

    # --- 尝试 2: Yfinance (海外源，适合云端) ---
    try:
        # 转换代码格式：6开头加.SS，其他加.SZ
        yf_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
        stock = yf.Ticker(yf_code)
        df = stock.history(period="1y")
        
        if not df.empty:
            # YF数据带时区，需要移除以便后续绘图
            df.index = df.index.tz_localize(None)
            return df.tail(300), f"{code}(YF)"
            
    except Exception:
        pass
        
    return None, code

def get_live_data(code):
    """双源获取实时数据"""
    code = str(code).strip()
    
    # --- 尝试 1: AkShare ---
    try:
        df = ak.stock_zh_a_hist_min_em(symbol=code, period='1', adjust='qfq')
        if not df.empty:
            r = df.iloc[-1]
            return {"price": float(r['收盘']), "high": float(r['最高']), "low": float(r['最低'])}
    except: pass

    # --- 尝试 2: Yfinance ---
    try:
        yf_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
        stock = yf.Ticker(yf_code)
        info = stock.fast_info
        return {
            "price": float(info.last_price),
            "high": float(info.day_high), 
            "low": float(info.day_low)
        }
    except: pass
    
    return None

def get_news(code):
    try:
        df = ak.stock_news_em(symbol=code)
        return df[['发布时间','新闻标题']].head(5).to_dict('records')
    except: return []

def calc_indicators(df):
    df = df.copy()
    df['MA50'] = ta.trend.sma_indicator(df['Close'], 50)
    df['RSI'] = ta.momentum.rsi(df['Close'], 14)
    bb = ta.volatility.BollingerBands(df['Close'], 20, 2)
    df['B_High'] = bb.bollinger_hband()
    df['B_Low'] = bb.bollinger_lband()
    rec = df.tail(60)
    return df, rec['Low'].min(), rec['High'].max()

# --- 5. Gemini AI 分析模块 ---

def call_gemini(prompt, model_name="gemini-1.5-flash"):
    """调用 Gemini API"""
    if not GEMINI_AVAILABLE: return None
    try:
        model = genai.GenerativeModel(model_name)
        # 设置 generation_config 强制让它尽量输出 JSON (Gemini 1.5 支持)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2
            )
        )
        return response.text
    except Exception as e:
        st.error(f"Gemini API 调用失败: {e}")
        return None

def ai_analyze_static(code, name, df, sup, res, model_choice):
    curr = df.iloc[-1]
    
    # 1. 兜底值
    def_buy_min = min(curr['B_Low'], sup)
    def_buy_max = max(curr['B_Low'], sup)
    def_sell_min = min(curr['B_High'], res)
    def_sell_max = max(curr['B_High'], res)
    
    data = {
        "p": clean_num(curr['Close']), "rsi": clean_num(curr['RSI']),
        "sup": clean_num(sup), "res": clean_num(res),
        "bl": clean_num(curr['B_Low']), "bh": clean_num(curr['B_High'])
    }

    prompt = f"""
    分析A股 {name}({code})。现价:{data['p']}。
    指标: 支撑{data['sup']}, 压力{data['res']}, 布林带{data['bl']}-{data['bh']}, RSI:{data['rsi']}。
    
    任务: 给出高抛低吸【价格区间】。
    请务必返回纯 JSON 格式，不要包含 Markdown 代码块(```json):
    {{
        "score": 0-100, "trend": "看涨/看跌/震荡",
        "buy_min": 买入下限(数字), "buy_max": 买入上限(数字),
        "sell_min": 卖出下限(数字), "sell_max": 卖出上限(数字),
        "reason": "简述理由"
    }}
    """
    
    res_text = call_gemini(prompt, model_choice)
    
    # 解析结果
    result = {
        "score": 50, "trend": "震荡",
        "buy_min": def_buy_min, "buy_max": def_buy_max,
        "sell_min": def_sell_min, "sell_max": def_sell_max,
        "reason": "AI未响应，使用技术指标兜底。"
    }
    
    if res_text:
        try:
            # 清洗 markdown 标记
            clean_text = res_text.replace("```json", "").replace("```", "").strip()
            # 提取 JSON
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match:
                ai_data = json.loads(match.group(0))
                # 覆盖兜底值
                result.update(ai_data)
        except: pass
            
    return result

def ai_analyze_news(news, model_choice):
    if not news or not GEMINI_AVAILABLE: return news
    txt = "\n".join([n['新闻标题'] for n in news])
    prompt = f"""
    分析以下新闻利好/利空/中性。
    {txt}
    请返回纯 JSON 列表: [{{ "index": 1, "s": "利好" }}, ...]
    """
    res_text = call_gemini(prompt, model_choice)
    if res_text:
        try:
            clean_text = res_text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\[.*\]', clean_text, re.DOTALL)
            if match:
                sents = json.loads(match.group(0))
                for i, n in enumerate(news):
                    n['s'] = '中性'
                    for s in sents:
                        if s.get('index') == i+1: n['s'] = s.get('s')
        except: pass
    return news

# --- 6. 界面主逻辑 ---

with st.sidebar:
    st.header("控制台")
    
    # 注意：下面这行必须缩进，与上面的 st.header 保持垂直对齐
    model_map = {
        "Gemini 1.5 Flash (快速)": "gemini-1.5-flash",
        "Gemini 1.5 Pro (强力)": "gemini-1.5-pro",
        "Gemini 1.0 Pro (备用)": "gemini-pro"
    }
    sel_label = st.selectbox("AI 模型", list(model_map.keys()))
    sel_model = model_map[sel_label]
    
    mode = st.radio("模式", ["静态分析", "实时盯盘"])
    ticker = st.text_input("股票代码", "600519")

if mode == "静态分析":
    st.title("Gemini 股票分析师 ☁️")
    if st.button("开始分析", type="primary"):
        with st.spinner("Gemini 正在思考..."):
            df, name = get_static_data(ticker)
            if df is not None:
                df, sup, res = calc_indicators(df)
                # 调用 AI
                ai_res = ai_analyze_static(ticker, name, df, sup, res, sel_model)
                news_raw = get_news(ticker)
                news_res = ai_analyze_news(news_raw, sel_model)
                
                # 展示
                curr = df['Close'].iloc[-1]
                st.subheader(f"{name} ({ticker})")
                c1, c2, c3 = st.columns(3)
                c1.metric("现价", f"{curr:.2f}")
                c2.metric("评分", ai_res.get('score'))
                c3.metric("趋势", ai_res.get('trend'))
                
                cb, cs = st.columns(2)
                # 辅助转换函数
                def f(x): 
                    try: return float(x)
                    except: return 0.0
                
                with cb:
                    st.markdown(f"""<div class="range-buy">
                        <div>🎯 建议低吸</div>
                        <div class="range-val">{f(ai_res.get('buy_min')):.2f} ~ {f(ai_res.get('buy_max')):.2f}</div>
                        <small>支撑: {sup:.2f}</small></div>""", unsafe_allow_html=True)
                with cs:
                    st.markdown(f"""<div class="range-sell">
                        <div>🛑 建议高抛</div>
                        <div class="range-val">{f(ai_res.get('sell_min')):.2f} ~ {f(ai_res.get('sell_max')):.2f}</div>
                        <small>压力: {res:.2f}</small></div>""", unsafe_allow_html=True)
                
                st.info(f"分析逻辑: {ai_res.get('reason')}")
                
                # 新闻
                st.subheader("新闻情感")
                if news_res:
                    for n in news_res:
                        s = n.get('s', '中性')
                        cls = "news-bull" if "利好" in s else ("news-bear" if "利空" in s else "news-neu")
                        st.markdown(f"<div class='{cls}'>[{s}] {n['新闻标题']} <span style='float:right'>{n['发布时间']}</span></div>", unsafe_allow_html=True)
                
                # 图表
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='orange'), name='MA50'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("数据获取失败")

elif mode == "实时盯盘":
    st.title(f"实时盯盘 ({ticker})")
    is_open, msg = get_market_status()
    if not is_open:
        st.warning(f"当前市场状态: {msg}，数据不刷新。")
    
    ph = st.empty()
    chart_ph = st.empty()
    hist = []
    
    if st.button("停止"): st.stop()
    
    while True:
        d = get_live_data(ticker)
        if d:
            hist.append(d['price'])
            if len(hist)>60: hist.pop(0)
            
            # 异动
            alert = ""
            if len(hist)>5:
                if d['price'] > min(hist[-5:])*1.01: alert="急速拉升!"
                elif d['price'] < max(hist[-5:])*0.99: alert="快速跳水!"
            
            with ph.container():
                c1, c2 = st.columns([2,1])
                c1.markdown(f"<div class='live-price'>{d['price']:.2f}</div>", unsafe_allow_html=True)
                c2.metric("最高", d['high'])
                c2.metric("最低", d['low'])
                if alert: st.markdown(f"<div class='alert-box'>{alert}</div>", unsafe_allow_html=True)
            
            with chart_ph.container():
                fig = go.Figure(go.Scatter(y=hist, mode='lines+markers'))
                fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
        
        time.sleep(3)


