import streamlit as st
import akshare as ak
import pandas as pd
import ta
import plotly.graph_objects as go
import yfinance as yf
from openai import OpenAI  # 使用 OpenAI 标准库调用 Groq
import json
import re
import time
import datetime
import calendar
import pytz
import os

# --- 1. 页面配置 & CSS样式 ---
st.set_page_config(page_title="AI 全能操盘手 (Groq极速版)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .main { background-color: #fdfdfd; }
    
    /* 价格区间卡片 */
    .range-buy { 
        background: linear-gradient(to right, #f0fff4, #c6f6d5); 
        border-left: 5px solid #2f855a; border-radius: 8px; padding: 15px; 
        color: #22543d; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .range-sell { 
        background: linear-gradient(to right, #fff5f5, #fed7d7); 
        border-left: 5px solid #c53030; border-radius: 8px; padding: 15px; 
        color: #742a2a; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .range-val { font-size: 1.6em; font-weight: 800; margin: 5px 0; }
    
    /* 新闻情感 (红涨绿跌) */
    .news-bull { color: #d20000; background-color:#fff5f5; padding:8px; border-radius:5px; margin-bottom:6px; border-left:4px solid #d20000; }
    .news-bear { color: #008000; background-color:#f0fff0; padding:8px; border-radius:5px; margin-bottom:6px; border-left:4px solid #008000; }
    .news-neu { color: #555; padding:8px; margin-bottom:6px; border-left:4px solid #ccc; }
    
    /* 实时盯盘 */
    .live-price { font-size: 3.5em; font-weight: 900; color: #d93025; line-height: 1; }
    .live-tag { background-color: #d93025; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; animation: blink 1.5s infinite; }
    .closed-tag { background-color: #999; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
    @keyframes blink { 50% { opacity: 0.6; } }
    
    /* 历史记录按钮适配 */
    .stButton button { width: 100%; padding: 0.2rem 0.5rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. Groq (OpenAI兼容) 客户端初始化 ---
try:
    # Groq 的 Base URL 是固定的
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    
    # 从 Secrets 获取 Key
    api_key = st.secrets["GROQ_API_KEY"]
    
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    AI_AVAILABLE = True
except Exception as e:
    st.error(f"⚠️ AI 配置失败: 未找到 GROQ_API_KEY，请在 Streamlit Secrets 中配置。错误: {e}")
    AI_AVAILABLE = False

# --- 3. 本地配置存储 (适配云端临时会话) ---
CONFIG_FILE = "stock_config.json"

def load_config():
    default = {"last_ticker": "600519", "last_model": "llama-3.3-70b-versatile", "history": []}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
        except: return default
    return default

def save_config(ticker, model, history):
    # 历史记录去重并前置
    if ticker in history: history.remove(ticker)
    history.insert(0, ticker)
    history = history[:10]
    
    # 注意: Streamlit Cloud 重启后文件会重置，这里主要用于当次运行体验
    data = {"last_ticker": ticker, "last_model": model, "history": history}
    try:
        with open(CONFIG_FILE, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except: pass 
    return history

# 初始化 Session State
if 'init_done' not in st.session_state:
    cfg = load_config()
    st.session_state.ticker = cfg['last_ticker']
    st.session_state.model = cfg['last_model']
    st.session_state.history = cfg['history']
    st.session_state.init_done = True

# --- 4. 核心工具函数 ---

def get_market_status():
    """判断A股交易状态"""
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    if now.weekday() >= 5: return False, "休市(周末)"
    t = now.time()
    # 简单判定: 9:15~11:30, 13:00~15:00
    is_trade = (datetime.time(9,15)<=t<=datetime.time(11,30)) or (datetime.time(13,0)<=t<=datetime.time(15,0))
    return is_trade, "交易中" if is_trade else "休市"

def check_delivery_day():
    """股指期货交割日预警"""
    today = datetime.date.today()
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    monthcal = c.monthdatescalendar(today.year, today.month)
    # 获取当月所有周五
    fridays = [d for week in monthcal for d in week if d.weekday() == calendar.FRIDAY and d.month == today.month]
    
    if len(fridays) < 3: return False, ""
    delivery_day = fridays[2] # 第三个周五
    delta = (delivery_day - today).days
    
    if 0 <= delta <= 2:
        msgs = {0: "⚠️ 今日是股指交割日，谨防剧烈波动！", 1: "⚠️ 明日是股指交割日！", 2: "⚠️ 后天是股指交割日！"}
        return True, msgs[delta]
    return False, ""

def clean_num(n):
    """数据清洗"""
    try: return 0.0 if (pd.isna(n) or n==float('inf')) else round(float(n), 2)
    except: return 0.0

# --- 5. 数据获取模块 (AkShare + YFinance 双源容错) ---

@st.cache_data(ttl=3600)
def get_static_data(code):
    """获取静态日线数据 (双源)"""
    code = str(code).strip()
    
    # [方案 A] 优先尝试 AkShare (国内源)
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        df.rename(columns={'日期':'Date','开盘':'Open','收盘':'Close','最高':'High','最低':'Low','成交量':'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 获取名称 (AkShare)
        name = code
        try:
            spot = ak.stock_zh_a_spot_em()
            row = spot[spot['代码']==code]
            if not row.empty: name = row.iloc[0]['名称']
        except: pass
        
        return df.tail(300), name
    except: 
        pass # 失败则静默进入方案B

    # [方案 B] 降级尝试 YFinance (海外源)
    try:
        # 格式转换: 600519 -> 600519.SS
        yf_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
        stock = yf.Ticker(yf_code)
        df = stock.history(period="1y")
        
        if not df.empty:
            # 移除时区信息
            df.index = df.index.tz_localize(None)
            return df.tail(300), f"{code}(YF)"
    except: 
        pass
        
    return None, code

def get_live_data(code):
    """获取实时分时数据 (双源)"""
    code = str(code).strip()
    
    # [方案 A] AkShare 分时
    try:
        df = ak.stock_zh_a_hist_min_em(symbol=code, period='1', adjust='qfq')
        if not df.empty:
            r = df.iloc[-1]
            return {"price": float(r['收盘']), "high": float(r['最高']), "low": float(r['最低'])}
    except: pass

    # [方案 B] YFinance 实时
    try:
        yf_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
        stock = yf.Ticker(yf_code)
        info = stock.fast_info
        return {"price": float(info.last_price), "high": float(info.day_high), "low": float(info.day_low)}
    except: pass
    
    return None

def get_news(code):
    """获取新闻 (仅限 AkShare，YF新闻解析较难)"""
    try:
        df = ak.stock_news_em(symbol=code)
        return df[['发布时间','新闻标题']].head(5).to_dict('records')
    except: return []

def calc_indicators(df):
    """计算技术指标"""
    df = df.copy()
    # 均线
    df['MA50'] = ta.trend.sma_indicator(df['Close'], 50)
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], 14)
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    # 布林带
    bb = ta.volatility.BollingerBands(df['Close'], 20, 2)
    df['B_High'] = bb.bollinger_hband()
    df['B_Low'] = bb.bollinger_lband()
    
    # 计算历史支撑压力 (近60天)
    rec = df.tail(60)
    return df, rec['Low'].min(), rec['High'].max()

# --- 6. AI 分析模块 (Groq/OpenAI 通用) ---

def call_ai_openai(prompt, model_name):
    """通用 AI 调用函数"""
    if not AI_AVAILABLE: return None
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                # 关键: 系统提示词强制 JSON 格式
                {"role": "system", "content": "You are a financial analyst. Output strictly in JSON format without Markdown blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # 低温以保证格式稳定
            # 如果模型支持 json_object 模式可开启，为了兼容性这里靠 Prompt 约束
            # response_format={"type": "json_object"} 
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"AI API 调用失败: {e}")
        return None

def ai_analyze_static(code, name, df, sup, res, model_choice):
    """静态策略：生成买卖区间 (带强力兜底)"""
    curr = df.iloc[-1]
    
    # [1. 预计算兜底值] 防止 AI 挂掉或返回 0
    # 低吸兜底: 布林下轨 ~ 历史支撑
    def_buy_min = min(curr['B_Low'], sup)
    def_buy_max = max(curr['B_Low'], sup)
    # 高抛兜底: 历史压力 ~ 布林上轨
    def_sell_min = min(curr['B_High'], res)
    def_sell_max = max(curr['B_High'], res)
    
    data = {
        "p": clean_num(curr['Close']), "rsi": clean_num(curr['RSI']),
        "sup": clean_num(sup), "res": clean_num(res),
        "bl": clean_num(curr['B_Low']), "bh": clean_num(curr['B_High'])
    }

    prompt = f"""
    分析A股 {name}({code})。现价:{data['p']}。
    技术指标: 支撑{data['sup']}, 压力{data['res']}, 布林带{data['bl']}-{data['bh']}, RSI:{data['rsi']}。
    
    任务: 给出高抛低吸的【价格区间】。
    请严格输出纯 JSON 格式 (不要Markdown):
    {{
        "score": 0-100, "trend": "看涨/看跌/震荡",
        "buy_min": 买入下限(数字), "buy_max": 买入上限(数字),
        "sell_min": 卖出下限(数字), "sell_max": 卖出上限(数字),
        "reason": "简述理由(50字内)"
    }}
    """
    
    # 初始化结果 (先填入兜底值)
    result = {
        "score": 50, "trend": "震荡",
        "buy_min": def_buy_min, "buy_max": def_buy_max,
        "sell_min": def_sell_min, "sell_max": def_sell_max,
        "reason": "AI响应超时或格式错误，已自动切换为纯技术指标策略。"
    }
    
    # 调用 AI
    res_text = call_ai_openai(prompt, model_choice)
    
    if res_text:
        try:
            # 清洗可能存在的 Markdown 标记
            clean_text = res_text.replace("```json", "").replace("```", "").strip()
            # 正则提取 JSON
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match:
                ai_data = json.loads(match.group(0))
                
                # 辅助函数：只有当 AI 返回有效正数时，才覆盖兜底值
                def get_valid(key, default_val):
                    try:
                        val = float(ai_data.get(key, 0))
                        return val if val > 0 else default_val
                    except: return default_val

                result['score'] = ai_data.get('score', 50)
                result['trend'] = ai_data.get('trend', '震荡')
                result['reason'] = ai_data.get('reason', result['reason'])
                
                # 尝试覆盖价格
                result['buy_min'] = get_valid('buy_min', def_buy_min)
                result['buy_max'] = get_valid('buy_max', def_buy_max)
                result['sell_min'] = get_valid('sell_min', def_sell_min)
                result['sell_max'] = get_valid('sell_max', def_sell_max)
        except: pass
            
    return result

def ai_analyze_news(news, model_choice):
    """新闻情感分析"""
    if not news or not AI_AVAILABLE: return news
    
    txt = "\n".join([n['新闻标题'] for n in news])
    prompt = f"""
    分析以下新闻是利好、利空还是中性。
    {txt}
    请返回纯 JSON 列表格式: [{{ "index": 1, "s": "利好" }}, {{ "index": 2, "s": "利空" }}...]
    """
    
    res_text = call_ai_openai(prompt, model_choice)
    if res_text:
        try:
            clean_text = res_text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\[.*\]', clean_text, re.DOTALL)
            if match:
                sents = json.loads(match.group(0))
                for i, n in enumerate(news):
                    n['s'] = '中性' # 默认
                    for s in sents:
                        if s.get('index') == i+1: n['s'] = s.get('s')
        except: pass
    return news

# --- 7. 界面主逻辑 ---

# === 侧边栏 (控制台) ===
with st.sidebar:
    st.header("🎮 控制台")
    
    # Groq 免费模型列表 (推荐 Llama 3.3)
    model_map = {
        "Llama 3.3 70B (最新/推荐)": "llama-3.3-70b-versatile",
        "Llama 3.1 70B (稳定)": "llama-3.1-70b-versatile",
        "Llama 3.1 8B (极速)": "llama-3.1-8b-instant",
        "Mixtral 8x7b (均衡)": "mixtral-8x7b-32768",
        "Gemma 2 9B (Google)": "gemma2-9b-it"
    }
    
    # 恢复上次的模型选择
    default_idx = 0
    # 获取字典的 value 列表
    model_values = list(model_map.values())
    if st.session_state.model in model_values:
        default_idx = model_values.index(st.session_state.model)
        
    sel_label = st.selectbox("AI 模型 (Groq)", list(model_map.keys()), index=default_idx)
    sel_model = model_map[sel_label]
    
    # 模式与代码
    mode = st.radio("模式", ["📊 静态深度分析", "🔴 实时盯盘 (Live)"])
    ticker_input = st.text_input("A股代码", value=st.session_state.ticker)
    
    # 历史记录按钮
    st.markdown("### 🕒 最近查询")
    if st.session_state.history:
        cols = st.columns(3)
        for i, h_code in enumerate(st.session_state.history):
            if cols[i%3].button(h_code, key=f"h_{h_code}"):
                st.session_state.ticker = h_code
                st.rerun()
    else:
        st.caption("暂无记录")
    
    st.divider()
    st.info("提示: 数据源优先 AkShare，云端自动切换 YFinance。")

# === 主界面逻辑 ===

# [模式 A] 静态深度分析
if mode == "📊 静态深度分析":
    st.title(f"📊 AI 深度复盘 (Groq版)")
    
    # 交割日预警
    is_del, del_msg = check_delivery_day()
    if is_del: st.warning(del_msg)
    
    if st.button("🚀 开始深度分析", type="primary", use_container_width=True):
        # 保存配置
        new_hist = save_config(ticker_input, sel_model, st.session_state.history)
        st.session_state.history = new_hist
        st.session_state.ticker = ticker_input
        
        with st.spinner(f"正在请求 Groq ({sel_model}) 进行极速分析..."):
            df, name = get_static_data(ticker_input)
            
            if df is not None:
                # 计算指标
                df, sup, res = calc_indicators(df)
                
                # 并行获取新闻
                news_raw = get_news(ticker_input)
                
                # AI 分析 (策略 + 新闻)
                ai_res = ai_analyze_static(ticker_input, name, df, sup, res, sel_model)
                news_res = ai_analyze_news(news_raw, sel_model)
                
                curr_price = df['Close'].iloc[-1]
                
                # --- 结果展示 ---
                st.header(f"{name} ({ticker_input})")
                
                # 1. 顶部指标
                c1, c2, c3 = st.columns(3)
                c1.metric("最新收盘价", f"{curr_price:.2f}")
                c2.metric("AI 评分", ai_res.get('score', 0))
                c3.metric("趋势判定", ai_res.get('trend', '-'))
                
                # 2. 策略卡片 (价格区间)
                st.divider()
                col_b, col_s = st.columns(2)
                
                # 安全转换
                def f(v): 
                    try: return float(v)
                    except: return 0.0
                
                with col_b:
                    st.markdown(f"""
                    <div class="range-buy">
                        <div style="opacity:0.9">🎯 建议低吸区间 (承接)</div>
                        <div class="range-val">{f(ai_res.get('buy_min')):.2f} ~ {f(ai_res.get('buy_max')):.2f}</div>
                        <small>强支撑参考: {sup:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_s:
                    st.markdown(f"""
                    <div class="range-sell">
                        <div style="opacity:0.9">🛑 建议高抛区间 (压力)</div>
                        <div class="range-val">{f(ai_res.get('sell_min')):.2f} ~ {f(ai_res.get('sell_max')):.2f}</div>
                        <small>强压力参考: {res:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 3. 分析理由
                st.info(f"🧠 **AI 逻辑分析**: {ai_res.get('reason')}")
                
                # 4. 新闻情报
                st.subheader("📢 消息面情报")
                if news_res:
                    for n in news_res:
                        s = n.get('s', '中性')
                        # 样式选择
                        cls = "news-neu"
                        if "利好" in s: cls = "news-bull"
                        elif "利空" in s: cls = "news-bear"
                        
                        st.markdown(f"<div class='{cls}'>[{s}] {n['新闻标题']} <span style='float:right;font-size:0.8em'>{n['发布时间']}</span></div>", unsafe_allow_html=True)
                else:
                    st.caption("暂无近期重大新闻 (或数据源受限)")
                
                # 5. K线图表
                st.subheader("📈 技术走势")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K线'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='orange'), name='MA50'))
                fig.add_trace(go.Scatter(x=df.index, y=df['B_High'], line=dict(color='gray', width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=df.index, y=df['B_Low'], line=dict(color='gray', width=0), fill='tonexty', showlegend=False))
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"❌ 数据获取失败: {name}。请检查代码是否正确。")

# [模式 B] 实时盯盘
elif mode == "🔴 实时盯盘 (Live)":
    st.title(f"⚡ 智能盯盘终端 ({ticker_input})")
    
    # 保存一次配置
    save_config(ticker_input, sel_model, st.session_state.history)
    
    # 市场状态
    is_open, status_msg = get_market_status()
    
    # 布局
    tag_box = st.empty()
    metric_box = st.empty()
    chart_box = st.empty()
    
    # 临时历史数据
    if 'live_hist' not in st.session_state: st.session_state.live_hist = []
    
    if not is_open:
        tag_box.markdown(f"<span class='closed-tag'>💤 {status_msg}</span>", unsafe_allow_html=True)
        # 显示最后一次静态数据作为参考
        d = get_live_data(ticker_input)
        if d:
            metric_box.metric("当前价格 (休市)", f"{d['price']:.2f}")
        st.warning("当前市场已休市，停止自动刷新。")
    else:
        tag_box.markdown(f"<span class='live-tag'>🔴 交易中</span>", unsafe_allow_html=True)
        
        if st.button("🛑 停止监控"): st.stop()
        
        while True:
            d = get_live_data(ticker_input)
            if d:
                price = d['price']
                # 记录走势
                st.session_state.live_hist.append(price)
                if len(st.session_state.live_hist) > 60: st.session_state.live_hist.pop(0)
                
                # 简单异动检测
                alert = ""
                if len(st.session_state.live_hist) > 5:
                    recent = st.session_state.live_hist[-5:]
                    if price > min(recent) * 1.01: alert = "🚀 突发异动：急速拉升！"
                    elif price < max(recent) * 0.99: alert = "🌊 突发异动：快速跳水！"
                
                # 刷新界面
                with metric_box.container():
                    c1, c2 = st.columns([2, 1])
                    c1.markdown(f"<div class='live-price'>{price:.2f}</div>", unsafe_allow_html=True)
                    c2.metric("最高", d['high'])
                    c2.metric("最低", d['low'])
                    if alert:
                        st.markdown(f"<div class='alert-box' style='border:2px solid red;background:#ffebeb;color:red;padding:10px;font-weight:bold'>{alert}</div>", unsafe_allow_html=True)
                
                # 绘制简易分时图
                with chart_box.container():
                    fig = go.Figure(go.Scatter(y=st.session_state.live_hist, mode='lines+markers', line=dict(color='red')))
                    fig.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), title="监控时段走势")
                    st.plotly_chart(fig, use_container_width=True)
                
                time.sleep(3) # 3秒刷新一次
            else:
                st.error("数据获取超时...")
                time.sleep(5)
