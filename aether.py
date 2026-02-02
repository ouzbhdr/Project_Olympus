import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import numpy as np
import json
import subprocess
import signal
from datetime import datetime  
import time
import os
from odysseus import Odysseus

# --- PAGE CONFIG ---
st.set_page_config(page_title="Project Olympus - Aether", layout="wide", page_icon="⚡")
st.title("⚡ Aether: Command & Control Center")

TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]

# --- DATABASE UTILS ---
def get_db_history():
    """Fetches trade history from Mnemosyne's 'trades' table."""
    db_path = "olympus.db"
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM trades ORDER BY id DESC LIMIT 100"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"DB Error: {e}")
        return pd.DataFrame()

# --- Backtest Engine ---
def backtest_run(df, start_balance=1000, leverage=20, enable_tp=False, commission_rate=0.05):
    """
    Simulates a trading session based on signals provided in the DataFrame.
    
    Parameters:
    - df: DataFrame containing 'signal', 'sl_price', 'tp_price', and 'close'
    - start_balance: Initial capital (USDT)
    - leverage: Used to calculate position size (not affecting liquidation in this sim)
    - enable_tp: Toggle for Take Profit logic
    - commission_rate: Percent fee charged per trade (Entry + Exit)
    """
    balance = start_balance
    position = None  # Current state: 'LONG', 'SHORT', or None
    entry_price = 0.0
    amount = 0.0
    
    # Performance Metrics
    trade_count = 0
    wins = 0
    total_commission = 0
    trade_log = []

    # Iterate through the DataFrame (itertuples is faster for large datasets)
    for row in df.itertuples():
        
        # --- 1. EXIT LOGIC (If in a position) ---
        if position:
            is_exit = False
            exit_reason = ""

            # Check for Signal Reversal or Stop Loss
            if position == 'LONG':
                if row.signal == -1:
                    is_exit = True
                    exit_reason = "Signal Reversal"
                elif row.close <= row.sl_price:
                    is_exit = True
                    exit_reason = "Stop Loss"
                elif enable_tp and row.tp_price > 0 and row.close >= row.tp_price:
                    is_exit = True
                    exit_reason = "Take Profit"

            elif position == 'SHORT':
                if row.signal == 1:
                    is_exit = True
                    exit_reason = "Signal Reversal"
                elif row.close >= row.sl_price:
                    is_exit = True
                    exit_reason = "Stop Loss"
                elif enable_tp and row.tp_price > 0 and row.close <= row.tp_price:
                    is_exit = True
                    exit_reason = "Take Profit"

            # Execute Exit
            if is_exit:
                # Calculate Raw PnL
                if position == 'LONG':
                    pnl_raw = (row.close - entry_price) * amount
                else:
                    pnl_raw = (entry_price - row.close) * amount
                
                # Calculate Commission (Fee on Entry value + Fee on Exit value)
                entry_value = entry_price * amount
                exit_value = row.close * amount
                fee = (entry_value + exit_value) * (commission_rate / 100)
                
                net_pnl = pnl_raw - fee
                balance += net_pnl
                total_commission += fee
                
                trade_count += 1
                if net_pnl > 0: wins += 1
                
                trade_log.append({
                    'timestamp': row.timestamp,
                    'action': 'EXIT',
                    'type': position,
                    'price': row.close,
                    'pnl': net_pnl,
                    'reason': exit_reason,
                    'balance': balance
                })
                position = None
                amount = 0.0
                continue # Skip entry on the same bar

        # --- 2. ENTRY LOGIC (If no active position) ---
        if not position and row.signal != 0:
            position = 'LONG' if row.signal == 1 else 'SHORT'
            entry_price = row.close
            
            # Position Sizing: Use full balance with leverage
            # Note: In a real bot (Orion), we'd use risk-based sizing.
            amount = (balance * leverage) / entry_price
            
            trade_log.append({
                'timestamp': row.timestamp,
                'action': 'ENTRY',
                'type': position,
                'price': entry_price
            })

    # --- 3. FINAL RESULTS ---
    return {
        "Net Profit": balance - start_balance,
        "Final Balance": balance,
        "Trade Count": trade_count,
        "Win Rate": (wins / trade_count * 100) if trade_count > 0 else 0,
        "Total Commission": total_commission,
        "Log": trade_log
    }

# --- History ---
def deep_history_download(symbol, timeframe, total_limit=5000):
    import ccxt
    exchange = ccxt.okx({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=total_limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    filename = f"data_{symbol}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    return df

def fetch_logs(limit=50):
    import glob
    log_dir = "logs"
    if not os.path.exists(log_dir): return "No logs found."
    
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files: return "Log file is empty."
    
    latest_log = max(log_files, key=os.path.getmtime)
    with open(latest_log, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return "".join(lines[-limit:])


# --- TAB NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Strategy Lab", 
    "🛠️ Grid Search", 
    "📜 Mnemosyne", 
    "🎮 Zeus Cockpit"
])

with tab1:
    st.header("📈 Strategy Lab & Backtest")
    
    # 1. Initialize Strategy for Schema
    strategy_engine = Odysseus() 
    schema = strategy_engine.get_params_schema()
    
    # --- DATA SETTINGS ---
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        t1_symbol = st.text_input("📊 Symbol", "BTC-USDT-SWAP", key="t1_sym_input")
    with col_input2:
        t1_tf = st.selectbox("⏱️ Timeframe", TIMEFRAMES, index=3, key="t1_tf_select")
    with col_input3:
        data_source = st.radio("Data Source", ["📡 Live (Fast)", "💾 Local CSV (Full)"], horizontal=True, key="t1_source_radio")

    # Data Download Logic
    if "Local" in data_source:
        t1_limit = st.number_input("Candles to Download", value=50000, step=10000, min_value=1000, key="t1_download_limit")
        if st.button("📥 Download & Update Local Data", use_container_width=True):
            with st.spinner("Downloading..."):
                df_downloaded = deep_history_download(t1_symbol, t1_tf, total_limit=t1_limit)
                st.success(f"Saved: data_{t1_symbol}_{t1_tf}.csv")
    else:
        t1_limit = st.slider("Live Candle Count", 100, 2000, 1000, key="t1_live_slider")

    st.divider()
    
    # --- STRATEGY INPUTS ---
    st.subheader("🤖 Strategy Parameters")
    dynamic_params = {}
    cols = st.columns(3)
    keys = list(schema.keys())
    
    for i, key in enumerate(keys):
        props = schema[key]
        col_idx = i % 3
        with cols[col_idx]:
            if props["type"] == "int":
                val = st.number_input(props["label"], value=int(props["default"]), min_value=int(props.get("min", 1)), max_value=int(props.get("max", 1000)), step=1, key=f"t1_param_{key}")
            else:
                val = st.number_input(props["label"], value=float(props["default"]), min_value=float(props.get("min", 0.0)), max_value=float(props.get("max", 1000.0)), step=float(props.get("step", 0.1)), format="%.2f", key=f"t1_param_{key}")
            dynamic_params[key] = val
    
    # --- GLOBAL FILTERS (CRITICAL FIX) ---
    st.markdown("**🛡️ Global Filters**")
    t1_enable_tp = st.checkbox("Enable TP", value=True, key="t1_tp_active")
    
    st.divider()
    
    # --- RUN BACKTEST ---
    if st.button("🚀 Launch Backtest", type="primary", use_container_width=True):
        df = None
        
        # 1. Data Prep
        if "Local" in data_source:
            filename = f"data_{t1_symbol}_{t1_tf}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                st.error("❌ File not found! Download data first.")
        else:
            with st.spinner("Fetching live data..."):
                try:
                    import ccxt
                    exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}, 'verify': False})
                    ohlcv = exchange.fetch_ohlcv(t1_symbol, t1_tf, limit=t1_limit + 100)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                except Exception as e:
                    st.error(f"Exchange Error: {e}")

        # 2. Process Strategy & Run Simulation
        if df is not None:
            with st.spinner("Calculating..."):
                tester = Odysseus(dynamic_params)
                df_results = tester.calculate(df)
                
                # --- CONNECTED TO UI VARIABLE ---
                # 
                sim_result = backtest_run(
                    df_results, 
                    start_balance=1000, 
                    leverage=20, 
                    enable_tp=t1_enable_tp, 
                    commission_rate=0.05
                )
                
            # 3. Display Metrics
            st.success("✅ Backtest Finished!")
            m1, m2, m3, m4, m5 = st.columns(5)
            net_profit = sim_result['Net Profit']
            m1.metric("💰 Net Profit", f"${net_profit:.2f}", delta_color="normal" if net_profit >= 0 else "inverse")
            m2.metric("📊 Trades", sim_result['Trade Count'])
            m3.metric("🎯 Win Rate", f"{sim_result['Win Rate']:.1f}%")
            m4.metric("💸 Commission", f"${sim_result['Total Commission']:.2f}")
            m5.metric("💵 Final Balance", f"${sim_result['Final Balance']:.2f}")
            
            # --- 4. TRADINGVIEW CHART ---
            st.divider()
            
            # Clean visuals
            plot_df = df_results.copy()
            for col in ['fast_ma', 'slow_ma', 'sl_price']:
                if col in plot_df.columns:
                    plot_df.loc[plot_df[col] < 0.00001, col] = np.nan
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=plot_df['timestamp'], open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='Price'))
            
            # Indicators
            if 'fast_ma' in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df['fast_ma'], line=dict(color='cyan', width=1), name='Fast MA'))
            if 'slow_ma' in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df['slow_ma'], line=dict(color='orange', width=1.5), name='Slow MA'))
            
            # Markers
            trade_log = sim_result.get('Log', [])
            if trade_log:
                longs = [t for t in trade_log if t['action']=='ENTRY' and t['type']=='LONG']
                shorts = [t for t in trade_log if t['action']=='ENTRY' and t['type']=='SHORT']
                exits = [t for t in trade_log if t['action']=='EXIT']
                
                if longs: fig.add_trace(go.Scatter(x=[t['timestamp'] for t in longs], y=[t['price'] for t in longs], mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00E676'), name='LONG'))
                if shorts: fig.add_trace(go.Scatter(x=[t['timestamp'] for t in shorts], y=[t['price'] for t in shorts], mode='markers', marker=dict(symbol='triangle-down', size=12, color='#FF1744'), name='SHORT'))
                if exits: fig.add_trace(go.Scatter(x=[t['timestamp'] for t in exits], y=[t['price'] for t in exits], mode='markers', marker=dict(symbol='x', size=8, color='white'), name='Exit'))

            fig.update_layout(height=750, template='plotly_dark', xaxis_rangeslider_visible=True, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🛠️ Grid Search - Strategy Optimization")
    
    # 1. Initialize Strategy for Schema
    gs_strategy = Odysseus() 
    schema = gs_strategy.get_params_schema()
    
    col_set1, col_set2 = st.columns([1, 2])
    
    with col_set1:
        st.subheader("1. Scan Configuration")
        
        # A) Data Source Selection
        gs_source = st.radio(
            "Data Source", 
            ["📡 Live (Fast - Max 2k)", "💾 Local CSV (Deep - Max 100k)"],
            help="To use local files, download the data in Tab 1 first."
        )

        # B) Assets & Timeframes
        gs_symbols = st.text_area("Symbols (Comma separated)", "BTC-USDT-SWAP,ETH-USDT-SWAP", height=70)
        gs_tfs = st.multiselect("Timeframes", TIMEFRAMES, default=["15m"])
        
        # Limit adjustment based on source
        if "Local" in gs_source:
            gs_limit = st.number_input("Candles to Test", value=50000, step=5000, min_value=1000)
        else:
            gs_limit = st.slider("Data Amount (Candles)", 200, 2000, 1000)
        
        st.divider()
        st.markdown("**🤖 Parameter Ranges**")
        
        # --- DYNAMIC PARAMETER RANGES (FIXED) ---
        grid_ranges = {}
        for key, props in schema.items():
            if props["type"] in ["int", "float"]:
                with st.expander(f"🔹 {props['label']} ({key})", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    
                    is_int = props["type"] == "int"
                    
                    # Force types to match to avoid StreamlitMixedNumericTypesError
                    if is_int:
                        def_val = int(props["default"])
                        step_d = 1
                        min_step = 1
                    else:
                        def_val = float(props["default"])
                        step_d = float(props.get("step", 0.1))
                        min_step = 0.01

                    # Inputs
                    g_min = c1.number_input("Min", value=def_val, step=step_d, key=f"gs_min_{key}")
                    g_max = c2.number_input("Max", value=def_val, step=step_d, key=f"gs_max_{key}")
                    g_step = c3.number_input("Step", value=step_d, min_value=min_step, key=f"gs_step_{key}")
                    
                    # Range Creation
                    if is_int:
                        r = np.arange(int(g_min), int(g_max) + 1, max(1, int(g_step)), dtype=int)
                    else:
                        r = np.arange(g_min, g_max + (g_step/1000), g_step)
                        r = np.round(r, 2)
                    
                    grid_ranges[key] = r
                    st.caption(f"Iterations: {len(r)}")

        st.divider()
        st.markdown("**🛡️ Global Filters**")
        
        # Backtest Constants (TP Range Support)
        tp_options = st.multiselect(
            "TP Optimization", 
            options=["Enabled", "Disabled"], 
            default=["Enabled", "Disabled"],
            help="Select both to test strategy with and without Take Profit."
        )
        gs_tp_range = []
        if "Enabled" in tp_options: gs_tp_range.append(True)
        if "Disabled" in tp_options: gs_tp_range.append(False)

        gs_commission = st.number_input("Commission %", value=0.05, step=0.01)
        
        start_scan = st.button("🚀 Start Grid Search", type="primary", use_container_width=True)
    
    with col_set2:
        if start_scan:
            # 1. Prepare Inputs
            symbol_list = [s.strip() for s in gs_symbols.split(",") if s.strip()]
            
            if not symbol_list:
                st.error("❌ Please enter at least one symbol.")
            elif not gs_tfs:
                st.error("❌ Please select at least one timeframe.")
            elif not gs_tp_range:
                st.error("❌ Please select at least one TP option.")
            else:
                import itertools
                import time
                
                # 2. Prepare Combinations
                keys = list(grid_ranges.keys())
                values = list(grid_ranges.values())
                param_combinations = list(itertools.product(*values))
                
                # FULL GRID: (Symbol, Timeframe, TP_Status, Strategy_Params)
                full_grid = list(itertools.product(symbol_list, gs_tfs, gs_tp_range, param_combinations))
                total_combs = len(full_grid)
                
                st.info(f"📊 Scanning {total_combs} combinations. Source: {gs_source}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                gs_results = []
                data_cache = {} 
                
                start_time = time.time()
                
                try:
                    for idx, (symbol, tf, tp_status, params_tuple) in enumerate(full_grid):
                        current_params = dict(zip(keys, params_tuple))
                        
                        if idx % 5 == 0:
                            status_text.text(f"⏳ Processing ({idx+1}/{total_combs}): {symbol} | {tf} | TP: {tp_status}")
                            progress_bar.progress(min((idx + 1) / total_combs, 1.0))
                        
                        # --- A) DATA FETCHING (WITH CACHING) ---
                        cache_key = f"{symbol}_{tf}"
                        if cache_key not in data_cache:
                            df_raw = None
                            if "Local" in gs_source:
                                filename = f"data_{symbol}_{tf}.csv"
                                if os.path.exists(filename):
                                    df_raw = pd.read_csv(filename)
                                    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                                    if len(df_raw) > gs_limit:
                                        df_raw = df_raw.iloc[-int(gs_limit):].reset_index(drop=True)
                                else:
                                    if idx == 0: st.warning(f"File missing: {filename}")
                            else:
                                import ccxt
                                exchange = ccxt.okx({'timeout': 10000, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
                                ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=int(gs_limit))
                                df_raw = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
                            
                            if df_raw is not None:
                                data_cache[cache_key] = df_raw
                        
                        # --- B) EXECUTION ---
                        if cache_key in data_cache:
                            df_base = data_cache[cache_key].copy()
                            tester = Odysseus(current_params)
                            df_calc = tester.calculate(df_base)
                            
                            res = backtest_run(
                                df_calc,
                                start_balance=1000,
                                leverage=20,
                                enable_tp=tp_status,
                                commission_rate=gs_commission
                            )
                            
                            res_row = {
                                "Symbol": symbol,
                                "TF": tf,
                                "TP": "ON" if tp_status else "OFF",
                                "Net Profit ($)": round(res["Net Profit"], 2),
                                "Win Rate %": round(res["Win Rate"], 1),
                                "Trades": res["Trade Count"]
                            }
                            res_row.update(current_params)
                            gs_results.append(res_row)
                    
                    # --- C) REPORTING ---
                    progress_bar.progress(1.0)
                    elapsed = time.time() - start_time
                    status_text.success(f"✅ Scan Finished in {elapsed:.1f}s")
                    
                    if gs_results:
                        df_final = pd.DataFrame(gs_results)
                        metrics = ["Symbol", "TF", "TP", "Net Profit ($)", "Win Rate %", "Trades"]
                        param_cols = [k for k in df_final.columns if k not in metrics]
                        df_final = df_final[metrics + param_cols]
                        
                        st.subheader("🏆 Leaderboard")
                        st.dataframe(
                            df_final.sort_values(by="Net Profit ($)", ascending=False).head(100)
                            .style.background_gradient(subset=["Net Profit ($)"], cmap="RdYlGn"),
                            use_container_width=True
                        )
                        
                        csv_data = df_final.to_csv(index=False)
                        st.download_button("📥 Download Results (CSV)", data=csv_data, file_name=f"grid_results_{int(time.time())}.csv")
                        
                        varying_keys = [k for k, v in grid_ranges.items() if len(v) > 1]
                        if len(varying_keys) >= 2:
                            st.subheader(f"🔥 Heatmap: {varying_keys[0]} vs {varying_keys[1]}")
                            pivot = df_final.pivot_table(index=varying_keys[1], columns=varying_keys[0], values="Net Profit ($)", aggfunc='mean')
                            import plotly.graph_objects as go
                            fig_hm = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Viridis'))
                            st.plotly_chart(fig_hm, use_container_width=True)
                    else:
                        st.warning("No results were generated.")

                except Exception as e:
                    st.error(f"Critical Grid Search Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
with tab3:
    st.header("📜 Mnemosyne: System Memory & Performance")
    
    # Fetch data from the updated 'trades' table
    df_db = get_db_history() # English-named utility function
    
    # Layout: Left (Performance & Charts), Right (System Logs)
    col_hist, col_logs = st.columns([3, 1])
    
    with col_hist:
        st.subheader("💰 Trade History & Analysis")
        
        if df_db.empty:
            st.info("📭 No trade history found. Mnemosyne will start recording once trades are executed.")
        else:
            # --- 1. FILTERS ---
            c_flt1, c_flt2 = st.columns(2)
            with c_flt1:
                unique_symbols = ["All"] + list(df_db['symbol'].unique())
                selected_symbol = st.selectbox("Filter by Symbol", unique_symbols)
            with c_flt2:
                unique_status = ["All", "OPEN", "CLOSED"]
                selected_status = st.selectbox("Filter by Status", unique_status)
            
            # Filtering Logic
            df_show = df_db.copy()
            if selected_symbol != "All":
                df_show = df_show[df_show['symbol'] == selected_symbol]
            if selected_status != "All":
                df_show = df_show[df_show['status'] == selected_status]

            # --- 2. KEY METRICS ---
            closed_trades = df_show[df_show['status'] == 'CLOSED']
            if not closed_trades.empty:
                total_pnl = closed_trades['pnl'].sum()
                wins = len(closed_trades[closed_trades['pnl'] > 0])
                total_count = len(closed_trades)
                wr = (wins / total_count * 100)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total PnL", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")
                m2.metric("Total Trades", total_count)
                m3.metric("Win Rate", f"%{wr:.1f}")
                # Handling date display for the last trade
                last_exit = closed_trades.iloc[0]['exit_time']
                m4.metric("Last Exit", last_exit[11:16] if isinstance(last_exit, str) else "-")
                
                st.divider()

                # --- 3. EQUITY CURVE (PnL Growth) ---
                # Sort by exit time for chronological chart
                df_chart = closed_trades.sort_values(by="exit_time", ascending=True).copy()
                df_chart['cumulative_pnl'] = df_chart['pnl'].cumsum()
                
                fig_equity = go.Figure()
                
                # Area Chart for Equity
                fig_equity.add_trace(go.Scatter(
                    x=df_chart['exit_time'], 
                    y=df_chart['cumulative_pnl'],
                    mode='lines+markers',
                    fill='tozeroy', 
                    name='Cumulative PnL',
                    line=dict(color='#00e676' if total_pnl >= 0 else '#ff1744', width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Total PnL:</b> $%{y:.2f}<extra></extra>'
                ))
                
                fig_equity.update_layout(
                    title="📈 Equity Curve (Cumulative PnL)",
                    height=300,
                    template='plotly_dark',
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="",
                    yaxis_title="USDT",
                    hovermode="x unified",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_equity, use_container_width=True)
            
            # --- 4. DETAILED LOG TABLE ---
            st.subheader("📋 Execution Details")
            
            # Professional Formatting & Coloring
            def style_trades(row):
                styles = [''] * len(row)
                # PnL Coloring
                if row['status'] == 'CLOSED':
                    pnl_idx = row.index.get_loc('pnl')
                    styles[pnl_idx] = 'color: #00e676' if row['pnl'] > 0 else 'color: #ff1744'
                
                # Side Coloring
                side_idx = row.index.get_loc('side')
                if row['side'] == 'LONG':
                    styles[side_idx] = 'color: #29b6f6; font-weight: bold;'
                else:
                    styles[side_idx] = 'color: #ffca28; font-weight: bold;'
                return styles

            st.dataframe(
                df_show.style.apply(style_trades, axis=1)
                       .format({
                           "entry_price": "{:.4f}", 
                           "exit_price": "{:.4f}", 
                           "pnl": "{:+.2f}", 
                           "amount": "{:.4f}",
                           "commission": "{:.4f}"
                       }),
                use_container_width=True,
                height=400
            )

    # --- RIGHT PANEL: SYSTEM LOGS ---
    with col_logs:
        st.subheader("🛠️ System Logs")
        
        # Log Controls
        l_btn1, l_btn2 = st.columns(2)
        with l_btn1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with l_btn2:
            # log_fetcher is our standardized utility
            full_logs = fetch_logs(limit=1000) 
            st.download_button(
                "💾 Export", 
                data=full_logs, 
                file_name=f"olympus_system_log_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Log Viewer with auto-scroll focus
        display_logs = fetch_logs(limit=50)
        st.text_area(
            "Recent Activity (Last 50 Lines)",
            value=display_logs,
            height=600,
            disabled=True,
            help="Real-time system events and error tracking."
        )

with tab4:
    st.header("🎮 Zeus Command Center")
    
    # 1. Load Strategy Schema and Current Configuration
    # We use Odysseus to dynamically generate the UI based on its internal schema
    temp_odysseus = Odysseus() 
    strategy_schema = temp_odysseus.get_params_schema()
    
    # Load existing configuration if it exists
    current_config = {}
    if os.path.exists("bot_config.json"):
        try:
            with open("bot_config.json", "r") as f:
                current_config = json.load(f)
        except:
            pass
            
    active_strategy_settings = current_config.get("strategy", {})
    
    col_left, col_right = st.columns([1, 1])
    
    # --- LEFT COLUMN: LIVE PARAMETERS ---
    with col_left:
        st.subheader("1. Target & Parameters")
        
        # A) Assets and Timeframe
        selected_symbols = st.text_area(
            "Symbols to Trade (Comma separated)", 
            value=",".join(current_config.get("coins", ["BTC-USDT-SWAP"])),
            height=70,
            help="Enter parities like BTC-USDT-SWAP, ETH-USDT-SWAP"
        )
        
        selected_tf = st.selectbox(
            "Timeframe", TIMEFRAMES, 
            index=TIMEFRAMES.index(current_config.get("timeframe", "15m")) if current_config.get("timeframe") in TIMEFRAMES else 3
        )
        
        st.divider()
        
        # B) Dynamic Strategy Parameters (Fetched from Odysseus)
        st.markdown("**🤖 Strategy Settings** (Live Values)")
        
        live_params = {}
        param_cols = st.columns(2)
        
        for i, (key, props) in enumerate(strategy_schema.items()):
            col_idx = i % 2
            
            # Default value priority: bot_config.json -> Odysseus Default
            saved_val = active_strategy_settings.get(key)
            default_val = saved_val if saved_val is not None else props["default"]
            
            with param_cols[col_idx]:
                if props["type"] == "int":
                    val = st.number_input(
                        props["label"], 
                        value=int(default_val),
                        min_value=props.get("min"), 
                        max_value=props.get("max"),
                        step=1,
                        key=f"live_ui_{key}"
                    )
                else:
                    val = st.number_input(
                        props["label"], 
                        value=float(default_val),
                        min_value=float(props.get("min", 0.0)), 
                        max_value=float(props.get("max", 100.0)),
                        step=props.get("step", 0.1),
                        format="%.2f",
                        key=f"live_ui_{key}"
                    )
                live_params[key] = val

        st.divider()
        
        # C) Risk & Application Settings
        st.markdown("**🛡️ Risk Management**")
        risk_c1, risk_c2 = st.columns(2)
        
        with risk_c1:
            live_risk_percent = st.number_input(
                "Risk % (Per Trade)", 
                value=float(active_strategy_settings.get("risk_percent", 1.0)), 
                step=0.1, format="%.1f"
            )
        with risk_c2:
            live_tp_active = st.checkbox(
                "Enable TP", 
                value=bool(active_strategy_settings.get("enable_tp", True))
            )

    # --- RIGHT COLUMN: ENGINE CONTROL & STATUS ---
    with col_right:
        st.subheader("2. Engine Control")
        
        # Monitor Bot Process
        is_active = False
        bot_pid = None
        bot_start_time = None
        
        if os.path.exists("bot.pid"):
            try:
                with open("bot.pid", "r") as f:
                    pid_content = f.read().strip().split("|")
                    bot_pid = int(pid_content[0])
                    if len(pid_content) > 1: bot_start_time = pid_content[1]
                
                # Check if process is still alive
                os.kill(bot_pid, 0) 
                is_active = True
            except:
                if os.path.exists("bot.pid"): os.remove("bot.pid")
        
        # Status Visualization
        status_box = st.container(border=True)
        if is_active:
            status_box.success(f"✅ ZEUS IS ONLINE (PID: {bot_pid})")
            status_box.caption(f"🚀 Started at: {bot_start_time}")
            status_box.markdown("**📡 Live Logs (Last 5):**")
            # Assuming get_recent_logs() is our standardized log utility
            status_box.code(fetch_logs(limit=5), language="text")
        else:
            status_box.error("🔴 ZEUS IS OFFLINE")
            status_box.info("Configure the settings on the left and launch the engine.")

        st.write("") # Spacer
        
        # Control Buttons
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("🚀 SAVE & LAUNCH", type="primary", use_container_width=True):
                target_coins = [c.strip() for c in selected_symbols.split(",") if c.strip()]
                
                if not target_coins:
                    st.error("❌ Symbol list cannot be empty!")
                else:
                    # 1. Prepare Final Config Data
                    strategy_data = live_params.copy()
                    strategy_data["risk_percent"] = live_risk_percent
                    strategy_data["enable_tp"] = live_tp_active
                    
                    final_config = {
                        "coins": target_coins,
                        "timeframe": selected_tf,
                        "strategy": strategy_data
                    }
                    
                    # 2. Save to Disk
                    with open("bot_config.json", "w") as f:
                        json.dump(final_config, f, indent=4)
                    
                    # 3. Restart Logic (Kill if already running)
                    if is_active and bot_pid:
                        try:
                            os.kill(bot_pid, signal.SIGTERM)
                            time.sleep(1)
                        except: pass
                    
                    # 4. Launch Zeus Engine
                    try:
                        # Standardized launch command
                        new_proc = subprocess.Popen(
                            ["python", "zeus.py"],
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
                        )
                        
                        launch_ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        with open("bot.pid", "w") as f:
                            f.write(f"{new_proc.pid}|{launch_ts}")
                        
                        st.toast("⚡ Zeus has been summoned!", icon="✅")
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Launch Failure: {e}")

        with btn_col2:
            if st.button("🛑 SHUTDOWN", type="secondary", disabled=not is_active, use_container_width=True):
                if bot_pid:
                    try:
                        os.kill(bot_pid, signal.SIGTERM)
                    except: pass
                    
                    if os.path.exists("bot.pid"): os.remove("bot.pid")
                    
                    st.toast("🛑 Zeus has returned to sleep.", icon="⚠️")
                    time.sleep(1)
                    st.rerun()
