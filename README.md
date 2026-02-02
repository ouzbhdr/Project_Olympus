# Project_Olympus
Infrastructure for a mid-level trading bot system.

⚡ Project Olympus: Modular Algo-Trading Framework
Project Olympus is a professional-grade, modular algorithmic trading skeleton built with Python and Streamlit. Designed for the OKX exchange (Futures/Swap), it decouples strategy logic from execution, providing a robust environment for backtesting, optimization, and live trading.

🏗️ Architecture (The Gods of Olympus)
The project is divided into specialized modules named after Greek deities:

Zeus: The King. Controls the main execution loop and bot life cycle.

Odysseus: The Strategist. Contains the trading logic and technical indicators.

Orion: The Hunter. Handles order execution, position management, and exchange connectivity.

Mnemosyne: The Memory. Manages SQLite database for trade logs and performance history.

Hekate: The Seer. Responsible for high-fidelity data fetching and processing.

Aether: The Atmosphere. A high-performance Streamlit dashboard for monitoring and analysis.

Hermes: The Messenger. Delivers real-time notifications via Telegram.

🚀 Key Features
Strategy Lab: Single-asset backtesting with interactive Plotly charts.

Deep Grid Search: Multi-symbol, multi-timeframe parameter optimization with Heatmap visualization.

Dynamic UI: The dashboard automatically generates input fields based on your strategy's schema. You have to figure out what the strategy is yourself.

Hedge Mode Support: Fully compatible with OKX's two-way position mode.

Execution Safety: Built-in precision handling and error catching for orders.

🧠 Creating Your Custom Strategy
Project Olympus uses a dynamic schema system. When you define parameters in odysseus.py, the Aether dashboard automatically generates the corresponding UI elements.

1. Define Your Parameters
In odysseus.py, update the get_params_schema() method. Supported types are int and float.

def get_params_schema(self):
    return {
        "fast_ma": {"label": "Fast MA Period", "type": "int", "default": 14, "min": 5, "max": 50},
        "slow_ma": {"label": "Slow MA Period", "type": "int", "default": 50, "min": 20, "max": 200},
        "threshold": {"label": "Signal Threshold", "type": "float", "default": 0.05, "step": 0.01}
    }

2. Implement the Logic
Update the calculate(self, df) method. Your logic must populate these 3 columns for the backtester and live bot to work:

df['signal']: 1 (Long), -1 (Short), 0 (None)

df['sl_price']: Stop loss price for the position.

df['tp_price']: Take profit price for the position.

3. Automatic Integration
Once saved, restart the dashboard. You will see your new parameters under the "Strategy Parameters" section in both Strategy Lab and Grid Search tabs.

streamlit run aether.py


