# Project_Olympus
Infrastructure for a mid-level trading bot system.
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)

##⚡ Project Olympus: Modular Algo-Trading Framework
Project Olympus is a professional-grade, modular algorithmic trading skeleton built with Python and Streamlit. Designed for the OKX exchange (Futures/Swap), it decouples strategy logic from execution, providing a robust environment for backtesting, optimization, and live trading.

The software provided in "Project Olympus" is for educational and research purposes only. 
Trading cryptocurrencies involves significant risk and can result in the loss of your entire capital. 

* **Not Financial Advice:** The developers of this project are not financial advisors. 
* **Use at Your Own Risk:** Backtest results do not guarantee future performance. 
* **No Liability:** The authors and contributors shall not be held liable for any financial losses incurred through the use of this software. 
Always perform your own due diligence and never trade with money you cannot afford to lose.

## 🏗️ Architecture (The Gods of Olympus)
The project is divided into specialized modules named after Greek deities:

**Zeus**: The King. Controls the main execution loop and bot life cycle.
**Odysseus**: The Strategist. Contains the trading logic and technical indicators.
**Orion**: The Hunter. Handles order execution, position management, and exchange connectivity.
**Mnemosyne**: The Memory. Manages SQLite database for trade logs and performance history.
**Hekate**: The Seer. Responsible for high-fidelity data fetching and processing.
**Aether**: The Atmosphere. A high-performance Streamlit dashboard for monitoring and analysis.
**Hermes**: The Messenger. Delivers real-time notifications via Telegram.

## 🚀 Key Features
- Strategy Lab: Single-asset backtesting with interactive Plotly charts.
- Deep Grid Search: Multi-symbol, multi-timeframe parameter optimization with Heatmap visualization.
- Dynamic UI: The dashboard automatically generates input fields based on your strategy's schema. You have to figure out what the strategy is yourself.
- Hedge Mode Support: Fully compatible with OKX's two-way position mode.
- Execution Safety: Built-in precision handling and error catching for orders.


## 🧠 Custom Strategy Configuration

To add your own parameters, update `odysseus.py` using this syntax:

```python
def get_params_schema(self):
    return {
        "fast_ma": {"label": "Fast MA Period", "type": "int", "default": 14},
        "slow_ma": {"label": "Slow MA Period", "type": "int", "default": 50}
    }
```

2. Implement the Logic
Update the calculate(self, df) method. Your logic must populate these 3 columns for the backtester and live bot to work:
```python
df['signal']: 1 (Long), -1 (Short), 0 (None)
df['sl_price']: Stop loss price for the position.
df['tp_price']: Take profit price for the position.
```

3. Automatic Integration
Once saved, restart the dashboard. You will see your new parameters under the "Strategy Parameters" section in both Strategy Lab and Grid Search tabs.

## 📦 Installation 
Install requirements. 
Just run the code.

```bash
pip install -r requirements.txt

streamlit run aether.py
```
🏛️ Architectural Inspiration & Credits
The modular architecture and "ecosystem" philosophy of Project Olympus were inspired by the work of Melih Tuna and his comprehensive study, Trade Bot Ekosistemim.
The following core principles from his ecosystem model have been integrated into this framework:
Decoupled Task Management: Isolation of scanning (Hekate/Aether), hunting (Odysseus/Zeus), and execution (Orion) processes to ensure system stability.
Fault Tolerance: A design where each module operates independently, allowing the system to resume seamlessly after any interruption.
Transparent Monitoring: Real-time tracking of decision-making logic via a "Watcher" (Aether Dashboard) interface.
You can explore the original inspiration and source materials here:

Medium Article: https://medium.com/@a.melihtuna/trade-bot-ekosistemim-c1698177272c
GitHub Repository: melihtuna/trade-bot-ekosistemim 
https://github.com/melihtuna/trade-bot-ekosistemim/tree/master


