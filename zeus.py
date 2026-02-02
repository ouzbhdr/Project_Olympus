import asyncio
import pandas as pd
import json
import os
import config
from datetime import datetime

# --- MODULES (Gods) ---
from hekate import Hekate           
from odysseus import Odysseus       
from orion import Orion     
from hermes import Hermes   
from mnemosyne import Mnemosyne     

class Zeus:
    def __init__(self):
        print("⚡ ZEUS: Ascending to the throne on Mount Olympus...")
        
        # 1. LOAD CONFIGURATION
        self.config_data = self._load_config()
        
        # Update global config module
        if "risk_percent" in self.config_data.get("strategy", {}):
            config.RISK_PERCENT = float(self.config_data["strategy"]["risk_percent"])
        
        self.symbols = self.config_data.get("coins", ["BTC-USDT-SWAP"])
        self.timeframe = self.config_data.get("timeframe", "15m")
        self.strategy_settings = self.config_data.get("strategy", {})

        # 2. INITIALIZE GODS
        self.mnemosyne = Mnemosyne()
        self.hermes = Hermes()
        self.orion = Orion(config)
        
        self.agents = {}
        for symbol in self.symbols:
            self.agents[symbol] = {
                "hekate": Hekate({'symbol': symbol, 'timeframe': self.timeframe}),
                "odysseus": Odysseus(self.strategy_settings)
            }
        
        print(f"⚙️ Config: {len(self.symbols)} Symbols | {self.timeframe} Timeframe")

    def _load_config(self):
        if os.path.exists("bot_config.json"):
            with open("bot_config.json", "r") as f:
                return json.load(f)
        return {"coins": ["BTC-USDT-SWAP"], "timeframe": "15m", "strategy": {}}

    async def decision_mechanism(self, symbol, df, bar_closed):
        """
        The core engine. Decides whether to open or close trades.
        """
        try:
            agent = self.agents[symbol]
            odysseus = agent["odysseus"]
            
            # 1. CALCULATE STRATEGY (English Method)
            df_with_signals = odysseus.calculate(df)
            last_row = df_with_signals.iloc[-1]
            signal = last_row['signal']
            
            # 2. CHECK ACTIVE POSITION
            active_pos = self.mnemosyne.aktif_pozisyon_getir(symbol) # Keeping DB method for now
            
            # 3. ACTION LOGIC
            # Case A: Close position if signal changes
            if active_pos:
                should_close = False
                if active_pos['tip'] == 'LONG' and signal == -1: should_close = True
                elif active_pos['tip'] == 'SHORT' and signal == 1: should_close = True
                
                if should_close:
                    await self._execute_close(symbol, active_pos, last_row['close'], "SIGNAL REVERSAL")
                    active_pos = None

            # Case B: Open position if signal exists and no active position
            if not active_pos and signal != 0:
                side = "LONG" if signal == 1 else "SHORT"
                await self._execute_open(symbol, side, last_row)

        except Exception as e:
            print(f"❌ ZEUS ERROR ({symbol}): {e}")
            self.hermes._gonder(f"⚠️ ZEUS CRITICAL ERROR: {e}")

    async def _execute_open(self, symbol, side, row):
        price = row['close']
        sl = row['sl_price']
        
        # Calculate amount (simplified for skeleton)
        amount = 0.01 # This would be calculated via Orion based on risk
        
        success = await self.orion.islem_ac(symbol, side, amount, sl)
        if success:
            self.mnemosyne.islem_kaydet(symbol, side, price, sl, amount)
            self.hermes.islem_bildirimi(symbol, side, price, amount, "ENTRY")
            print(f"🚀 {side} POSITION OPENED: {symbol} @ {price}")

    async def _execute_close(self, symbol, pos, price, reason):
        success = await self.orion.islem_kapat(symbol, pos['tip'], pos['miktar'])
        if success:
            # PnL Calculation
            pnl = (price - pos['giris_fiyati']) * pos['miktar'] if pos['tip'] == 'LONG' else (pos['giris_fiyati'] - price) * pos['miktar']
            self.mnemosyne.islem_kapat(pos['id'], price, pnl)
            self.hermes.islem_bildirimi(symbol, pos['tip'], price, pos['miktar'], "EXIT")
            print(f"✅ POSITION CLOSED: {pnl:.2f} USDT ({reason})")

    async def start(self):
        print("\n⚡ ZEUS: System Starting...")
        self.hermes._gonder("⚡ <b>PROJECT OLYMPUS V2 ACTIVE</b>")
        
        # Test Connection
        if not await self.orion.borsa_baglantisini_test_et():
            return

        tasks = []
        for symbol, agents in self.agents.items():
            hekate = agents["hekate"]
            await hekate.gecmisi_topla()
            
            async def wrapper(df, bar_closed, s=symbol):
                await self.decision_mechanism(s, df, bar_closed)
            
            tasks.append(hekate.gozculuge_basla(wrapper))

        await asyncio.gather(*tasks)

if __name__ == "__main__":
    zeus = Zeus()
    asyncio.run(zeus.start())
