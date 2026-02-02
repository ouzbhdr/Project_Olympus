"""
ODYSSEUS - STRATEGY ENGINE (MA CROSS)
=====================================
Standardized skeleton for Project Olympus. 
Calculates signals based on Moving Average crossovers.
"""
import pandas as pd
import numpy as np

class Odysseus:
    def __init__(self, user_config=None):
        # 1. PARAMETER SCHEMA (Feeds the Aether UI)
        self.param_schema = {
            "fast_period": {"type": "int",   "default": 14,  "min": 1, "max": 100, "label": "Fast MA Period"},
            "slow_period": {"type": "int",   "default": 50,  "min": 1, "max": 200, "label": "Slow MA Period"},
            "rr_ratio":    {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1, "label": "Risk/Reward Ratio"}
        }

        # 2. LOAD CONFIGURATION
        self.config = {}
        for key, props in self.param_schema.items():
            val = user_config.get(key) if user_config else None
            if val is None: val = props["default"]
            self.config[key] = int(val) if props["type"] == "int" else float(val)
        
        self.warmup = self.config["slow_period"] + 10

    def get_params_schema(self):
        return self.param_schema

    def calculate(self, df):
        if len(df) < self.warmup:
            return self._empty_df(df)
        
        df = df.copy()
        
        # --- 1. INDICATOR CALCULATIONS ---
        df['fast_ma'] = df['close'].rolling(window=self.config["fast_period"]).mean()
        df['slow_ma'] = df['close'].rolling(window=self.config["slow_period"]).mean()
        
        # --- 2. SIGNAL LOGIC (MA Cross) ---
        df['signal'] = 0
        
        # Bullish Cross (Golden Cross)
        df.loc[(df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)), 'signal'] = 1
        
        # Bearish Cross (Death Cross)
        df.loc[(df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)), 'signal'] = -1

        # --- 3. RISK MANAGEMENT (SL / TP) ---
        # Using Slow MA as a Trailing Stop (Dynamic)
        df['sl_price'] = df['slow_ma']
        df['tp_price'] = 0.0 # Placeholder for static TP
        
        # Technical values for visualization
        df['s2'] = df['slow_ma'] 

        # Clean up warmup bars
        df.iloc[:self.warmup, df.columns.get_loc('signal')] = 0
        
        return df

    def _empty_df(self, df):
        cols = ['signal', 'sl_price', 'tp_price', 's2', 'fast_ma', 'slow_ma']
        for c in cols: df[c] = 0.0
        return df
