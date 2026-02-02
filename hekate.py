import ccxt
import pandas as pd
import asyncio
import functools
import warnings
import urllib3

# --- 🔇 SILENCE MODES ---
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
pd.options.mode.chained_assignment = None

class Hekate:
    def __init__(self, config_dict):
        """
        HEKATE: The Data Goddess.
        Solely responsible for fetching and preparing OHLCV data.
        """
        self.symbol = config_dict['symbol']
        self.timeframe = config_dict['timeframe']
        self.limit = config_dict.get('limit', 100)
        
        # Exchange Connection (Global Standards)
        exchange_config = {
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': 'swap'}, 
            'verify': False, 
        }
        
        # Inject API keys if provided in the config
        if 'apiKey' in config_dict:
            exchange_config.update({
                'apiKey': config_dict['apiKey'],
                'secret': config_dict['secret'],
                'password': config_dict['password'],
            })

        self.exchange = ccxt.okx(exchange_config)
        self.exchange.ssl_verification = False
        self.df = pd.DataFrame()

    async def _run_sync(self, func, *args, **kwargs):
        """Bridge to run synchronous CCXT calls in an async loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    async def fetch_history(self):
        """
        Initial data load when the bot starts.
        Populates self.df with historical candles.
        """
        try:
            print(f"🌑 HEKATE: Fetching historical data for {self.symbol}...")
            
            ohlcv = await self._run_sync(
                self.exchange.fetch_ohlcv, 
                self.symbol, 
                self.timeframe, 
                limit=self.limit
            )
            
            if not ohlcv:
                print(f"⚠️ HEKATE: Received empty data for {self.symbol}!")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure numeric integrity
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # Drop the last (unclosed) candle to keep strategy calculations accurate
            self.df = df.iloc[:-1]
            print(f"🌕 HEKATE: {len(self.df)} candles loaded for {self.symbol}.")
            return self.df

        except Exception as e:
            print(f"❌ HEKATE (History) Error: {e}")
            return None

    async def start_monitoring(self, on_tick_callback):
        """
        The main loop. Watches for new candles and triggers Zeus via callback.
        """
        print(f"👁️ HEKATE: Monitoring {self.symbol}...")
        
        while True:
            try:
                # Fetch last 2 candles (one closed, one live)
                ohlcv = await self._run_sync(
                    self.exchange.fetch_ohlcv, 
                    self.symbol, 
                    self.timeframe, 
                    limit=2
                )
                
                if ohlcv:
                    last_candle = ohlcv[-1]
                    ts = pd.to_datetime(last_candle[0], unit='ms')
                    
                    new_data = {
                        'timestamp': ts,
                        'open': float(last_candle[1]), 'high': float(last_candle[2]), 
                        'low': float(last_candle[3]), 'close': float(last_candle[4]), 
                        'volume': float(last_candle[5])
                    }
                    
                    # Detect Bar Closure
                    bar_closed = False
                    
                    if not self.df.empty and self.df.iloc[-1]['timestamp'] == ts:
                        # We are still in the same candle, update it
                        self.df.iloc[-1] = new_data
                    else:
                        # A new candle has appeared, meaning the previous one is now CLOSED
                        bar_closed = True
                        new_row_df = pd.DataFrame([new_data])
                        self.df = pd.concat([self.df, new_row_df], ignore_index=True).tail(self.limit)
                        print(f"🕯️ HEKATE: New Candle Closed -> {ts}")

                    # Trigger the callback (Zeus)
                    await on_tick_callback(self.df, bar_closed)

                # Wait to prevent rate limiting
                await asyncio.sleep(5) 

            except Exception as e:
                print(f"⚠️ HEKATE (Live) Warning: {e}")
                await asyncio.sleep(10)
