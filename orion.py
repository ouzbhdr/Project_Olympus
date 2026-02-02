import ccxt
import config
import asyncio
import functools
import warnings
import urllib3

# --- 🔇 SILENCE MODES ---
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Orion:
    def __init__(self, bot_config):
        """
        ORION: The Order Execution Engine.
        Handles API connections, risk calculation, and order placement.
        """
        self.config = bot_config 
        self.exchange = ccxt.okx({
            'apiKey': bot_config.API_KEY, 
            'secret': bot_config.API_SECRET, 
            'password': bot_config.API_PASSPHRASE,
            'enableRateLimit': True, 
            'timeout': 30000, 
            'options': {'defaultType': 'swap'}, 
            'verify': False
        })
        self.is_hedge_mode = False 

    async def _run_sync(self, func, *args, **kwargs):
        """Asynchronous bridge for synchronous CCXT calls."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    async def test_connection(self):
        """Tests exchange connectivity and sets account modes."""
        try:
            print("🏹 ORION: Checking weapons and armor...")
            await self._run_sync(self.exchange.load_markets)
            
            # --- ACCOUNT MODE SETUP ---
            try:
                # Attempt to set Hedge Mode (Long/Short positions simultaneously)
                await self._run_sync(self.exchange.set_position_mode, 'long_short_mode')
                self.is_hedge_mode = True
                print("   ✅ Mode: HEDGE (Long/Short)")
            except Exception:
                print("   🔄 Mode: NET (One-Way) - Defaulting.")
                self.is_hedge_mode = False

            # Set Leverage
            try:
                await self._run_sync(self.exchange.set_leverage, self.config.LEVERAGE, self.config.SYMBOL)
                print(f"   ✅ Leverage: {self.config.LEVERAGE}x")
            except: pass 

            print("✅ ORION: Ready for duty.")
            return True
        except Exception as e:
            print(f"❌ ORION CONNECTION ERROR: {e}")
            return False

    async def open_position(self, symbol, side, amount, stop_loss_price):
        """
        Opens a market order and sets a stop-loss.
        side: 'LONG' or 'SHORT'
        """
        try:
            # Convert internal terms to CCXT terms
            order_side = 'buy' if side == 'LONG' else 'sell'
            pos_side = 'long' if side == 'LONG' else 'short'
            
            print(f"\n⚔️ ORION FIRING: {symbol} {side} | Amount: {amount}")

            # Prepare Params based on Mode
            params = {'tdMode': 'cross'}
            if self.is_hedge_mode:
                params['posSide'] = pos_side

            # 1. Place Market Order
            order = await self._run_sync(
                self.exchange.create_market_order, 
                symbol, order_side, amount, params
            )

            # 2. Place Stop Loss Order
            if order:
                sl_trigger = self.exchange.price_to_precision(symbol, stop_loss_price)
                sl_params = {'tdMode': 'cross', 'reduceOnly': True, 'triggerPrice': sl_trigger}
                if self.is_hedge_mode:
                    sl_params['posSide'] = pos_side
                
                # SL side is opposite of entry
                sl_side = 'sell' if side == 'LONG' else 'buy'
                
                try:
                    await self._run_sync(self.exchange.create_order, symbol, 'stop', amount, sl_trigger, sl_params)
                    print(f"   🛡️ Stop Loss set at {sl_trigger}.")
                except Exception as e:
                    print(f"   ⚠️ Could not set Stop Loss: {e}")

                return {
                    "success": True, 
                    "entry_price": order.get('average', 0),
                    "amount": amount
                }

        except Exception as e:
            print(f"⚠️ ORION EXECUTION ERROR: {e}")
            return {"success": False}

    async def close_position(self, symbol, side, amount):
        """
        Closes an active position using market order.
        """
        try:
            print(f"👋 ORION: Closing {side} position for {symbol}...")
            
            # Close side is opposite of current side
            close_side = 'sell' if side == 'LONG' else 'buy'
            pos_side = 'long' if side == 'LONG' else 'short'

            params = {'tdMode': 'cross', 'reduceOnly': True}
            if self.is_hedge_mode:
                params['posSide'] = pos_side

            await self._run_sync(self.exchange.create_market_order, symbol, close_side, amount, params)
            
            # Clean up pending orders (SL etc.)
            try:
                await self._run_sync(self.exchange.cancel_all_orders, symbol)
            except: pass

            print(f"✅ {symbol} {side} position closed successfully.")
            return True

        except Exception as e:
            print(f"⚠️ ORION CLOSE ERROR: {e}")
            return False
