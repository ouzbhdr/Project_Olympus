import requests
from datetime import datetime
import config

class Hermes:
    def __init__(self):
        """
        HERMES: The Messenger of the Gods.
        Responsible for sending real-time notifications via Telegram.
        """
        self.base_url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        self.chat_id = config.TELEGRAM_CHAT_ID

    def _send(self, message):
        """Internal function to dispatch requests to Telegram API."""
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            # Timeout is critical to prevent the bot from hanging if Telegram is slow
            requests.post(self.base_url, data=payload, timeout=5)
        except Exception as e:
            print(f"⚠️ HERMES: Failed to deliver message. Error: {e}")

    def notify_trade(self, symbol, side, price, amount, event_type="ENTRY"):
        """
        Sends formatted trade notifications.
        event_type: ENTRY, EXIT, TP, STOP
        """
        now = datetime.now().strftime("%H:%M:%S")
        
        # Select emoji based on trade event
        emoji = "🔵"
        if event_type == "ENTRY":
            emoji = "🟢" if side == "LONG" else "🔴"
        elif event_type == "TP":
            emoji = "💰"
        elif event_type == "STOP":
            emoji = "🛑"

        msg = (
            f"{emoji} <b>TRADE {event_type}</b>\n"
            f"--------------------------\n"
            f"🪙 <b>Symbol:</b> {symbol}\n"
            f"⚡ <b>Side:</b> {side}\n"
            f"💵 <b>Price:</b> ${price}\n"
            f"⚖️ <b>Amount:</b> {amount}\n"
            f"⏰ <b>Time:</b> {now}"
        )
        self._send(msg)

    def send_daily_report(self, balance, daily_pnl, trade_count, win_rate):
        """Dispatches the daily 'Z-Report' summary."""
        today = datetime.now().strftime("%Y-%m-%d")
        status_emoji = "✅" if daily_pnl >= 0 else "🔻"
        
        msg = (
            f"📜 <b>HERMES DAILY REPORT ({today})</b>\n"
            f"========================\n"
            f"💰 <b>Total Balance:</b> ${balance:.2f}\n"
            f"{status_emoji} <b>Daily PnL:</b> ${daily_pnl:.2f}\n"
            f"📊 <b>Trade Count:</b> {trade_count}\n"
            f"🎯 <b>Win Rate:</b> %{win_rate}\n\n"
            f"<i>Olympus is watching you...</i> ⚡"
        )
        self._send(msg)

    def notify_error(self, error_msg):
        """Wakes you up in case of a critical system failure."""
        self._send(f"⚠️ <b>CRITICAL SYSTEM ERROR!</b>\n{error_msg}")
