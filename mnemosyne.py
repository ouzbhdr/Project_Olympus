import sqlite3
import logging
import os
from datetime import datetime

class Mnemosyne:
    def __init__(self, db_name="olympus.db"):
        self.db_name = db_name
        self._initialize_logging()
        self._build_database()

    def _initialize_logging(self):
        """Sets up the system logging in English."""
        if not os.path.exists('logs'):
            os.makedirs('logs')

        logging.basicConfig(
            filename=f'logs/system_{datetime.now().strftime("%Y%m")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger("Olympus")

    def _get_connection(self):
        """Creates a secure connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_name, timeout=10)
        conn.row_factory = sqlite3.Row 
        return conn

    def _build_database(self):
        """Creates the tables if they don't exist with English naming."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # TRADES TABLE
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    entry_time TEXT,
                    amount REAL,
                    tp_price REAL,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL,
                    commission REAL,
                    status TEXT
                )
            ''')
            
            # SIGNAL LOGS (Technical details)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    indicator_data TEXT,
                    FOREIGN KEY(trade_id) REFERENCES trades(id)
                )
            ''')
            conn.commit()

    # --- TRADE MANAGEMENT ---

    def save_trade(self, symbol, side, price, amount, sl=None, tp=None):
        """Records a new entry in the database."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (symbol, side, entry_price, entry_time, amount, tp_price, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
                ''', (symbol, side, price, now, amount, tp))
                trade_id = cursor.lastrowid
                conn.commit()
                
                self.log(f"NEW TRADE: ID={trade_id} | {symbol} {side} at {price}")
                return trade_id
        except Exception as e:
            self.error("Database Save Error", str(e))
            return None

    def close_trade(self, trade_id, exit_price, pnl, commission=0):
        """Updates the trade record upon exit."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE trades 
                    SET exit_price = ?, exit_time = ?, pnl = ?, commission = ?, status = 'CLOSED'
                    WHERE id = ?
                ''', (exit_price, now, pnl, commission, trade_id))
                conn.commit()
                self.log(f"TRADE CLOSED: ID={trade_id} | PnL: ${pnl:.2f}")
        except Exception as e:
            self.error("Trade Close Error", str(e))

    def get_active_position(self, symbol):
        """Retrieves an open position for a specific symbol to recover after crash."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM trades 
                    WHERE symbol = ? AND status = 'OPEN'
                    ORDER BY id DESC LIMIT 1
                ''', (symbol,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            self.error("Recovery Error", str(e))
            return None

    # --- LOGGING UTILS ---

    def log(self, message, level="INFO"):
        print(f"📝 {message}")
        if level == "INFO": self.logger.info(message)
        elif level == "WARNING": self.logger.warning(message)

    def error(self, error_msg, detail=""):
        full_msg = f"{error_msg} | {detail}"
        print(f"❌ ERROR: {full_msg}")
        self.logger.error(full_msg)
