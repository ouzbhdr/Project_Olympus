"""
PROJECT OLYMPUS - INFRASTRUCTURE CONFIGURATION
----------------------------------------------
Core connection settings only. 
Trading logic and dynamic risk parameters are managed via strategy config.
"""

# --- 1. EXCHANGE AUTHENTICATION (OKX) ---
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"
API_PASSPHRASE = "your_api_passphrase_here"

# --- 2. TELEGRAM NOTIFICATIONS (HERMES) ---
TELEGRAM_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"

# --- 3. SYSTEM DEFAULTS ---
# These are used only as a fallback for the engine.
DB_NAME = "olympus.db"

