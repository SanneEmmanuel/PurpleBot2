import os
import json
import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode

from celery import Celery
from deriv_api import DerivAPI
from cryptography.fernet import Fernet
import sqlite3
import joblib # For saving scikit-learn models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import pandas as pd # For data manipulation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Celery App Setup ---
# The Celery broker URL will come from Render's Redis add-on
celery_app = Celery(
    'engine',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

# Configure Celery to import tasks from this module
celery_app.conf.imports = ['engine']

# --- Encryption Setup ---
# Generate a Fernet key. In a real app, this should be loaded from an environment variable.
# For development, you can generate one and store it securely.
# key = Fernet.generate_key()
# print(key.decode()) # Print this once to get your key
# Ensure this key is consistent across deployments!
FERNET_KEY = os.getenv("FERNET_KEY", b'YOUR_FERNET_KEY_HERE_REPLACE_THIS_WITH_A_REAL_KEY').decode()
fernet = Fernet(FERNET_KEY)

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./deriv_bot.db") # Render persistent storage
# For SQLite, it's just the file path
DB_FILE = DATABASE_URL.replace("sqlite:///", "")

def init_db():
    """Initializes the SQLite database tables."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            encrypted_token TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            trade_id TEXT UNIQUE,
            asset TEXT,
            direction TEXT,
            amount REAL,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Trial user data can be simplified or integrated into 'tokens' table if needed
    # For now, let's assume 'tokens' covers user identification.
    conn.commit()
    conn.close()

# Initialize database on startup (or when this module is imported)
init_db()

# --- Deriv API Wrapper ---
class DerivTradingBot:
    def __init__(self, token: str, app_id: int = 1089): # Using app_id 1089 as a placeholder
        self.token = token
        self.app_id = app_id
        self.api = None
        self.max_concurrent_positions = 3
        self.max_daily_trades = 10
        self.current_open_positions = 0
        self.trades_today = 0
        self._last_trade_date = datetime.now().date()

    async def connect(self):
        """Connects to the Deriv WebSocket API and authorizes."""
        if self.api is None:
            self.api = DerivAPI(app_id=self.app_id)
            await self.api.connect()
            logger.info("Connected to Deriv API.")
            await self.api.authorize(self.token)
            logger.info("Authorized with Deriv API.")
            # Set up subscription to track open positions if needed
            # await self.api.subscribe(passthrough={'req_id': 1}, proposal_open_contract=1)
            # This would require an event listener to update self.current_open_positions

    async def disconnect(self):
        """Disconnects from the Deriv WebSocket API."""
        if self.api:
            await self.api.disconnect()
            self.api = None
            logger.info("Disconnected from Deriv API.")

    def reset_daily_trade_count(self):
        """Resets the daily trade count if a new day has started."""
        today = datetime.now().date()
        if today > self._last_trade_date:
            self.trades_today = 0
            self._last_trade_date = today
            logger.info("Daily trade count reset.")

    async def place_trade(self, symbol: str, amount: float, contract_type: str = 'CALL', duration: int = 5, duration_unit: str = 't'):
        """
        Places a trade on Deriv.
        symbol: e.g., 'R_100' (Volatility 100 Index)
        amount: Stake amount
        contract_type: 'CALL' for rise, 'PUT' for fall
        duration: Trade duration
        duration_unit: 's' for seconds, 'm' for minutes, 'h' for hours, 'd' for days, 't' for ticks
        """
        self.reset_daily_trade_count()

        if self.current_open_positions >= self.max_concurrent_positions:
            logger.warning(f"Max concurrent positions ({self.max_concurrent_positions}) reached. Cannot place new trade.")
            return None

        if self.trades_today >= self.max_daily_trades:
            logger.warning(f"Daily trade limit ({self.max_daily_trades}) reached. Cannot place new trade today.")
            return None

        try:
            # Get proposal for the trade to check price
            proposal = await self.api.proposal(
                amount=amount,
                basis='stake',
                contract_type=contract_type,
                currency='USD',
                duration=duration,
                duration_unit=duration_unit,
                symbol=symbol
            )
            logger.info(f"Proposal received: {proposal}")

            # Buy the contract
            buy_response = await self.api.buy(proposal['proposal']['id'], proposal['proposal']['ask_price'])
            logger.info(f"Buy response: {buy_response}")

            if buy_response and buy_response.get('buy') and buy_response['buy'].get('contract_id'):
                contract_id = buy_response['buy']['contract_id']
                logger.info(f"Trade placed successfully! Contract ID: {contract_id}")
                self.current_open_positions += 1
                self.trades_today += 1

                # Log the trade
                self._log_trade(
                    user_id="default_user", # Placeholder for actual user ID
                    trade_id=contract_id,
                    asset=symbol,
                    direction=contract_type,
                    amount=amount,
                    entry_price=buy_response['buy'].get('buy_price'),
                    exit_price=None,
                    pnl=None,
                    status="open"
                )
                return buy_response['buy']
            else:
                logger.error(f"Failed to place trade: {buy_response}")
                return None
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return None

    def _log_trade(self, user_id, trade_id, asset, direction, amount, entry_price, exit_price, pnl, status):
        """Logs a trade to the database."""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trade_logs (user_id, trade_id, asset, direction, amount, entry_price, exit_price, pnl, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, trade_id, asset, direction, amount, entry_price, exit_price, pnl, status))
            conn.commit()
            conn.close()
            logger.info(f"Trade {trade_id} logged to DB.")
        except Exception as e:
            logger.error(f"Error logging trade {trade_id}: {e}")

    async def get_historical_data(self, symbol: str, granularity: int = 60, count: int = 5000):
        """
        Fetches historical candlestick data from Deriv API.
        granularity: candlestick period in seconds (e.g., 60 for 1 minute)
        count: number of candles to retrieve
        """
        try:
            # Fetch latest prices for a symbol, or use ticks_history for more raw data
            # For model training, ticks_history with aggregation is more common
            response = await self.api.ticks_history(
                ticks_history=symbol,
                end='latest',
                count=count,
                adjust_start_time=1,
                style='candles',
                granularity=granularity
            )
            candles = response.get('candles', [])
            logger.info(f"Fetched {len(candles)} historical candles for {symbol}.")
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
            df.set_index('epoch', inplace=True)
            df = df.astype(float) # Ensure numeric types
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

# --- Helper Functions for Token Management ---
def encrypt_token(token: str) -> bytes:
    """Encrypts a token using Fernet."""
    return fernet.encrypt(token.encode())

def decrypt_token(encrypted_token: bytes) -> str:
    """Decrypts a token using Fernet."""
    return fernet.decrypt(encrypted_token).decode()

def store_token_in_db(user_id: str, encrypted_token: bytes):
    """Stores an encrypted token in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO tokens (user_id, encrypted_token)
        VALUES (?, ?)
    """, (user_id, encrypted_token))
    conn.commit()
    conn.close()
    logger.info(f"Token stored for user {user_id}.")

def get_token_from_db(user_id: str) -> str | None:
    """Retrieves and decrypts a token from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT encrypted_token FROM tokens WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return decrypt_token(result[0])
    return None

# --- Celery Tasks ---

@celery_app.task
def store_token_and_start_trading(api_token: str):
    """
    Celery task to encrypt and store the API token, then initiate trading.
    This task will be called from main.py.
    """
    user_id = "default_user" # Placeholder: In a real app, this would be a unique user ID
    encrypted_tok = encrypt_token(api_token)
    store_token_in_db(user_id, encrypted_tok)
    logger.info(f"Token encrypted and stored for {user_id}. Starting trading process...")
    # Trigger the actual trading task after token storage
    trade_task.delay(user_id) # Pass the user_id to retrieve token later

@celery_app.task
def trade_task(user_id: str):
    """
    Celery task for the main trading loop.
    This task will run continuously or be periodically triggered.
    """
    token = get_token_from_db(user_id)
    if not token:
        logger.error(f"No token found for user {user_id}. Cannot start trading.")
        return

    bot = DerivTradingBot(token)
    asyncio.run(bot.connect()) # Connect to Deriv API
    logger.info(f"Trading bot started for user {user_id}.")

    try:
        # Example: Implement a simple trading strategy
        # In a real bot, this would involve strategy logic,
        # checking market conditions, model predictions, etc.
        symbol = 'R_100' # Volatility 100 Index
        amount = 10.0 # Stake USD
        contract_type = 'CALL' # Example: always buy CALL

        # This loop will run continuously, placing trades based on logic
        # For a real bot, you'd have more sophisticated triggers and stop conditions.
        while True:
            # Check market conditions, make predictions using a trained model
            # For demonstration, let's just try to place a trade periodically
            logger.info(f"Attempting to place a trade for {user_id}...")
            trade_info = asyncio.run(bot.place_trade(symbol, amount, contract_type))
            if trade_info:
                logger.info(f"Trade successfully initiated: {trade_info.get('contract_id')}")
                # You might want to update the dashboard via Redis Pub/Sub here
            else:
                logger.warning("Trade attempt failed or limits reached.")

            # Sleep for a bit before the next trade attempt
            # In a real bot, this sleep duration would depend on your strategy
            asyncio.run(asyncio.sleep(30)) # Wait 30 seconds before next attempt
    except Exception as e:
        logger.error(f"Trading bot encountered an error for user {user_id}: {e}", exc_info=True)
    finally:
        asyncio.run(bot.disconnect()) # Ensure disconnection

@celery_app.task
def train_trading_model():
    """
    Celery task to train the trading model.
    """
    logger.info("Starting model training task...")
    # Placeholder token for historical data fetching if no user is active
    # In a real scenario, this would use a dedicated API token for data fetching
    # or the bot would use the active user's token.
    # For now, let's just use a dummy token, assuming 'authorize' might not be strictly needed for public data.
    # If it fails, a valid token for historical data is required.
    data_fetch_token = os.getenv("DERIV_DATA_API_TOKEN", "YOUR_DERIV_READONLY_TOKEN")
    if data_fetch_token == "YOUR_DERIV_READONLY_TOKEN":
        logger.warning("Using placeholder Deriv data API token. Data fetching might fail without a real token.")

    bot = DerivTradingBot(data_fetch_token)
    asyncio.run(bot.connect())

    try:
        symbol = 'R_100' # Example symbol
        historical_data = asyncio.run(bot.get_historical_data(symbol, granularity=60, count=2000))

        if historical_data.empty:
            logger.error("No historical data fetched for training. Aborting model training.")
            return

        # --- Feature Engineering (Example) ---
        # Simple features: price change, moving averages
        historical_data['open_close_diff'] = historical_data['close'] - historical_data['open']
        historical_data['high_low_diff'] = historical_data['high'] - historical_data['low']
        historical_data['volume_diff'] = historical_data['high_low_diff'] * historical_data['open_close_diff'] # Dummy volume

        # Create a target variable: '1' if price rises in the next candle, '0' otherwise
        # Shift the 'close' price to predict the *next* candle's close
        historical_data['next_close'] = historical_data['close'].shift(-1)
        # Assuming a simple rise/fall prediction
        historical_data['target'] = (historical_data['next_close'] > historical_data['close']).astype(int)

        # Drop the last row as it has NaN for 'next_close' and 'target'
        historical_data.dropna(inplace=True)

        features = ['open', 'high', 'low', 'close', 'open_close_diff', 'high_low_diff'] # Add other engineered features
        X = historical_data[features]
        y = historical_data['target']

        # Ensure we have enough data for splitting
        if len(X) < 2:
            logger.error("Not enough data to train the model after feature engineering.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- Scikit-learn Pipeline ---
        # Example pipeline: Scaling + Classifier
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('transformer', QuantileTransformer(output_distribution='normal')), # Optional: make features Gaussian
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ])

        logger.info("Fitting model pipeline...")
        model_pipeline.fit(X_train, y_train)
        logger.info("Model training complete.")

        # Evaluate the model
        y_pred = model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy on test set: {accuracy:.2f}")

        # --- Save the trained model ---
        model_filename = f"trading_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model_pipeline, model_path)
        logger.info(f"Model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Error during model training task: {e}", exc_info=True)
    finally:
        asyncio.run(bot.disconnect())
