
## The Purple Platform: Deriv Trading Bot

This automated trading bot for Deriv.com uses FastAPI for its web interface, Celery for background tasks, HTMX for dynamic UI, and includes machine learning capabilities. It manages secure token storage and live trade monitoring.

-----

## Crucial Code Modifications & Configuration

Before you start, you **must** make these vital changes in your code and environment:

### 1\. `engine.py` Modifications

  * **Fernet Encryption Key (`FERNET_KEY`)**: This key encrypts your Deriv API tokens.

      * **Generate Key**: Run this Python code once to get your key:
        ```python
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        print(key.decode()) # Copy this output!
        ```
      * **Set Environment Variable**: Wherever your app runs (local shell, Render dashboard), set this key as an environment variable named `FERNET_KEY`.
          * **Code Line**: `FERNET_KEY = os.getenv("FERNET_KEY", b'YOUR_FERNET_KEY_HERE_REPLACE_THIS_WITH_A_REAL_KEY').decode()`
          * **Action**: **Never hardcode your generated key directly into the Python file for production\!**

  * **Deriv Data API Token (`DERIV_DATA_API_TOKEN`)**: Used for historical data fetching.

      * **Code Line**: `data_fetch_token = os.getenv("DERIV_DATA_API_TOKEN", "YOUR_DERIV_READONLY_TOKEN")`
      * **Action**: Set an environment variable `DERIV_DATA_API_TOKEN` with a valid **read-only** Deriv API token. If not set, data fetching may fail.

  * **Deriv App ID for Trading Bot (`app_id`)**:

      * **Code Line**: `class DerivTradingBot: def __init__(self, token: str, app_id: int = 1089):`
      * **Action**: While `1089` is a placeholder, it's best to match this with your actual Deriv App ID used in `oauth.py`. You could also make this an environment variable (`os.getenv("DERIV_APP_ID", 1089)`).

### 2\. `oauth.py` Modifications

  * **Deriv App ID for OAuth (`DERIV_APP_ID`)**: This is your app's registered ID with Deriv, crucial for the OAuth URL.
      * **Code Line**: `DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")`
      * **Action**: **Replace `"1089"` with your actual Deriv App ID.** Register your app at [Deriv Developers App Registration](https://www.google.com/search?q=https://developers.deriv.com/docs/app-registration/).

-----

## Running on Your Desktop (Local Development)

This is the easiest way to get started.

1.  **Clone the Repo**: `git clone <repo_url> && cd the-purple-platform`
2.  **Create & Activate Virtual Environment**: `python -m venv venv` then `source venv/bin/activate` (macOS/Linux) or `.\venv\Scripts\activate` (Windows).
3.  **Install Dependencies**: `pip install -r requirements.txt`
4.  **Set Environment Variables**: Use `export KEY="value"` (macOS/Linux) or `$env:KEY="value"` (PowerShell Windows) for `FERNET_KEY`, `DERIV_APP_ID`, `DERIV_DATA_API_TOKEN`, and `PYTHONUNBUFFERED=1`.
5.  **Run Services (in separate terminal windows/tabs)**:
      * **FastAPI Web Server**: `uvicorn main:app --reload` (runs on `http://127.0.0.1:8000`)
      * **Celery Worker**: `celery -A engine worker --loglevel=info` (requires a running Redis server if you want robust task queuing, otherwise it will use a simple in-memory default that's not for production).
      * **OAuth Server (if needed)**: `python oauth.py` (follows prompts to open browser).
6.  **Access**: Open your browser to `http://127.0.0.1:8000`.

-----

## Running on a General Web Server (e.g., Linux VPS)

For a production environment, you'll need more robust setup.

1.  **Server Setup**: Provision a Linux VPS, install Python, `git`, and a production-grade **Redis server**.
2.  **Database**: **Migrate from SQLite to PostgreSQL or MySQL.** You'll need to update `DATABASE_URL` environment variable to point to your new database (e.g., `postgresql://user:password@host:port/dbname`) and install the appropriate database driver (`psycopg2-binary` or `mysqlclient`).
3.  **Clone & Install**: Clone the repo and install dependencies in a virtual environment.
4.  **Set Environment Variables**: Configure `FERNET_KEY`, `DERIV_APP_ID`, `DERIV_DATA_API_TOKEN`, `REDIS_URL`, and `DATABASE_URL` using your server's process manager (e.g., `systemd` or `Supervisor`).
5.  **Process Management**: Use `Gunicorn` (or `uWSGI`) to run `main:app` and `systemd` (or `Supervisor`) to manage the Celery worker process.
6.  **Reverse Proxy**: Set up `Nginx` (or `Apache`) as a reverse proxy for your FastAPI app, handling SSL.

-----

## Running on Render.com

Render.com simplifies cloud deployment using a `render.yaml` blueprint.

### Shared Prerequisites

  * Render.com account.
  * Your code pushed to a GitHub, GitLab, or Bitbucket repo.

-----

### Render Free Plan (Limitations & Setup)

**⚠️ Major Limitations:** The free plan **cannot fully support this bot** due to:

  * **No Persistent Disks**: Your SQLite DB (`deriv_bot.db`) and models will **be lost on every deploy/restart.**
  * **No Free Worker Services**: You **cannot run a dedicated Celery worker** for continuous trading or background model training.
  * **Ephemeral Redis**: The free Redis is not persistent.
  * **Web Services Spin Down**: Free web services go idle after 15 mins of inactivity.

**What works**: The web interface (API token input, dashboard, admin, models list) will be accessible. However, any trading actions will be non-persistent and short-lived.

**Code Adjustment (`engine.py` for Free Plan)**:
To avoid errors from missing persistent storage, modify `engine.py`:

```python
# In engine.py, change:
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./deriv_bot.db")
DB_FILE = DATABASE_URL.replace("sqlite:///", "")

# TO:
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///:memory:") # Use in-memory SQLite
DB_FILE = ":memory:"
```

**`render.yaml` for Free Plan**:
Create a `render.yaml` at your repo root. **Note the absence of `worker`, `disk`, and `databases` sections.**

```yaml
services:
  - type: web
    name: purple-platform-web
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn main:app --host 0.0.0.0 --port $PORT"
    healthCheckPath: /
    autoDeploy: true
    envVars:
      - key: PYTHONUNBUFFERED
        value: "1"
      - key: REDIS_URL # Dummy value; Celery worker cannot run on free tier
        value: "redis://localhost:6379/0"
      - key: DATABASE_URL # Set to in-memory as no persistent disk is free
        value: "sqlite:///:memory:"
      - key: FERNET_KEY # IMPORTANT: Replace with your secure key
        value: "YOUR_SECURE_FERNET_KEY_HERE"
      - key: DERIV_APP_ID # IMPORTANT: Replace with your Deriv App ID
        value: "YOUR_DERIV_APP_ID_HERE"
      - key: DERIV_DATA_API_TOKEN # Optional; replace if needed
        value: "YOUR_DERIV_READONLY_TOKEN_HERE"
```

**Deployment Steps (Free Plan)**:

1.  Make the `engine.py` adjustment.
2.  Commit the `render.yaml` and code changes.
3.  In Render Dashboard, create a "New Web Service", connect your repo.
4.  Manually add `FERNET_KEY`, `DERIV_APP_ID`, and `DERIV_DATA_API_TOKEN` environment variables for the web service.
5.  Deploy.

-----

### Render Paid Plan (Full Functionality & Setup)

This is the recommended path for full bot functionality, allowing persistent data and background tasks.

**Code Adjustment (`engine.py` for Paid Plan)**:
**Revert the `engine.py` change** made for the free plan to enable persistent SQLite:

```python
# In engine.py, change back to:
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./deriv_bot.db")
DB_FILE = DATABASE_URL.replace("sqlite:///", "")
```

**`render.yaml` for Paid Plan**:
Create a `render.yaml` at your repo root. This includes all services and add-ons.

```yaml
services:
  - type: web
    name: purple-platform-web
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn main:app --host 0.0.0.0 --port $PORT"
    healthCheckPath: /
    autoDeploy: true
    envVars:
      - key: PYTHONUNBUFFERED
        value: "1"
      - key: REDIS_URL
        fromService:
          type: redis
          name: deriv-redis
          property: connectionString
      - key: DATABASE_URL
        value: "sqlite:///./deriv_bot.db"
      - key: FERNET_KEY # IMPORTANT: Replace
        value: "YOUR_SECURE_FERNET_KEY_HERE"
      - key: DERIV_APP_ID # IMPORTANT: Replace
        value: "YOUR_DERIV_APP_ID_HERE"
      - key: DERIV_DATA_API_TOKEN # Optional; replace
        value: "YOUR_DERIV_READONLY_TOKEN_HERE"

  - type: worker
    name: purple-platform-worker
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "celery -A engine worker --loglevel=info"
    autoDeploy: true
    envVars:
      - key: PYTHONUNBUFFERED
        value: "1"
      - key: REDIS_URL
        fromService:
          type: redis
          name: deriv-redis
          property: connectionString
      - key: DATABASE_URL
        value: "sqlite:///./deriv_bot.db"
      - key: FERNET_KEY # IMPORTANT: Replace
        value: "YOUR_SECURE_FERNET_KEY_HERE"
      - key: DERIV_APP_ID # IMPORTANT: Replace
        value: "YOUR_DERIV_APP_ID_HERE"
      - key: DERIV_DATA_API_TOKEN # Optional; replace
        value: "YOUR_DERIV_READONLY_TOKEN_HERE"

disk:
  - name: sqlite-data
    mountPath: /opt/render/project/src/deriv_bot.db # Path for persistent DB
    sizeGB: 1

databases:
  - type: redis
    name: deriv-redis
```

**Deployment Steps (Paid Plan)**:

1.  **Revert `engine.py`** to use persistent SQLite.
2.  Commit the `render.yaml` (paid config) and code changes.
3.  In Render Dashboard, create a "New Blueprint", connect your repo.
4.  Render will detect the blueprint and create all services (web, worker, Redis, disk).
5.  Manually add `FERNET_KEY`, `DERIV_APP_ID`, and `DERIV_DATA_API_TOKEN` environment variables for **each** service (web and worker).
6.  Deploy the blueprint.

-----

I hope this detailed guide helps you get your Deriv Trading Bot up and 
Dr Sanne Karibo
