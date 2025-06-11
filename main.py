import os
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, Request, Form, Response, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from celery import Celery
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI()

# Mount static files (for models and other static assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Ensure the models directory exists
MODELS_DIR = "static/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Celery App Setup ---
# The Celery broker URL will come from Render's Redis add-on
# The result backend is optional but good for monitoring task states
celery_app = Celery(
    'engine',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)
# Load tasks from engine.py
celery_app.autodiscover_tasks(['engine'])


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Root route: Serves the API key input form using HTMX.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-trading", response_class=HTMLResponse)
async def start_trading(request: Request, api_token: str = Form(...)):
    """
    Starts the trading bot for a given API token.
    This route will encrypt the token and store it, then launch a Celery task.
    """
    if not api_token:
        return HTMLResponse("<p class='text-red-500'>API Token cannot be empty.</p>")

    # In a real application, you'd associate this token with a user securely.
    # For this example, we'll assume a single user context or handle it via OAuth flow.
    # Here, we'll call a Celery task to handle token encryption and trading.
    try:
        # Assuming `store_token_and_start_trading` is a Celery task defined in engine.py
        task = celery_app.send_task(
            'engine.store_token_and_start_trading',
            args=[api_token],
            kwargs={}
        )
        logger.info(f"Started trading task with ID: {task.id}")
        return HTMLResponse(
            f"""
            <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative" role="alert">
                <strong class="font-bold">Success!</strong>
                <span class="block sm:inline">Trading started. Monitoring on dashboard.</span>
            </div>
            <script>window.location.href = '/dashboard';</script>
            """
        )
    except Exception as e:
        logger.error(f"Failed to start trading task: {e}")
        return HTMLResponse(f"<p class='text-red-500'>Error starting trading: {e}</p>")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Dashboard route: Displays live trade data.
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/events")
async def sse_events(request: Request) -> Response:
    """
    Server-Sent Events (SSE) endpoint for pushing live trade data to the dashboard.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        # This is a placeholder. In a real scenario, this would fetch data
        # from a shared data store (e.g., Redis Pub/Sub, database) updated by Celery tasks.
        counter = 0
        while True:
            # Check if client has disconnected
            if await request.is_disconnected():
                logger.info("SSE client disconnected.")
                break

            # Simulate sending live trade data
            # In a real app, this would query a database or Redis for new trade logs
            trade_data = {
                "id": counter,
                "asset": f"Asset_{counter % 3}",
                "direction": "buy" if counter % 2 == 0 else "sell",
                "amount": round(10 + counter * 0.5, 2),
                "status": "open" if counter % 5 != 0 else "closed",
                "pnl": round((counter - 5) * 1.2, 2)
            }
            yield f"data: {json.dumps(trade_data)}\n\n"
            counter += 1
            await asyncio.sleep(2) # Send updates every 2 seconds

    # Ensure json is imported for dumps
    import json
    return Response(content=event_generator(), media_type="text/event-stream")


@app.get("/admin/train", response_class=HTMLResponse)
async def admin_train(request: Request):
    """
    Admin route for model training. (Protected - Placeholder for authentication)
    This route will trigger a Celery task to train the trading model.
    """
    # For now, no authentication. In a real app, add proper auth.
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/train_model")
async def train_model_trigger():
    """
    Triggers the model training Celery task.
    """
    try:
        task = celery_app.send_task('engine.train_trading_model')
        logger.info(f"Model training task initiated: {task.id}")
        return {"message": "Model training initiated successfully!"}
    except Exception as e:
        logger.error(f"Error initiating model training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate model training: {e}")


@app.get("/models/download", response_class=HTMLResponse)
async def list_models(request: Request):
    """
    Lists available trained models for download.
    """
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    return templates.TemplateResponse(
        "model_list.html",
        {"request": request, "model_files": model_files, "models_dir": MODELS_DIR}
    )

@app.get("/models/download/{filename}")
async def download_model(filename: str):
    """
    Serves a specific trained model file for download.
    """
    file_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Model not found.")
    if not filename.endswith(".joblib"): # Basic security check
        raise HTTPException(status_code=403, detail="Only .joblib files are allowed for download.")

    return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")

# Add a simple model_list.html template for listing models (it will be created next)
# This isn't specified in the file structure but is necessary for the /models/download route.
# I'll add this to the templates section later.

