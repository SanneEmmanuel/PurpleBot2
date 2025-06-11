import os
import sys
import webbrowser
import threading
from flask import Flask, request, redirect, url_for, render_template_string
import sqlite3
from cryptography.fernet import Fernet
import logging
from engine import encrypt_token, store_token_in_db, get_token_from_db, DB_FILE # Import functions from engine.py

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask App Setup for OAuth Callback ---
oauth_app = Flask(__name__)

# Derive APP_ID from environment or use a placeholder
# IMPORTANT: Replace YOUR_APP_ID with your actual Deriv API App ID
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089") # Placeholder, use your actual Deriv App ID
REDIRECT_URI = "http://localhost:5000/callback"

# HTML for the simple success page after OAuth
SUCCESS_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OAuth Success</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-lg shadow-md w-full max-w-md text-center">
        <h1 class="text-2xl font-bold text-green-600 mb-4">Authentication Successful!</h1>
        <p class="text-gray-700 mb-6">Your Deriv API token has been captured and securely stored.</p>
        <p class="text-gray-600">You can now close this window and proceed with the trading bot.</p>
        <button onclick="window.close()" class="mt-6 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50">
            Close Window
        </button>
    </div>
</body>
</html>
"""

@oauth_app.route("/callback")
def callback():
    """
    Callback route for Deriv OAuth2.
    Captures the access token from the URL fragment (client-side flow).
    """
    # The Deriv OAuth flow returns the token in the URL fragment (e.g., #access_token=...).
    # Flask/server-side code cannot directly read URL fragments.
    # We need a client-side JavaScript to extract it and send it to the server.
    # This HTML includes JS to do that.
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Deriv OAuth Callback</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                body { font-family: 'Inter', sans-serif; }
            </style>
        </head>
        <body class="bg-gray-100 flex items-center justify-center min-h-screen">
            <div class="bg-white p-8 rounded-lg shadow-md w-full max-w-md text-center">
                <h1 class="text-2xl font-bold text-blue-600 mb-4">Processing Authentication...</h1>
                <p class="text-gray-700">Please wait while we secure your token.</p>
                <div class="mt-4 animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
            </div>
            <script>
                // This script runs in the browser when the callback URL is loaded
                const hash = window.location.hash;
                if (hash) {
                    const params = new URLSearchParams(hash.substring(1)); // Remove '#'
                    const token = params.get('access_token');
                    if (token) {
                        // Send the token to the server
                        fetch('/store-token', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ token: token })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                window.location.href = '/oauth-success'; // Redirect to a success page
                            } else {
                                alert('Failed to store token: ' + data.message);
                            }
                        })
                        .catch(error => {
                            console.error('Error sending token to server:', error);
                            alert('An error occurred during token storage.');
                        });
                    } else {
                        alert('Access token not found in URL fragment.');
                    }
                } else {
                    alert('No URL fragment found.');
                }
            </script>
        </body>
        </html>
    """)

@oauth_app.route("/store-token", methods=["POST"])
def store_token():
    """
    Endpoint to receive the token from the client-side JavaScript.
    Encrypts and stores the token in the database.
    """
    data = request.json
    token = data.get("token")
    if token:
        try:
            user_id = "default_user_oauth" # Use a distinct user ID for OAuth captured tokens
            encrypted_tok = encrypt_token(token)
            store_token_in_db(user_id, encrypted_tok)
            logger.info(f"OAuth token successfully stored for user {user_id}.")
            return {"status": "success", "message": "Token stored successfully"}
        except Exception as e:
            logger.error(f"Error storing OAuth token: {e}")
            return {"status": "error", "message": str(e)}, 500
    return {"status": "error", "message": "No token provided"}, 400

@oauth_app.route("/oauth-success")
def oauth_success():
    """
    Displays a success message after the OAuth flow is complete and token is stored.
    """
    return SUCCESS_PAGE_HTML


def start_oauth_flow():
    """
    Launches the browser to the Deriv OAuth authorization page and starts the Flask server.
    """
    auth_url = f"https://oauth.deriv.com/oauth2/authorize?app_id={DERIV_APP_ID}&redirect_uri={REDIRECT_URI}"
    logger.info(f"Opening browser to: {auth_url}")
    webbrowser.open(auth_url)

    # Start the Flask app in a separate thread so it doesn't block the main process
    # and can be stopped programmatically if needed.
    # Using 'debug=False' and 'use_reloader=False' for thread safety.
    # The server needs to run on port 5000 as specified in REDIRECT_URI.
    try:
        logger.info("Starting Flask OAuth callback server on http://localhost:5000...")
        # Use a non-blocking way to run Flask.
        # This will be run in a separate thread.
        # To stop it cleanly, you might need to send a request to /shutdown.
        threading.Thread(target=oauth_app.run, kwargs={'host': '127.0.0.1', 'port': 5000, 'debug': False, 'use_reloader': False}).start()
        logger.info("Flask server started in background.")
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
        sys.exit(1)

# To run this file directly for testing the OAuth flow:
if __name__ == "__main__":
    print(f"Deriv App ID: {DERIV_APP_ID}")
    print(f"Redirect URI: {REDIRECT_URI}")
    input("Press Enter to start the OAuth flow and open your browser...")
    start_oauth_flow()
    print("OAuth flow initiated. Check your browser.")
    print("Once authenticated, your token will be stored in deriv_bot.db.")
    print("You can close this window after authentication is successful.")
    # Keep the main thread alive for the Flask server to run in its thread
    # For a persistent app, this would be integrated differently.
    # For this simple script, we'll just let it run.
