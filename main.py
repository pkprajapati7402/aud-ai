import os
import pkgutil
pkgutil.ImpImporter = pkgutil.zipimporter

from app import create_app
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = create_app()
CORS(app, resources={r"/api/*": {"origins": "https://audihealth-backend.onrender.com"}})

if __name__ == "__main__":
    # Use the PORT environment variable provided by Render, default to 5000 if not set
    app.run(host='0.0.0.0', port=8000, use_reloader=False)
