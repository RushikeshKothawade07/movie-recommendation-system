import os
import sys

# Dynamic root directory resolution for Vercel Serverless environment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

for p in [ROOT_DIR, CURRENT_DIR, os.path.join(ROOT_DIR, "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import the FastAPI application
from backend.server import app

# Vercel ASGI entry point handler
handler = app
