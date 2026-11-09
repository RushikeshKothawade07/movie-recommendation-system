import os
import sys

# Ensure repository root is on sys.path for Vercel Serverless environment
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.server import app

# Vercel serverless function entrypoint
# Exposes the FastAPI ASGI app directly
handler = app
