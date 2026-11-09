import os
import sys
import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  [BollyWise AI] Movie Recommendation System")
    print("  Serving React UI + FastAPI Backend")
    print("  Open in browser: http://127.0.0.1:8000")
    print("=" * 55 + "\n")
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
