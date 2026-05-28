"""
Nexus CV — Entry Point
Run with: python run.py
"""
import sys
import os

# Add project root to path so all absolute imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
