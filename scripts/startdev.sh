#!/bin/bash

# ───────────────────────────────────────────────
#   🚀 CasaLingua Dev Launcher for Mac M4
# ───────────────────────────────────────────────

clear
echo ""
echo " ██████╗ █████╗ ███████╗ █████╗ ██╗     ██╗███╗   ██╗ ██████╗ ██╗   ██╗  "
echo "██╔════╝██╔══██╗╚══███╔╝██╔══██╗██║     ██║████╗  ██║██╔═══██╗██║   ██║  "
echo "██║     ███████║  ███╔╝ ███████║██║     ██║██╔██╗ ██║██║   ██║██║   ██║  "
echo "██║     ██╔══██║ ███╔╝  ██╔══██║██║     ██║██║╚██╗██║██║   ██║██║   ██║  "
echo "╚██████╗██║  ██║███████╗██║  ██║███████╗██║██║ ╚████║╚██████╔╝╚██████╔╝  "
echo " ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝   "
echo "  🧠 Powered by M4 Max | CasaLingua - DEV Mode"
echo ""

# Check for venv
if [ ! -d ".venv" ]; then
  echo "🔍 Python virtual environment not found."
  read -p "Would you like to create and install dependencies now? (y/n): " confirm
  if [[ "$confirm" == "y" ]]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
  else
    echo "❌ Aborting. Please set up the environment manually."
    exit 1
  fi
else
  echo "✅ Virtual environment found. Activating..."
  source .venv/bin/activate
fi

echo "🚀 Launching development server using Uvicorn..."
echo ""
echo "🧪 Dev Mode | Local Debug UI at http://127.0.0.1:8000"
echo "📡 Watching for file changes..."

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000