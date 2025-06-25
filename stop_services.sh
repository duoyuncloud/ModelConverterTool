#!/bin/bash

echo "🛑 Stopping Model Converter Tool Services..."
echo "=============================================="

# Stop Celery workers
echo "🔧 Stopping Celery Workers..."
pkill -f "celery.*worker" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Celery Workers stopped"
else
    echo "ℹ️  No Celery Workers found"
fi

# Stop FastAPI server
echo "🌐 Stopping FastAPI Server..."
pkill -f "uvicorn.*app.main:app" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ FastAPI Server stopped"
else
    echo "ℹ️  No FastAPI Server found"
fi

# Stop Streamlit UI
echo "🎨 Stopping Streamlit UI..."
pkill -f "streamlit.*streamlit_app.py" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Streamlit UI stopped"
else
    echo "ℹ️  No Streamlit UI found"
fi

# Stop Redis
echo "📡 Stopping Redis..."
redis-cli shutdown 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Redis stopped"
else
    echo "ℹ️  Redis was not running or already stopped"
fi

# Wait a moment for processes to stop
sleep 2

# Check if any processes are still running
echo "🔍 Checking for remaining processes..."
REMAINING=$(ps aux | grep -E "(celery|uvicorn|streamlit)" | grep -v grep | wc -l)
if [ $REMAINING -eq 0 ]; then
    echo "✅ All services stopped successfully"
else
    echo "⚠️  $REMAINING processes still running"
    ps aux | grep -E "(celery|uvicorn|streamlit)" | grep -v grep
fi

echo ""
echo "🎉 Services stopped!"
echo "==============================================" 