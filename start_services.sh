#!/bin/bash

echo "🚀 Starting Model Converter Tool Services..."
echo "=============================================="

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Function to kill processes on a port
kill_port() {
    if lsof -ti:$1 > /dev/null 2>&1; then
        echo "🔄 Stopping processes on port $1..."
        lsof -ti:$1 | xargs kill -9
        sleep 2
    fi
}

# Start Redis
echo "📡 Starting Redis..."
redis-server --daemonize yes
sleep 2

# Check if Redis is running
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Failed to start Redis"
    exit 1
fi

# Start Celery Worker
echo "🔧 Starting Celery Worker..."
# Kill any existing Celery processes
pkill -f "celery.*worker" 2>/dev/null
sleep 2

# Start Celery worker in background
python3 -m celery -A app.tasks worker --loglevel=info &
CELERY_PID=$!
echo "✅ Celery Worker started (PID: $CELERY_PID)"

# Start FastAPI Server
echo "🌐 Starting FastAPI Server..."
kill_port 8000
sleep 2

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
echo "✅ FastAPI Server started (PID: $API_PID)"

# Wait for API to start
sleep 5

# Test API health
echo "🔍 Testing API health..."
if curl -s http://localhost:8000/api/v1/health > /dev/null; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed"
fi

# Start Streamlit UI
echo "🎨 Starting Streamlit UI..."
kill_port 8501
sleep 2

python3 -m streamlit run streamlit_app.py --server.port 8501 &
STREAMLIT_PID=$!
echo "✅ Streamlit UI started (PID: $STREAMLIT_PID)"

# Wait for Streamlit to start
sleep 5

echo ""
echo "🎉 All services started successfully!"
echo "=============================================="
echo "📊 Service Status:"
echo "   Redis:        http://localhost:6379"
echo "   FastAPI:      http://localhost:8000"
echo "   Streamlit UI: http://localhost:8501"
echo "   API Docs:     http://localhost:8000/docs"
echo ""
echo "🔧 Test the API:"
echo "   curl -X GET http://localhost:8000/api/v1/health"
echo "   curl -X GET http://localhost:8000/api/v1/formats"
echo ""
echo "📝 To stop all services, run:"
echo "   pkill -f 'celery\|uvicorn\|streamlit'"
echo "   redis-cli shutdown"
echo ""
echo "💡 The services are running in the background."
echo "   Check the logs for any issues." 