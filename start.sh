#!/bin/bash
# =============================================================================
# Start script for running both FastAPI and Gradio in development mode
# =============================================================================

set -e

echo "Starting Agentic RAG System..."

# Check for .env file
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating from template..."
    echo "OPENAI_API_KEY=your-api-key-here" > .env
    echo "Please edit .env and add your OpenAI API key!"
fi

# Create necessary directories
mkdir -p data/documents
mkdir -p data/chroma_db
mkdir -p logs

# Check if OpenAI API key is set
if ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo "Warning: OPENAI_API_KEY may not be set. Please check your .env file."
fi

# Function to cleanup background processes on exit
cleanup() {
    echo "Shutting down services..."
    kill $API_PID $GRADIO_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start FastAPI in background
echo "Starting FastAPI server on port 8000..."
python -m api.main &
API_PID=$!

# Wait for API to start
sleep 5

# Start Gradio in background
echo "Starting Gradio frontend on port 7860..."
python -m frontend.app &
GRADIO_PID=$!

echo ""
echo "=============================================="
echo "Agentic RAG System is running!"
echo "=============================================="
echo "FastAPI Backend: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Gradio Frontend: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=============================================="

# Wait for both processes
wait
