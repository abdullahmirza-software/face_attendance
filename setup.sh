#!/bin/bash

# Enable verbose debugging mode if you pass -v or if specified below
# To run normally: ./script.sh
# To run with deep verbosity: ./script.sh -v
if [ "$1" == "-v" ]; then
    echo "⚙️ Verbose mode enabled. Printing every command executed..."
    set -x  # This prints every command before running it
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Copying .env.example to .env..."
    cp -v .env.example .env
else
    echo "✅ .env file already exists. Skipping copy."
fi

# Load .env file into the script
if [ -f .env ]; then
    echo "🔄 Loading environment variables..."
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ Error: .env file not found. Exiting..."
    exit 1
fi

# Clean up Python cache to prevent Docker volume permission conflicts
echo "🧹 Cleaning up local Python cache..."
find . -type d -name "__pycache__" -exec rm -rv {} + 2>/dev/null || true

# Remove existing containers
echo "🛑 Stopping and removing existing containers..."
docker compose down --volumes --remove-orphans

# Build containers cleanly
echo "🏗️ Building containers..."
docker compose build --no-cache

# Start containers
echo "🚀 Starting containers in background..."
docker compose up -d

# Wait 3 seconds for the application to fully initialize
echo "⏳ Waiting 3 seconds for containers to initialize..."
sleep 3

# Check if the container is actually running
IS_RUNNING=$(docker inspect -f '{{.State.Running}}' con_sentry_app 2>/dev/null)

if [ "$IS_RUNNING" = "true" ]; then
    echo "🟢 Container is running successfully! Dropping into terminal..."
    # Disable command printing right before jumping into the interactive bash prompt
    set +x 2>/dev/null
    docker exec -it con_sentry_app bash
else
    echo "❌ Error: con_sentry_app crashed on startup."
    echo "📋 Printing full application container logs to find the issue:"
    echo "--------------------------------------------------------"
    docker logs con_sentry_app
    echo "--------------------------------------------------------"
    
    echo "🧐 Inspecting container exit code and details:"
    docker inspect --format='ExitCode: {{.State.ExitCode}} | Error: {{.State.Error}}' con_sentry_app
fi
