@echo off
setlocal
cd /d "%~dp0"

if not exist ".env" (
  echo Missing .env. Copy .env.example to .env and fill in the required settings first.
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not running. Start Docker Desktop, then run this file again.
  pause
  exit /b 1
)

docker compose -f deploy/docker-compose.yml --profile operations --profile observability --profile code-agent --profile rag-progressive up -d --build
if errorlevel 1 (
  echo Full stack startup failed. Check the Docker Compose output above.
  pause
  exit /b 1
)

echo.
echo Full Docker stack is running.
echo Web app and API: http://127.0.0.1:28080
echo Phoenix traces:  http://127.0.0.1:6006
echo Prometheus:      http://127.0.0.1:9090
echo Unleash:         http://127.0.0.1:4242
echo.
echo To stop the stack, run:
echo docker compose -f deploy/docker-compose.yml --profile operations --profile observability --profile code-agent --profile rag-progressive down
echo.
pause
endlocal
