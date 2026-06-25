@echo off
REM Batch script to start Neo4j via Docker
REM Make sure Docker Desktop is running before executing this script

echo Starting Neo4j container...

REM Check if Docker is running
docker ps >nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Docker is running.

REM Check if container already exists
docker ps -a --filter "name=food_health_neo4j" --format "{{.Names}}" | findstr /C:"food_health_neo4j" >nul
if errorlevel 1 (
    echo Creating and starting Neo4j container...
    docker-compose up -d
) else (
    echo Container exists. Starting it...
    docker start food_health_neo4j
)

REM Wait a few seconds for Neo4j to start
echo Waiting for Neo4j to start...
timeout /t 10 /nobreak >nul

REM Check container status
docker ps --filter "name=food_health_neo4j" --format "{{.Status}}" | findstr /C:"Up" >nul
if errorlevel 1 (
    echo Neo4j failed to start. Check logs with: docker logs food_health_neo4j
) else (
    echo.
    echo Neo4j is running!
    echo   - Browser: http://localhost:7474
    echo   - Bolt: neo4j://localhost:7687
    echo   - Username: neo4j
    echo   - Password: sakuni200211
)

pause
