# PowerShell script to start Neo4j via Docker
# Make sure Docker Desktop is running before executing this script

Write-Host "Starting Neo4j container..." -ForegroundColor Green

# Check if Docker is running
try {
    docker ps | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if container already exists
$existing = docker ps -a --filter "name=food_health_neo4j" --format "{{.Names}}"
if ($existing -eq "food_health_neo4j") {
    Write-Host "Container exists. Starting it..." -ForegroundColor Yellow
    docker start food_health_neo4j
} else {
    Write-Host "Creating and starting Neo4j container..." -ForegroundColor Yellow
    docker-compose up -d
}

# Wait a few seconds for Neo4j to start
Write-Host "Waiting for Neo4j to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check container status
$status = docker ps --filter "name=food_health_neo4j" --format "{{.Status}}"
if ($status) {
    Write-Host "✓ Neo4j is running!" -ForegroundColor Green
    Write-Host "  - Browser: http://localhost:7474" -ForegroundColor Cyan
    Write-Host "  - Bolt: neo4j://localhost:7687" -ForegroundColor Cyan
    Write-Host "  - Username: neo4j" -ForegroundColor Cyan
    Write-Host "  - Password: sakuni200211" -ForegroundColor Cyan
} else {
    Write-Host "✗ Neo4j failed to start. Check logs with: docker logs food_health_neo4j" -ForegroundColor Red
}
