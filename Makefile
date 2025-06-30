# OpenPerturbation Platform Makefile
# Build automation, testing, and deployment
#
# Author: Nik Jois
# Email: nikjois@llamasearch.ai

.PHONY: help install test lint format build run clean docker deploy

# Default target
help:
	@echo "OpenPerturbation Platform - Build Automation"
	@echo "============================================"
	@echo ""
	@echo "Available targets:"
	@echo "  help         - Show this help message"
	@echo "  install      - Install dependencies and setup environment"
	@echo "  test         - Run all tests"
	@echo "  test-api     - Run API tests only"
	@echo "  test-models  - Run model tests only"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black and isort"
	@echo "  build        - Build the package"
	@echo "  run          - Run the API server locally"
	@echo "  run-jupyter  - Start Jupyter Lab"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-stop  - Stop Docker containers"
	@echo "  deploy       - Deploy to production"
	@echo "  clean        - Clean up temporary files"
	@echo ""

# Environment setup
install:
	@echo "🔧 Installing OpenPerturbation dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ Installation complete!"

# Development setup
dev-install: install
	@echo "🔧 Installing development dependencies..."
	pip install pytest pytest-asyncio pytest-cov black isort mypy flake8
	pip install jupyter jupyterlab
	pip install fastapi[all] uvicorn[standard]
	@echo "✅ Development setup complete!"

# Testing
test:
	@echo "🧪 Running all tests..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ All tests completed!"

test-api:
	@echo "🧪 Running API tests..."
	python -m pytest tests/test_api.py -v
	@echo "✅ API tests completed!"

test-models:
	@echo "🧪 Running model tests..."
	python -m pytest tests/test_models.py -v
	@echo "✅ Model tests completed!"

test-integration:
	@echo "🧪 Running integration tests..."
	python tests/test_api.py
	@echo "✅ Integration tests completed!"

# Code quality
lint:
	@echo "🔍 Running code linting..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
	@echo "✅ Linting completed!"

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black
	@echo "✅ Code formatting completed!"

# Build
build:
	@echo "🏗️ Building OpenPerturbation package..."
	python setup.py sdist bdist_wheel
	@echo "✅ Build completed!"

# Local development
run:
	@echo "🚀 Starting OpenPerturbation API server..."
	# Allow custom PORT env variable, default 8000
	PORT := $(or $(PORT),8000)
	@if lsof -i :$$PORT -sTCP:LISTEN -t >/dev/null ; then \
	  echo "⚠️  Port $$PORT already in use. Please set PORT=<port> make run or free the port." ; \
	  exit 1 ; \
	else \
	  uvicorn src.api.server:app --host 0.0.0.0 --port $$PORT --reload ; \
	fi

run-jupyter:
	@echo "🚀 Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

run-main:
	@echo "🚀 Running main analysis pipeline..."
	python main.txt

# Docker operations
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build
	@echo "✅ Docker build completed!"

docker-run:
	@echo "🐳 Starting Docker containers..."
	docker-compose up -d
	@echo "✅ Docker containers started!"
	@echo "📊 Services available at:"
	@echo "  API: http://localhost:8000"
	@echo "  Jupyter: http://localhost:8888"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Grafana: http://localhost:3000"

docker-stop:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down
	@echo "✅ Docker containers stopped!"

docker-logs:
	@echo "📋 Showing Docker logs..."
	docker-compose logs -f

docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Docker cleanup completed!"

# Production deployment
deploy-staging:
	@echo "🚀 Deploying to staging..."
	@echo "⚠️  Staging deployment not configured yet"

deploy-production:
	@echo "🚀 Deploying to production..."
	@echo "⚠️  Production deployment not configured yet"

# Database operations
db-init:
	@echo "🗄️ Initializing database..."
	docker-compose exec postgres psql -U openperturbation -d openperturbation -f /docker-entrypoint-initdb.d/init.sql
	@echo "✅ Database initialized!"

db-backup:
	@echo "💾 Creating database backup..."
	docker-compose exec postgres pg_dump -U openperturbation openperturbation > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backup created!"

# Monitoring and logs
logs:
	@echo "📋 Showing application logs..."
	tail -f logs/openperturbation.log

monitor:
	@echo "📊 Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

# Development utilities
create-config:
	@echo "⚙️ Creating default configuration..."
	mkdir -p configs
	@echo "✅ Configuration directory created!"

create-notebooks:
	@echo "📓 Creating notebooks directory..."
	mkdir -p notebooks
	@echo "✅ Notebooks directory created!"

# Performance testing
benchmark:
	@echo "⚡ Running performance benchmarks..."
	python -m pytest tests/test_performance.py -v
	@echo "✅ Benchmarks completed!"

load-test:
	@echo "⚡ Running load tests..."
	@echo "⚠️  Load testing not configured yet"

# Security
security-scan:
	@echo "🔒 Running security scan..."
	pip install bandit safety
	bandit -r src/
	safety check
	@echo "✅ Security scan completed!"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "⚠️  Documentation generation not configured yet"

# Cleanup
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "✅ Cleanup completed!"

# Full workflow
all: clean install format lint test build
	@echo "🎉 Full build workflow completed successfully!"

# Quick development workflow
dev: format lint test-api
	@echo "🎉 Development workflow completed!"

# CI/CD workflow
ci: install lint test build
	@echo "🎉 CI/CD workflow completed!" 