# Quick Start

This guide shows how to get **OpenPerturbation** running locally or in Docker within a few minutes.

## Prerequisites
* Python ≥ 3.10
* Git
* Optional: Docker ≥ 24 (for container deployment)

---

## 1. Clone the Repository
```bash
git clone https://github.com/llamasearchai/OpenPerturbation.git
cd OpenPerturbation
```

## 2. Create & Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## 3. Install the Package
```bash
pip install -e ".[dev]"
```

## 4. Run the Test-Suite *(optional)*
```bash
pytest -q
```

## 5. Launch the API Server
```bash
openperturbation                # or: python -m src.api.server
```

OpenAPI docs available at `http://localhost:8000/docs`.

---

## Docker Deployment
```bash
docker compose up --build -d
```
API will start on port **8000**. Stop with `docker compose down`. 