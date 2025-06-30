# Deployment Guide

This page documents production deployment options for **OpenPerturbation**.

## Docker Compose *(recommended)*
```bash
docker compose up --build -d
```
* **API** exposed on `0.0.0.0:8000`
* Uses the official Python base image and installs pinned dependencies.

## Docker (stand-alone)
```bash
docker build -t openperturbation .
docker run -p 8000:8000 openperturbation
```

## Kubernetes
A `kubernetes/` folder containing Helm charts will be added in a future release. Contributions welcome. 