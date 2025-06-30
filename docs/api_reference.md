# API Reference

OpenPerturbation exposes a fully typed REST API built with FastAPI. The interactive Swagger UI is served at `/docs` when the server is running.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service liveness probe |
| `/causal-discovery` | POST | Run causal-discovery on tabular data |
| `/explainability` | POST | Generate explanation artefacts for trained models |
| `/intervention-design` | POST | Recommend optimal interventions given a causal graph |
| `/analysis/start` | POST | Launch multi-step analysis pipeline |
| `/analysis/{job_id}/status` | GET | Poll analysis job status & results |

For detailed request / response models see the automatically generated OpenAPI schema. 