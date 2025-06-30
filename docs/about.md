# About

**OpenPerturbation** is an end-to-end research and engineering toolkit designed to accelerate *perturbation biology* workflows – from raw multi-omic data ingestion to causal graph discovery, intervention optimisation and explainability.

## Vision
We aim to empower researchers and engineers to:
1. **Reveal causal mechanisms** behind genetic or chemical perturbations.
2. **Design optimal interventions** that maximise a desired phenotype or minimise toxicity.
3. **Integrate heterogeneous data** (images, sequences, graphs) via modern multimodal models.
4. **Move rapidly from prototype to production** with reproducible, typed Python code.

## Architecture Overview
```
            +-------------------+
            |    API Server     | (FastAPI, Pydantic v2)
            +---------+---------+
                      |
   +------------------+------------------+
   |                                     |
+--v--+                               +--v--+
|Data |                               |ML   |
|IO   |                               |Models|
+--+--+                               +--+--+
   |                                     |
   +----------+--------------+-----------+
              |              |
        +-----v----+   +-----v----+
        | Causal   |   | Training |
        | Engine   |   | Pipeline |
        +----------+   +----------+
```

*Full high-resolution diagram available in the repository [`docs/assets/architecture.svg`](assets/architecture.svg).*

## Technology Stack
* **Python ≥ 3.10** – Strict type hints, Pydantic v2 models
* **FastAPI** – Async REST service with OpenAPI 3.1 docs
* **PyTorch / PyTorch-Lightning** – Deep-learning backbone
* **scikit-learn & SciPy** – Classical statistics & ML utilities
* **MkDocs-Material** – Beautiful, versioned documentation (this site)
* **GitHub Actions** – CI, CD & automated documentation deploy

## Governance
OpenPerturbation is an **open-source** project released under the MIT Licence.  Maintained by *Nik Jois* with support from the community.  All development follows semantic versioning and conventional commits. 