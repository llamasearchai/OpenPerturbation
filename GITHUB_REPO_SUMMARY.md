# OpenPerturbation – GitHub Repository Summary

**Author:** Nik Jois  
**Email:** <nikjois@llamasearch.ai>

---

## Mission
OpenPerturbation provides an end-to-end, production-ready software stack for perturbation biology, causal discovery, multimodal data fusion and optimal intervention design.  Built with typed Python ≥ 3.10, FastAPI, PyTorch-Lightning and a fully reproducible CI/CD pipeline, the platform empowers researchers and engineers to model, analyse and optimise complex biological systems at scale.

---

## Key Highlights
1. **Causal Discovery Engine** – PC, GES, LiNGAM & correlation-based algorithms with configurable hyper-parameters.
2. **Intervention Design Framework** – Data-driven recommendation of gene or compound perturbations via causal graph reasoning.
3. **Multimodal Fusion Models** – Vision Transformers, Molecular GNNs and Multimodal Transformers for joint representation learning.
4. **Explainability Toolkit** – Attention maps, concept activation vectors and pathway analysis for biological interpretability.
5. **Extensible REST API** – 25 fully-typed endpoints with autogenerated OpenAPI 3.1 & Swagger UI.
6. **Reproducible CI/CD** – Dockerised deployment, GitHub Actions and ≥ 90 % test coverage.
7. **Comprehensive Documentation** – MkDocs site hosted on GitHub Pages with tutorials, API reference and design notes.

---

## Repository Anatomy
```text
OpenPerturbation/
├── src/            # Application & ML code
│   ├── api/        # FastAPI app & routes
│   ├── causal/     # Discovery & intervention logic
│   ├── models/     # Vision / graph / fusion models
│   ├── training/   # PyTorch-Lightning modules
│   └── utils/      # Helper utilities
├── tests/          # Unit, integration & benchmark suites
├── docs/           # Documentation source (MkDocs)
├── configs/        # YAML configuration files
├── docker/         # Container assets & compose files
└── ...
```

---

## Release Tags
| Tag | Date | Notes |
|-----|------|-------|
| `v1.0.0` | 2025-01-03 | First production-ready release – stable API & 90 %+ coverage |

All tags follow [Semantic Versioning](https://semver.org).  Future releases will increment the **MAJOR** version for breaking changes, **MINOR** for new features and **PATCH** for bug fixes.

---

## Professional Commit History
The project adheres to the **Conventional Commits** specification.  Commit messages follow the pattern:

```
<type>: <short imperative subject>

<body – optional, wrapped at 72 chars>
```

**Allowed `<type>` values** include `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`, and `build`.  This style enables automatic changelog generation, semantic release tooling and clear project archaeology.

---

## Contributing
Contributions are welcome – please read `CONTRIBUTING.md` before opening a pull request.  All submissions must:
* Pass the full test‐suite (`pytest -q`).
* Conform to `black`-formatted code and pass `flake8` & `mypy`.
* Include relevant unit or integration tests.
* Use professional commit messages authored as **Nik Jois**.

---

## Contact
For questions, feature requests or collaboration, open a GitHub issue or contact **Nik Jois** at <nikjois@llamasearch.ai>. 