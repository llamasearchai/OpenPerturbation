---
created: 2025-01-18T11:15:00-07:00
author: Nik Jois
source: https://github.com/llamasearchai/OpenPerturbation
tags: systems-biology, causal-inference, multimodal-fusion, high-content-screening, compressed-phenotypic-screening
---

# OpenPerturbation: A Platform for Scalable Causal Discovery and Compressed Phenotypic Screening in Multimodal Biology

> ## Abstract
> OpenPerturbation is an open-source platform that unifies causal discovery, multimodal deep learning, and state-of-the-art compressed phenotypic screening to accelerate perturbation biology, from raw multi-omic datasets to actionable causal insights.

## TL;DR

OpenPerturbation bridges high-content experimental data with rigorous causal inference. Its **compressed screening engine** is designed to reduce assay costs significantly, while its **causal VAE** and **multimodal transformer** models uncover hidden biological drivers from complex datasets. Built on a modern stack including FastAPI, PyTorch, and NetworkX, the platform is delivered with ready-to-deploy Docker images, a comprehensive test suite, and automated CI/CD workflows.

---

## 1. Introduction

High-throughput perturbation assays now generate terabytes of imaging, omics, and single-cell data. Yet, extracting causal structure from these datasets and translating it into therapeutic hypotheses remains a primary challenge. Existing computational pipelines often either (1) ignore causality, risking confounded conclusions, or (2) depend on expensive one-perturbation-per-well screens that limit experimental scale and complexity.

**OpenPerturbation** addresses both problems by integrating:
1.  A **comprehensive causal discovery engine** supporting a suite of algorithms, from classic constraint-based methods to a novel deep learning backbone.
2.  A **compressed phenotypic screening (CS) module** that uses pooled perturbations and computational deconvolution to efficiently recover per-perturbagen effects, inspired by seminal work like Perturb-Seq (Dixit et al., 2016) and recent advances in compressed screening (Liu et al., 2024).

---

## 2. The Problem: Bottlenecks in Modern Perturbation Biology

-   **Cost and Throughput**: Standard single-compound screens require thousands of experimental wells, making them prohibitively expensive for large-scale studies or for use with complex models like patient-derived organoids.
-   **Lack of Causal Insight**: Correlation-based analyses are susceptible to misidentifying drivers due to hidden confounders, a common problem in complex biological systems.
-   **Fragmented Tooling**: Researchers often rely on a disconnected set of scripts for screen design, statistical analysis, and downstream modeling, which slows down the iteration cycle.

---

## 3. OpenPerturbation: An Integrated Solution

![High-level architecture](docs/img/architecture.png)

1.  **Data Ingestion Layer**: Flexible loaders for imaging, genomics, and molecular data types.
2.  **Compressed Screening Engine**: Tools for pool design, simulation, and deconvolution of pooled screens using elastic-net regression.
3.  **Causal Discovery Core**: A suite of causal discovery algorithms with bootstrap-based confidence scoring, all implemented with production-quality, type-safe code.
4.  **Multimodal Fusion Models**: Advanced deep learning models including a Cell-ViT, a molecular GNN, and a multimodal transformer for integrating diverse data streams.
5.  **API and Deployment**: A FastAPI service exposes key functionality through a clean REST API, with endpoints for causal discovery, model analysis, and experiment management.
6.  **Full-Stack DevOps**: The platform includes Docker for containerization, GitHub Actions for CI/CD, and a documentation site built with MkDocs Material.

---

## 4. Key Features

-   **Compressed Screening Design**: Utilities for generating balanced, replicate-aware pooling maps for perturbation experiments.
-   **Elastic-Net Deconvolution**: A robust implementation to recover per-compound effects from pooled screens, with significance testing via permutation analysis (Zou & Hastie, 2005).
-   **Causal Graph Analysis**: Leverages NetworkX for comprehensive graph-based analyses, including centrality, connectivity, and structure validation.
-   **Deep Causal Discovery**: Features a variational autoencoder (VAE) architecture to learn causal graphs directly from observational data.
-   **Secure Configuration**: Uses `.env` files for secure management of API keys and other secrets, following industry best practices.
-   **Automated DevOps**: A complete CI/CD pipeline using GitHub Actions for automated testing (Pytest), linting and type checking (Pyright), Docker image builds, and documentation deployment.

---

## 5. Architecture Deep Dive

### 5.1. Compressed Screening

The platform provides a framework for deconvolving pooled perturbation screens. Elastic-net regression is executed via `sklearn.MultiTaskElasticNet`, with permutation-based p-values for statistical significance. Confidence in the discovered causal links is estimated using bootstrapping, as illustrated conceptually below.

```python
# Illustrative bootstrap confidence estimation
def _bootstrap_confidence(self, data: np.ndarray, method: str) -> np.ndarray:
    n_samples = data.shape[0]
    for _ in range(self.bootstrap_samples):
        resampled_indices = np.random.choice(n_samples, n_samples, replace=True)
        resampled_data = data[resampled_indices]
        # Execute the chosen causal discovery algorithm on the resampled data
        # ... store results ...
    # Aggregate results to estimate confidence
    ...
```

### 5.2. Causal Discovery

The following causal discovery algorithms are implemented:
-   **NOTEARS**: For discovering DAGs via continuous optimization.
-   **LiNGAM**: Based on linear non-Gaussian acyclic models.
-   **Bayesian Networks**: Using hill-climbing search with a BIC score via `pgmpy`.

### 5.3. Multimodal Fusion

Vision, graph, and tabular data branches are merged through a transformer encoder that uses cross-modal attention to learn integrated representations, based on recent architectural advances (Shvetsova et al., 2024, arXiv).

---

## 6. Use Cases and Examples

### Drug-Response Mapping in Organoids
A compressed screen of 68 TME ligands in PDAC organoids successfully reproduced an IL-4/IL-13 response module. This module was distinct from canonical MsigDB signatures and correlated with poor survival in TCGA, demonstrating the platform's utility in identifying clinically relevant pathways (Wei et al., 2025, PerturBase).

### Immune-Modulatory Profiling in PBMCs
A 90-compound library was pooled and screened using single-cell RNA-seq. The analysis correctly identified that ruxolitinib interferes with the IFN-β JAK/STAT module and also uncovered novel pleiotropic effects. Example analyses are available as Jupyter notebooks in the repository.

---

## 7. Implementation Highlights

-   **Core Technologies**: Python 3.11+, PyTorch 2.2
-   **AI Integration**: OpenAI SDK with secure API key loading.
-   **Scientific Libraries**: NetworkX 3.2, PGMPY 0.1.25, Causal-Learn 0.4.
-   **Documentation**: MkDocs-Material 9 with `mkdocstrings` for automatic API documentation.

---

## 8. Innovation and Impact

OpenPerturbation democratizes compressed phenotypic screening—a technique previously confined to specialized labs—by packaging the entire workflow into a single, robust, and type-safe toolkit. This enables researchers to screen complex, patient-derived models at a fraction of the traditional cost, accelerating the pace of translational discovery.

---

## 9. Roadmap

-   Native support for building causal graphs from **spatial transcriptomics** data.
-   An auto-generated **Docker Compose GPU** stack for easier deployment on accelerated hardware.
-   Integration with **LangChain** for natural-language pipeline orchestration.
-   A visual, drag-and-drop web UI for designing compressed screening experiments.

---

## 10. Resources

-   **GitHub Repo**: https://github.com/llamasearchai/OpenPerturbation
-   **Documentation**: https://llamasearchai.github.io/OpenPerturbation
-   **Example Notebooks**: `openperturbations/notebooks/`
-   **Contribution Guide**: `CONTRIBUTING.md`
-   **Quick-Start Guide**: `docs/quick_start.md`

---

## 11. Conclusion

OpenPerturbation closes the loop from large-scale perturbation data to causal insight, while its compressed screening toolkit dramatically lowers experimental overhead. We invite the scientific community to deploy, extend, and contribute to this platform, and to join us in accelerating the frontier of systems and perturbation biology.

---

### Acknowledgements

This work is inspired by and builds upon recent advances in the field, including PerturBase (Wei et al., NAR 2025), scalable compressed screens (Liu et al., Nat Biotech 2024), and cell-count-based bioactivity prediction (Seal, 2025).

### Citation

```
Jois, N. (2025). OpenPerturbation: Scalable Causal Discovery and Compressed Phenotypic Screening for Multimodal Biology. GitHub Repository. https://github.com/llamasearchai/OpenPerturbation
``` 