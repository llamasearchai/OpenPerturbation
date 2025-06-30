# Cookbook Gallery

Welcome to the OpenPerturbation **Cookbook Gallery** – a curated collection of end-to-end, runnable examples that showcase the most common workflows in the platform.  Each notebook can be opened directly on GitHub or downloaded locally.

> **Tip:** The fastest way to get started is to clone the repository and launch JupyterLab:
>
> ```bash
> git clone https://github.com/llamasearchai/OpenPerturbation.git
> cd OpenPerturbation
> pip install -e .[dev]  # or: make install
> jupyter lab openperturbations/notebooks/
> ```

---

## 1. Loading Multimodal Data

| Notebook | Description |
|----------|-------------|
| [`01_loading_multimodal_data.ipynb`](../openperturbations/notebooks/01_loading_multimodal_data.ipynb) | Demonstrates how to generate **synthetic** genomics, imaging and molecular-structure datasets and load them with the library's data-loader API. |

---

## 2. Training a Model

| Notebook | Description |
|----------|-------------|
| [`02_training_a_model.ipynb`](../openperturbations/notebooks/02_training_a_model.ipynb) | Shows how to configure and train a **CausalVAE** model with PyTorch Lightning, including data preparation and a minimal training loop. |

---

## 3. Causal Discovery

| Notebook | Description |
|----------|-------------|
| [`03_causal_discovery.ipynb`](../openperturbations/notebooks/03_causal_discovery.ipynb) | Provides an end-to-end workflow: extract latent causal factors from a trained VAE, learn a causal graph with the **PC algorithm**, and visualise the resulting network. |

---

### How to Contribute a Cookbook

We welcome new examples!  If you have a cool workflow, submit a PR adding your notebook to `openperturbations/notebooks/` and update this page.  Please follow the existing style – short markdown explanations + clean, runnable code cells.

Happy experimenting! 🎉 