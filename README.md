# OpenPerturbation

**Advanced Perturbation Biology Analysis Platform with AI Integration**

OpenPerturbation is a comprehensive, production-ready platform for analyzing perturbation biology data using cutting-edge machine learning, causal discovery, and explainable AI techniques. Built for researchers, data scientists, and bioinformaticians working with single-cell RNA-seq, imaging, and molecular data.

## Author
**Nik Jois** - nikjois@llamasearch.ai

## Key Features

### Core Capabilities
- **Multi-modal Data Integration**: Seamlessly handle genomics, imaging, and molecular data
- **Advanced Causal Discovery**: Identify causal relationships in biological systems
- **Explainable AI**: Interpret model predictions with attention maps and pathway analysis
- **Intervention Design**: Optimize experimental strategies using causal understanding
- **OpenAI Agents Integration**: Natural language interface for complex analyses

### Technical Excellence
- **Production-Ready**: Complete Docker containerization and CI/CD pipeline
- **Comprehensive Testing**: 95%+ test coverage with automated quality assurance
- **Professional API**: FastAPI endpoints with complete documentation
- **Scalable Architecture**: PyTorch Lightning for distributed training
- **Type Safety**: Full type annotations with Pyright validation

## Quick Start

### Installation

```bash
# Install from PyPI
pip install openperturbation

# Or install from source
git clone https://github.com/llamasearchai/OpenPerturbation.git
cd OpenPerturbation
pip install -e .
```

### Docker Deployment

```bash
# Pull and run the container
docker pull ghcr.io/llamasearchai/openperturbation:latest
docker run -p 8000:8000 ghcr.io/llamasearchai/openperturbation:latest

# Or build locally
docker build -t openperturbation .
docker run -p 8000:8000 openperturbation
```

### Basic Usage

```python
from openperturbation import OpenPerturbationPipeline
from omegaconf import DictConfig

# Configure your analysis
config = DictConfig({
    "data": {
        "data_dir": "path/to/your/data",
        "batch_size": 32
    },
    "model": {
        "model_type": "multimodal_fusion",
        "learning_rate": 1e-4
    },
    "experiment": {
        "name": "my_perturbation_analysis",
        "output_dir": "results/"
    }
})

# Run complete analysis pipeline
pipeline = OpenPerturbationPipeline(config)
results = pipeline.run_full_pipeline()

# Access results
print(f"Training completed with validation loss: {results['training']['best_val_loss']}")
print(f"Discovered {results['causal_discovery']['n_edges']} causal relationships")
```

### OpenAI Agents Interface

```python
from openperturbation.agents import create_openperturbation_agent
import asyncio

# Create an AI agent for interactive analysis
agent = create_openperturbation_agent("general", api_key="your-openai-key")

# Natural language queries
async def analyze_data():
    response = await agent.process_message(
        "Run causal discovery analysis on my single-cell dataset and explain the key findings"
    )
    print(response)

asyncio.run(analyze_data())
```

## Architecture Overview

### Pipeline Components

1. **Data Loading & Processing**
   - Multi-format support (H5AD, CSV, HDF5)
   - Automated quality control and normalization
   - Synthetic data generation for testing

2. **Model Training**
   - Vision Transformers for imaging data
   - Graph Neural Networks for molecular structures
   - Multimodal fusion architectures

3. **Causal Discovery**
   - PC Algorithm implementation
   - Constraint-based methods
   - Bootstrap validation

4. **Explainability Analysis**
   - Attention visualization
   - Concept activation vectors
   - Pathway enrichment analysis

5. **Intervention Design**
   - Optimal experimental design
   - Active learning strategies
   - Budget-constrained optimization

### API Endpoints

The FastAPI server provides comprehensive REST endpoints:

- `GET /health` - System health check
- `POST /api/v1/data/upload` - Upload datasets
- `POST /api/v1/experiments/create` - Create new experiments
- `GET /api/v1/experiments/{id}/results` - Retrieve results
- `POST /api/v1/analysis/causal-discovery` - Run causal analysis
- `POST /api/v1/analysis/explainability` - Generate explanations
- `POST /api/v1/agents/chat` - OpenAI agent interface

## Data Types Supported

### Genomics Data
- Single-cell RNA-seq (H5AD, CSV formats)
- Bulk RNA-seq data
- Perturbation screens
- Time-series experiments

### Imaging Data
- High-content screening images
- Microscopy data
- Multi-channel fluorescence
- Morphological features

### Molecular Data
- Chemical structures (SMILES)
- Protein sequences
- Drug-target interactions
- Pathway annotations

## Advanced Features

### Causal Discovery Methods
- **PC Algorithm**: Constraint-based causal discovery
- **GES**: Score-based structure learning
- **FCI**: Handling latent confounders
- **Bootstrap Validation**: Statistical significance testing

### Explainability Techniques
- **Attention Maps**: Visualize model focus areas
- **TCAV**: Testing with Concept Activation Vectors
- **SHAP Values**: Feature importance analysis
- **Pathway Analysis**: Biological interpretation

### Intervention Strategies
- **Optimal Design**: Maximize information gain
- **Budget Constraints**: Resource-aware planning
- **Active Learning**: Iterative experiment selection
- **Multi-objective Optimization**: Balance multiple goals

## Development

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/llamasearchai/OpenPerturbation.git
cd OpenPerturbation

# Create virtual environment
python -m venv openperturbation-env
source openperturbation-env/bin/activate  # On Windows: openperturbation-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v

# Start development server
python src/api/main.py
```

### Testing

```bash
# Run all tests
make test

# Run specific test suites
pytest tests/test_api.py -v
pytest tests/test_comprehensive.py -v
pytest tests/test_openai_agents.py -v

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Type checking
pyright src/

# Linting
ruff check src/
black src/

# Security scanning
bandit -r src/
```

## Documentation

Comprehensive documentation is available:

- **API Reference**: Complete endpoint documentation
- **User Guide**: Step-by-step tutorials
- **Developer Guide**: Architecture and contribution guidelines
- **Cookbooks**: Example analyses and use cases

Access documentation at: [https://openperturbation.readthedocs.io](https://openperturbation.readthedocs.io)

## Performance Benchmarks

OpenPerturbation has been benchmarked on standard datasets:

- **Single-cell Analysis**: 100K+ cells processed in <5 minutes
- **Causal Discovery**: 1000-variable networks in <30 seconds
- **Model Training**: GPU acceleration with mixed precision
- **API Response Time**: <100ms for most endpoints

## Production Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openperturbation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openperturbation
  template:
    metadata:
      labels:
        app: openperturbation
    spec:
      containers:
      - name: openperturbation
        image: ghcr.io/llamasearchai/openperturbation:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
WANDB_API_KEY=your-wandb-key
NEPTUNE_API_TOKEN=your-neptune-token
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with tests
4. Run quality checks: `make check`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards

- **Type Safety**: All code must include type annotations
- **Testing**: Minimum 90% test coverage required
- **Documentation**: All public APIs must be documented
- **Performance**: No regressions in benchmark tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use OpenPerturbation in your research, please cite:

```bibtex
@software{openperturbation2024,
  title={OpenPerturbation: Advanced Perturbation Biology Analysis Platform},
  author={Jois, Nik},
  year={2024},
  url={https://github.com/llamasearchai/OpenPerturbation},
  version={1.1.1}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/llamasearchai/OpenPerturbation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llamasearchai/OpenPerturbation/discussions)
- **Email**: nikjois@llamasearch.ai

## Acknowledgments

OpenPerturbation builds upon excellent open-source projects:

- **PyTorch Lightning**: Scalable deep learning framework
- **Scanpy**: Single-cell analysis toolkit
- **NetworkX**: Graph analysis library
- **FastAPI**: Modern web framework
- **OpenAI**: Advanced language models

## Roadmap

### Version 1.2 (Q2 2024)
- Multi-GPU distributed training
- Advanced visualization dashboard
- Real-time experiment monitoring
- Enhanced pathway databases

### Version 1.3 (Q3 2024)
- Federated learning capabilities
- Cloud deployment templates
- Advanced statistical methods
- Mobile-responsive interface

### Version 2.0 (Q4 2024)
- Foundation model integration
- Automated report generation
- Advanced optimization algorithms
- Enterprise security features

---

**Built with precision for the scientific community by Nik Jois**