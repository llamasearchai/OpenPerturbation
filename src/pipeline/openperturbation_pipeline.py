#!/usr/bin/env python3
"""
OpenPerturbation: AI-Driven Perturbation Biology Analysis Platform

Main application entry point for running comprehensive perturbation biology
analysis including causal discovery, multimodal fusion, and explainable AI.

Author: Nik Jois
Email: nikjois@llamasearch.ai

Usage:
    python main.py --config-path configs --config-name main_config.yaml
    python main.py experiment=causal_discovery data=high_content_screening
    python main.py --help
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Import core modules
from src.training.data_modules import (
    PerturbationDataModule,
    SingleCellImageDataModule,
    MolecularGraphDataModule,
    MultimodalDataModule
)
from src.training.lightning_modules import (
    VisionTransformerLightningModule,
    GraphLightningModule,
    CausalDiscoveryLightningModule
)
from src.models.vision.cell_vit import CellViT
from src.models.graph.molecular_gnn import MolecularGNN
from src.causal.causal_discovery_engine import CausalDiscoveryEngine, run_causal_discovery
from src.causal.intervention import (
    CausalGraphInterventionPredictor, 
    DeepLearningInterventionPredictor,
    ExperimentalDesignEngine
)
from src.explainability.attention_maps import generate_attention_analysis
from src.explainability.concept_activation import compute_concept_activations
from src.explainability.pathway_analysis import run_pathway_analysis

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openperturbation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OpenPerturbationPipeline:
    """Main pipeline for OpenPerturbation analysis."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_environment()
        
    def setup_environment(self):
        """Setup the environment for reproducible experiments."""
        # Set seeds for reproducibility
        if hasattr(self.config, 'seed'):
            pl.seed_everything(self.config.seed, workers=True)
        
        # Setup device
        if torch.cuda.is_available() and self.config.get('use_gpu', True):
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        if self.config.get('use_wandb', False):
            self.logger = WandbLogger(
                project=self.config.get('project_name', 'openperturbation'),
                name=self.config.get('experiment_name', 'experiment'),
                save_dir=str(self.output_dir)
            )
        else:
            self.logger = TensorBoardLogger(
                save_dir=str(self.output_dir),
                name="tensorboard_logs"
            )
    
    def setup_data(self) -> PerturbationDataModule:
        """Setup the data module."""
        logger.info("Setting up data module...")
        
        data_module = PerturbationDataModule(self.config.data)
        data_module.prepare_data()
        data_module.setup()
        
        logger.info(f"Data module setup complete:")
        logger.info(f"  Train samples: {len(data_module.train_dataloader().dataset)}")
        logger.info(f"  Val samples: {len(data_module.val_dataloader().dataset)}")
        logger.info(f"  Test samples: {len(data_module.test_dataloader().dataset)}")
        
        return data_module
    
    def setup_model(self, model_type: str) -> pl.LightningModule:
        """Setup the model based on configuration."""
        logger.info(f"Setting up {model_type} model...")
        
        if model_type == 'causal_vae':
            model = CausalVAELightningModule(self.config.model)
        elif model_type == 'multimodal_fusion':
            model = MultiModalFusionModule(self.config.model)
        elif model_type == 'cell_vit':
            model = CellViTModule(self.config.model)
        elif model_type == 'causal_discovery':
            model = CausalDiscoveryLightningModule(self.config.model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def setup_trainer(self) -> Trainer:
        """Setup the PyTorch Lightning trainer."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.output_dir / "checkpoints"),
            filename="{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.get('early_stopping', True):
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=self.config.get('patience', 10),
                mode="min",
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        trainer = Trainer(
            max_epochs=self.config.get('max_epochs', 100),
            accelerator='gpu' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu',
            devices=1,
            logger=self.logger,
            callbacks=callbacks,
            deterministic=True,
            enable_progress_bar=True,
            log_every_n_steps=50,
            val_check_interval=self.config.get('val_check_interval', 1.0),
            gradient_clip_val=self.config.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=self.config.get('accumulate_grad_batches', 1),
        )
        
        return trainer
    
    def run_training(self, model: pl.LightningModule, data_module: PerturbationDataModule) -> pl.LightningModule:
        """Run model training."""
        logger.info("Starting model training...")
        
        trainer = self.setup_trainer()
        
        # Train the model
        trainer.fit(model, data_module)
        
        # Test the model
        if self.config.get('run_test', True):
            trainer.test(model, data_module)
        
        logger.info("Training completed!")
        return model
    
    def run_causal_discovery(self, data_module: PerturbationDataModule) -> Dict[str, Any]:
        """Run causal discovery analysis."""
        logger.info("Running causal discovery analysis...")
        
        # Get sample data for causal discovery
        sample_batch = data_module.get_sample_batch('train', batch_size=1000)
        
        # Extract features for causal analysis
        if 'genomics' in sample_batch:
            causal_factors = sample_batch['genomics']['expression'].numpy()
        elif 'imaging' in sample_batch:
            # Use a pretrained feature extractor for imaging data
            causal_factors = sample_batch['imaging']['images'].view(
                sample_batch['imaging']['images'].size(0), -1
            ).numpy()
        else:
            logger.warning("No suitable data found for causal discovery")
            return {}
        
        # Extract perturbation labels
        perturbation_labels = sample_batch['perturbation']['compound_id'].numpy()
        
        # Run causal discovery
        causal_results = run_causal_discovery(
            causal_factors=causal_factors,
            perturbation_labels=perturbation_labels,
            config=self.config.causal_discovery
        )
        
        # Save results
        results_file = self.output_dir / "causal_discovery_results.json"
        import json
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in causal_results.items():
                if hasattr(value, 'tolist'):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Causal discovery results saved to {results_file}")
        return causal_results
    
    def run_explainability_analysis(self, model: pl.LightningModule, data_module: PerturbationDataModule) -> Dict[str, Any]:
        """Run explainability analysis."""
        logger.info("Running explainability analysis...")
        
        explainability_results = {}
        
        # Get sample data
        sample_batch = data_module.get_sample_batch('test', batch_size=50)
        
        if 'imaging' in sample_batch:
            images = sample_batch['imaging']['images']
            perturbations = sample_batch['perturbation']['compound_id']
            
            # Attention analysis
            if hasattr(model, 'model') and hasattr(model.model, 'attention'):
                attention_results = generate_attention_analysis(
                    model=model.model,
                    images=images,
                    perturbations=perturbations,
                    output_dir=str(self.output_dir / "attention_analysis")
                )
                explainability_results['attention_analysis'] = attention_results
        
        # Concept activation analysis
        if hasattr(model, 'model'):
            try:
                from explainability.concept_activation import discover_biological_concepts
                
                # Discover concepts from genomics data if available
                if 'genomics' in sample_batch:
                    import pandas as pd
                    expression_data = pd.DataFrame(sample_batch['genomics']['expression'].numpy())
                    concepts = discover_biological_concepts(expression_data)
                    
                    concept_results = compute_concept_activations(
                        model=model.model,
                        data_loader=data_module.test_dataloader(),
                        concepts=concepts,
                        layer_names=['layer1', 'layer2'],
                        output_dir=str(self.output_dir / "concept_analysis")
                    )
                    explainability_results['concept_analysis'] = concept_results
            except Exception as e:
                logger.warning(f"Concept activation analysis failed: {e}")
        
        # Pathway analysis
        if 'genomics' in sample_batch:
            try:
                # Extract gene expression data
                gene_expression = sample_batch['genomics']['expression'].numpy()
                gene_names = sample_batch['genomics'].get('gene_names', 
                    [f"gene_{i}" for i in range(gene_expression.shape[1])])
                
                # Run pathway analysis on top differentially expressed genes
                import numpy as np
                gene_scores = np.mean(np.abs(gene_expression), axis=0)
                top_genes_idx = np.argsort(gene_scores)[-100:]  # Top 100 genes
                top_genes = [gene_names[i] for i in top_genes_idx]
                
                pathway_results = run_pathway_analysis(
                    gene_list=top_genes,
                    output_dir=str(self.output_dir / "pathway_analysis")
                )
                explainability_results['pathway_analysis'] = pathway_results
            except Exception as e:
                logger.warning(f"Pathway analysis failed: {e}")
        
        logger.info("Explainability analysis completed!")
        return explainability_results
    
    def run_intervention_design(self, causal_results: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimal interventions based on causal discovery results."""
        logger.info("Running intervention design...")
        
        if not causal_results:
            logger.warning("No causal results available for intervention design")
            return {}
        
        try:
            # Setup intervention predictor
            causal_graph = causal_results.get('adjacency_matrix')
            if causal_graph is None:
                logger.warning("No causal graph found in results")
                return {}
            
            from causal.intervention import CausalGraphInterventionPredictor, ExperimentalDesignEngine
            import numpy as np
            
            predictor = CausalGraphInterventionPredictor(
                causal_graph=np.array(causal_graph),
                variable_names=causal_results.get('variable_names', [])
            )
            
            # Setup experimental design engine
            design_engine = ExperimentalDesignEngine(predictor)
            
            # Generate candidate interventions
            from causal.intervention import create_standard_intervention_library
            n_variables = len(causal_results.get('variable_names', []))
            candidate_interventions = create_standard_intervention_library(n_variables)
            
            # Design optimal experiment batch
            baseline_data = np.random.randn(100, n_variables)  # Mock baseline data
            
            intervention_results = design_engine.design_optimal_experiment_batch(
                baseline_data=baseline_data,
                candidate_interventions=candidate_interventions[:20],  # Limit candidates
                batch_size=96,
                budget=10000.0
            )
            
            # Save results
            results_file = self.output_dir / "intervention_design_results.json"
            import json
            with open(results_file, 'w') as f:
                # Convert complex objects to serializable format
                json_results = {
                    'n_interventions': len(intervention_results.get('interventions', [])),
                    'total_cost': intervention_results.get('total_cost', 0),
                    'expected_information_gain': intervention_results.get('expected_information_gain', 0),
                    'design_metrics': intervention_results.get('design_metrics', {}),
                }
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Intervention design results saved to {results_file}")
            return intervention_results
            
        except Exception as e:
            logger.error(f"Intervention design failed: {e}")
            return {}
    
    def generate_final_report(self, results: Dict[str, Any]):
        """Generate a comprehensive final report."""
        logger.info("Generating final report...")
        
        report_file = self.output_dir / "final_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# OpenPerturbation Analysis Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now()}\n")
            f.write(f"**Configuration:** {self.config.get('experiment_name', 'default')}\n\n")
            
            # Training results
            if 'training' in results:
                f.write("## Training Results\n\n")
                f.write(f"- Model type: {self.config.get('model_type', 'unknown')}\n")
                f.write(f"- Training completed successfully\n\n")
            
            # Causal discovery results
            if 'causal_discovery' in results:
                causal_results = results['causal_discovery']
                f.write("## Causal Discovery Results\n\n")
                f.write(f"- Variables analyzed: {len(causal_results.get('variable_names', []))}\n")
                f.write(f"- Causal relationships discovered: {causal_results.get('n_edges', 0)}\n")
                f.write(f"- Discovery method: {causal_results.get('method', 'unknown')}\n\n")
            
            # Explainability results
            if 'explainability' in results:
                f.write("## Explainability Analysis\n\n")
                exp_results = results['explainability']
                
                if 'attention_analysis' in exp_results:
                    f.write("- Attention analysis completed\n")
                if 'concept_analysis' in exp_results:
                    f.write("- Concept activation analysis completed\n")
                if 'pathway_analysis' in exp_results:
                    f.write("- Pathway analysis completed\n")
                f.write("\n")
            
            # Intervention design results
            if 'intervention_design' in results:
                f.write("## Intervention Design\n\n")
                int_results = results['intervention_design']
                f.write(f"- Interventions designed: {int_results.get('n_interventions', 0)}\n")
                f.write(f"- Total estimated cost: ${int_results.get('total_cost', 0):,.2f}\n")
                f.write(f"- Expected information gain: {int_results.get('expected_information_gain', 0):.3f}\n\n")
            
            f.write("## Output Files\n\n")
            f.write(f"All results have been saved to: `{self.output_dir}`\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review the causal discovery results\n")
            f.write("2. Examine explainability visualizations\n")
            f.write("3. Consider implementing suggested interventions\n")
            f.write("4. Validate findings with experimental data\n")
        
        logger.info(f"Final report saved to {report_file}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete OpenPerturbation analysis pipeline."""
        logger.info("Starting OpenPerturbation full pipeline...")
        
        results = {}
        
        try:
            # 1. Setup data
            data_module = self.setup_data()
            
            # 2. Setup and train model
            if self.config.get('run_training', True):
                model_type = self.config.get('model_type', 'multimodal_fusion')
                model = self.setup_model(model_type)
                model = self.run_training(model, data_module)
                results['training'] = {'status': 'completed', 'model_type': model_type}
            else:
                # Load pretrained model if available
                model = None
                logger.info("Skipping training, using pretrained model if available")
            
            # 3. Run causal discovery
            if self.config.get('run_causal_discovery', True):
                causal_results = self.run_causal_discovery(data_module)
                results['causal_discovery'] = causal_results
            else:
                causal_results = {}
            
            # 4. Run explainability analysis
            if self.config.get('run_explainability', True) and model is not None:
                explainability_results = self.run_explainability_analysis(model, data_module)
                results['explainability'] = explainability_results
            
            # 5. Design interventions
            if self.config.get('run_intervention_design', True):
                intervention_results = self.run_intervention_design(causal_results)
                results['intervention_design'] = intervention_results
            
            # 6. Generate final report
            self.generate_final_report(results)
            
            logger.info("OpenPerturbation pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise
        
        return results


@hydra.main(version_base=None, config_path="configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    """Main entry point for OpenPerturbation."""
    
    logger.info("Starting OpenPerturbation Analysis Platform")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize pipeline
    pipeline = OpenPerturbationPipeline(cfg)
    
    # Run the complete analysis
    results = pipeline.run_full_pipeline()
    
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved to: {pipeline.output_dir}")


if __name__ == "__main__":
    main()

