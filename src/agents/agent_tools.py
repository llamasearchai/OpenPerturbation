"""
Agent tools for perturbation analysis.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import torch

logger = logging.getLogger(__name__)


class PerturbationAnalysisTools:
    """Tools for perturbation analysis used by OpenAI agents."""
    
    def __init__(self):
        """Initialize analysis tools."""
        self.logger = logger
        
    def analyze_gene_expression(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze gene expression data."""
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
            
        return {
            'mean_expression': float(np.mean(data_array)),
            'std_expression': float(np.std(data_array)),
            'num_genes': data_array.shape[1] if len(data_array.shape) > 1 else 1,
            'num_samples': data_array.shape[0],
            'expression_range': [float(np.min(data_array)), float(np.max(data_array))]
        }
    
    def identify_differentially_expressed_genes(self, 
                                              control_data: np.ndarray,
                                              treatment_data: np.ndarray,
                                              threshold: float = 2.0) -> Dict[str, Any]:
        """Identify differentially expressed genes."""
        
        # Calculate fold changes
        control_mean = np.mean(control_data, axis=0)
        treatment_mean = np.mean(treatment_data, axis=0)
        
        # Avoid division by zero
        fold_changes = np.divide(treatment_mean, control_mean, 
                               out=np.ones_like(treatment_mean), 
                               where=control_mean!=0)
        
        # Log2 fold changes
        log2_fc = np.log2(fold_changes + 1e-10)
        
        # Identify significant changes
        upregulated = np.where(log2_fc > np.log2(threshold))[0]
        downregulated = np.where(log2_fc < -np.log2(threshold))[0]
        
        return {
            'upregulated_genes': upregulated.tolist(),
            'downregulated_genes': downregulated.tolist(),
            'log2_fold_changes': log2_fc.tolist(),
            'num_upregulated': len(upregulated),
            'num_downregulated': len(downregulated)
        }
    
    def calculate_pathway_enrichment(self, gene_list: List[str]) -> Dict[str, Any]:
        """Calculate pathway enrichment for a gene list."""
        
        # Mock pathway analysis - in real implementation would use proper databases
        mock_pathways = [
            'Cell cycle regulation',
            'Apoptosis signaling',
            'DNA repair mechanisms',
            'Metabolic pathways',
            'Immune response'
        ]
        
        # Simulate enrichment scores
        enrichment_scores = np.random.uniform(0.1, 3.0, len(mock_pathways))
        p_values = np.random.uniform(0.001, 0.1, len(mock_pathways))
        
        return {
            'pathways': mock_pathways,
            'enrichment_scores': enrichment_scores.tolist(),
            'p_values': p_values.tolist(),
            'significant_pathways': [p for i, p in enumerate(mock_pathways) 
                                   if p_values[i] < 0.05]
        }
    
    def predict_drug_targets(self, gene_signature: List[str]) -> Dict[str, Any]:
        """Predict potential drug targets based on gene signature."""
        
        # Mock drug target prediction
        mock_targets = [
            {'target': 'EGFR', 'confidence': 0.85, 'mechanism': 'Kinase inhibition'},
            {'target': 'TP53', 'confidence': 0.72, 'mechanism': 'Tumor suppressor activation'},
            {'target': 'VEGFA', 'confidence': 0.68, 'mechanism': 'Angiogenesis inhibition'}
        ]
        
        return {
            'predicted_targets': mock_targets,
            'num_targets': len(mock_targets),
            'high_confidence_targets': [t for t in mock_targets if t['confidence'] > 0.8]
        }
    
    def analyze_cellular_morphology(self, image_features: np.ndarray) -> Dict[str, Any]:
        """Analyze cellular morphology features."""
        
        return {
            'mean_cell_area': float(np.mean(image_features[:, 0]) if image_features.shape[1] > 0 else 0),
            'mean_cell_perimeter': float(np.mean(image_features[:, 1]) if image_features.shape[1] > 1 else 0),
            'morphology_diversity': float(np.std(image_features.flatten())),
            'num_cells': image_features.shape[0],
            'feature_dimensions': image_features.shape[1]
        }
    
    def generate_experimental_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate experimental recommendations based on analysis."""
        
        recommendations = []
        
        if 'upregulated_genes' in analysis_results:
            num_up = len(analysis_results['upregulated_genes'])
            if num_up > 10:
                recommendations.append(f"Investigate {num_up} upregulated genes for potential therapeutic targets")
        
        if 'downregulated_genes' in analysis_results:
            num_down = len(analysis_results['downregulated_genes'])
            if num_down > 10:
                recommendations.append(f"Study {num_down} downregulated genes for pathway disruption")
        
        if 'significant_pathways' in analysis_results:
            pathways = analysis_results['significant_pathways']
            if pathways:
                recommendations.append(f"Focus on {len(pathways)} significantly enriched pathways")
        
        if not recommendations:
            recommendations.append("Perform additional validation experiments")
            
        return recommendations
