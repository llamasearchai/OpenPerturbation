"""
Agent tools for perturbation analysis.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import torch

logger = logging.getLogger(__name__)


class AgentTools:
    """Static utility tools for AI agents."""
    
    @staticmethod
    def format_data_for_ai(data: Dict[str, Any]) -> str:
        """Format experimental data for AI consumption."""
        if isinstance(data, dict):
            formatted_lines = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if 'viability' in key.lower() or 'percent' in key.lower():
                        formatted_lines.append(f"{key}: {value * 100:.1f}%")
                    else:
                        formatted_lines.append(f"{key}: {value}")
                elif isinstance(value, list):
                    if len(value) <= 5:
                        formatted_lines.append(f"{key}: {value}")
                    else:
                        formatted_lines.append(f"{key}: {len(value)} values (range: {min(value):.3f} - {max(value):.3f})")
                else:
                    formatted_lines.append(f"{key}: {str(value)}")
            return "\n".join(formatted_lines)
        else:
            return str(data)
    
    @staticmethod
    def parse_ai_response(response: str) -> Dict[str, Any]:
        """Parse AI response, attempting JSON first, then fallback."""
        response = response.strip()
        
        # Try to extract JSON from response
        try:
            # Look for JSON blocks
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try parsing the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Fallback: create structured response
        return {
            "analysis": response,
            "confidence": 0.7,
            "recommendations": [],
            "parsed": False
        }
    
    @staticmethod
    def build_analysis_prompt(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None, query: Optional[str] = None) -> str:
        """Build analysis prompt for AI."""
        formatted_data = AgentTools.format_data_for_ai(data)
        
        prompt = f"""
        Analyze the following perturbation biology data:
        
        Data:
        {formatted_data}
        """
        
        if context:
            prompt += f"\nContext: {json.dumps(context, indent=2)}"
        
        if query:
            prompt += f"\nSpecific Question: {query}"
        
        prompt += """
        
        Please provide:
        1. Key biological insights
        2. Statistical interpretation
        3. Potential mechanisms
        4. Follow-up recommendations
        5. Confidence assessment (0.0-1.0)
        
        Format as JSON when possible.
        """
        
        return prompt
    
    @staticmethod
    def validate_experiment_design(experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experimental design parameters."""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['type']
        for field in required_fields:
            if field not in experiment:
                errors.append(f"Missing required field: {field}")
        
        # Validate experiment type
        valid_types = ['dose_response', 'time_course', 'compound_screening', 'genetic_perturbation']
        if 'type' in experiment and experiment['type'] not in valid_types:
            errors.append(f"Invalid experiment type: {experiment['type']}")
        
        # Validate concentrations if present
        if 'concentrations' in experiment:
            concs = experiment['concentrations']
            if not isinstance(concs, list) or len(concs) == 0:
                errors.append("Concentrations must be a non-empty list")
            elif any(c < 0 for c in concs if isinstance(c, (int, float))):
                errors.append("Concentrations must be non-negative")
        
        # Validate timepoints
        if 'timepoints' in experiment:
            timepoints = experiment['timepoints']
            if not isinstance(timepoints, list) or len(timepoints) == 0:
                warnings.append("No timepoints specified")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


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
            
        # Safe shape access
        num_samples = data_array.size if data_array.ndim == 1 else data_array.shape[0]
        num_genes = 1 if data_array.ndim == 1 else data_array.shape[1]
        
        return {
            'mean_expression': float(np.mean(data_array)),
            'std_expression': float(np.std(data_array)),
            'num_genes': num_genes,
            'num_samples': num_samples,
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
        
        # Safe array indexing with shape checking
        mean_area = 0.0
        mean_perimeter = 0.0
        
        if len(image_features.shape) >= 2 and image_features.shape[1] > 0:
            mean_area = float(np.mean(image_features[:, 0]))
            if image_features.shape[1] > 1:
                mean_perimeter = float(np.mean(image_features[:, 1]))
        
        # Safe shape access for num_cells
        num_cells = image_features.shape[0] if len(image_features.shape) > 0 else 0
        feature_dimensions = image_features.shape[1] if len(image_features.shape) >= 2 else 0
        
        return {
            'mean_cell_area': mean_area,
            'mean_cell_perimeter': mean_perimeter,
            'morphology_diversity': float(np.std(image_features.flatten())),
            'num_cells': num_cells,
            'feature_dimensions': feature_dimensions
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
