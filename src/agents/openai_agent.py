"""
OpenAI-powered agents for perturbation analysis.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import os
from pathlib import Path

# OpenAI imports
try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    ChatCompletionMessageParam = None
    OPENAI_AVAILABLE = False

from .agent_tools import PerturbationAnalysisTools

try:
    from dotenv import load_dotenv
    # Load .env from project root
    _env_path = Path(__file__).resolve().parents[2] / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except Exception:
    # dotenv is optional; ignore errors silently
    pass

logger = logging.getLogger(__name__)


class OpenPerturbationAgent:
    """Base class for OpenAI-powered perturbation analysis agents."""
    
    def __init__(
        self, api_key: Optional[str] = None, model: str = "gpt-4", temperature: float = 0.1
    ):
        """Initialize the OpenAI agent."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")

        # Retrieve API key from parameter, environment variable, or .env file in that priority
        _env_key = os.getenv('OPENAI_API_KEY')
        api_key_final = api_key or _env_key

        if api_key_final and OpenAI is not None:
            self.client = OpenAI(api_key=api_key_final)
        else:
            self.client = None
            logger.warning("OpenAI API key not found. Agent will run in mock mode. Set OPENAI_API_KEY in environment or .env file.")
            
        self.model = model
        self.temperature = temperature
        self.tools = PerturbationAnalysisTools()

    async def analyze_data(
        self, data_description: str, analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze perturbation data using OpenAI."""
        
        if not self.client:
            return self._mock_analysis_response(data_description, analysis_type)

        try:
            # Create properly formatted messages
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": "You are an expert in perturbation biology and data analysis. Provide detailed, scientific analysis of biological data."
                },
                {
                    "role": "user", 
                    "content": f"Analyze this perturbation data: {data_description}. Analysis type: {analysis_type}"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, max_tokens=2000
            )

            analysis_plan = response.choices[0].message.content

            return {
                "analysis_plan": analysis_plan,
                "analysis_type": analysis_type,
                "confidence": 0.85,
                "estimated_runtime": self._estimate_runtime(analysis_type),
                "recommendations": analysis_plan.split('\n')[-3:] if analysis_plan else []
            }

        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._mock_analysis_response(data_description, analysis_type)

    async def interpret_results(self, results: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Interpret analysis results using OpenAI."""
        
        if not self.client:
            return self._mock_interpretation_response(results, context)

        try:
            # Create properly formatted messages  
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": "You are an expert in interpreting biological analysis results. Provide clear, actionable insights."
                },
                {
                    "role": "user",
                    "content": f"Interpret these results: {str(results)[:1000]}. Context: {context}"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, max_tokens=2000
            )

            interpretation = response.choices[0].message.content

            return {
                "interpretation": interpretation,
                "key_findings": self._extract_key_findings(results),
                "confidence": self._assess_confidence(results),
                "next_steps": self._generate_followup_recommendations(results),
                "biological_significance": interpretation
            }

        except Exception as e:
            logger.error(f"OpenAI interpretation failed: {e}")
            return self._mock_interpretation_response(results, context)

    def _estimate_runtime(self, analysis_type: str) -> str:
        """Estimate analysis runtime."""
        runtime_map = {
            "comprehensive": "2-4 hours",
            "quick": "15-30 minutes", 
            "detailed": "4-8 hours",
            "exploratory": "1-2 hours"
        }
        return runtime_map.get(analysis_type, "1-2 hours")

    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from results."""
        findings = []
        
        if "upregulated_genes" in results:
            findings.append(f"Found {len(results['upregulated_genes'])} upregulated genes")
        if "downregulated_genes" in results:
            findings.append(f"Found {len(results['downregulated_genes'])} downregulated genes")
        if "significant_pathways" in results:
            findings.append(f"Identified {len(results['significant_pathways'])} significant pathways")
            
        return findings or ["No significant findings detected"]

    def _assess_confidence(self, results: Dict[str, Any]) -> float:
        """Assess confidence in results."""
        confidence_factors = []
        
        if "p_values" in results:
            p_vals = results["p_values"]
            if isinstance(p_vals, list) and p_vals:
                confidence_factors.append(float(1.0 - np.mean(p_vals)))
        
        if "num_samples" in results:
            sample_factor = min(1.0, results["num_samples"] / 100)
            confidence_factors.append(float(sample_factor))
            
        return float(np.mean(confidence_factors)) if confidence_factors else 0.7

    def _generate_followup_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate follow-up recommendations."""
        recommendations = []
        
        if "upregulated_genes" in results and len(results["upregulated_genes"]) > 5:
            recommendations.append("Validate upregulated genes with qPCR")
        if "significant_pathways" in results:
            recommendations.append("Perform pathway-specific functional assays")
        if "predicted_targets" in results:
            recommendations.append("Test predicted drug targets experimentally")
            
        return recommendations or ["Perform additional validation experiments"]

    def _mock_analysis_response(self, data_description: str, analysis_type: str) -> Dict[str, Any]:
        """Mock response when OpenAI is not available."""
        return {
            "analysis_plan": f"Mock analysis of {data_description} using {analysis_type} approach",
            "analysis_type": analysis_type,
            "confidence": 0.5,
            "estimated_runtime": self._estimate_runtime(analysis_type),
            "recommendations": ["Validate results", "Perform controls", "Replicate experiments"]
        }

    def _mock_interpretation_response(self, results: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Mock interpretation when OpenAI is not available."""
        return {
            "interpretation": f"Mock interpretation of results in context: {context}",
            "key_findings": self._extract_key_findings(results),
            "confidence": self._assess_confidence(results),
            "next_steps": self._generate_followup_recommendations(results),
            "biological_significance": "Moderate biological significance detected"
        }


class AnalysisAgent(OpenPerturbationAgent):
    """Specialized agent for data analysis tasks."""
    
    def __init__(self, **kwargs):
        """Initialize analysis agent."""
        super().__init__(**kwargs)
        self.analysis_history = []

    async def recommend_preprocessing(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend preprocessing steps for data."""
        
        recommendations = {
            "normalization": "Apply quantile normalization for gene expression data",
            "filtering": "Remove genes with low expression across all samples",
            "batch_correction": "Apply ComBat for batch effect correction if needed",
            "quality_control": "Remove outlier samples based on PCA analysis"
        }
        
        if data_info.get("data_type") == "imaging":
            recommendations.update({
                "image_preprocessing": "Apply intensity normalization and background subtraction",
                "segmentation": "Use watershed algorithm for cell segmentation",
                "feature_extraction": "Extract morphological and intensity features"
            })
        elif data_info.get("data_type") == "molecular":
            recommendations.update({
                "descriptor_calculation": "Calculate molecular descriptors and fingerprints",
                "scaffold_analysis": "Perform Murcko scaffold analysis",
                "similarity_analysis": "Calculate Tanimoto similarity matrices"
            })
        
        confidence = 0.8 if data_info.get("num_samples", 0) > 50 else 0.6
        
        return {
            "preprocessing_steps": recommendations,
            "confidence": confidence,
            "estimated_time": "30-60 minutes",
            "priority_order": list(recommendations.keys())
        }


class ExperimentDesignAgent(OpenPerturbationAgent):
    """Specialized agent for experimental design."""
    
    def __init__(self, **kwargs):
        """Initialize experiment design agent."""
        super().__init__(**kwargs)
        self.design_history = []

    async def design_experiment(
        self, objectives: List[str], constraints: Dict[str, Any], budget: float = 10000.0
    ) -> Dict[str, Any]:
        """Design optimal experiments based on objectives and constraints."""
        
        # Estimate costs and timeline
        estimated_cost = budget * 0.8  # Use 80% of budget
        timeline = self._estimate_timeline(constraints)
        
        # Generate experimental conditions
        conditions = []
        for i, objective in enumerate(objectives[:5]):  # Limit to 5 objectives
            conditions.append({
                "condition_id": f"exp_{i+1}",
                "objective": objective,
                "treatment": f"Treatment_{i+1}",
                "controls": ["Vehicle", "Untreated"],
                "replicates": 3,
                "timepoints": ["6h", "24h", "48h"],
                "estimated_cost": estimated_cost / len(objectives)
            })
        
        return {
            "experimental_design": {
                "conditions": conditions,
                "total_experiments": len(conditions) * 3 * 3,  # conditions * replicates * timepoints
                "estimated_cost": estimated_cost,
                "timeline_weeks": timeline,
                "success_probability": 0.75
            },
            "recommendations": [
                "Include appropriate controls for each condition",
                "Randomize sample processing to avoid batch effects", 
                "Plan for interim analysis at 50% completion"
            ],
            "risk_assessment": {
                "technical_risk": "Medium",
                "timeline_risk": "Low",
                "budget_risk": "Low"
            }
        }

    def _estimate_timeline(self, constraints: Dict[str, Any]) -> int:
        """Estimate experiment timeline in weeks."""
        base_timeline = 8  # Base 8 weeks
        
        if constraints.get("complexity", "medium") == "high":
            base_timeline += 4
        if constraints.get("sample_size", 100) > 500:
            base_timeline += 2
        if constraints.get("num_conditions", 5) > 10:
            base_timeline += 3
            
        return min(base_timeline, 20)  # Cap at 20 weeks


def create_agent(agent_type: str = "general", **kwargs) -> OpenPerturbationAgent:
    """Factory function to create different types of agents."""
    
    agent_classes = {
        "general": OpenPerturbationAgent,
        "analysis": AnalysisAgent,
        "experiment_design": ExperimentDesignAgent
    }
    
    agent_class = agent_classes.get(agent_type, OpenPerturbationAgent)
    return agent_class(**kwargs)
