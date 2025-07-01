#!/usr/bin/env python3
"""
Agent Tools for OpenPerturbation Platform

Tools and utilities for AI agents to interact with the OpenPerturbation system,
execute analyses, and provide automated insights.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import json
import asyncio
from datetime import datetime
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTool:
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for AI agent integration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameter_schema()
        }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for the tool."""
        return {"type": "object", "properties": {}}


class DataAnalysisTool(BaseTool):
    """Tool for performing data analysis."""
    
    def __init__(self):
        super().__init__(
            name="data_analysis",
            description="Perform statistical analysis and data exploration on datasets"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis."""
        start_time = time.time()
        
        try:
            dataset_id = parameters.get("dataset_id")
            analysis_type = parameters.get("analysis_type", "descriptive")
            include_plots = parameters.get("include_plots", True)
            
            # Mock data analysis results
            results = {
                "dataset_id": dataset_id,
                "analysis_type": analysis_type,
                "dataset_size": np.random.randint(100, 10000),
                "num_features": np.random.randint(10, 1000),
                "summary_stats": {
                    "mean": np.random.uniform(0, 100),
                    "std": np.random.uniform(1, 50),
                    "min": np.random.uniform(-10, 0),
                    "max": np.random.uniform(100, 1000)
                },
                "missing_values": np.random.randint(0, 100),
                "outliers_detected": np.random.randint(0, 50),
                "data_quality_score": np.random.uniform(0.7, 1.0),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            if include_plots:
                results["visualizations"] = [
                    "histogram", "boxplot", "correlation_matrix", "pca_plot"
                ]
            
            logger.info(f"Data analysis completed for dataset {dataset_id}")
            return results
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for data analysis."""
        return {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["descriptive", "exploratory", "statistical"],
                    "description": "Type of analysis to perform"
                },
                "include_plots": {
                    "type": "boolean",
                    "description": "Whether to generate visualizations"
                }
            },
            "required": ["dataset_id"]
        }


class CausalDiscoveryTool(BaseTool):
    """Tool for causal discovery analysis."""
    
    def __init__(self):
        super().__init__(
            name="causal_discovery",
            description="Discover causal relationships in data using various algorithms"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute causal discovery analysis."""
        start_time = time.time()
        
        try:
            dataset_id = parameters.get("dataset_id")
            method = parameters.get("method", "pc")
            alpha = parameters.get("alpha", 0.05)
            max_vars = parameters.get("max_vars", 100)
            
            # Mock causal discovery results
            n_variables = min(max_vars, np.random.randint(10, 100))
            n_edges = np.random.randint(5, n_variables * 2)
            
            results = {
                "dataset_id": dataset_id,
                "method": method,
                "alpha": alpha,
                "n_variables": n_variables,
                "n_edges": n_edges,
                "variable_names": [f"var_{i}" for i in range(n_variables)],
                "causal_graph": np.random.rand(n_variables, n_variables).tolist(),
                "edge_weights": np.random.uniform(0, 1, n_edges).tolist(),
                "causal_strength": np.random.uniform(0.3, 0.9),
                "confidence_scores": np.random.uniform(0.6, 0.99, n_edges).tolist(),
                "significant_relationships": n_edges,
                "algorithm_convergence": True,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Causal discovery completed: {n_edges} relationships found")
            return results
            
        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for causal discovery."""
        return {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset for causal discovery"
                },
                "method": {
                    "type": "string",
                    "enum": ["pc", "ges", "fci", "lingam"],
                    "description": "Causal discovery algorithm to use"
                },
                "alpha": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.1,
                    "description": "Significance level for statistical tests"
                },
                "max_vars": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 1000,
                    "description": "Maximum number of variables to analyze"
                }
            },
            "required": ["dataset_id"]
        }


class ExplainabilityTool(BaseTool):
    """Tool for explainability analysis."""
    
    def __init__(self):
        super().__init__(
            name="explainability",
            description="Generate explainability reports for trained models"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute explainability analysis."""
        start_time = time.time()
        
        try:
            model_id = parameters.get("model_id")
            dataset_id = parameters.get("dataset_id")
            analysis_types = parameters.get("analysis_types", ["attention", "concept", "pathway"])
            num_samples = parameters.get("num_samples", 100)
            
            results = {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "analysis_types": analysis_types,
                "num_samples": num_samples,
                "results": {},
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Mock attention analysis
            if "attention" in analysis_types:
                results["results"]["attention_analysis"] = {
                    "n_attention_maps": np.random.randint(5, 20),
                    "top_features": [f"feature_{i}" for i in range(10)],
                    "attention_scores": np.random.uniform(0, 1, 10).tolist(),
                    "visualization_files": ["attention_map_1.png", "attention_map_2.png"]
                }
            
            # Mock concept analysis
            if "concept" in analysis_types:
                results["results"]["concept_analysis"] = {
                    "n_concepts": np.random.randint(10, 50),
                    "biological_concepts": [
                        "cell_cycle", "apoptosis", "metabolism", "signaling", "transcription"
                    ],
                    "concept_scores": np.random.uniform(0, 1, 5).tolist(),
                    "concept_activations": np.random.uniform(0, 1, (num_samples, 5)).tolist()
                }
            
            # Mock pathway analysis
            if "pathway" in analysis_types:
                results["results"]["pathway_analysis"] = {
                    "enriched_pathways": [
                        "KEGG_PATHWAY_1", "GO_BIOLOGICAL_PROCESS_1", "REACTOME_PATHWAY_1"
                    ],
                    "p_values": [0.001, 0.005, 0.01],
                    "fold_enrichment": [2.5, 1.8, 3.2],
                    "pathway_genes": {
                        "KEGG_PATHWAY_1": ["gene1", "gene2", "gene3"],
                        "GO_BIOLOGICAL_PROCESS_1": ["gene2", "gene4", "gene5"],
                        "REACTOME_PATHWAY_1": ["gene1", "gene3", "gene6"]
                    }
                }
            
            results["key_findings"] = [
                "Model shows high attention to cell cycle genes",
                "Strong activation of apoptosis-related concepts",
                "Enrichment in metabolic pathways"
            ]
            
            logger.info(f"Explainability analysis completed for model {model_id}")
            return results
            
        except Exception as e:
            logger.error(f"Explainability analysis failed: {e}")
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for explainability analysis."""
        return {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "ID of the trained model to explain"
                },
                "dataset_id": {
                    "type": "string",
                    "description": "ID of the dataset used for analysis"
                },
                "analysis_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["attention", "concept", "pathway", "feature_importance"]
                    },
                    "description": "Types of explainability analysis to perform"
                },
                "num_samples": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 1000,
                    "description": "Number of samples to analyze"
                }
            },
            "required": ["model_id", "dataset_id"]
        }


class InterventionDesignTool(BaseTool):
    """Tool for intervention design and optimization."""
    
    def __init__(self):
        super().__init__(
            name="intervention_design",
            description="Design optimal interventions based on causal models"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intervention design."""
        start_time = time.time()
        
        try:
            causal_results_id = parameters.get("causal_results_id")
            target_outcomes = parameters.get("target_outcomes", {})
            num_interventions = parameters.get("num_interventions", 10)
            budget_constraints = parameters.get("budget_constraints", {})
            
            # Mock intervention design results
            interventions = []
            total_cost = 0
            
            for i in range(num_interventions):
                cost = np.random.uniform(100, 1000)
                total_cost += cost
                
                intervention = {
                    "intervention_id": f"intervention_{i+1}",
                    "description": f"Intervention strategy {i+1}",
                    "target_variables": [f"var_{j}" for j in np.random.choice(10, np.random.randint(1, 4), replace=False)],
                    "intervention_values": np.random.uniform(-2, 2, np.random.randint(1, 4)).tolist(),
                    "expected_effect": np.random.uniform(0.1, 1.0),
                    "confidence": np.random.uniform(0.6, 0.99),
                    "estimated_cost": cost,
                    "feasibility_score": np.random.uniform(0.5, 1.0),
                    "risk_assessment": np.random.choice(["low", "medium", "high"]),
                    "expected_outcomes": {
                        outcome: np.random.uniform(0, 1) 
                        for outcome in target_outcomes.keys()
                    }
                }
                interventions.append(intervention)
            
            results = {
                "causal_results_id": causal_results_id,
                "target_outcomes": target_outcomes,
                "num_interventions": num_interventions,
                "budget_constraints": budget_constraints,
                "interventions": interventions,
                "total_cost": total_cost,
                "avg_effectiveness": np.mean([i["expected_effect"] for i in interventions]),
                "avg_confidence": np.mean([i["confidence"] for i in interventions]),
                "optimization_metrics": {
                    "cost_effectiveness": np.random.uniform(0.6, 0.95),
                    "outcome_coverage": np.random.uniform(0.7, 1.0),
                    "feasibility": np.random.uniform(0.8, 1.0)
                },
                "recommendations": [
                    "Prioritize interventions with high confidence scores",
                    "Consider combining low-cost interventions",
                    "Validate top 3 interventions experimentally"
                ],
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Intervention design completed: {num_interventions} strategies generated")
            return results
            
        except Exception as e:
            logger.error(f"Intervention design failed: {e}")
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for intervention design."""
        return {
            "type": "object",
            "properties": {
                "causal_results_id": {
                    "type": "string",
                    "description": "ID of causal discovery results to base interventions on"
                },
                "target_outcomes": {
                    "type": "object",
                    "description": "Target outcomes and their desired values"
                },
                "num_interventions": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of intervention strategies to generate"
                },
                "budget_constraints": {
                    "type": "object",
                    "description": "Budget and resource constraints"
                }
            },
            "required": ["causal_results_id", "target_outcomes"]
        }


# Registry of available tools
TOOL_REGISTRY = {
    "data_analysis": DataAnalysisTool(),
    "causal_discovery": CausalDiscoveryTool(),
    "explainability": ExplainabilityTool(),
    "intervention_design": InterventionDesignTool()
}


def get_available_tools() -> List[Dict[str, Any]]:
    """Get list of available tools and their schemas."""
    return [tool.get_schema() for tool in TOOL_REGISTRY.values()]


async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool with given parameters."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool = TOOL_REGISTRY[tool_name]
    return await tool.execute(parameters)


def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """Get schema for a specific tool."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    return TOOL_REGISTRY[tool_name].get_schema()


def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> bool:
    """Validate parameters for a tool."""
    if tool_name not in TOOL_REGISTRY:
        return False
    
    schema = TOOL_REGISTRY[tool_name]._get_parameter_schema()
    required_params = schema.get("required", [])
    
    # Basic validation - check required parameters
    for param in required_params:
        if param not in parameters:
            logger.error(f"Missing required parameter: {param}")
            return False
    
    return True


# Utility functions for agent integration
def format_data_for_ai(data: Dict[str, Any]) -> str:
    """Format data for AI consumption."""
    try:
        # Limit data size for AI processing
        if isinstance(data, dict):
            # Sample large data structures
            formatted = {}
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 100:
                    formatted[key] = f"[List with {len(value)} items - showing first 5: {value[:5]}]"
                elif isinstance(value, dict) and len(value) > 50:
                    formatted[key] = f"[Dict with {len(value)} keys - showing first 5: {dict(list(value.items())[:5])}]"
                else:
                    formatted[key] = value
            return json.dumps(formatted, indent=2, default=str)
        else:
            return str(data)
    except Exception:
        return str(data)


async def test_tools():
    """Test all available tools."""
    logger.info("Testing agent tools...")
    
    test_parameters = {
        "data_analysis": {
            "dataset_id": "test_dataset_001",
            "analysis_type": "descriptive",
            "include_plots": True
        },
        "causal_discovery": {
            "dataset_id": "test_dataset_001",
            "method": "pc",
            "alpha": 0.05,
            "max_vars": 50
        },
        "explainability": {
            "model_id": "test_model_001",
            "dataset_id": "test_dataset_001",
            "analysis_types": ["attention", "concept", "pathway"],
            "num_samples": 100
        },
        "intervention_design": {
            "causal_results_id": "test_causal_001",
            "target_outcomes": {"cell_viability": 0.8, "pathway_activity": 0.6},
            "num_interventions": 5,
            "budget_constraints": {"max_cost": 5000}
        }
    }
    
    for tool_name, params in test_parameters.items():
        logger.info(f"Testing {tool_name}...")
        try:
            result = await execute_tool(tool_name, params)
            logger.info(f"✅ {tool_name} completed successfully")
            logger.info(f"   Execution time: {result.get('execution_time', 'N/A')} seconds")
        except Exception as e:
            logger.error(f"❌ {tool_name} failed: {e}")
    
    logger.info("Tool testing completed!")


if __name__ == "__main__":
    # Run tool tests
    asyncio.run(test_tools())
