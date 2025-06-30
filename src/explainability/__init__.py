"""
OpenPerturbations Explainability Module

This module provides comprehensive explainability analysis for perturbation biology models,
including attention visualization, concept activation mapping, and pathway analysis.
"""

from .attention_maps import (
    AttentionVisualization as AttentionVisualizer,
    BiologicalAttentionAnalyzer as AttentionAnalyzer,
    generate_attention_analysis,
    create_attention_dashboard,
)

from .concept_activation import (
    ConceptActivationMapper,
    BiologicalConcept,
    ConceptAnalysisResult,
    compute_concept_activations,
    discover_biological_concepts,
)

from .pathway_analysis import (
    PathwayDatabase,
    PathwayEnrichmentAnalyzer,
    PathwayEnrichmentResult,
    NetworkAnalyzer,
    NetworkAnalysisResult,
    PathwayVisualizer,
    run_pathway_analysis,
    # validate_gene_symbols,
    # compare_pathway_analyses,
    # create_pathway_database_from_gmt,
    # export_results_to_excel
)

__all__ = [
    # Attention Analysis
    "AttentionVisualizer",
    "AttentionAnalyzer",
    "generate_attention_analysis",
    "create_attention_dashboard",
    # Concept Activation
    "ConceptActivationMapper",
    "BiologicalConcept",
    "ConceptAnalysisResult",
    "compute_concept_activations",
    "discover_biological_concepts",
    # Pathway Analysis
    "PathwayDatabase",
    "PathwayEnrichmentAnalyzer",
    "PathwayEnrichmentResult",
    "NetworkAnalyzer",
    "NetworkAnalysisResult",
    "PathwayVisualizer",
    "run_pathway_analysis",
    "validate_gene_symbols",
    "compare_pathway_analyses",
    "create_pathway_database_from_gmt",
    "export_results_to_excel",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "OpenPerturbations Team"
__email__ = "contact@openperturbations.org"
__description__ = "Explainability analysis for perturbation biology AI models"
