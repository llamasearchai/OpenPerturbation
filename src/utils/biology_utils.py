"""Biological utilities for pathway analysis and annotation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
import requests
import json
from pathlib import Path
import pickle
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import torch

try:
    import community as community_louvain  # type: ignore
except ImportError:  # fallback stub
    import types, sys as _sys
    community_louvain = types.ModuleType("community")
    def _best_partition(graph):
        return {n: 0 for n in graph.nodes}
    # Attribute assignment on stub – ignore type checker
    community_louvain.best_partition = _best_partition  # type: ignore[attr-defined]
    _sys.modules["community"] = community_louvain

@dataclass
class BiologicalPathway:
    """Represents a biological pathway."""
    pathway_id: str
    name: str
    description: str
    genes: List[str]
    pathway_type: str  # e.g., 'metabolic', 'signaling', 'disease'
    source_database: str  # e.g., 'KEGG', 'Reactome', 'GO'
    confidence_score: float = 1.0

@dataclass
class GeneAnnotation:
    """Gene annotation information."""
    gene_symbol: str
    gene_id: str
    description: str
    pathways: List[str]
    protein_families: List[str]
    cellular_locations: List[str]
    molecular_functions: List[str]

@dataclass
class DrugTargetInteraction:
    """Drug-target interaction information."""
    drug_name: str
    drug_id: str
    target_gene: str
    interaction_type: str  # 'inhibitor', 'activator', 'modulator'
    binding_affinity: Optional[float]
    clinical_phase: Optional[str]
    indication: Optional[str]

class BiologicalKnowledgeBase:
    """Knowledge base for biological annotations and pathways."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pathways = {}
        self.gene_annotations = {}
        self.drug_targets = {}
        self.protein_interactions = {}
        self.pathway_networks = {}
        
        # Initialize with default databases
        self._initialize_default_knowledge()
        
    def _initialize_default_knowledge(self):
        """Initialize with default biological knowledge."""
        
        # Cell cycle pathways
        self.pathways['cell_cycle'] = BiologicalPathway(
            pathway_id='GO:0007049',
            name='Cell Cycle',
            description='The progression of biochemical and morphological phases and events that occur in a cell during successive cell replication or nuclear replication events.',
            genes=['CDK1', 'CDK2', 'CDK4', 'CDK6', 'CCNA1', 'CCNA2', 'CCNB1', 'CCNB2', 'CCND1', 'CCNE1', 'TP53', 'RB1'],
            pathway_type='cell_process',
            source_database='GO'
        )
        
        # DNA damage response
        self.pathways['dna_damage_response'] = BiologicalPathway(
            pathway_id='GO:0006974',
            name='DNA Damage Response',
            description='Any process that results in a change in state or activity of a cell or an organism as a result of a DNA damage stimulus.',
            genes=['ATM', 'ATR', 'CHEK1', 'CHEK2', 'TP53', 'BRCA1', 'BRCA2', 'H2AFX', 'MDM2'],
            pathway_type='stress_response',
            source_database='GO'
        )
        
        # Apoptosis
        self.pathways['apoptosis'] = BiologicalPathway(
            pathway_id='GO:0006915',
            name='Apoptotic Process',
            description='A form of programmed cell death that begins when a cell receives an internal or external signal that triggers the activation of proteolytic caspases.',
            genes=['CASP3', 'CASP8', 'CASP9', 'BAX', 'BCL2', 'CYTC', 'APAF1', 'TP53', 'FAS', 'FADD'],
            pathway_type='cell_death',
            source_database='GO'
        )
        
        # PI3K/AKT signaling
        self.pathways['pi3k_akt'] = BiologicalPathway(
            pathway_id='hsa04151',
            name='PI3K-Akt Signaling Pathway',
            description='The PI3K-Akt signaling pathway is activated by many types of cellular stimuli and regulates fundamental cellular functions such as transcription, translation, proliferation, growth, and survival.',
            genes=['PIK3CA', 'PIK3CB', 'PIK3CD', 'AKT1', 'AKT2', 'PTEN', 'MTOR', 'TSC1', 'TSC2', 'FOXO1', 'GSK3B'],
            pathway_type='signaling',
            source_database='KEGG'
        )
        
        # MAPK signaling
        self.pathways['mapk_signaling'] = BiologicalPathway(
            pathway_id='hsa04010',
            name='MAPK Signaling Pathway',
            description='The MAPK signaling pathways mediate a wide variety of cellular behaviors in response to extracellular stimuli.',
            genes=['EGFR', 'KRAS', 'BRAF', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'JUN', 'FOS', 'ELK1'],
            pathway_type='signaling',
            source_database='KEGG'
        )
        
        # Initialize gene annotations
        self._initialize_gene_annotations()
        
        # Initialize drug-target interactions
        self._initialize_drug_targets()
    
    def _initialize_gene_annotations(self):
        """Initialize gene annotation database."""
        
        # Key genes with annotations
        gene_data = {
            'TP53': {
                'gene_id': 'ENSG00000141510',
                'description': 'Tumor protein p53, guardian of the genome',
                'pathways': ['cell_cycle', 'dna_damage_response', 'apoptosis'],
                'protein_families': ['DNA-binding transcription factor'],
                'cellular_locations': ['nucleus'],
                'molecular_functions': ['transcription regulation', 'DNA damage detection']
            },
            'BRCA1': {
                'gene_id': 'ENSG00000012048',
                'description': 'BRCA1 DNA repair associated',
                'pathways': ['dna_damage_response', 'dna_repair'],
                'protein_families': ['RING finger protein'],
                'cellular_locations': ['nucleus'],
                'molecular_functions': ['DNA repair', 'transcription regulation']
            },
            'EGFR': {
                'gene_id': 'ENSG00000146648',
                'description': 'Epidermal growth factor receptor',
                'pathways': ['mapk_signaling', 'pi3k_akt'],
                'protein_families': ['Receptor tyrosine kinase'],
                'cellular_locations': ['plasma membrane'],
                'molecular_functions': ['signal transduction', 'protein kinase activity']
            },
            'KRAS': {
                'gene_id': 'ENSG00000133703',
                'description': 'KRAS proto-oncogene, GTPase',
                'pathways': ['mapk_signaling', 'pi3k_akt'],
                'protein_families': ['Small GTPase'],
                'cellular_locations': ['plasma membrane', 'cytoplasm'],
                'molecular_functions': ['GTPase activity', 'signal transduction']
            }
        }
        
        for gene_symbol, data in gene_data.items():
            self.gene_annotations[gene_symbol] = GeneAnnotation(
                gene_symbol=gene_symbol,
                gene_id=data['gene_id'],
                description=data['description'],
                pathways=data['pathways'],
                protein_families=data['protein_families'],
                cellular_locations=data['cellular_locations'],
                molecular_functions=data['molecular_functions']
            )
    
    def _initialize_drug_targets(self):
        """Initialize drug-target interaction database."""
        
        drug_target_data = [
            {
                'drug_name': 'Paclitaxel',
                'drug_id': 'DB01229',
                'target_gene': 'TUBB',
                'interaction_type': 'inhibitor',
                'binding_affinity': 8.2,  # pKd
                'clinical_phase': 'Approved',
                'indication': 'Cancer chemotherapy'
            },
            {
                'drug_name': 'Gefitinib',
                'drug_id': 'DB00317',
                'target_gene': 'EGFR',
                'interaction_type': 'inhibitor',
                'binding_affinity': 7.8,
                'clinical_phase': 'Approved',
                'indication': 'Non-small cell lung cancer'
            },
            {
                'drug_name': 'Rapamycin',
                'drug_id': 'DB00877',
                'target_gene': 'MTOR',
                'interaction_type': 'inhibitor',
                'binding_affinity': 9.1,
                'clinical_phase': 'Approved',
                'indication': 'Immunosuppression, cancer'
            }
        ]
        
        for data in drug_target_data:
            key = f"{data['drug_name']}_{data['target_gene']}"
            self.drug_targets[key] = DrugTargetInteraction(**data)
    
    def get_pathway_genes(self, pathway_id: str) -> List[str]:
        """Get genes associated with a pathway."""
        if pathway_id in self.pathways:
            return self.pathways[pathway_id].genes
        return []
    
    def get_gene_pathways(self, gene_symbol: str) -> List[str]:
        """Get pathways associated with a gene."""
        if gene_symbol in self.gene_annotations:
            return self.gene_annotations[gene_symbol].pathways
        return []
    
    def find_pathway_overlap(self, gene_list: List[str]) -> Dict[str, Dict]:
        """Find pathway overlap for a list of genes."""
        
        pathway_overlap = {}
        
        for pathway_id, pathway in self.pathways.items():
            overlap_genes = list(set(gene_list) & set(pathway.genes))
            if overlap_genes:
                overlap_ratio = len(overlap_genes) / len(pathway.genes)
                coverage_ratio = len(overlap_genes) / len(gene_list)
                
                pathway_overlap[pathway_id] = {
                    'pathway_name': pathway.name,
                    'overlap_genes': overlap_genes,
                    'overlap_count': len(overlap_genes),
                    'pathway_size': len(pathway.genes),
                    'query_size': len(gene_list),
                    'overlap_ratio': overlap_ratio,
                    'coverage_ratio': coverage_ratio,
                    'enrichment_score': overlap_ratio * coverage_ratio,
                    'pathway_type': pathway.pathway_type,
                    'source_database': pathway.source_database
                }
        
        # Sort by enrichment score
        sorted_pathways = dict(sorted(
            pathway_overlap.items(), 
            key=lambda x: x[1]['enrichment_score'], 
            reverse=True
        ))
        
        return sorted_pathways

class PathwayEnrichmentAnalyzer:
    """Analyzer for pathway enrichment analysis."""
    
    def __init__(self, knowledge_base: BiologicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        
    def hypergeometric_test(self, 
                          query_genes: List[str],
                          pathway_genes: List[str],
                          total_genes: int = 20000) -> Tuple[float, float]:
        """
        Perform hypergeometric test for pathway enrichment.
        
        Args:
            query_genes: Genes in query set
            pathway_genes: Genes in pathway
            total_genes: Total number of genes in genome
            
        Returns:
            Tuple of (p_value, fold_enrichment)
        """
        from scipy.stats import hypergeom
        
        # Calculate overlap
        overlap = len(set(query_genes) & set(pathway_genes))
        query_size = len(query_genes)
        pathway_size = len(pathway_genes)
        
        if overlap == 0:
            return 1.0, 0.0
        
        # Hypergeometric test
        p_value = hypergeom.sf(overlap - 1, total_genes, pathway_size, query_size)
        
        # Fold enrichment
        expected = (query_size * pathway_size) / total_genes
        fold_enrichment = overlap / max(expected, 1e-10)
        
        # Ensure pure Python floats for precise type checking
        return float(p_value), float(fold_enrichment)
    
    def enrichment_analysis(self, 
                          gene_list: List[str],
                          p_threshold: float = 0.05,
                          min_overlap: int = 2) -> Dict:
        """
        Perform comprehensive pathway enrichment analysis.
        
        Args:
            gene_list: List of genes to analyze
            p_threshold: P-value threshold for significance
            min_overlap: Minimum overlap required
            
        Returns:
            Enrichment analysis results
        """
        
        enrichment_results = {
            'significant_pathways': [],
            'all_pathways': [],
            'summary_stats': {}
        }
        
        total_pathways_tested = 0
        significant_count = 0
        
        for pathway_id, pathway in self.knowledge_base.pathways.items():
            # Calculate enrichment
            p_value, fold_enrichment = self.hypergeometric_test(
                gene_list, pathway.genes
            )
            
            # Calculate overlap
            overlap_genes = list(set(gene_list) & set(pathway.genes))
            overlap_count = len(overlap_genes)
            
            if overlap_count >= min_overlap:
                result = {
                    'pathway_id': pathway_id,
                    'pathway_name': pathway.name,
                    'pathway_type': pathway.pathway_type,
                    'source_database': pathway.source_database,
                    'p_value': p_value,
                    'fold_enrichment': fold_enrichment,
                    'overlap_count': overlap_count,
                    'pathway_size': len(pathway.genes),
                    'query_size': len(gene_list),
                    'overlap_genes': overlap_genes,
                    'is_significant': p_value < p_threshold
                }
                
                enrichment_results['all_pathways'].append(result)
                total_pathways_tested += 1
                
                if p_value < p_threshold:
                    enrichment_results['significant_pathways'].append(result)
                    significant_count += 1
        
        # Sort by p-value
        enrichment_results['significant_pathways'].sort(key=lambda x: x['p_value'])
        enrichment_results['all_pathways'].sort(key=lambda x: x['p_value'])
        
        # Summary statistics
        enrichment_results['summary_stats'] = {
            'total_query_genes': len(gene_list),
            'total_pathways_tested': total_pathways_tested,
            'significant_pathways_count': significant_count,
            'enrichment_ratio': significant_count / max(total_pathways_tested, 1),
            'top_pathway': enrichment_results['significant_pathways'][0] if significant_count > 0 else None
        }
        
        return enrichment_results
    
    def multiple_testing_correction(self, 
                                  enrichment_results: Dict,
                                  method: str = 'bonferroni') -> Dict:
        """
        Apply multiple testing correction to enrichment results.
        
        Args:
            enrichment_results: Results from enrichment_analysis
            method: Correction method ('bonferroni', 'fdr_bh')
            
        Returns:
            Corrected enrichment results
        """
        from scipy.stats import false_discovery_control
        
        all_pathways = enrichment_results['all_pathways']
        
        if not all_pathways:
            return enrichment_results
        
        # Extract p-values
        p_values = [result['p_value'] for result in all_pathways]
        
        if method == 'bonferroni':
            # Bonferroni correction
            corrected_p_values = [p * len(p_values) for p in p_values]
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR correction
            corrected_p_values = false_discovery_control(p_values)
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Update results with corrected p-values
        corrected_results = enrichment_results.copy()
        corrected_results['significant_pathways'] = []
        
        for i, result in enumerate(all_pathways):
            result_copy = result.copy()
            result_copy['corrected_p_value'] = min(1.0, corrected_p_values[i])
            result_copy['is_significant'] = result_copy['corrected_p_value'] < 0.05
            
            if result_copy['is_significant']:
                corrected_results['significant_pathways'].append(result_copy)
        
        # Update summary statistics
        corrected_results['summary_stats']['significant_pathways_count'] = len(
            corrected_results['significant_pathways']
        )
        corrected_results['summary_stats']['correction_method'] = method
        
        return corrected_results

class ProteinInteractionNetwork:
    """Protein-protein interaction network analysis."""
    
    def __init__(self, knowledge_base: BiologicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.network = nx.Graph()
        self._build_network()
        
    def _build_network(self):
        """Build protein interaction network from knowledge base."""
        
        # Add nodes for all genes
        all_genes = set()
        for pathway in self.knowledge_base.pathways.values():
            all_genes.update(pathway.genes)
        
        self.network.add_nodes_from(all_genes)
        
        # Add edges based on pathway co-membership
        for pathway in self.knowledge_base.pathways.values():
            genes = pathway.genes
            # Add edges between all genes in the same pathway
            for i, gene1 in enumerate(genes):
                for gene2 in genes[i+1:]:
                    if self.network.has_edge(gene1, gene2):
                        # Increase weight if already connected
                        self.network[gene1][gene2]['weight'] += 1
                    else:
                        self.network.add_edge(gene1, gene2, weight=1)
    
    def find_network_modules(self, 
                           gene_list: List[str],
                           algorithm: str = 'louvain') -> Dict:
        """
        Find network modules/communities containing query genes.
        
        Args:
            gene_list: Genes of interest
            algorithm: Community detection algorithm
            
        Returns:
            Network module analysis results
        """
        
        # Create subgraph with query genes and their neighbors
        subgraph_nodes = set(gene_list)
        for gene in gene_list:
            if gene in self.network:
                subgraph_nodes.update(self.network.neighbors(gene))
        
        subgraph = self.network.subgraph(subgraph_nodes)
        
        # Community detection
        if algorithm == 'louvain':
            communities = community_louvain.best_partition(subgraph)
        else:
            # Default to connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(subgraph)):
                for node in component:
                    communities[node] = i
        
        # Organize results
        module_dict = defaultdict(list)
        for gene, module_id in communities.items():
            module_dict[module_id].append(gene)
        
        # Analyze modules containing query genes
        relevant_modules = []
        for module_id, module_genes in module_dict.items():
            query_overlap = list(set(module_genes) & set(gene_list))
            if query_overlap:
                # Calculate module connectivity
                module_subgraph = subgraph.subgraph(module_genes)
                connectivity = nx.density(module_subgraph)
                
                relevant_modules.append({
                    'module_id': module_id,
                    'module_genes': module_genes,
                    'module_size': len(module_genes),
                    'query_overlap': query_overlap,
                    'overlap_count': len(query_overlap),
                    'connectivity': connectivity,
                    'centrality_scores': self._calculate_centrality(module_subgraph)
                })
        
        return {
            'modules': relevant_modules,
            'total_modules': len(module_dict),
            'subgraph_stats': {
                'nodes': subgraph.number_of_nodes(),
                'edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph)
            }
        }
    
    def _calculate_centrality(self, graph: nx.Graph) -> Dict:
        """Calculate centrality measures for nodes in graph."""
        
        centrality_measures = {}
        
        if graph.number_of_nodes() > 0:
            # Degree centrality
            centrality_measures['degree'] = nx.degree_centrality(graph)
            
            # Betweenness centrality (for larger graphs)
            if graph.number_of_nodes() < 100:
                centrality_measures['betweenness'] = nx.betweenness_centrality(graph)
            
            # Closeness centrality
            if nx.is_connected(graph) and graph.number_of_nodes() < 50:
                centrality_measures['closeness'] = nx.closeness_centrality(graph)
        
        return centrality_measures
    
    def shortest_path_analysis(self, 
                             source_genes: List[str], 
                             target_genes: List[str]) -> Dict:
        """
        Analyze shortest paths between source and target gene sets.
        
        Args:
            source_genes: Source gene set
            target_genes: Target gene set
            
        Returns:
            Shortest path analysis results
        """
        
        path_results = {
            'paths': [],
            'intermediate_genes': set(),
            'path_statistics': {}
        }
        
        path_lengths = []
        
        for source in source_genes:
            for target in target_genes:
                if source in self.network and target in self.network:
                    try:
                        if nx.has_path(self.network, source, target):
                            path = nx.shortest_path(self.network, source, target)
                            path_length = len(path) - 1
                            
                            path_results['paths'].append({
                                'source': source,
                                'target': target,
                                'path': path,
                                'length': path_length,
                                'intermediate_nodes': path[1:-1]
                            })
                            
                            path_lengths.append(path_length)
                            path_results['intermediate_genes'].update(path[1:-1])
                    except nx.NetworkXNoPath:
                        continue
        
        # Calculate statistics
        if path_lengths:
            path_results['path_statistics'] = {
                'mean_path_length': np.mean(path_lengths),
                'median_path_length': np.median(path_lengths),
                'min_path_length': min(path_lengths),
                'max_path_length': max(path_lengths),
                'total_paths_found': len(path_lengths)
            }
        
        path_results['intermediate_genes'] = list(path_results['intermediate_genes'])
        
        return path_results

def annotate_biological_pathways(concepts: torch.Tensor, config: Dict) -> Dict:
    """
    High-level function to annotate biological pathways from concept activations.
    
    Args:
        concepts: Concept activation tensor
        config: Configuration dictionary
        
    Returns:
        Dictionary with pathway annotations and enrichment analysis
    """
    print("  PATH: Annotating biological pathways...")
    
    # Initialize knowledge base
    knowledge_base = BiologicalKnowledgeBase(config)
    
    # Convert concept activations to gene-like features
    # This is a simplified mapping - in practice, you'd have:
    # 1. Trained mappings from concepts to genes
    # 2. Literature-based concept-gene associations
    # 3. Experimental validation of concept-pathway relationships
    
    # For demonstration, create synthetic gene list based on top concepts
    n_concepts = concepts.shape[1] if concepts.dim() > 1 else concepts.shape[0]
    
    # Simulate concept-to-gene mapping
    concept_gene_mapping = {
        0: 'TP53',    # Cell cycle/DNA damage concept
        1: 'EGFR',    # Growth signaling concept
        2: 'CASP3',   # Apoptosis concept
        3: 'BRCA1',   # DNA repair concept
        4: 'KRAS',    # Oncogene concept
        5: 'AKT1',    # Survival signaling concept
        6: 'MTOR',    # Growth control concept
        7: 'MAPK1',   # MAPK signaling concept
    }
    
    # Identify activated concepts (top 30%)
    if concepts.dim() > 1:
        concept_scores = torch.mean(torch.abs(concepts), dim=0)
    else:
        concept_scores = torch.abs(concepts)
    
    top_concept_indices = torch.topk(concept_scores, k=min(8, len(concept_gene_mapping))).indices
    
    # Map concepts to genes
    activated_genes = []
    for idx in top_concept_indices:
        key = int(idx.item())  # Cast ensures proper int key for mypy/pyright
        if key in concept_gene_mapping:
            activated_genes.append(concept_gene_mapping[key])
    
    # Perform pathway enrichment analysis
    enrichment_analyzer = PathwayEnrichmentAnalyzer(knowledge_base)
    enrichment_results = enrichment_analyzer.enrichment_analysis(
        activated_genes,
        p_threshold=0.05,
        min_overlap=1  # Relaxed for demo
    )
    
    # Apply multiple testing correction
    corrected_results = enrichment_analyzer.multiple_testing_correction(
        enrichment_results,
        method='fdr_bh'
    )
    
    # Network analysis
    network_analyzer = ProteinInteractionNetwork(knowledge_base)
    network_modules = network_analyzer.find_network_modules(activated_genes)
    
    # Compile results
    pathway_annotations = {
        'activated_genes': activated_genes,
        'enriched_pathways': corrected_results['significant_pathways'],
        'pathway_summary': corrected_results['summary_stats'],
        'network_modules': network_modules,
        'annotations': generate_pathway_descriptions(corrected_results, knowledge_base),
        'confidence_scores': calculate_annotation_confidence(corrected_results, network_modules)
    }
    
    print(f"    STATS: Found {len(corrected_results['significant_pathways'])} significantly enriched pathways")
    print(f"    LINK: Identified {len(network_modules['modules'])} network modules")
    
    return pathway_annotations

def generate_pathway_descriptions(enrichment_results: Dict, 
                                knowledge_base: BiologicalKnowledgeBase) -> List[Dict]:
    """Generate human-readable descriptions of pathway enrichment."""
    
    descriptions = []
    
    for pathway_result in enrichment_results['significant_pathways'][:5]:  # Top 5
        pathway_id = pathway_result['pathway_id']
        pathway_info = knowledge_base.pathways.get(pathway_id)
        
        if pathway_info:
            description = {
                'pathway_name': pathway_result['pathway_name'],
                'description': pathway_info.description,
                'significance': 'High' if pathway_result['p_value'] < 0.001 else 'Moderate',
                'genes_involved': pathway_result['overlap_genes'],
                'biological_interpretation': generate_biological_interpretation(
                    pathway_info, pathway_result
                ),
                'clinical_relevance': assess_clinical_relevance(pathway_info),
                'fold_enrichment': pathway_result['fold_enrichment']
            }
            descriptions.append(description)
    
    return descriptions

def generate_biological_interpretation(pathway_info: BiologicalPathway, 
                                     pathway_result: Dict) -> str:
    """Generate biological interpretation for pathway enrichment."""
    
    fold_enrichment = pathway_result['fold_enrichment']
    overlap_genes = pathway_result['overlap_genes']
    
    interpretation = f"The {pathway_info.name} pathway shows "
    
    if fold_enrichment > 3:
        interpretation += "strong enrichment "
    elif fold_enrichment > 1.5:
        interpretation += "moderate enrichment "
    else:
        interpretation += "weak enrichment "
    
    interpretation += f"({fold_enrichment:.1f}-fold) with {len(overlap_genes)} genes affected. "
    
    # Add pathway-specific interpretations
    if pathway_info.pathway_type == 'cell_process':
        interpretation += "This suggests significant alterations in fundamental cellular processes."
    elif pathway_info.pathway_type == 'signaling':
        interpretation += "This indicates perturbation of key signaling cascades."
    elif pathway_info.pathway_type == 'stress_response':
        interpretation += "This suggests activation of cellular stress response mechanisms."
    elif pathway_info.pathway_type == 'cell_death':
        interpretation += "This indicates modulation of cell death pathways."
    
    return interpretation

def assess_clinical_relevance(pathway_info: BiologicalPathway) -> Dict:
    """Assess clinical relevance of pathway."""
    
    clinical_relevance = {
        'disease_associations': [],
        'therapeutic_targets': [],
        'drug_development_potential': 'Unknown',
        'biomarker_potential': 'Unknown'
    }
    
    # Pathway-specific clinical assessments
    if 'cancer' in pathway_info.name.lower() or 'tumor' in pathway_info.name.lower():
        clinical_relevance['disease_associations'].append('Cancer')
        clinical_relevance['drug_development_potential'] = 'High'
        clinical_relevance['biomarker_potential'] = 'High'
    
    if 'cell_cycle' in pathway_info.pathway_id.lower():
        clinical_relevance['disease_associations'].extend(['Cancer', 'Developmental disorders'])
        clinical_relevance['therapeutic_targets'].extend(['CDK inhibitors', 'Cell cycle checkpoints'])
        clinical_relevance['drug_development_potential'] = 'High'
    
    if 'dna_damage' in pathway_info.pathway_id.lower():
        clinical_relevance['disease_associations'].extend(['Cancer', 'Aging', 'Neurodegeneration'])
        clinical_relevance['therapeutic_targets'].extend(['DNA repair enhancers', 'Checkpoint inhibitors'])
        clinical_relevance['drug_development_potential'] = 'High'
    
    if 'apoptosis' in pathway_info.pathway_id.lower():
        clinical_relevance['disease_associations'].extend(['Cancer', 'Neurodegeneration', 'Autoimmune'])
        clinical_relevance['therapeutic_targets'].extend(['Apoptosis modulators', 'Caspase inhibitors'])
        clinical_relevance['drug_development_potential'] = 'Medium'
    
    return clinical_relevance

def calculate_annotation_confidence(enrichment_results: Dict, 
                                  network_modules: Dict) -> Dict:
    """Calculate confidence scores for pathway annotations."""
    
    confidence_scores = {}
    
    # Enrichment-based confidence
    if enrichment_results['significant_pathways']:
        top_p_value = enrichment_results['significant_pathways'][0]['p_value']
        top_fold_enrichment = enrichment_results['significant_pathways'][0]['fold_enrichment']
        
        enrichment_confidence = min(1.0, -np.log10(top_p_value) / 5.0)  # Scale log p-value
        fold_confidence = min(1.0, top_fold_enrichment / 5.0)  # Scale fold enrichment
        
        confidence_scores['enrichment_confidence'] = (enrichment_confidence + fold_confidence) / 2
    else:
        confidence_scores['enrichment_confidence'] = 0.0
    
    # Network-based confidence
    if network_modules['modules']:
        max_connectivity = max(module['connectivity'] for module in network_modules['modules'])
        max_overlap = max(module['overlap_count'] for module in network_modules['modules'])
        
        network_confidence = (max_connectivity + min(1.0, max_overlap / 5.0)) / 2
        confidence_scores['network_confidence'] = network_confidence
    else:
        confidence_scores['network_confidence'] = 0.0
    
    # Overall confidence
    confidence_scores['overall_confidence'] = (
        confidence_scores['enrichment_confidence'] * 0.7 +
        confidence_scores['network_confidence'] * 0.3
    )
    
    return confidence_scores

class DrugTargetPredictor:
    """Predictor for drug-target interactions based on pathway analysis."""
    
    def __init__(self, knowledge_base: BiologicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        
    def predict_drug_targets(self, 
                           enriched_pathways: List[Dict],
                           target_confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Predict potential drug targets based on enriched pathways.
        
        Args:
            enriched_pathways: List of enriched pathway results
            target_confidence_threshold: Minimum confidence for target prediction
            
        Returns:
            List of predicted drug targets
        """
        
        target_predictions = []
        
        for pathway_result in enriched_pathways:
            pathway_genes = pathway_result['overlap_genes']
            pathway_id = pathway_result['pathway_id']
            
            for gene in pathway_genes:
                # Check if gene has known drug interactions
                target_info = self._assess_target_potential(gene, pathway_result)
                
                if target_info['confidence'] >= target_confidence_threshold:
                    target_predictions.append({
                        'target_gene': gene,
                        'pathway_context': pathway_result['pathway_name'],
                        'target_confidence': target_info['confidence'],
                        'druggability_score': target_info['druggability'],
                        'known_drugs': target_info['known_drugs'],
                        'target_class': target_info['target_class'],
                        'development_stage': target_info['development_stage'],
                        'rationale': target_info['rationale']
                    })
        
        # Sort by confidence
        target_predictions.sort(key=lambda x: x['target_confidence'], reverse=True)
        
        return target_predictions
    
    def _assess_target_potential(self, gene: str, pathway_result: Dict) -> Dict:
        """Assess the potential of a gene as a drug target."""
        
        # Check known drug interactions
        known_drugs = []
        for drug_key, drug_interaction in self.knowledge_base.drug_targets.items():
            if drug_interaction.target_gene == gene:
                known_drugs.append({
                    'drug_name': drug_interaction.drug_name,
                    'interaction_type': drug_interaction.interaction_type,
                    'clinical_phase': drug_interaction.clinical_phase
                })
        
        # Assess druggability based on protein class
        gene_annotation = self.knowledge_base.gene_annotations.get(gene)
        druggability_score = 0.5  # Default
        target_class = 'Unknown'
        
        if gene_annotation:
            protein_families = gene_annotation.protein_families
            
            if any('kinase' in family.lower() for family in protein_families):
                druggability_score = 0.8
                target_class = 'Protein Kinase'
            elif any('receptor' in family.lower() for family in protein_families):
                druggability_score = 0.7
                target_class = 'Receptor'
            elif any('transcription' in family.lower() for family in protein_families):
                druggability_score = 0.4
                target_class = 'Transcription Factor'
            elif any('enzyme' in family.lower() for family in protein_families):
                druggability_score = 0.6
                target_class = 'Enzyme'
        
        # Calculate overall confidence
        pathway_confidence = min(1.0, -np.log10(pathway_result['p_value']) / 5.0)
        fold_enrichment_factor = min(1.0, pathway_result['fold_enrichment'] / 3.0)
        
        overall_confidence = (
            druggability_score * 0.4 +
            pathway_confidence * 0.3 +
            fold_enrichment_factor * 0.2 +
            (0.1 if known_drugs else 0.0)
        )
        
        # Determine development stage
        if known_drugs:
            max_phase = max(drug['clinical_phase'] for drug in known_drugs if drug['clinical_phase'])
            if 'Approved' in max_phase:
                development_stage = 'Approved drugs available'
            elif 'Phase' in max_phase:
                development_stage = f'In clinical trials ({max_phase})'
            else:
                development_stage = 'Preclinical'
        else:
            development_stage = 'No known drugs'
        
        # Generate rationale
        rationale = f"{gene} shows potential as a drug target due to "
        rationale += f"enrichment in {pathway_result['pathway_name']} pathway "
        rationale += f"({pathway_result['fold_enrichment']:.1f}-fold enrichment). "
        if target_class != 'Unknown':
            rationale += f"The protein belongs to the {target_class} class, "
            rationale += f"which is generally {'highly' if druggability_score > 0.7 else 'moderately'} druggable."
        
        return {
            'confidence': overall_confidence,
            'druggability': druggability_score,
            'known_drugs': known_drugs,
            'target_class': target_class,
            'development_stage': development_stage,
            'rationale': rationale
        }

class BiologicalInsightGenerator:
    """Generator for high-level biological insights."""
    
    def __init__(self, knowledge_base: BiologicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.drug_predictor = DrugTargetPredictor(knowledge_base)
        
    def generate_comprehensive_insights(self, 
                                      pathway_annotations: Dict,
                                      perturbation_effects: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive biological insights from analysis results.
        
        Args:
            pathway_annotations: Results from pathway annotation analysis
            perturbation_effects: Optional perturbation effect predictions
            
        Returns:
            Comprehensive biological insights
        """
        
        insights = {
            'summary': self._generate_executive_summary(pathway_annotations),
            'mechanistic_insights': self._generate_mechanistic_insights(pathway_annotations),
            'therapeutic_implications': self._generate_therapeutic_implications(pathway_annotations),
            'experimental_recommendations': self._generate_experimental_recommendations(pathway_annotations),
            'literature_connections': self._find_literature_connections(pathway_annotations),
            'risk_assessment': self._assess_biological_risks(pathway_annotations)
        }
        
        if perturbation_effects:
            insights['perturbation_analysis'] = self._analyze_perturbation_effects(
                perturbation_effects, pathway_annotations
            )
        
        return insights
    
    def _generate_executive_summary(self, pathway_annotations: Dict) -> str:
        """Generate executive summary of biological findings."""
        
        n_pathways = len(pathway_annotations['enriched_pathways'])
        n_genes = len(pathway_annotations['activated_genes'])
        
        summary = f"Analysis identified {n_genes} key genes affecting {n_pathways} biological pathways. "
        
        if n_pathways > 0:
            top_pathway = pathway_annotations['enriched_pathways'][0]
            summary += f"The most significantly affected pathway is {top_pathway['pathway_name']} "
            summary += f"(p-value: {top_pathway['p_value']:.2e}, "
            summary += f"{top_pathway['fold_enrichment']:.1f}-fold enrichment). "
            
            # Add pathway type context
            pathway_types = [p['pathway_type'] for p in pathway_annotations['enriched_pathways']]
            most_common_type = max(set(pathway_types), key=pathway_types.count)
            
            if most_common_type == 'signaling':
                summary += "The analysis suggests significant perturbation of cellular signaling networks."
            elif most_common_type == 'cell_process':
                summary += "The results indicate substantial changes in fundamental cellular processes."
            elif most_common_type == 'stress_response':
                summary += "The findings point to activation of cellular stress response mechanisms."
            
        confidence = pathway_annotations['confidence_scores']['overall_confidence']
        summary += f" Overall confidence in these findings is {confidence:.1%}."
        
        return summary
    
    def _generate_mechanistic_insights(self, pathway_annotations: Dict) -> List[Dict]:
        """Generate mechanistic insights from pathway analysis."""
        
        insights = []
        
        for pathway in pathway_annotations['enriched_pathways'][:3]:  # Top 3 pathways
            pathway_id = pathway['pathway_id']
            pathway_info = self.knowledge_base.pathways.get(pathway_id)
            
            if pathway_info:
                insight = {
                    'pathway': pathway['pathway_name'],
                    'mechanism': self._describe_pathway_mechanism(pathway_info, pathway),
                    'key_players': pathway['overlap_genes'],
                    'upstream_regulators': self._identify_upstream_regulators(pathway['overlap_genes']),
                    'downstream_effects': self._predict_downstream_effects(pathway_info),
                    'crosstalk_pathways': self._identify_pathway_crosstalk(pathway_id)
                }
                insights.append(insight)
        
        return insights
    
    def _describe_pathway_mechanism(self, pathway_info: BiologicalPathway, pathway_result: Dict) -> str:
        """Describe the mechanism of pathway perturbation."""
        
        mechanism = f"Perturbation of the {pathway_info.name} pathway likely occurs through "
        
        affected_genes = pathway_result['overlap_genes']
        
        # Identify key regulatory nodes
        key_regulators = []
        for gene in affected_genes:
            if gene in ['TP53', 'MYC', 'RB1', 'EGFR', 'KRAS']:
                key_regulators.append(gene)
        
        if key_regulators:
            mechanism += f"modulation of key regulatory genes including {', '.join(key_regulators)}. "
        else:
            mechanism += f"effects on {len(affected_genes)} pathway components. "
        
        # Add pathway-specific mechanisms
        if pathway_info.pathway_type == 'signaling':
            mechanism += "This may involve altered protein phosphorylation cascades and transcriptional changes."
        elif pathway_info.pathway_type == 'cell_process':
            mechanism += "This suggests direct interference with essential cellular machinery."
        elif pathway_info.pathway_type == 'stress_response':
            mechanism += "This indicates activation of damage sensing and repair mechanisms."
        
        return mechanism
    
    def _identify_upstream_regulators(self, genes: List[str]) -> List[str]:
        """Identify potential upstream regulators of affected genes."""
        
        # Simplified upstream regulator identification
        # In practice, this would use regulatory network databases
        
        known_regulators = {
            'TP53': ['ATM', 'ATR', 'MDM2'],
            'EGFR': ['EGF', 'TGF-alpha'],
            'KRAS': ['RTKs', 'GEFs'],
            'MYC': ['E2F', 'TCF4'],
            'BRCA1': ['ATM', 'ATR']
        }
        
        upstream_regulators = set()
        for gene in genes:
            if gene in known_regulators:
                upstream_regulators.update(known_regulators[gene])
        
        return list(upstream_regulators)
    
    def _predict_downstream_effects(self, pathway_info: BiologicalPathway) -> List[str]:
        """Predict downstream effects of pathway perturbation."""
        
        effects = []
        
        if 'cell_cycle' in pathway_info.name.lower():
            effects.extend(['Cell cycle arrest', 'Altered proliferation', 'DNA damage checkpoint activation'])
        
        if 'apoptosis' in pathway_info.name.lower():
            effects.extend(['Cell death induction', 'Caspase activation', 'DNA fragmentation'])
        
        if 'dna_damage' in pathway_info.name.lower():
            effects.extend(['DNA repair activation', 'Cell cycle checkpoint', 'Genomic instability'])
        
        if 'signaling' in pathway_info.pathway_type:
            effects.extend(['Transcriptional changes', 'Metabolic reprogramming', 'Phenotypic alterations'])
        
        return effects
    
    def _identify_pathway_crosstalk(self, pathway_id: str) -> List[str]:
        """Identify pathways that crosstalk with the given pathway."""
        
        # Simplified crosstalk identification based on shared genes
        target_pathway = self.knowledge_base.pathways.get(pathway_id)
        if not target_pathway:
            return []
        
        crosstalk_pathways = []
        target_genes = set(target_pathway.genes)
        
        for other_id, other_pathway in self.knowledge_base.pathways.items():
            if other_id != pathway_id:
                other_genes = set(other_pathway.genes)
                overlap = len(target_genes & other_genes)
                
                # Significant overlap suggests crosstalk
                if overlap >= 2:
                    crosstalk_pathways.append(other_pathway.name)
        
        return crosstalk_pathways[:5]  # Top 5
    
    def _generate_therapeutic_implications(self, pathway_annotations: Dict) -> Dict:
        """Generate therapeutic implications from pathway analysis."""
        
        # Predict drug targets
        predicted_targets = self.drug_predictor.predict_drug_targets(
            pathway_annotations['enriched_pathways']
        )
        
        # Assess therapeutic opportunities
        therapeutic_opportunities = []
        for target in predicted_targets[:5]:  # Top 5 targets
            opportunity = {
                'target': target['target_gene'],
                'confidence': target['target_confidence'],
                'rationale': target['rationale'],
                'development_feasibility': self._assess_development_feasibility(target),
                'market_potential': self._assess_market_potential(target),
                'competitive_landscape': self._assess_competitive_landscape(target)
            }
            therapeutic_opportunities.append(opportunity)
        
        return {
            'predicted_targets': predicted_targets,
            'therapeutic_opportunities': therapeutic_opportunities,
            'drug_repurposing_candidates': self._identify_repurposing_candidates(pathway_annotations),
            'combination_therapy_suggestions': self._suggest_combination_therapies(predicted_targets)
        }
    
    def _assess_development_feasibility(self, target: Dict) -> Dict:
        """Assess the feasibility of drug development for a target."""
        
        feasibility = {
            'druggability': target['druggability_score'],
            'development_stage': target['development_stage'],
            'estimated_timeline': 'Unknown',
            'estimated_cost': 'Unknown',
            'technical_challenges': []
        }
        
        # Estimate timeline based on target class and current stage
        if 'Approved' in target['development_stage']:
            feasibility['estimated_timeline'] = '2-5 years (optimization/repurposing)'
            feasibility['estimated_cost'] = '$50-200M'
        elif 'Phase' in target['development_stage']:
            feasibility['estimated_timeline'] = '5-10 years'
            feasibility['estimated_cost'] = '$100-500M'
        else:
            feasibility['estimated_timeline'] = '10-15 years'
            feasibility['estimated_cost'] = '$500M-2B'
        
        # Identify technical challenges
        if target['target_class'] == 'Transcription Factor':
            feasibility['technical_challenges'].append('Challenging protein-protein interaction target')
        if target['druggability_score'] < 0.5:
            feasibility['technical_challenges'].append('Low druggability score')
        if not target['known_drugs']:
            feasibility['technical_challenges'].append('No validated chemical starting points')
        
        return feasibility
    
    def _assess_market_potential(self, target: Dict) -> Dict:
        """Assess market potential for a drug target."""
        
        # Simplified market assessment
        market_potential = {
            'indication_areas': [],
            'market_size_estimate': 'Unknown',
            'competition_level': 'Unknown',
            'patent_landscape': 'Unknown'
        }
        
        # Map pathway to indication areas
        pathway_context = target['pathway_context'].lower()
        
        if any(term in pathway_context for term in ['cancer', 'tumor', 'oncology']):
            market_potential['indication_areas'].append('Oncology')
            market_potential['market_size_estimate'] = 'Large ($10B+)'
            market_potential['competition_level'] = 'High'
        
        if any(term in pathway_context for term in ['neural', 'brain', 'neuro']):
            market_potential['indication_areas'].append('Neurology')
            market_potential['market_size_estimate'] = 'Medium ($1-10B)'
            market_potential['competition_level'] = 'Medium'
        
        if any(term in pathway_context for term in ['immune', 'inflammation']):
            market_potential['indication_areas'].append('Immunology')
            market_potential['market_size_estimate'] = 'Large ($5B+)'
            market_potential['competition_level'] = 'High'
        
        return market_potential
    
    def _assess_competitive_landscape(self, target: Dict) -> Dict:
        """Assess competitive landscape for a target."""
        
        landscape = {
            'known_competitors': [],
            'competitive_advantage': [],
            'differentiation_opportunities': []
        }
        
        # Identify known drugs/competitors
        for drug in target['known_drugs']:
            landscape['known_competitors'].append({
                'drug': drug['drug_name'],
                'stage': drug['clinical_phase'],
                'mechanism': drug['interaction_type']
            })
        
        # Assess competitive advantages
        if target['target_confidence'] > 0.8:
            landscape['competitive_advantage'].append('High confidence prediction')
        
        if target['druggability_score'] > 0.7:
            landscape['competitive_advantage'].append('Highly druggable target')
        
        # Identify differentiation opportunities
        if not target['known_drugs']:
            landscape['differentiation_opportunities'].append('First-in-class opportunity')
        
        landscape['differentiation_opportunities'].append('Novel pathway-based approach')
        
        return landscape
    
    def _identify_repurposing_candidates(self, pathway_annotations: Dict) -> List[Dict]:
        """Identify drug repurposing candidates."""
        
        repurposing_candidates = []
        
        for pathway in pathway_annotations['enriched_pathways']:
            affected_genes = pathway['overlap_genes']
            
            # Find drugs targeting affected genes
            for gene in affected_genes:
                for drug_key, drug_interaction in self.knowledge_base.drug_targets.items():
                    if drug_interaction.target_gene == gene and drug_interaction.clinical_phase == 'Approved':
                        
                        candidate = {
                            'drug_name': drug_interaction.drug_name,
                            'current_indication': drug_interaction.indication,
                            'target_gene': gene,
                            'pathway_context': pathway['pathway_name'],
                            'repurposing_rationale': f"Targets {gene} which is affected in {pathway['pathway_name']} pathway",
                            'repurposing_confidence': pathway['fold_enrichment'] / 5.0  # Simplified
                        }
                        
                        repurposing_candidates.append(candidate)
        
        # Remove duplicates and sort by confidence
        unique_candidates = {cand['drug_name']: cand for cand in repurposing_candidates}
        sorted_candidates = sorted(unique_candidates.values(), 
                                 key=lambda x: x['repurposing_confidence'], 
                                 reverse=True)
        
        return sorted_candidates[:10]  # Top 10
    
    def _suggest_combination_therapies(self, predicted_targets: List[Dict]) -> List[Dict]:
        """Suggest combination therapy strategies."""
        
        combinations = []
        
        # Identify targets in different pathways for combination
        pathway_targets = defaultdict(list)
        for target in predicted_targets:
            pathway_targets[target['pathway_context']].append(target)
        
        # Suggest inter-pathway combinations
        pathway_list = list(pathway_targets.keys())
        for i, pathway1 in enumerate(pathway_list):
            for pathway2 in pathway_list[i+1:]:
                if len(pathway_targets[pathway1]) > 0 and len(pathway_targets[pathway2]) > 0:
                    target1 = pathway_targets[pathway1][0]
                    target2 = pathway_targets[pathway2][0]
                    
                    combination = {
                        'target_1': target1['target_gene'],
                        'target_2': target2['target_gene'],
                        'pathway_1': pathway1,
                        'pathway_2': pathway2,
                        'combination_rationale': f"Dual targeting of {pathway1} and {pathway2} pathways",
                        'synergy_potential': min(target1['target_confidence'], target2['target_confidence']),
                        'development_challenges': self._assess_combination_challenges(target1, target2)
                    }
                    
                    combinations.append(combination)
        
        return combinations[:5]  # Top 5 combinations
    
    def _assess_combination_challenges(self, target1: Dict, target2: Dict) -> List[str]:
        """Assess challenges for combination therapy development."""
        
        challenges = []
        
        if target1['druggability_score'] < 0.5 or target2['druggability_score'] < 0.5:
            challenges.append('Low druggability of one or both targets')
        
        if not target1['known_drugs'] or not target2['known_drugs']:
            challenges.append('Limited chemical starting points')
        
        challenges.append('Need to optimize dosing and scheduling')
        challenges.append('Potential for increased toxicity')
        challenges.append('Complex regulatory pathway')
        
        return challenges
    
    def _generate_experimental_recommendations(self, pathway_annotations: Dict) -> List[Dict]:
        """Generate experimental recommendations for follow-up studies."""
        
        recommendations = []
        
        # Pathway validation experiments
        for pathway in pathway_annotations['enriched_pathways'][:3]:
            recommendation = {
                'experiment_type': 'Pathway Validation',
                'objective': f"Validate perturbation effects on {pathway['pathway_name']} pathway",
                'methods': [
                    'qRT-PCR analysis of pathway genes',
                    'Western blot for key pathway proteins',
                    'Pathway reporter assays',
                    'Functional rescue experiments'
                ],
                'target_genes': pathway['overlap_genes'],
                'expected_outcomes': [
                    f"Confirm {pathway['fold_enrichment']:.1f}-fold enrichment",
                    'Validate pathway activation/inhibition',
                    'Identify key regulatory nodes'
                ],
                'priority': 'High' if pathway['p_value'] < 0.001 else 'Medium',
                'estimated_duration': '4-6 weeks',
                'estimated_cost': '$15,000-25,000'
            }
            recommendations.append(recommendation)
        
        # Mechanistic studies
        high_confidence_genes = [
            gene for pathway in pathway_annotations['enriched_pathways']
            for gene in pathway['overlap_genes']
            if pathway['fold_enrichment'] > 2.0
        ]
        
        if high_confidence_genes:
            recommendation = {
                'experiment_type': 'Mechanistic Studies',
                'objective': 'Elucidate molecular mechanisms of perturbation effects',
                'methods': [
                    'CRISPR-Cas9 knockout/knockdown studies',
                    'Overexpression rescue experiments',
                    'Protein-protein interaction mapping',
                    'ChIP-seq for transcription factor binding',
                    'Time-course expression analysis'
                ],
                'target_genes': high_confidence_genes[:5],  # Top 5
                'expected_outcomes': [
                    'Define causal relationships',
                    'Identify direct vs. indirect effects',
                    'Map regulatory networks'
                ],
                'priority': 'High',
                'estimated_duration': '8-12 weeks',
                'estimated_cost': '$35,000-50,000'
            }
            recommendations.append(recommendation)
        
        # Drug screening experiments
        if pathway_annotations['enriched_pathways']:
            recommendation = {
                'experiment_type': 'Drug Screening',
                'objective': 'Identify compounds that modulate affected pathways',
                'methods': [
                    'Small molecule library screening',
                    'Pathway-specific reporter assays',
                    'Cell viability and phenotype analysis',
                    'Dose-response characterization'
                ],
                'target_pathways': [p['pathway_name'] for p in pathway_annotations['enriched_pathways'][:3]],
                'expected_outcomes': [
                    'Identify lead compounds',
                    'Validate pathway druggability',
                    'Generate structure-activity relationships'
                ],
                'priority': 'Medium',
                'estimated_duration': '12-16 weeks',
                'estimated_cost': '$75,000-100,000'
            }
            recommendations.append(recommendation)
        
        # Biomarker discovery
        recommendation = {
            'experiment_type': 'Biomarker Discovery',
            'objective': 'Identify biomarkers for pathway perturbation',
            'methods': [
                'Proteomics analysis',
                'Metabolomics profiling',
                'Phosphoproteomics',
                'Single-cell RNA sequencing'
            ],
            'target_pathways': [p['pathway_name'] for p in pathway_annotations['enriched_pathways'][:3]],
            'expected_outcomes': [
                'Identify predictive biomarkers',
                'Develop pathway activity signatures',
                'Enable patient stratification'
            ],
            'priority': 'Medium',
            'estimated_duration': '16-20 weeks',
            'estimated_cost': '$100,000-150,000'
        }
        recommendations.append(recommendation)
        
        return recommendations

    # ------------------------------------------------------------------
    # Placeholder helper methods to satisfy static type checker
    # ------------------------------------------------------------------
    def _find_literature_connections(self, pathway_annotations: Dict) -> List[str]:  # noqa: D401, pylint: disable=unused-argument
        """Stub: return empty list in lightweight build."""
        return []

    def _assess_biological_risks(self, pathway_annotations: Dict) -> Dict[str, Any]:  # noqa: D401
        return {}

    def _analyze_perturbation_effects(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # noqa: D401
        return {}