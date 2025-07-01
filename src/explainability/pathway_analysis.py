"""
Biological pathway analysis and network interpretation for perturbation biology.

This module provides tools for analyzing biological pathways, gene networks,
and molecular interactions in the context of perturbation experiments.
"""

import os
import sys
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
# Remove problematic scipy and sklearn imports - will be imported lazily when needed

# Visualization - lazy imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False
    sns = None
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Biological databases
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:  # Catch all exceptions during requests import
    REQUESTS_AVAILABLE = False
    warnings.warn("Requests not available. Online database features disabled.")
    
    # Create dummy requests functionality
    class DummyResponse:
        def __init__(self, status_code=200, text="", json_data=None):
            self.status_code = status_code
            self.text = text
            self._json_data = json_data or {}
        
        def json(self):
            return self._json_data
        
        def raise_for_status(self):
            pass
    
    class DummyRequests:
        @staticmethod
        def get(*args, **kwargs):
            return DummyResponse()
        
        @staticmethod
        def post(*args, **kwargs):
            return DummyResponse()
    
    requests = DummyRequests()

try:
    from bioservices import KEGG, Reactome, UniProt, PathwayCommons, PathwayCommonsClient  # type: ignore
    BIOSERVICES_AVAILABLE = True
except ImportError:
    # Create mock classes if bioservices is not available
    class KEGG:
        def pathwayIds(self, organism='hsa'): return []
        def parse(self, data): return {}
        def get(self, pathway_id): return ""
    
    class Reactome:
        def content_query(self, organism, content_type): return []
        def participants(self, pathway_id): return []
    
    class UniProt: pass
    class PathwayCommons: pass
    class PathwayCommonsClient: pass
    
    BIOSERVICES_AVAILABLE = False
    warnings.warn("bioservices not available. Some pathway features will be limited.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PathwayEnrichmentResult:
    """Results from pathway enrichment analysis."""
    pathway_id: str
    pathway_name: str
    category: str
    genes_in_pathway: List[str]
    genes_in_query: List[str]
    overlap_genes: List[str]
    p_value: float
    adjusted_p_value: float
    odds_ratio: float
    enrichment_score: float
    effect_size: float = 0.0  # Add missing effect_size attribute
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PathwayEnrichmentResult':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class NetworkAnalysisResult:
    """Results from network analysis."""
    network_id: str
    nodes: List[str]
    edges: List[Tuple[str, str]]
    communities: Dict[int, List[str]]
    centrality_scores: Dict[str, float]
    clustering_coefficient: float
    modularity: float
    average_path_length: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

class PathwayDatabase:
    """Database for biological pathways and gene sets."""
    
    def __init__(self, data_dir: str = "data/pathways"):
        """Initialize pathway database."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.pathways = {}
        self.gene_to_pathways = defaultdict(set)
        self.pathway_hierarchy = {}
        
        # Load built-in pathways
        self._load_builtin_pathways()
        
        # Load external databases if available
        if BIOSERVICES_AVAILABLE:
            self._load_external_databases()
    
    def _load_builtin_pathways(self):
        """Load built-in pathway definitions."""
        # Cell cycle pathways
        cell_cycle_genes = [
            'TP53', 'RB1', 'CDKN1A', 'CDKN2A', 'MDM2', 'ATM', 'CHEK1', 'CHEK2',
            'CDC25A', 'CDC25B', 'CDC25C', 'CDK1', 'CDK2', 'CDK4', 'CDK6',
            'CCNA1', 'CCNA2', 'CCNB1', 'CCNB2', 'CCND1', 'CCNE1', 'CCNE2'
        ]
        
        apoptosis_genes = [
            'TP53', 'BCL2', 'BAX', 'BAK1', 'BCL2L1', 'CASP3', 'CASP8', 'CASP9',
            'APAF1', 'CYTC', 'FAS', 'FADD', 'TNFRSF1A', 'BID', 'BAD', 'PUMA',
            'NOXA', 'MCL1', 'XIAP', 'SURVIVIN'
        ]
        
        dna_repair_genes = [
            'BRCA1', 'BRCA2', 'ATM', 'ATR', 'CHEK1', 'CHEK2', 'TP53BP1',
            'MDC1', 'H2AFX', 'XRCC1', 'XRCC3', 'XRCC4', 'XRCC5', 'XRCC6',
            'PARP1', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'ERCC1', 'ERCC4'
        ]
        
        metabolism_genes = [
            'PFKP', 'PFKM', 'PFKL', 'ALDOA', 'GAPDH', 'PGK1', 'PGAM1',
            'ENO1', 'PKM', 'LDHA', 'LDHB', 'G6PD', 'TKT', 'TALDO1',
            'ACLY', 'FASN', 'ACC1', 'SCD1', 'CPT1A', 'ACOX1'
        ]
        
        # Add pathways to database
        self.add_pathway('CELL_CYCLE', 'Cell Cycle', 'Cell Cycle & Division', cell_cycle_genes)
        self.add_pathway('APOPTOSIS', 'Apoptosis', 'Cell Death', apoptosis_genes)
        self.add_pathway('DNA_REPAIR', 'DNA Repair', 'DNA Damage Response', dna_repair_genes)
        self.add_pathway('METABOLISM', 'Metabolism', 'Metabolic Processes', metabolism_genes)
        
        # Signaling pathways
        pi3k_akt_genes = [
            'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG', 'AKT1', 'AKT2', 'AKT3',
            'PTEN', 'mTOR', 'RICTOR', 'RAPTOR', 'TSC1', 'TSC2', 'RHEB',
            'S6K1', 'S6K2', '4EBP1', 'GSK3B', 'FOXO1', 'FOXO3'
        ]
        
        mapk_genes = [
            'KRAS', 'HRAS', 'NRAS', 'RAF1', 'BRAF', 'MEK1', 'MEK2',
            'ERK1', 'ERK2', 'JNK1', 'JNK2', 'JNK3', 'p38', 'MKK3',
            'MKK4', 'MKK6', 'MKK7', 'ELK1', 'FOS', 'JUN'
        ]
        
        self.add_pathway('PI3K_AKT', 'PI3K/AKT Signaling', 'Signal Transduction', pi3k_akt_genes)
        self.add_pathway('MAPK', 'MAPK Signaling', 'Signal Transduction', mapk_genes)
        
        logger.info(f"Loaded {len(self.pathways)} built-in pathways")
    
    def _load_external_databases(self):
        """Load pathways from external databases."""
        try:
            # Load KEGG pathways
            self._load_kegg_pathways()
            
            # Load Reactome pathways
            self._load_reactome_pathways()
            
        except Exception as e:
            logger.warning(f"Could not load external databases: {str(e)}")
    
    def _load_kegg_pathways(self):
        """Load KEGG pathway data."""
        try:
            kegg = KEGG()
            
            # Get human pathways
            pathways = kegg.pathwayIds(organism='hsa')
            
            for pathway_id in pathways[:10]:  # Limit for demo
                try:
                    pathway_info = kegg.parse(kegg.get(pathway_id))
                    genes = pathway_info.get('GENE', {})
                    
                    if genes:
                        gene_symbols = [gene.split()[1] for gene in genes.values() if ' ' in gene]
                        
                        self.add_pathway(
                            pathway_id.replace('hsa', 'KEGG_'),
                            pathway_info.get('NAME', pathway_id),
                            'KEGG',
                            gene_symbols
                        )
                except Exception as e:
                    logger.debug(f"Could not load KEGG pathway {pathway_id}: {str(e)}")
            
            logger.info("Loaded KEGG pathways")
            
        except Exception as e:
            logger.debug(f"KEGG loading failed: {str(e)}")
    
    def _load_reactome_pathways(self):
        """Load Reactome pathway data."""
        try:
            reactome = Reactome()
            
            # Get human pathways
            pathways = reactome.content_query('homo sapiens', 'Pathway')
            
            for pathway in pathways[:10]:  # Limit for demo
                try:
                    pathway_id = pathway['stId']
                    pathway_name = pathway['displayName']
                    
                    # Get genes in pathway
                    genes = reactome.participants(pathway_id)
                    gene_symbols = [gene['displayName'] for gene in genes if 'displayName' in gene]
                    
                    if gene_symbols:
                        self.add_pathway(
                            f"REACTOME_{pathway_id}",
                            pathway_name,
                            'Reactome',
                            gene_symbols
                        )
                        
                except Exception as e:
                    logger.debug(f"Could not load Reactome pathway {pathway_id}: {str(e)}")
            
            logger.info("Loaded Reactome pathways")
            
        except Exception as e:
            logger.debug(f"Reactome loading failed: {str(e)}")
    
    def add_pathway(self, 
                   pathway_id: str, 
                   pathway_name: str, 
                   category: str, 
                   genes: List[str]):
        """Add a pathway to the database."""
        self.pathways[pathway_id] = {
            'id': pathway_id,
            'name': pathway_name,
            'category': category,
            'genes': list(set(genes))  # Remove duplicates
        }
        
        # Update gene-to-pathway mapping
        for gene in genes:
            self.gene_to_pathways[gene].add(pathway_id)
    
    def get_pathway(self, pathway_id: str) -> Optional[Dict]:
        """Get pathway information."""
        return self.pathways.get(pathway_id)
    
    def get_pathways_for_gene(self, gene: str) -> List[str]:
        """Get pathways containing a specific gene."""
        return list(self.gene_to_pathways.get(gene, set()))
    
    def search_pathways(self, query: str) -> List[Dict]:
        """Search pathways by name or gene."""
        results = []
        query_lower = query.lower()
        
        for pathway_id, pathway_info in self.pathways.items():
            # Search in pathway name
            if query_lower in pathway_info['name'].lower():
                results.append(pathway_info)
            # Search in genes
            elif any(query_lower in gene.lower() for gene in pathway_info['genes']):
                results.append(pathway_info)
        
        return results
    
    def get_pathway_overlap(self, pathway_id1: str, pathway_id2: str) -> Dict:
        """Calculate overlap between two pathways."""
        pathway1 = self.get_pathway(pathway_id1)
        pathway2 = self.get_pathway(pathway_id2)
        
        if not pathway1 or not pathway2:
            return {}
        
        genes1 = set(pathway1['genes'])
        genes2 = set(pathway2['genes'])
        
        overlap = genes1.intersection(genes2)
        union = genes1.union(genes2)
        
        jaccard_index = len(overlap) / len(union) if union else 0
        
        return {
            'pathway1': pathway1['name'],
            'pathway2': pathway2['name'],
            'overlap_genes': list(overlap),
            'overlap_count': len(overlap),
            'jaccard_index': jaccard_index,
            'genes1_only': list(genes1 - genes2),
            'genes2_only': list(genes2 - genes1)
        }
    
    def export_database(self, output_file: str):
        """Export database to JSON file."""
        export_data = {
            'pathways': self.pathways,
            'gene_to_pathways': {gene: list(pathways) for gene, pathways in self.gene_to_pathways.items()},
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Database exported to {output_file}")
    
    def import_database(self, input_file: str):
        """Import database from JSON file."""
        with open(input_file, 'r') as f:
            import_data = json.load(f)
        
        self.pathways = import_data['pathways']
        self.gene_to_pathways = defaultdict(set)
        
        for gene, pathways in import_data['gene_to_pathways'].items():
            self.gene_to_pathways[gene] = set(pathways)
        
        logger.info(f"Database imported from {input_file}")

class PathwayEnrichmentAnalyzer:
    """Analyze pathway enrichment in gene sets."""
    
    def __init__(self, pathway_database: PathwayDatabase):
        """Initialize analyzer."""
        self.pathway_database = pathway_database
        self.background_genes = self._get_background_genes()
    
    def _get_background_genes(self) -> Set[str]:
        """Get background gene set from all pathways."""
        background = set()
        for pathway_info in self.pathway_database.pathways.values():
            background.update(pathway_info['genes'])
        return background
    
    def analyze_enrichment(self, 
                         query_genes: List[str],
                         pathway_ids: Optional[List[str]] = None,
                         background_genes: Optional[List[str]] = None,
                         min_pathway_size: int = 5,
                         max_pathway_size: int = 500,
                         method: str = 'hypergeometric') -> List[PathwayEnrichmentResult]:
        """
        Analyze pathway enrichment for a set of genes.
        
        Args:
            query_genes: List of genes to analyze
            pathway_ids: Specific pathways to test (if None, test all)
            background_genes: Background gene set (if None, use all pathway genes)
            min_pathway_size: Minimum pathway size to consider
            max_pathway_size: Maximum pathway size to consider
            method: Statistical method ('hypergeometric' or 'fisher')
            
        Returns:
            List of enrichment results
        """
        # Prepare gene sets
        query_set = set(query_genes)
        background_set = set(background_genes) if background_genes else self.background_genes
        
        # Filter query genes to background
        query_filtered = query_set.intersection(background_set)
        
        if not query_filtered:
            logger.warning("No query genes found in background. Check gene symbols.")
            return []
        
        # Determine pathways to test
        if pathway_ids is None:
            pathway_ids = list(self.pathway_database.pathways.keys())
        
        # Test enrichment for each pathway
        results = []
        
        for pathway_id in pathway_ids:
            pathway_info = self.pathway_database.get_pathway(pathway_id)
            if not pathway_info:
                continue
            
            pathway_genes = set(pathway_info['genes']).intersection(background_set)
            
            # Filter by pathway size
            if len(pathway_genes) < min_pathway_size or len(pathway_genes) > max_pathway_size:
                continue
            
            # Calculate overlap
            overlap_genes = query_filtered.intersection(pathway_genes)
            
            if not overlap_genes:
                continue
            
            # Statistical test
            if method == 'hypergeometric':
                p_value, odds_ratio = self._hypergeometric_test(
                    len(overlap_genes),
                    len(query_filtered),
                    len(pathway_genes),
                    len(background_set)
                )
            elif method == 'fisher':
                p_value, odds_ratio = self._fisher_exact_test(
                    len(overlap_genes),
                    len(query_filtered),
                    len(pathway_genes),
                    len(background_set)
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Calculate enrichment score
            expected = (len(query_filtered) * len(pathway_genes)) / len(background_set)
            enrichment_score = len(overlap_genes) / expected if expected > 0 else 0
            
            # Create result
            result = PathwayEnrichmentResult(
                pathway_id=pathway_id,
                pathway_name=pathway_info['name'],
                category=pathway_info['category'],
                genes_in_pathway=list(pathway_genes),
                genes_in_query=list(query_filtered),
                overlap_genes=list(overlap_genes),
                p_value=p_value,
                adjusted_p_value=p_value,  # Will be adjusted later
                odds_ratio=odds_ratio,
                enrichment_score=enrichment_score
            )
            
            results.append(result)
        
        # Multiple testing correction
        if results:
            p_values = [r.p_value for r in results]
            adjusted_p_values = self._benjamini_hochberg_correction(p_values)
            
            for result, adj_p in zip(results, adjusted_p_values):
                result.adjusted_p_value = adj_p
        
        # Sort by adjusted p-value
        results.sort(key=lambda x: x.adjusted_p_value)
        
        logger.info(f"Analyzed {len(results)} pathways for {len(query_filtered)} genes")
        return results
    
    def _hypergeometric_test(self, 
                           overlap: int, 
                           query_size: int, 
                           pathway_size: int, 
                           background_size: int) -> Tuple[float, float]:
        """Perform hypergeometric test."""
        # P(X >= overlap)
        p_value = hypergeom.sf(overlap - 1, background_size, pathway_size, query_size)
        
        # Odds ratio
        a = overlap  # overlap
        b = query_size - overlap  # query not in pathway
        c = pathway_size - overlap  # pathway not in query
        d = background_size - pathway_size - query_size + overlap  # neither
        
        odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')
        
        return float(p_value), float(odds_ratio)
    
    def _fisher_exact_test(self, 
                          overlap: int, 
                          query_size: int, 
                          pathway_size: int, 
                          background_size: int) -> Tuple[float, float]:
        """Perform Fisher's exact test."""
        # Create contingency table
        a = overlap
        b = query_size - overlap
        c = pathway_size - overlap
        d = background_size - pathway_size - query_size + overlap
        
        # Unpack the tuple properly to avoid type issues
        result = fisher_exact([[a, b], [c, d]], alternative='greater')
        odds_ratio = float(result[0])  # type: ignore
        p_value = float(result[1])  # type: ignore
        
        return p_value, odds_ratio
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg correction for multiple testing."""
        n = len(p_values)
        if n == 0:
            return []
        
        # Sort p-values with indices
        sorted_indices = sorted(range(n), key=lambda i: p_values[i])
        sorted_p_values = [p_values[i] for i in sorted_indices]
        
        # Apply correction - use float list explicitly
        adjusted_p_values: List[float] = [0.0] * n
        for i in range(n - 1, -1, -1):
            rank = i + 1
            bh_value = sorted_p_values[i] * n / rank
            
            if i == n - 1:
                adjusted_p_values[i] = float(min(bh_value, 1.0))
            else:
                adjusted_p_values[i] = float(min(bh_value, adjusted_p_values[i + 1]))
        
        # Restore original order
        result: List[float] = [0.0] * n
        for i, original_idx in enumerate(sorted_indices):
            result[original_idx] = adjusted_p_values[i]
        
        return result
    
    def compare_gene_sets(self, 
                         gene_set1: List[str], 
                         gene_set2: List[str],
                         set1_name: str = "Set1",
                         set2_name: str = "Set2") -> Dict[str, Any]:
        """Compare pathway enrichment between two gene sets."""
        # Analyze enrichment for both sets
        results1 = self.analyze_enrichment(gene_set1)
        results2 = self.analyze_enrichment(gene_set2)
        
        # Create comparison
        pathways1 = {r.pathway_id: r for r in results1 if r.adjusted_p_value < 0.05}
        pathways2 = {r.pathway_id: r for r in results2 if r.adjusted_p_value < 0.05}
        
        common_pathways = set(pathways1.keys()).intersection(set(pathways2.keys()))
        unique_to_set1 = set(pathways1.keys()) - set(pathways2.keys())
        unique_to_set2 = set(pathways2.keys()) - set(pathways1.keys())
        
        comparison = {
            'set1_name': set1_name,
            'set2_name': set2_name,
            'set1_genes': len(set(gene_set1)),
            'set2_genes': len(set(gene_set2)),
            'common_genes': len(set(gene_set1).intersection(set(gene_set2))),
            'set1_pathways': len(pathways1),
            'set2_pathways': len(pathways2),
            'common_pathways': list(common_pathways),
            'unique_to_set1': list(unique_to_set1),
            'unique_to_set2': list(unique_to_set2),
            'pathway_details': {
                'common': [(pid, pathways1[pid], pathways2[pid]) for pid in common_pathways],
                'set1_unique': [pathways1[pid] for pid in unique_to_set1],
                'set2_unique': [pathways2[pid] for pid in unique_to_set2]
            }
        }
        
        return comparison
    
    def generate_enrichment_report(self, 
                                 results: List[PathwayEnrichmentResult],
                                 output_file: str = "pathway_enrichment_report.txt",
                                 significance_threshold: float = 0.05) -> str:
        """Generate pathway enrichment report."""
        significant_results = [r for r in results if r.adjusted_p_value < significance_threshold]
        
        report_lines = [
            "PATHWAY ENRICHMENT ANALYSIS REPORT",
            "=" * 50,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total pathways tested: {len(results)}",
            f"Significant pathways (FDR < {significance_threshold}): {len(significant_results)}",
            "",
            "SIGNIFICANT PATHWAYS",
            "-" * 30,
        ]
        
        for i, result in enumerate(significant_results[:20], 1):  # Top 20
            report_lines.extend([
                f"{i}. {result.pathway_name} ({result.pathway_id})",
                f"   Category: {result.category}",
                f"   Genes in overlap: {len(result.overlap_genes)}/{len(result.genes_in_pathway)}",
                f"   P-value: {result.p_value:.2e}",
                f"   Adjusted P-value: {result.adjusted_p_value:.2e}",
                f"   Enrichment score: {result.enrichment_score:.2f}",
                f"   Odds ratio: {result.odds_ratio:.2f}",
                f"   Overlap genes: {', '.join(result.overlap_genes[:10])}{'...' if len(result.overlap_genes) > 10 else ''}",
                ""
            ])
        
        # Category summary
        category_counts = Counter(r.category for r in significant_results)
        report_lines.extend([
            "ENRICHMENT BY CATEGORY",
            "-" * 25,
        ])
        
        for category, count in category_counts.most_common():
            report_lines.append(f"{category}: {count} pathways")
        
        report_content = '\n'.join(report_lines)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Enrichment report saved to {output_file}")
        return report_content

class NetworkAnalyzer:
    """Analyze gene/protein networks and interactions."""
    
    def __init__(self, pathway_database: PathwayDatabase):
        """Initialize network analyzer."""
        self.pathway_database = pathway_database
        self.interaction_networks = {}
    
    def build_pathway_network(self, pathway_ids: List[str]) -> nx.Graph:
        """Build network from pathway gene sets."""
        G = nx.Graph()
        
        # Add nodes and edges based on pathway membership
        for pathway_id in pathway_ids:
            pathway_info = self.pathway_database.get_pathway(pathway_id)
            if not pathway_info:
                continue
            
            genes = pathway_info['genes']
            
            # Add nodes
            for gene in genes:
                if not G.has_node(gene):
                    G.add_node(gene, pathways=[pathway_id])
                else:
                    G.nodes[gene]['pathways'].append(pathway_id)
            
            # Add edges between genes in same pathway
            for i, gene1 in enumerate(genes):
                for gene2 in genes[i+1:]:
                    if G.has_edge(gene1, gene2):
                        G.edges[gene1, gene2]['weight'] += 1
                        G.edges[gene1, gene2]['pathways'].append(pathway_id)
                    else:
                        G.add_edge(gene1, gene2, weight=1, pathways=[pathway_id])
        
        logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def analyze_network_properties(self, G: nx.Graph) -> NetworkAnalysisResult:
        """Analyze network properties and structure."""
        if G.number_of_nodes() == 0:
            logger.warning("Empty network provided")
            return NetworkAnalysisResult(
                network_id="empty", nodes=[], edges=[], communities={},
                centrality_scores={}, clustering_coefficient=0.0,
                modularity=0.0, average_path_length=0.0
            )
        
        # Basic properties
        nodes = list(G.nodes())
        edges = list(G.edges())
        
        # Centrality measures
        try:
            centrality_scores = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            
            # Combine centrality measures
            for node in centrality_scores:
                centrality_scores[node] = (
                    centrality_scores[node] + 
                    betweenness.get(node, 0) + 
                    closeness.get(node, 0)
                ) / 3
        except Exception as e:
            logger.warning(f"Could not compute centrality: {str(e)}")
            centrality_scores = {node: 0.0 for node in nodes}
        
        # Clustering coefficient
        try:
            clustering_coefficient = nx.average_clustering(G)
        except:
            clustering_coefficient = 0.0
        
        # Average path length
        try:
            if nx.is_connected(G):
                average_path_length = nx.average_shortest_path_length(G)
            else:
                # For disconnected graphs, compute for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                average_path_length = nx.average_shortest_path_length(subgraph)
        except:
            average_path_length = 0.0
        
        # Community detection
        communities = self._detect_communities(G)
        
        # Modularity
        try:
            community_list = [communities.get(node, 0) for node in G.nodes()]
            modularity = nx.algorithms.community.modularity(G, [
                {node for node, comm in communities.items() if comm == i}
                for i in set(communities.values())
            ])
        except:
            modularity = 0.0
        
        result = NetworkAnalysisResult(
            network_id=f"network_{len(nodes)}nodes",
            nodes=nodes,
            edges=edges,
            communities=self._group_communities(communities),
            centrality_scores=centrality_scores,
            clustering_coefficient=clustering_coefficient,
            modularity=modularity,
            average_path_length=average_path_length
        )
        
        return result
    
    def _detect_communities(self, G: nx.Graph) -> Dict[str, int]:
        """Detect communities in network."""
        if G.number_of_nodes() < 3:
            return {node: 0 for node in G.nodes()}
        
        try:
            # Use spectral clustering for community detection
            adjacency_matrix = nx.adjacency_matrix(G).toarray()
            
            # Determine optimal number of clusters
            n_nodes = adjacency_matrix.shape[0]
            max_clusters = min(10, n_nodes // 2)
            
            if max_clusters < 2:
                return {node: 0 for node in G.nodes()}
            
            best_score = -1
            best_labels = None
            
            for n_clusters in range(2, max_clusters + 1):
                try:
                    clustering = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity='precomputed',
                        random_state=42
                    )
                    labels = clustering.fit_predict(adjacency_matrix)
                    
                    # Calculate silhouette score
                    score = silhouette_score(adjacency_matrix, labels, metric='precomputed')
                    
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        
                except Exception as e:
                    logger.debug(f"Clustering with {n_clusters} clusters failed: {str(e)}")
                    continue
            
            if best_labels is not None:
                return {node: int(label) for node, label in zip(G.nodes(), best_labels)}
            else:
                return {node: 0 for node in G.nodes()}
                
        except Exception as e:
            logger.debug(f"Community detection failed: {str(e)}")
            return {node: 0 for node in G.nodes()}
    
    def _group_communities(self, communities: Dict[str, int]) -> Dict[int, List[str]]:
        """Group nodes by community."""
        grouped = defaultdict(list)
        for node, community in communities.items():
            grouped[community].append(node)
        return dict(grouped)
    
    def find_network_hubs(self, G: nx.Graph, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find network hub genes based on centrality measures."""
        if G.number_of_nodes() == 0:
            return []
        
        # Compute multiple centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = {}
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            # Fallback if eigenvector centrality fails
            eigenvector_centrality = {node: 0.0 for node in G.nodes()}
        
        # Combine centrality scores
        combined_scores = {}
        for node in G.nodes():
            combined_scores[node] = (
                degree_centrality.get(node, 0) +
                betweenness_centrality.get(node, 0) +
                closeness_centrality.get(node, 0) +
                eigenvector_centrality.get(node, 0)
            ) / 4
        
        # Safe handling of degree values - convert to plain int
        node_degrees = []
        for node in G.nodes():
            degree_val = G.degree(node)
            # NetworkX degree returns int, but type checker sees DiDegreeView
            node_degrees.append(int(degree_val))  # type: ignore
        
        max_degree = max(node_degrees) if node_degrees else 1
        avg_degree = float(sum(node_degrees) / len(node_degrees)) if node_degrees else 0.0
        
        # Sort by combined score
        hub_genes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return hub_genes[:top_k]
    
    def analyze_pathway_crosstalk(self, pathway_ids: List[str]) -> Dict[str, Any]:
        """Analyze crosstalk between pathways."""
        crosstalk_matrix = np.zeros((len(pathway_ids), len(pathway_ids)))
        pathway_genes = {}
        
        # Get genes for each pathway
        for i, pathway_id in enumerate(pathway_ids):
            pathway_info = self.pathway_database.get_pathway(pathway_id)
            if pathway_info:
                pathway_genes[i] = set(pathway_info['genes'])
        
        # Calculate pairwise overlaps
        for i in range(len(pathway_ids)):
            for j in range(i + 1, len(pathway_ids)):
                if i in pathway_genes and j in pathway_genes:
                    overlap = len(pathway_genes[i].intersection(pathway_genes[j]))
                    union = len(pathway_genes[i].union(pathway_genes[j]))
                    jaccard = overlap / union if union > 0 else 0
                    
                    crosstalk_matrix[i, j] = jaccard
                    crosstalk_matrix[j, i] = jaccard
        
        # Find strongest crosstalk pairs
        crosstalk_pairs = []
        for i in range(len(pathway_ids)):
            for j in range(i + 1, len(pathway_ids)):
                if crosstalk_matrix[i, j] > 0:
                    crosstalk_pairs.append({
                        'pathway1': pathway_ids[i],
                        'pathway2': pathway_ids[j],
                        'jaccard_index': crosstalk_matrix[i, j],
                        'overlap_genes': list(pathway_genes.get(i, set()).intersection(
                            pathway_genes.get(j, set())
                        ))
                    })
        
        # Sort by crosstalk strength
        crosstalk_pairs.sort(key=lambda x: x['jaccard_index'], reverse=True)
        
        return {
            'crosstalk_matrix': crosstalk_matrix,
            'pathway_ids': pathway_ids,
            'crosstalk_pairs': crosstalk_pairs,
            'strong_crosstalk': [p for p in crosstalk_pairs if p['jaccard_index'] > 0.1]
        }

class PathwayVisualizer:
    """Visualize pathway analysis results."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_enrichment_results(self, 
                               results: List[PathwayEnrichmentResult],
                               top_k: int = 20,
                               save_path: Optional[str] = None) -> go.Figure:
        """Plot pathway enrichment results."""
        # Take top results
        top_results = results[:top_k]
        
        if not top_results:
            logger.warning("No results to plot")
            return go.Figure()
        
        # Prepare data
        pathway_names = [r.pathway_name[:50] + "..." if len(r.pathway_name) > 50 
                        else r.pathway_name for r in top_results]
        enrichment_scores = [r.enrichment_score for r in top_results]
        p_values = [-np.log10(r.adjusted_p_value) for r in top_results]
        categories = [r.category for r in top_results]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars colored by category
        unique_categories = list(set(categories))
        color_map = {cat: self.color_palette[i % len(self.color_palette)] 
                    for i, cat in enumerate(unique_categories)}
        
        colors = [color_map[cat] for cat in categories]
        
        fig.add_trace(go.Bar(
            y=pathway_names,
            x=enrichment_scores,
            orientation='h',
            marker=dict(color=colors),
            text=[f"p={r.adjusted_p_value:.2e}" for r in top_results],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>" +
                         "Enrichment Score: %{x:.2f}<br>" +
                         "P-value: %{text}<br>" +
                         "Genes: %{customdata}<extra></extra>",
            customdata=[len(r.overlap_genes) for r in top_results]
        ))
        
        fig.update_layout(
            title="Pathway Enrichment Analysis",
            xaxis_title="Enrichment Score",
            yaxis_title="Pathway",
            height=max(400, len(top_results) * 25),
            showlegend=False,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Enrichment plot saved to {save_path}")
        
        return fig
    
    def plot_pathway_network(self, 
                           G: nx.Graph,
                           communities: Optional[Dict[str, int]] = None,
                           hub_genes: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> go.Figure:
        """Plot pathway network."""
        if G.number_of_nodes() == 0:
            logger.warning("Empty network provided")
            return go.Figure()
        
        # Use spring layout for node positions
        try:
            pos = nx.spring_layout(G, k=1, iterations=50)
        except:
            pos = {node: (i % 10, i // 10) for i, node in enumerate(G.nodes())}
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Node colors based on communities
        if communities:
            node_colors = [communities.get(node, 0) for node in G.nodes()]
            colorscale = 'Viridis'
        else:
            node_colors = 'lightblue'
            colorscale = None
        
        # Node sizes based on degree - convert to int to avoid DiDegreeView issues
        node_degrees = [int(G.degree(node)) for node in G.nodes()]  # type: ignore
        max_degree = max(node_degrees) if node_degrees else 1
        node_sizes = [10 + (float(degree) / float(max_degree)) * 20 for degree in node_degrees]
        
        # Highlight hub genes
        if hub_genes:
            hub_set = set(hub_genes)
            node_symbols = ['diamond' if node in hub_set else 'circle' for node in G.nodes()]
        else:
            node_symbols = 'circle'
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=colorscale,
                symbol=node_symbols,
                line=dict(width=2, color='white')
            ),
            hovertext=[f"{node}<br>Degree: {G.degree(node)}" for node in G.nodes()]
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title="Pathway Network",
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size represents degree centrality",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Network plot saved to {save_path}")
        
        return fig
    
    def plot_crosstalk_heatmap(self, 
                             crosstalk_data: Dict[str, Any],
                             save_path: Optional[str] = None) -> go.Figure:
        """Plot pathway crosstalk heatmap."""
        matrix = crosstalk_data['crosstalk_matrix']
        pathway_ids = crosstalk_data['pathway_ids']
        
        # Get pathway names
        pathway_names = [pid.replace('_', ' ').title() for pid in pathway_ids]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=pathway_names,
            y=pathway_names,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Jaccard Index"),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>" +
                         "Jaccard Index: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Pathway Crosstalk Analysis",
            xaxis_title="Pathway",
            yaxis_title="Pathway",
            width=800,
            height=800,
            template="plotly_white"
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Crosstalk heatmap saved to {save_path}")
        
        return fig
    
    def create_pathway_dashboard(self, 
                               enrichment_results: List[PathwayEnrichmentResult],
                               network_analysis: Optional[NetworkAnalysisResult] = None,
                               crosstalk_data: Optional[Dict[str, Any]] = None,
                               save_path: str = "pathway_dashboard.html") -> None:
        """Create comprehensive pathway analysis dashboard."""
        from plotly.subplots import make_subplots
        
        # Create subplots
        if network_analysis and crosstalk_data:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Pathway Enrichment', 'Network Communities', 
                               'Crosstalk Heatmap', 'Hub Genes'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "heatmap"}, {"type": "bar"}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Pathway Enrichment', 'Category Distribution'),
                specs=[[{"type": "bar"}, {"type": "pie"}]]
            )
        
        # Add enrichment plot
        top_results = enrichment_results[:15]
        pathway_names = [r.pathway_name[:30] + "..." if len(r.pathway_name) > 30 
                        else r.pathway_name for r in top_results]
        
        fig.add_trace(
            go.Bar(
                y=pathway_names,
                x=[r.enrichment_score for r in top_results],
                orientation='h',
                name="Enrichment Score",
                marker=dict(color='lightblue')
            ),
            row=1, col=1
        )
        
        # Add category pie chart
        categories = [r.category for r in enrichment_results if r.adjusted_p_value < 0.05]
        category_counts = Counter(categories)
        
        fig.add_trace(
            go.Pie(
                labels=list(category_counts.keys()),
                values=list(category_counts.values()),
                name="Categories"
            ),
            row=1, col=2
        )
        
        # Add network analysis if available
        if network_analysis:
            # Community sizes
            community_sizes = [len(genes) for genes in network_analysis.communities.values()]
            community_labels = [f"Community {i}" for i in range(len(community_sizes))]
            
            fig.add_trace(
                go.Bar(
                    x=community_labels,
                    y=community_sizes,
                    name="Community Size",
                    marker=dict(color='lightgreen')
                ),
                row=2, col=1
            )
        
        # Add crosstalk heatmap if available
        if crosstalk_data:
            matrix = crosstalk_data['crosstalk_matrix']
            pathway_ids = crosstalk_data['pathway_ids']
            
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=pathway_ids,
                    y=pathway_ids,
                    colorscale='RdYlBu_r',
                    showscale=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Pathway Analysis Dashboard",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        # Save dashboard
        fig.write_html(save_path)
        logger.info(f"Pathway dashboard saved to {save_path}")

def run_pathway_analysis(gene_list: List[str],
                        pathway_database: Optional[PathwayDatabase] = None,
                        output_dir: str = "pathway_analysis_results",
                        config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run comprehensive pathway analysis pipeline.
    
    Args:
        gene_list: List of genes to analyze
        pathway_database: Pathway database to use
        output_dir: Output directory for results
        config: Configuration dictionary
        
    Returns:
        Dictionary containing all analysis results
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if config is None:
        config = {
            'min_pathway_size': 5,
            'max_pathway_size': 500,
            'significance_threshold': 0.05,
            'top_pathways_plot': 20,
            'network_analysis': True,
            'crosstalk_analysis': True
        }
    
    # Initialize components
    if pathway_database is None:
        logger.info("DATABASE: Initializing pathway database...")
        pathway_database = PathwayDatabase()
    
    enrichment_analyzer = PathwayEnrichmentAnalyzer(pathway_database)
    network_analyzer = NetworkAnalyzer(pathway_database)
    visualizer = PathwayVisualizer()
    
    logger.info(f" Analyzing {len(gene_list)} genes...")
    
    # Step 1: Pathway Enrichment Analysis
    logger.info("STATS: Running pathway enrichment analysis...")
    enrichment_results = enrichment_analyzer.analyze_enrichment(
        gene_list,
        min_pathway_size=config['min_pathway_size'],
        max_pathway_size=config['max_pathway_size']
    )
    
    significant_results = [
        r for r in enrichment_results 
        if r.adjusted_p_value < config['significance_threshold']
    ]
    
    logger.info(f"SUCCESS: Found {len(significant_results)} significantly enriched pathways")
    
    # Step 2: Network Analysis
    network_analysis = None
    if config.get('network_analysis', True) and significant_results:
        logger.info("NETWORK: Building pathway network...")
        significant_pathway_ids = [r.pathway_id for r in significant_results[:20]]
        
        pathway_network = network_analyzer.build_pathway_network(significant_pathway_ids)
        network_analysis = network_analyzer.analyze_network_properties(pathway_network)
        
        # Find hub genes
        hub_genes = network_analyzer.find_network_hubs(pathway_network, top_k=10)
        
        logger.info(f"METRICS: Network analysis complete: {len(network_analysis.nodes)} nodes, "
                   f"{len(network_analysis.edges)} edges")
    
    # Step 3: Crosstalk Analysis
    crosstalk_data = None
    if config.get('crosstalk_analysis', True) and len(significant_results) > 1:
        logger.info("LINK: Analyzing pathway crosstalk...")
        significant_pathway_ids = [r.pathway_id for r in significant_results[:15]]
        crosstalk_data = network_analyzer.analyze_pathway_crosstalk(significant_pathway_ids)
        
        strong_crosstalk = len(crosstalk_data['strong_crosstalk'])
        logger.info(f"LINK: Found {strong_crosstalk} strong crosstalk pairs")
    
    # Step 4: Generate Visualizations
    logger.info("STATS: Generating visualizations...")
    
    # Enrichment plot
    enrichment_fig = visualizer.plot_enrichment_results(
        enrichment_results,
        top_k=config['top_pathways_plot'],
        save_path=str(output_path / "pathway_enrichment.html")
    )
    
    # Network plot
    network_fig = None
    if network_analysis:
        pathway_network = network_analyzer.build_pathway_network(
            [r.pathway_id for r in significant_results[:20]]
        )
        network_fig = visualizer.plot_pathway_network(
            pathway_network,
            communities={node: comm for comm, nodes in network_analysis.communities.items() 
                        for node in nodes},
            hub_genes=[gene for gene, score in hub_genes] if 'hub_genes' in locals() else None,
            save_path=str(output_path / "pathway_network.html")
        )
    
    # Crosstalk heatmap
    crosstalk_fig = None
    if crosstalk_data:
        crosstalk_fig = visualizer.plot_crosstalk_heatmap(
            crosstalk_data,
            save_path=str(output_path / "pathway_crosstalk.html")
        )
    
    # Comprehensive dashboard
    visualizer.create_pathway_dashboard(
        enrichment_results,
        network_analysis,
        crosstalk_data,
        save_path=str(output_path / "pathway_dashboard.html")
    )
    
    # Step 5: Generate Reports
    logger.info("REPORT: Generating analysis reports...")
    enrichment_analyzer.generate_enrichment_report(
        enrichment_results,
        output_file=str(output_path / "pathway_enrichment_report.txt"),
        significance_threshold=config['significance_threshold']
    )
    
    # Generate comprehensive summary report
    summary_report = generate_pathway_summary_report(
        enrichment_results, network_analysis, crosstalk_data,
        output_file=str(output_path / "pathway_analysis_summary.md")
    )
    
    # Compile results
    results = {
        'input_genes': gene_list,
        'enrichment_results': enrichment_results,
        'significant_pathways': significant_results,
        'network_analysis': network_analysis,
        'crosstalk_analysis': crosstalk_data,
        'hub_genes': hub_genes if 'hub_genes' in locals() else [],
        'visualizations': {
            'enrichment_plot': enrichment_fig,
            'network_plot': network_fig,
            'crosstalk_heatmap': crosstalk_fig
        },
        'config': config,
        'output_directory': str(output_path)
    }
    
    # Save results
    with open(output_path / "pathway_analysis_results.json", 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {
            'input_genes': gene_list,
            'enrichment_results': [r.to_dict() for r in enrichment_results],
            'significant_pathways': [r.to_dict() for r in significant_results],
            'network_analysis': network_analysis.to_dict() if network_analysis else None,
            'crosstalk_analysis': crosstalk_data,
            'hub_genes': hub_genes if 'hub_genes' in locals() else [],
            'config': config,
            'output_directory': str(output_path),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"COMPLETE: Pathway analysis completed! Results saved to: {output_path}")
    return results

def generate_pathway_summary_report(enrichment_results: List[PathwayEnrichmentResult],
                                  network_analysis: Optional[NetworkAnalysisResult],
                                  crosstalk_data: Optional[Dict[str, Any]],
                                  output_file: str) -> str:
    """Generate comprehensive pathway analysis summary report."""
    
    significant_results = [r for r in enrichment_results if r.adjusted_p_value < 0.05]
    
    report_lines = [
        "# Pathway Analysis Summary Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"- **Total pathways tested:** {len(enrichment_results)}",
        f"- **Significantly enriched pathways:** {len(significant_results)}",
        f"- **Top enrichment score:** {max([r.enrichment_score for r in enrichment_results], default=0):.2f}",
        "",
        "## Key Findings",
        "",
    ]
    
    if significant_results:
        # Top pathways
        report_lines.extend([
            "### Most Significantly Enriched Pathways",
            "",
        ])
        
        for i, result in enumerate(significant_results[:5], 1):
            report_lines.extend([
                f"{i}. **{result.pathway_name}**",
                f"   - Category: {result.category}",
                f"   - Enrichment Score: {result.enrichment_score:.2f}",
                f"   - P-value: {result.adjusted_p_value:.2e}",
                f"   - Genes: {len(result.overlap_genes)}/{len(result.genes_in_pathway)}",
                ""
            ])
        
        # Category breakdown
        categories = [r.category for r in significant_results]
        category_counts = Counter(categories)
        
        report_lines.extend([
            "### Enrichment by Category",
            "",
        ])
        
        for category, count in category_counts.most_common():
            percentage = (count / len(significant_results)) * 100
            report_lines.append(f"- **{category}:** {count} pathways ({percentage:.1f}%)")
        
        report_lines.append("")
    
    # Network analysis results
    if network_analysis:
        report_lines.extend([
            "## Network Analysis",
            "",
            f"- **Network size:** {len(network_analysis.nodes)} genes, {len(network_analysis.edges)} connections",
            f"- **Average clustering coefficient:** {network_analysis.clustering_coefficient:.3f}",
            f"- **Modularity:** {network_analysis.modularity:.3f}",
            f"- **Communities detected:** {len(network_analysis.communities)}",
            "",
            "### Network Communities",
            "",
        ])
        
        for comm_id, genes in network_analysis.communities.items():
            report_lines.append(f"- **Community {comm_id}:** {len(genes)} genes")
        
        report_lines.append("")
    
    # Crosstalk analysis
    if crosstalk_data:
        strong_crosstalk = crosstalk_data.get('strong_crosstalk', [])
        report_lines.extend([
            "## Pathway Crosstalk Analysis",
            "",
            f"- **Pathway pairs analyzed:** {len(crosstalk_data.get('crosstalk_pairs', []))}",
            f"- **Strong crosstalk pairs:** {len(strong_crosstalk)}",
            "",
        ])
        
        if strong_crosstalk:
            report_lines.extend([
                "### Strongest Crosstalk Pairs",
                "",
            ])
            
            for pair in strong_crosstalk[:5]:
                report_lines.extend([
                    f"- **{pair['pathway1']}** ↔ **{pair['pathway2']}**",
                    f"  - Jaccard Index: {pair['jaccard_index']:.3f}",
                    f"  - Shared genes: {len(pair['overlap_genes'])}",
                    ""
                ])
    
    # Biological insights
    report_lines.extend([
        "## Biological Insights & Interpretation",
        "",
        "### Key Biological Processes",
        "",
    ])
    
    if significant_results:
        # Group by category for insights
        category_pathways = defaultdict(list)
        for result in significant_results:
            category_pathways[result.category].append(result)
        
        for category, pathways in category_pathways.items():
            report_lines.extend([
                f"**{category}:**",
                f"- {len(pathways)} enriched pathway(s)",
                f"- Average enrichment: {np.mean([p.enrichment_score for p in pathways]):.2f}",
                f"- Key pathways: {', '.join([p.pathway_name for p in pathways[:3]])}",
                ""
            ])
    
    # Recommendations
    report_lines.extend([
        "## Recommendations for Follow-up",
        "",
        "### Experimental Validation",
        "",
    ])
    
    if significant_results:
        top_pathway = significant_results[0]
        report_lines.extend([
            f"1. **Validate {top_pathway.pathway_name}**: Priority pathway for experimental validation",
            f"   - P-value: {top_pathway.p_value:.2e}",
            f"   - Effect size: {top_pathway.effect_size:.3f}",
            "2. **Design targeted experiments** based on pathway predictions",
            "3. **Monitor key biomarkers** identified in the analysis"
        ])
    
    # Compile report content
    report_content = '\n'.join(report_lines)
    
    # Save report to file
    try:
        with open(output_file, 'w') as f:
            f.write(report_content)
        logger.info(f"Pathway summary report saved to {output_file}")
    except Exception as e:
        logger.warning(f"Could not save report to {output_file}: {str(e)}")
    
    return report_content  # Always return the report content as string

def fetch_kegg_pathways(gene_list: List[str]) -> Dict[str, Any]:
    BASE_URL = "http://rest.kegg.jp"
    pathways = {}
    
    try:
        # Convert gene list to KEGG format
        gene_ids = [f"hsa:{gene}" for gene in gene_list]
        
        # Get pathway information
        response = requests.post(
            f"{BASE_URL}/link/pathway",
            data="\n".join(gene_ids),
            headers={"Content-Type": "text/plain"}
        )
        
        if response.status_code == 200:
            for line in response.text.split('\n'):
                if line:
                    gene, pathway = line.split('\t')
                    pathway_id = pathway.split(':')[1]
                    pathways.setdefault(pathway_id, []).append(gene)
    except Exception as e:
        logger.error(f"KEGG API error: {str(e)}")
    
    return pathways