"""
Molecular Data Loader for OpenPerturbation

Comprehensive molecular data loading with RDKit integration and fallbacks.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import pickle
import json
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from omegaconf import DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    warnings.warn("OmegaConf not available")
    OMEGACONF_AVAILABLE = False
    DictConfig = dict

# Chemistry libraries with proper imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import DataStructs
    from rdkit.Chem.rdmolfiles import SDMolSupplier, MolFromSmiles, MolToSmiles
    from rdkit.Chem.rdmolops import SanitizeMol
    from rdkit.DataStructs import TanimotoSimilarity
    from rdkit.Chem.rdMolChemicalFeatures import BuildFeatureFactory
    from rdkit import RDConfig
    
    # Import Mol type
    Mol = Chem.Mol
    
    RDKIT_AVAILABLE = True
except Exception:  # Catch all exceptions during RDKit import
    warnings.warn("RDKit not available. Molecular features will be limited.")
    RDKIT_AVAILABLE = False
    
    # Create comprehensive dummy classes for RDKit
    class Mol:
        def __init__(self):
            self._atoms = [DummyAtom() for _ in range(10)]  # Dummy molecule with 10 atoms
            self._bonds = [DummyBond() for _ in range(9)]   # 9 bonds for linear chain
            
        def GetNumAtoms(self): 
            return len(self._atoms)
            
        def GetAtoms(self): 
            return self._atoms
            
        def GetBonds(self): 
            return self._bonds
            
        def GetPropNames(self): 
            return ["_Name", "Activity"]
            
        def GetProp(self, name): 
            if name == "_Name":
                return "dummy_compound"
            elif name == "Activity":
                return "1.0"
            return ""
    
    class DummyAtom:
        def GetAtomicNum(self): return 6
        def GetDegree(self): return 2
        def GetFormalCharge(self): return 0
        def GetChiralTag(self): return 0
        def GetTotalNumHs(self): return 1
        def GetIsAromatic(self): return False
        def GetHybridization(self): return 2
        def GetIdx(self): return 0
        
    class DummyBond:
        def GetBeginAtomIdx(self): return 0
        def GetEndAtomIdx(self): return 1
        def GetBondType(self): return 1
        def GetIsConjugated(self): return False
        def GetIsAromatic(self): return False
        def GetStereo(self): return 0

    # Dummy functions
    def MolFromSmiles(smiles):
        return Mol() if smiles else None
        
    def MolToSmiles(mol):
        return "CCO"  # dummy SMILES
        
    def SanitizeMol(mol):
        pass
        
    class Descriptors:
        @staticmethod
        def MolWt(mol): return 46.07
        @staticmethod
        def MolLogP(mol): return -0.31
        @staticmethod
        def NumHDonors(mol): return 1
        @staticmethod
        def NumHAcceptors(mol): return 1
        @staticmethod
        def TPSA(mol): return 20.23
        @staticmethod
        def NumRotatableBonds(mol): return 0
        @staticmethod
        def NumAromaticRings(mol): return 0
        @staticmethod
        def NumSaturatedRings(mol): return 0
        @staticmethod
        def NumAliphaticRings(mol): return 0
        @staticmethod
        def RingCount(mol): return 0
        @staticmethod
        def FractionCsp3(mol): return 0.4
        @staticmethod
        def BertzCT(mol): return 15.5
        
    class SDMolSupplier:
        def __init__(self, filename):
            self.filename = filename
            self._mols = [Mol() for _ in range(10)]  # Dummy data
            
        def __iter__(self):
            return iter(self._mols)
            
        def __len__(self):
            return len(self._mols)

# Graph libraries
try:
    import networkx as nx
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch Geometric not available. Graph features will be limited.")
    TORCH_GEOMETRIC_AVAILABLE = False
    
    # Create dummy classes
    class Data:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.num_nodes = kwargs.get('x', torch.zeros(10, 5)).size(0)
            self.num_edges = kwargs.get('edge_index', torch.zeros(2, 0)).size(1)

    class Batch:
        @staticmethod
        def from_data_list(data_list):
            if not data_list:
                return Data(x=torch.zeros(0, 5), edge_index=torch.zeros(2, 0))
            return data_list[0]


class MolecularFeatureExtractor:
    """Extract molecular features from chemical structures."""

    def __init__(self, config: Union[DictConfig, dict]):
        self.config = config
        self.include_descriptors = config.get("include_descriptors", True)
        self.include_fingerprints = config.get("include_fingerprints", True)
        self.include_graphs = config.get("include_graphs", True)

        # Descriptor settings
        self.descriptor_list = config.get(
            "descriptors",
            [
                "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
                "NumRotatableBonds", "NumAromaticRings", "NumSaturatedRings", 
                "NumAliphaticRings", "RingCount", "FractionCsp3", "BertzCT"
            ],
        )

        # Fingerprint settings
        self.fingerprint_type = config.get("fingerprint_type", "morgan")
        self.fingerprint_radius = config.get("fingerprint_radius", 2)
        self.fingerprint_bits = config.get("fingerprint_bits", 2048)

        # Graph settings
        self.max_atoms = config.get("max_atoms", 100)
        self.atom_features = config.get(
            "atom_features",
            ["atomic_num", "degree", "formal_charge", "chiral_tag", 
             "num_implicit_hs", "is_aromatic", "hybridization"]
        )
        self.bond_features = config.get(
            "bond_features", ["bond_type", "is_conjugated", "is_aromatic", "stereo"]
        )

        if RDKIT_AVAILABLE:
            self._init_rdkit_components()

    def _init_rdkit_components(self):
        """Initialize RDKit components."""
        try:
            # Fingerprint generator
            if self.fingerprint_type == "morgan":
                self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.fingerprint_radius, fpSize=self.fingerprint_bits
                )
            elif self.fingerprint_type == "rdkit":
                self.fp_generator = rdFingerprintGenerator.GetRDKitFPGenerator(
                    fpSize=self.fingerprint_bits
                )
            else:
                self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.fingerprint_radius, fpSize=self.fingerprint_bits
                )

            # Chemical feature factory
            try:
                fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
                if os.path.exists(fdef_path):
                    self.feature_factory = BuildFeatureFactory(fdef_path)
                else:
                    self.feature_factory = None
                    logger.warning("BaseFeatures.fdef not found")
            except Exception as e:
                logger.warning(f"Failed to initialize chemical features: {e}")
                self.feature_factory = None
        except Exception as e:
            logger.warning(f"Failed to initialize RDKit components: {e}")

    def smiles_to_mol(self, smiles: str) -> Optional[Mol]:
        """Convert SMILES string to RDKit molecule."""
        if not RDKIT_AVAILABLE:
            return Mol() if smiles else None
            
        try:
            mol = MolFromSmiles(smiles)
            if mol is not None:
                SanitizeMol(mol)
            return mol
        except Exception as e:
            logger.debug(f"Failed to parse SMILES {smiles}: {e}")
            return None

    def extract_descriptors(self, mol: Mol) -> Dict[str, float]:
        """Extract molecular descriptors."""
        descriptors = {}
        
        if not RDKIT_AVAILABLE:
            # Return dummy descriptors
            descriptor_values = {
                "MolWt": 180.16, "MolLogP": 1.2, "NumHDonors": 2, "NumHAcceptors": 3,
                "TPSA": 60.7, "NumRotatableBonds": 3, "NumAromaticRings": 1,
                "NumSaturatedRings": 0, "NumAliphaticRings": 0, "RingCount": 1,
                "FractionCsp3": 0.4, "BertzCT": 15.5
            }
            return {desc: descriptor_values.get(desc, 0.0) for desc in self.descriptor_list}
        
        for desc_name in self.descriptor_list:
            try:
                if hasattr(Descriptors, desc_name):
                    descriptor_func = getattr(Descriptors, desc_name)
                    descriptors[desc_name] = float(descriptor_func(mol))
                else:
                    descriptors[desc_name] = 0.0
            except Exception as e:
                logger.debug(f"Failed to compute descriptor {desc_name}: {e}")
                descriptors[desc_name] = 0.0
                
        return descriptors

    def extract_fingerprints(self, mol: Mol) -> np.ndarray:
        """Extract molecular fingerprints."""
        if not RDKIT_AVAILABLE:
            # Return dummy fingerprint
            return np.random.randint(0, 2, self.fingerprint_bits).astype(float)
        
        try:
            fp = self.fp_generator.GetFingerprint(mol)
            # Convert to numpy array
            fp_array = np.zeros(self.fingerprint_bits)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            return fp_array
        except Exception as e:
            logger.debug(f"Failed to extract fingerprint: {e}")
            return np.zeros(self.fingerprint_bits)

    def mol_to_graph(self, mol: Mol) -> Optional[Data]:
        """Convert molecule to PyTorch Geometric graph."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            # Return dummy graph
            num_atoms = min(mol.GetNumAtoms(), self.max_atoms)
            x = torch.randn(num_atoms, len(self.atom_features))
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            if num_atoms > 1:
                # Create simple linear chain for dummy
                edges = [(i, i+1) for i in range(num_atoms-1)]
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            return Data(x=x, edge_index=edge_index)
        
        try:
            # Get atom features
            atom_features = []
            for atom in mol.GetAtoms():
                if len(atom_features) >= self.max_atoms:
                    break
                features = self._get_atom_features(atom)
                atom_features.append(features)
            
            if not atom_features:
                return None
                
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # Get bond features and connectivity
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                if start_idx < self.max_atoms and end_idx < self.max_atoms:
                    # Add both directions for undirected graph
                    edge_indices.extend([[start_idx, end_idx], [end_idx, start_idx]])
                    
                    bond_feat = self._get_bond_features(bond)
                    edge_features.extend([bond_feat, bond_feat])
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                edge_attr = torch.zeros(0, len(self.bond_features), dtype=torch.float)
                
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            logger.debug(f"Failed to create molecular graph: {e}")
            return None

    def _get_atom_features(self, atom) -> List[float]:
        """Extract features for a single atom."""
        features = []
        
        try:
            if "atomic_num" in self.atom_features:
                features.append(float(atom.GetAtomicNum()))
            if "degree" in self.atom_features:
                features.append(float(atom.GetDegree()))
            if "formal_charge" in self.atom_features:
                features.append(float(atom.GetFormalCharge()))
            if "chiral_tag" in self.atom_features:
                features.append(float(atom.GetChiralTag()))
            if "num_implicit_hs" in self.atom_features:
                features.append(float(atom.GetTotalNumHs()))
            if "is_aromatic" in self.atom_features:
                features.append(float(atom.GetIsAromatic()))
            if "hybridization" in self.atom_features:
                features.append(float(atom.GetHybridization()))
        except Exception as e:
            logger.debug(f"Error extracting atom features: {e}")
            # Return zeros if extraction fails
            features = [0.0] * len(self.atom_features)
            
        return features

    def _get_bond_features(self, bond) -> List[float]:
        """Extract features for a single bond."""
        features = []
        
        try:
            if "bond_type" in self.bond_features:
                features.append(float(bond.GetBondType()))
            if "is_conjugated" in self.bond_features:
                features.append(float(bond.GetIsConjugated()))
            if "is_aromatic" in self.bond_features:
                features.append(float(bond.GetIsAromatic()))
            if "stereo" in self.bond_features:
                features.append(float(bond.GetStereo()))
        except Exception as e:
            logger.debug(f"Error extracting bond features: {e}")
            # Return zeros if extraction fails
            features = [0.0] * len(self.bond_features)
            
        return features

    def extract_all_features(self, smiles: str) -> Dict:
        """Extract all molecular features from SMILES."""
        mol = self.smiles_to_mol(smiles)
        
        if mol is None:
            # Return empty features
            return {
                "descriptors": {desc: 0.0 for desc in self.descriptor_list},
                "fingerprints": np.zeros(self.fingerprint_bits),
                "graph": None,
                "valid": False
            }
            
        features = {"valid": True}
        
        if self.include_descriptors:
            features["descriptors"] = self.extract_descriptors(mol)
            
        if self.include_fingerprints:
            features["fingerprints"] = self.extract_fingerprints(mol)
            
        if self.include_graphs:
            features["graph"] = self.mol_to_graph(mol)
            
        return features


class MolecularDataset(Dataset):
    """Dataset for molecular data with comprehensive feature extraction."""

    def __init__(
        self,
        config: Union[DictConfig, dict],
        data_file: str,
        metadata_file: Optional[str] = None,
        mode: str = "train",
        transform=None,
    ):
        """
        Initialize molecular dataset.
        
        Args:
            config: Configuration dictionary
            data_file: Path to molecular data file
            metadata_file: Optional metadata file
            mode: Dataset mode ('train', 'val', 'test')
            transform: Optional data transforms
        """
        self.config = config
        self.data_file = Path(data_file)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.mode = mode
        self.transform = transform
        
        # Initialize feature extractor
        self.feature_extractor = MolecularFeatureExtractor(config)
        
        # Load data
        self.data = self._load_molecular_data()
        self.metadata = self._load_metadata() if self.metadata_file else None
        
        # Merge metadata if available
        if self.metadata is not None:
            self._merge_metadata()
            
        # Precompute features if requested
        if config.get("precompute_features", False):
            self._precompute_features()
            
        logger.info(f"Loaded molecular dataset with {len(self.data)} compounds")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get molecular sample by index."""
        compound_data = self.data.iloc[idx]
        
        # Extract molecular features
        smiles = compound_data.get("SMILES", compound_data.get("smiles", ""))
        
        if self.config.get("precompute_features", False) and f"features_{idx}" in compound_data:
            features = compound_data[f"features_{idx}"]
        else:
            features = self.feature_extractor.extract_all_features(smiles)
        
        # Create sample dictionary
        sample = {
            "compound_id": compound_data.get("compound_id", str(idx)),
            "smiles": smiles,
            "features": features,
            "perturbation": self._get_perturbation_info(compound_data),
            "target": self._get_target_info(compound_data),
            "activity": self._get_activity_info(compound_data),
            "metadata": compound_data.to_dict()
        }
        
        # Apply transforms
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample

    def _load_molecular_data(self) -> pd.DataFrame:
        """Load molecular data from various file formats."""
        try:
            if self.data_file.suffix.lower() == '.csv':
                return pd.read_csv(self.data_file)
            elif self.data_file.suffix.lower() == '.sdf':
                return self._load_sdf_data()
            elif self.data_file.suffix.lower() == '.json':
                return pd.read_json(self.data_file)
            elif self.data_file.suffix.lower() in ['.pkl', '.pickle']:
                return pd.read_pickle(self.data_file)
            else:
                logger.warning(f"Unknown file format: {self.data_file.suffix}")
                # Try CSV as fallback
                return pd.read_csv(self.data_file)
        except Exception as e:
            logger.error(f"Failed to load molecular data: {e}")
            # Return dummy data
            return self._create_dummy_data()

    def _load_sdf_data(self) -> pd.DataFrame:
        """Load data from SDF file."""
        molecules_data = []
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, cannot load SDF files")
            return self._create_dummy_data()
        
        try:
            suppl = SDMolSupplier(str(self.data_file))
            
            for i, mol in enumerate(suppl):
                if mol is None:
                    continue
                    
                mol_data = {
                    "compound_id": f"mol_{i}",
                    "SMILES": MolToSmiles(mol)
                }
                
                # Extract properties from SDF
                for prop_name in mol.GetPropNames():
                    try:
                        mol_data[prop_name] = mol.GetProp(prop_name)
                    except Exception:
                        continue
                        
                molecules_data.append(mol_data)
                
            return pd.DataFrame(molecules_data)
            
        except Exception as e:
            logger.error(f"Failed to load SDF file: {e}")
            return self._create_dummy_data()

    def _create_dummy_data(self) -> pd.DataFrame:
        """Create dummy molecular data for testing."""
        dummy_smiles = [
            "CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCC(=O)O",
            "c1ccc(cc1)O", "CC(C)(C)O", "CCCC", "c1ccc2ccccc2c1", "CC(=O)N"
        ]
        
        data = []
        for i, smiles in enumerate(dummy_smiles):
            data.append({
                "compound_id": f"compound_{i}",
                "SMILES": smiles,
                "activity": np.random.uniform(0, 1),
                "target_protein": f"protein_{i % 3}",
                "assay_type": "binding" if i % 2 == 0 else "functional"
            })
        
        return pd.DataFrame(data)

    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """Load metadata file."""
        if not self.metadata_file.exists():
            return None
            
        try:
            if self.metadata_file.suffix.lower() == '.csv':
                return pd.read_csv(self.metadata_file)
            elif self.metadata_file.suffix.lower() == '.json':
                return pd.read_json(self.metadata_file)
            else:
                return pd.read_csv(self.metadata_file)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None

    def _merge_metadata(self):
        """Merge metadata with main data."""
        if self.metadata is None:
            return
            
        try:
            # Try to merge on compound_id
            if "compound_id" in self.data.columns and "compound_id" in self.metadata.columns:
                self.data = self.data.merge(self.metadata, on="compound_id", how="left")
            else:
                logger.warning("Cannot merge metadata: no common compound_id column")
        except Exception as e:
            logger.warning(f"Failed to merge metadata: {e}")

    def _precompute_features(self):
        """Precompute molecular features for faster loading."""
        logger.info("Precomputing molecular features...")
        
        for idx in range(len(self.data)):
            smiles = self.data.iloc[idx].get("SMILES", "")
            features = self.feature_extractor.extract_all_features(smiles)
            # Store features in data
            self.data.at[idx, f"features_{idx}"] = features

    def _get_perturbation_info(self, compound_data: pd.Series) -> Dict:
        """Extract perturbation information."""
        return {
            "compound_id": compound_data.get("compound_id", ""),
            "dose": compound_data.get("dose", compound_data.get("concentration", 1.0)),
            "time": compound_data.get("time", compound_data.get("duration", 24.0)),
            "perturbation_type": compound_data.get("perturbation_type", "chemical")
        }

    def _get_target_info(self, compound_data: pd.Series) -> Dict:
        """Extract target information."""
        return {
            "target_protein": compound_data.get("target_protein", compound_data.get("target", "unknown")),
            "pathway": compound_data.get("pathway", "unknown"),
            "mechanism": compound_data.get("mechanism", compound_data.get("mode_of_action", "unknown"))
        }

    def _get_activity_info(self, compound_data: pd.Series) -> Dict:
        """Extract activity/bioactivity information."""
        return {
            "activity_value": float(compound_data.get("activity", compound_data.get("bioactivity", 0.0))),
            "activity_type": compound_data.get("activity_type", "binding"),
            "unit": compound_data.get("unit", "uM"),
            "assay_type": compound_data.get("assay_type", "biochemical")
        }


class MolecularDataLoader:
    """Data loader for molecular datasets with train/val/test splits."""

    def __init__(self, config: Union[DictConfig, dict]):
        """Initialize molecular data loader."""
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 0)
        self.pin_memory = config.get("pin_memory", False)
        
        # Data splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets and data loaders."""
        data_dir = Path(self.config.get("data_dir", "data/molecular"))
        
        # Load datasets
        if stage == "fit" or stage is None:
            train_file = data_dir / self.config.get("train_file", "train.csv")
            val_file = data_dir / self.config.get("val_file", "val.csv")
            
            if train_file.exists():
                self.train_dataset = MolecularDataset(
                    self.config, str(train_file), mode="train"
                )
            
            if val_file.exists():
                self.val_dataset = MolecularDataset(
                    self.config, str(val_file), mode="val"
                )
            elif self.train_dataset is not None:
                # Split train dataset
                self._split_dataset(self.train_dataset)
        
        if stage == "test" or stage is None:
            test_file = data_dir / self.config.get("test_file", "test.csv")
            if test_file.exists():
                self.test_dataset = MolecularDataset(
                    self.config, str(test_file), mode="test"
                )
        
        # Create data loaders
        self._create_dataloaders()

    def _split_dataset(self, full_dataset: MolecularDataset):
        """Split dataset into train/val sets."""
        from torch.utils.data import random_split
        
        val_ratio = self.config.get("val_ratio", 0.2)
        dataset_size = len(full_dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def _create_dataloaders(self):
        """Create PyTorch data loaders."""
        
        def collate_fn(batch):
            """Custom collate function for molecular data."""
            # Separate different data types
            compound_ids = [item["compound_id"] for item in batch]
            smiles = [item["smiles"] for item in batch]
            perturbations = [item["perturbation"] for item in batch]
            targets = [item["target"] for item in batch]
            activities = [item["activity"] for item in batch]
            
            # Handle molecular features
            descriptors = []
            fingerprints = []
            graphs = []
            
            for item in batch:
                features = item["features"]
                if features["valid"]:
                    # Descriptors
                    if "descriptors" in features:
                        desc_values = list(features["descriptors"].values())
                        descriptors.append(desc_values)
                    
                    # Fingerprints
                    if "fingerprints" in features:
                        fingerprints.append(features["fingerprints"])
                    
                    # Graphs
                    if "graph" in features and features["graph"] is not None:
                        graphs.append(features["graph"])
                else:
                    # Handle invalid molecules
                    if descriptors:
                        descriptors.append([0.0] * len(descriptors[0]))
                    if fingerprints:
                        fingerprints.append(np.zeros_like(fingerprints[0]))
            
            # Convert to tensors
            batch_data = {
                "compound_ids": compound_ids,
                "smiles": smiles,
                "perturbations": perturbations,
                "targets": targets,
                "activities": activities
            }
            
            if descriptors:
                batch_data["descriptors"] = torch.tensor(descriptors, dtype=torch.float)
            
            if fingerprints:
                batch_data["fingerprints"] = torch.tensor(
                    np.array(fingerprints), dtype=torch.float
                )
            
            if graphs and TORCH_GEOMETRIC_AVAILABLE:
                batch_data["graphs"] = Batch.from_data_list(graphs)
            
            return batch_data
        
        # Create data loaders
        if self.train_dataset is not None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn
            )
        
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn
            )
        
        if self.test_dataset is not None:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn
            )

    def get_dataloader(self, split: str) -> Optional[DataLoader]:
        """Get data loader for specified split."""
        if split == "train":
            return self.train_loader
        elif split == "val":
            return self.val_loader
        elif split == "test":
            return self.test_loader
        else:
            logger.warning(f"Unknown split: {split}")
            return None

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the datasets."""
        stats = {"splits": {}}
        
        if self.train_dataset:
            stats["splits"]["train"] = {
                "size": len(self.train_dataset),
                "batch_size": self.batch_size
            }
        
        if self.val_dataset:
            stats["splits"]["val"] = {
                "size": len(self.val_dataset),
                "batch_size": self.batch_size
            }
        
        if self.test_dataset:
            stats["splits"]["test"] = {
                "size": len(self.test_dataset),
                "batch_size": self.batch_size
            }
        
        # Add feature statistics
        if self.train_dataset:
            sample = self.train_dataset[0]
            features = sample["features"]
            
            stats["features"] = {
                "descriptors": len(features.get("descriptors", {})),
                "fingerprint_bits": features.get("fingerprints", np.array([])).shape[0] if "fingerprints" in features else 0,
                "graph_available": features.get("graph") is not None
            }
        
        return stats

    def get_compound_by_id(self, compound_id: str, split: str = "train") -> Optional[Dict]:
        """Get compound data by ID."""
        dataset = None
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        
        if dataset is None:
            return None
        
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample["compound_id"] == compound_id:
                return sample
        
        return None


def create_synthetic_molecular_data(config: Union[DictConfig, dict], output_dir: Union[str, Path]):
    """Create synthetic molecular data for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate diverse SMILES
    smiles_templates = [
        "CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCC(=O)O",
        "c1ccc(cc1)O", "CC(C)(C)O", "CCCC", "c1ccc2ccccc2c1", "CC(=O)N",
        "c1ccncc1", "CC(=O)OC", "CCCN", "c1ccc(cc1)N", "CC(=O)C",
        "CCOC(=O)C", "c1ccc(cc1)C", "CCC(C)C", "c1ccc2[nH]ccc2c1", "CC(=O)NC"
    ]
    
    # Expand SMILES with modifications
    all_smiles = []
    for base_smiles in smiles_templates:
        all_smiles.append(base_smiles)
        # Add some variations
        for i in range(3):
            modified = base_smiles
            if "C" in modified:
                modified = modified.replace("C", "CC", 1)  # Add carbon
            all_smiles.append(modified)
    
    # Remove duplicates
    all_smiles = list(set(all_smiles))
    
    n_compounds = min(len(all_smiles), config.get("n_compounds", 1000))
    
    data = []
    for i in range(n_compounds):
        smiles = all_smiles[i % len(all_smiles)]
        
        compound_data = {
            "compound_id": f"compound_{i:05d}",
            "SMILES": smiles,
            "activity": np.random.uniform(0, 10),
            "target_protein": f"protein_{np.random.randint(0, 10)}",
            "assay_type": np.random.choice(["binding", "functional", "enzymatic"]),
            "dose": np.random.uniform(0.1, 100),
            "time": np.random.choice([6, 12, 24, 48]),
            "perturbation_type": "chemical",
            "pathway": f"pathway_{np.random.randint(0, 5)}",
            "mechanism": np.random.choice(["agonist", "antagonist", "inhibitor", "activator"]),
            "unit": "uM",
            "source": "synthetic"
        }
        
        data.append(compound_data)
    
    df = pd.DataFrame(data)
    
    # Split into train/val/test
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("val_ratio", 0.15)
    
    n_train = int(train_ratio * len(df))
    n_val = int(val_ratio * len(df))
    
    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]
    
    # Save files
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    # Create metadata file
    metadata = []
    for i in range(len(df)):
        metadata.append({
            "compound_id": f"compound_{i:05d}",
            "molecular_weight": np.random.uniform(100, 800),
            "logp": np.random.uniform(-3, 7),
            "hbd": np.random.randint(0, 10),
            "hba": np.random.randint(0, 15),
            "tpsa": np.random.uniform(0, 200),
            "rotatable_bonds": np.random.randint(0, 20)
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    
    logger.info(f"Created synthetic molecular data in {output_dir}")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def test_molecular_loader():
    """Test molecular data loader functionality."""
    logger.info("Testing molecular data loader...")
    
    # Create test config
    config = {
        "data_dir": "test_molecular_data",
        "batch_size": 4,
        "include_descriptors": True,
        "include_fingerprints": True,
        "include_graphs": True,
        "fingerprint_bits": 512,
        "precompute_features": False,
        "n_compounds": 50
    }
    
    # Create synthetic data
    create_synthetic_molecular_data(config, config["data_dir"])
    
    # Initialize data loader
    loader = MolecularDataLoader(config)
    loader.setup()
    
    # Test data loading
    train_loader = loader.get_dataloader("train")
    if train_loader:
        for batch in train_loader:
            logger.info(f"Batch keys: {batch.keys()}")
            if "descriptors" in batch:
                logger.info(f"Descriptors shape: {batch['descriptors'].shape}")
            if "fingerprints" in batch:
                logger.info(f"Fingerprints shape: {batch['fingerprints'].shape}")
            break
    
    # Print statistics
    stats = loader.get_dataset_statistics()
    logger.info(f"Dataset statistics: {stats}")
    
    logger.info("Molecular data loader test completed successfully!")


if __name__ == "__main__":
    test_molecular_loader()
