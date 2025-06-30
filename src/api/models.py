"""
Pydantic models for API requests and responses.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Annotated, Union, Literal
from datetime import datetime
import uuid

from pydantic import BaseModel, Field, field_validator

# -----------------------------------------------------------------------------
# Shared definitions
# -----------------------------------------------------------------------------

class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# -----------------------------------------------------------------------------
# Base models
# -----------------------------------------------------------------------------
class BaseRequest(BaseModel):
    """Base request model."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None
    disk_free_gb: float
    active_jobs: int


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
class AnalysisRequest(BaseRequest):
    """Request model for analysis operations."""
    experiment_type: Optional[str] = None
    data_paths: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    
    @field_validator('experiment_type')
    @classmethod
    def validate_experiment_type(cls, v):
        allowed_types = ['causal_discovery', 'intervention_design', 'explainability', 'multimodal_fusion']
        if v not in allowed_types:
            raise ValueError(f'experiment_type must be one of {allowed_types}')
        return v

class AnalysisResponse(BaseResponse):
    """Response model for analysis operations."""
    job_id: str
    status: str
    progress: int = Field(ge=0, le=100, description="Progress percentage")
    results: Optional[Dict[str, Any]] = None
    estimated_completion: Optional[datetime] = None


# -----------------------------------------------------------------------------
# Causal discovery
# -----------------------------------------------------------------------------
class CausalDiscoveryRequest(BaseRequest):
    """Request model for causal discovery."""
    data: Optional[List[List[float]]] = None
    method: Optional[str] = None
    alpha: Optional[float] = None
    variable_names: Optional[List[str]] = None
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        allowed_methods = ['pc', 'fci', 'ges', 'lingam', 'direct_lingam']
        if v not in allowed_methods:
            raise ValueError(f'method must be one of {allowed_methods}')
        return v

class CausalDiscoveryResponse(BaseResponse):
    """Response model for causal discovery."""
    job_id: str
    graph: Dict[str, Any]
    metrics: Dict[str, float]
    method_used: str
    execution_time: float
    node_count: int
    edge_count: int


# -----------------------------------------------------------------------------
# Intervention design
# -----------------------------------------------------------------------------
class InterventionDesignRequest(BaseRequest):
    """Request model for intervention design."""
    variable_names: Optional[List[str]] = None
    batch_size: Optional[int] = None
    budget: Optional[float] = None
    optimization_target: str = Field(default="maximize_effect", description="Optimization target")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Additional constraints")
    risk_tolerance: float = Field(default=0.1, ge=0, le=1, description="Risk tolerance level")
    
    @field_validator('variable_names')
    @classmethod
    def validate_variable_names(cls, v):
        if not v or len(v) == 0:
            raise ValueError('variable_names must contain at least one variable')
        return v
    
    @field_validator('optimization_target')
    @classmethod
    def validate_optimization_target(cls, v):
        allowed_targets = ['maximize_effect', 'minimize_cost', 'maximize_efficiency', 'balanced']
        if v not in allowed_targets:
            raise ValueError(f'optimization_target must be one of {allowed_targets}')
        return v

class InterventionDesignResponse(BaseResponse):
    """Response model for intervention design."""
    interventions_designed: int
    interventions: List[Dict[str, Any]]
    expected_effects: Dict[str, Dict[str, float]]
    total_estimated_cost: float
    design_confidence: float
    batch_size: int
    within_budget: bool
    risk_assessment: Dict[str, float]


# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------
class ExplainabilityRequest(BaseRequest):
    """Request model for explainability analysis."""
    model_path: Optional[str] = None
    data_path: Optional[str] = None
    analysis_types: Optional[List[str]] = None
    
    @field_validator('analysis_types')
    @classmethod
    def validate_analysis_types(cls, v):
        allowed_types = ['attention', 'grad_cam', 'shap', 'lime', 'integrated_gradients']
        if v and any(t not in allowed_types for t in v):
            raise ValueError(f'analysis_types must be one of {allowed_types}')
        return v

class ExplainabilityResponse(BaseResponse):
    """Response model for explainability analysis."""
    job_id: str
    explanations: Dict[str, Any]
    importance_scores: Dict[str, float]
    method_used: str
    sample_count: int
    confidence_scores: Dict[str, float]


# -----------------------------------------------------------------------------
# Data upload
# -----------------------------------------------------------------------------
class DataUploadRequest(BaseRequest):
    """Request model for data upload."""
    data_type: str = Field(..., description="Type of data being uploaded", min_length=1)
    file_format: str = Field(..., description="Format of the file", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    @field_validator('data_type')
    @classmethod
    def validate_data_type(cls, v):
        allowed_types = ['genomics', 'imaging', 'chemical', 'multimodal']
        if v not in allowed_types:
            raise ValueError(f'data_type must be one of {allowed_types}')
        return v
    
    @field_validator('file_format')
    @classmethod
    def validate_file_format(cls, v):
        allowed_formats = ['csv', 'tsv', 'h5', 'xlsx', 'png', 'jpg', 'jpeg', 'tiff', 'tif', 'sdf', 'mol']
        if v not in allowed_formats:
            raise ValueError(f'file_format must be one of {allowed_formats}')
        return v

class DataUploadResponse(BaseResponse):
    """Response model for data upload."""
    filename: str = ""
    file_path: str = ""
    file_size: int = 0
    format: str = ""
    status: str = ""
    message: str = ""


# -----------------------------------------------------------------------------
# Model catalogue
# -----------------------------------------------------------------------------
class ModelInfo(BaseModel):
    """Model information model."""
    name: str = ""
    description: str = ""
    version: str = Field(..., description="Model version")
    input_types: List[str] = Field(..., description="Supported input types")
    parameters: Dict[str, Any] = Field(..., description="Model parameters")
    performance_metrics: Optional[Dict[str, float]] = Field(default=None, description="Performance metrics")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class ModelListResponse(BaseResponse):
    """Response model for model listing."""
    models: List[ModelInfo]
    total_count: int
    available_types: List[str]


# -----------------------------------------------------------------------------
# Experiment management
# -----------------------------------------------------------------------------
class ExperimentInfo(BaseModel):
    """Experiment information model."""
    id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    data_sources: List[str] = Field(..., description="Data sources used")
    status: str = Field(..., description="Experiment status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    config: Dict[str, Any] = Field(..., description="Experiment configuration")
    results: Optional[Dict[str, Any]] = Field(default=None, description="Experiment results")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        allowed_statuses = ['active', 'completed', 'failed', 'paused', 'cancelled']
        if v not in allowed_statuses:
            raise ValueError(f'status must be one of {allowed_statuses}')
        return v

class ExperimentListResponse(BaseResponse):
    """Response model for experiment listing."""
    experiments: List[ExperimentInfo]
    total_count: int
    active_count: int
    completed_count: int


# -----------------------------------------------------------------------------
# Dataset management
# -----------------------------------------------------------------------------
class DatasetInfo(BaseModel):
    """Dataset information model."""
    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    data_type: str = Field(..., description="Data type")
    format: str = Field(..., description="File format")
    size: int = Field(..., description="Dataset size in bytes")
    source: str = Field(..., description="Data source")
    last_updated: datetime = Field(..., description="Last update timestamp")
    version: str = Field(default="1.0", description="Dataset version")
    license: Optional[str] = Field(default=None, description="Dataset license")
    citation: Optional[str] = Field(default=None, description="Citation information")

class DatasetListResponse(BaseResponse):
    """Response model for dataset listing."""
    datasets: List[DatasetInfo]
    total_count: int
    total_size: int
    available_types: List[str]


# -----------------------------------------------------------------------------
# Configuration validation
# -----------------------------------------------------------------------------
class ConfigValidationRequest(BaseRequest):
    """Request model for configuration validation."""
    config: Dict[str, Any] = Field(..., description="Configuration to validate")

class ConfigValidationResponse(BaseResponse):
    """Response model for configuration validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


# -----------------------------------------------------------------------------
# System information
# -----------------------------------------------------------------------------
class SystemInfo(BaseModel):
    """System information model."""
    python_version: str = Field(..., description="Python version")
    pytorch_version: str = Field(..., description="PyTorch version")
    platform: str = Field(..., description="Platform information")
    cpu_count: int = Field(..., description="Number of CPU cores")
    memory_available: int = Field(..., description="Available memory in bytes")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_info: Optional[Dict[str, Any]] = Field(default=None, description="GPU information")
    dependencies: Dict[str, bool] = Field(..., description="Dependency availability")
    system_load: Optional[Dict[str, float]] = Field(default=None, description="System load information")

class SystemInfoResponse(BaseResponse):
    """Response model for system information."""
    system_info: SystemInfo
    uptime: float
    api_version: str


# -----------------------------------------------------------------------------
# File operations
# -----------------------------------------------------------------------------
class FileUploadResponse(BaseResponse):
    """Response model for file upload."""
    file_id: str
    filename: str
    data_type: str
    file_size: int
    file_path: str
    upload_timestamp: datetime
    status: str
    checksum: Optional[str] = None


# -----------------------------------------------------------------------------
# Health and status
# -----------------------------------------------------------------------------
class HealthCheckResponse(BaseResponse):
    """Response model for health check."""
    service: str
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime: float
    memory_usage: Dict[str, Any]
    active_connections: int

class RootResponse(BaseResponse):
    """Response model for root endpoint."""
    service: str
    version: str
    status: str
    documentation: str
    health_check: str
    api_endpoints: List[str]


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    error_code: Optional[str] = None

class ValidationResult(BaseModel):
    """Validation result model."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Performance metrics
# -----------------------------------------------------------------------------
class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)
    f1_score: Optional[float] = Field(None, ge=0, le=1)
    auc: Optional[float] = Field(None, ge=0, le=1)
    mse: Optional[float] = Field(None, ge=0)
    mae: Optional[float] = Field(None, ge=0)
    custom_metrics: Optional[Dict[str, float]] = None


# -----------------------------------------------------------------------------
# Job management
# -----------------------------------------------------------------------------
class JobListResponse(BaseResponse):
    """Response model for job listing."""
    jobs: List[JobStatus]
    total_count: int
    active_count: int
    completed_count: int
    failed_count: int

class ExperimentConfig(BaseModel):
    """Experiment configuration model."""
    id: str = ""
    name: str = ""
    status: str = ""
    created_at: str = ""
    config: Dict[str, Any] = Field(default_factory=dict, description="Experiment configuration")
    results: Optional[Dict[str, Any]] = Field(default=None, description="Experiment results")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        allowed_statuses = ['active', 'completed', 'failed', 'paused', 'cancelled']
        if v not in allowed_statuses:
            raise ValueError(f'status must be one of {allowed_statuses}')
        return v

class ResultsSummary(BaseModel):
    """Results summary model."""
    job_id: str = ""
    status: str = ""
    total_compounds: int = 0
    active_compounds: int = 0
    completion_time: str = ""
    summary_stats: Optional[Dict[str, Any]] = Field(default=None, description="Summary statistics") 