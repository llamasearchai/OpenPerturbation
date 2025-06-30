"""
Pydantic models for OpenPerturbation API.

Author: Nik Jois
Email: nikjois@llamasearch.ai
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
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
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
    service: str = "OpenPerturbation API"
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str


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
class CausalDiscoveryRequest(BaseModel):
    data: List[List[float]]
    method: str = "correlation"
    alpha: float = 0.05
    variable_names: Optional[List[str]] = None
    max_lag: int = 0

class CausalDiscoveryResponse(BaseModel):
    adjacency_matrix: List[List[float]]
    method: str
    variable_names: List[str]
    confidence_scores: Optional[List[List[float]]] = None
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Intervention design
# -----------------------------------------------------------------------------
class InterventionDesignRequest(BaseModel):
    variable_names: List[str]
    batch_size: int = 10
    budget: float = 1000.0
    constraints: Optional[Dict[str, Any]] = None

class InterventionDesignResponse(BaseModel):
    recommended_interventions: List[Dict[str, Any]]
    intervention_ranking: List[Dict[str, float]]
    expected_outcomes: Dict[str, Any]
    execution_time: float


# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------
class ExplainabilityRequest(BaseModel):
    model_path: str
    data_path: str
    analysis_types: List[str] = Field(default_factory=lambda: ["attention"])
    output_dir: Optional[str] = None

class ExplainabilityResponse(BaseModel):
    attention_analysis: Optional[Dict[str, Any]] = None
    concept_analysis: Optional[Dict[str, Any]] = None
    pathway_analysis: Optional[Dict[str, Any]] = None
    execution_time: float
    output_paths: List[str] = Field(default_factory=list)


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

class DataUploadResponse(BaseModel):
    filename: str
    file_size: int
    upload_time: str
    data_type: str
    summary: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Model catalogue
# -----------------------------------------------------------------------------
class ModelInfo(BaseModel):
    name: str
    description: str
    version: str = "1.0.0"
    input_types: List[str] = Field(default_factory=lambda: ["multimodal"])
    output_types: List[str] = Field(default_factory=lambda: ["predictions"])
    parameters: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    last_updated: Optional[str] = None

class ModelListResponse(BaseResponse):
    """Response model for model listing."""
    models: List[ModelInfo]
    total_count: int
    available_types: List[str]


# -----------------------------------------------------------------------------
# Experiment management
# -----------------------------------------------------------------------------
class ExperimentInfo(BaseModel):
    experiment_id: str
    name: str
    description: str
    status: str
    created_at: str
    config: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None

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
    name: str
    description: str
    size: int
    format: str
    modalities: List[str]
    created_at: str
    tags: List[str] = Field(default_factory=list)

class DatasetListResponse(BaseResponse):
    """Response model for dataset listing."""
    datasets: List[DatasetInfo]
    total_count: int
    total_size: int
    available_types: List[str]


# -----------------------------------------------------------------------------
# Configuration validation
# -----------------------------------------------------------------------------
class ConfigValidationRequest(BaseModel):
    config: Dict[str, Any]

class ConfigValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# System information
# -----------------------------------------------------------------------------
class SystemInfo(BaseModel):
    platform: str
    python_version: str
    cpu_count: int
    memory_gb: float
    gpu_available: bool = False
    dependencies: Dict[str, str] = Field(default_factory=dict)
    timestamp: str

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
    error: str
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None

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


# -----------------------------------------------------------------------------
# Job management
# -----------------------------------------------------------------------------
class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Analysis models
# -----------------------------------------------------------------------------
class AnalysisModels(BaseModel):
    causal_discovery: List[str] = Field(default_factory=lambda: ["pc", "ges", "lingam", "correlation"])
    explainability: List[str] = Field(default_factory=lambda: ["attention", "concept", "pathway"])
    prediction: List[str] = Field(default_factory=lambda: ["multimodal_fusion", "causal_vae"]) 