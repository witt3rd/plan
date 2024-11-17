"""Capability metadata models

Improvements:
- Added validation rules for input/output schemas
- Enhanced performance metrics with statistical measures
- Added dependency tracking with version constraints
- Included capability tags for better organization
- Added configuration options specific to each capability type
- Included audit trail for changes and optimizations
- Added resource usage tracking
- Enhanced versioning with semantic version parsing
- Added capability lifecycle status tracking
- Included documentation and example usage
"""

import json
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import JsonSchemaValue


class CapabilityType(str, Enum):
    """Types of capabilities"""

    TOOL = "tool"
    INSTRUCTION = "instruction"
    PLAN = "plan"


class ResourceUsage(BaseModel):
    """Resource usage metrics"""

    avg_memory_mb: float = Field(default=0.0)
    peak_memory_mb: float = Field(default=0.0)
    avg_cpu_percent: float = Field(default=0.0)
    peak_cpu_percent: float = Field(default=0.0)
    total_execution_time_sec: float = Field(default=0.0)
    api_calls: int = Field(default=0)


class PerformanceMetrics(BaseModel):
    """Detailed performance tracking"""

    avg_execution_time: float = Field(default=0.0)
    min_execution_time: float = Field(default=float("inf"))
    max_execution_time: float = Field(default=0.0)
    std_dev_execution_time: float = Field(default=0.0)
    p95_execution_time: float = Field(default=0.0)
    error_rate: float = Field(default=0.0)
    timeout_rate: float = Field(default=0.0)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)


class DependencyRequirement(BaseModel):
    """Capability dependency with version constraints"""

    capability_name: str
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    exact_version: Optional[str] = None


class AuditEntry(BaseModel):
    """Record of changes to the capability"""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    change_type: str
    description: str
    previous_version: Optional[str] = None
    new_version: str
    changed_by: str


class LifecycleStatus(str, Enum):
    """Capability lifecycle states"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class CapabilityMetadata(BaseModel):
    """Metadata about a capability"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic information
    name: str
    type: CapabilityType
    version: str = Field(default="1.0.0")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    description: str

    # Schema information
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    validation_rules: List[str] = Field(default_factory=list)
    openai_schema: Optional[Dict[str, Any]] = None  # Store OpenAI function schema

    # Performance and usage
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    usage_count: int = Field(default=0)
    success_rate: float = Field(default=0.0)
    last_used: Optional[datetime] = None

    # Dependencies and relationships
    dependencies: List[DependencyRequirement] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)

    # Organization and documentation
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    owner: Optional[str] = None
    documentation_url: Optional[str] = None
    example_usage: Optional[str] = None

    # Lifecycle and history
    status: LifecycleStatus = Field(default=LifecycleStatus.DEVELOPMENT)
    audit_trail: List[AuditEntry] = Field(default_factory=list)

    # Type-specific configuration
    config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    def validate_version(cls, v: str) -> str:
        """Validate version string format"""
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError("Version must be in format X.Y.Z")
        return v

    def update_metrics(self, execution_time: float, success: bool) -> None:
        """Update performance metrics with new execution data"""
        self.usage_count += 1
        self.last_used = datetime.now(UTC)

        metrics = self.performance_metrics
        metrics.avg_execution_time = (
            metrics.avg_execution_time * (self.usage_count - 1) + execution_time
        ) / self.usage_count

        metrics.min_execution_time = min(metrics.min_execution_time, execution_time)
        metrics.max_execution_time = max(metrics.max_execution_time, execution_time)

        # Update success rate
        total_success = self.success_rate * (self.usage_count - 1)
        total_success += 1 if success else 0
        self.success_rate = total_success / self.usage_count

    def add_audit_entry(
        self,
        change_type: str,
        description: str,
        changed_by: str,
        new_version: Optional[str] = None,
    ) -> None:
        """Add an audit entry for capability changes"""
        entry = AuditEntry(
            timestamp=datetime.now(UTC),
            change_type=change_type,
            description=description,
            previous_version=self.version,
            new_version=new_version or self.version,
            changed_by=changed_by,
        )
        self.audit_trail.append(entry)
        if new_version:
            self.version = new_version

    def is_compatible_with(self, other: "CapabilityMetadata") -> bool:
        """Check if this capability is compatible with another"""
        # Check input/output schema compatibility
        if not set(self.input_schema.keys()).issubset(other.input_schema.keys()):
            return False
        if not set(self.output_schema.keys()).issubset(other.output_schema.keys()):
            return False
        return True

    @property
    def as_openai_tool(self) -> Optional[Dict[str, Any]]:
        """Get OpenAI tool definition if available

        Returns:
            Dictionary containing OpenAI tool definition if this is a tool capability,
            None otherwise
        """
        if self.type == CapabilityType.TOOL and self.openai_schema:
            return {"type": "function", "function": self.openai_schema}
        return None

    def model_dump_json(self, **kwargs) -> str:
        """Custom JSON serialization"""
        data = self.model_dump(**kwargs)

        # Convert datetime objects to ISO format strings
        def convert_datetime(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        # Recursively convert datetime objects
        def process_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = process_dict(v)
                elif isinstance(v, list):
                    result[k] = [convert_datetime(item) for item in v]
                else:
                    result[k] = convert_datetime(v)
            return result

        processed_data = process_dict(data)
        return json.dumps(processed_data)

    @classmethod
    def model_json_schema(cls, **kwargs) -> JsonSchemaValue:
        """Custom JSON schema generation"""
        schema = super().model_json_schema(**kwargs)
        # Add any custom schema modifications here if needed
        return schema

    def validate_schema(
        self, input_model: Type[BaseModel], output_model: Type[BaseModel]
    ) -> None:
        """Validate input/output models match schema"""
        input_fields = input_model.model_fields
        output_fields = output_model.model_fields

        # Validate input schema
        for name, schema in self.input_schema.items():
            if name not in input_fields:
                raise ValueError(f"Missing input field: {name}")
            # Add type validation

        # Validate output schema
        for name, schema in self.output_schema.items():
            if name not in output_fields:
                raise ValueError(f"Missing output field: {name}")
            # Add type validation
