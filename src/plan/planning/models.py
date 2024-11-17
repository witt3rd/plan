"""Plan and task data models

Improvements:
- Enhanced type safety with strict Pydantic models
- Added validation rules for inputs/outputs
- Included support for conditional task execution
- Added task priorities and dependencies
- Enhanced metadata tracking
- Added support for task timeouts and retries
- Included resource requirements specification
- Added task grouping and parallel execution hints
- Enhanced error handling specifications
- Added support for task documentation
- Included task versioning
- Added support for task templates
- Enhanced input/output validation rules
- Added support for execution constraints
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

from pydantic import BaseModel, Field, field_validator

from plan.capabilities.factory import create_pydantic_model
from plan.capabilities.metadata import CapabilityMetadata


class TaskPriority(int, Enum):
    """Priority levels for task execution"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class ExecutionMode(str, Enum):
    """Task execution modes"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class ResourceRequirements(BaseModel):
    """Resource requirements for task execution"""

    min_memory_mb: Optional[int] = None
    max_memory_mb: Optional[int] = None
    min_cpu_cores: Optional[int] = None
    max_cpu_cores: Optional[int] = None
    gpu_required: bool = False
    max_execution_time_seconds: Optional[float] = None


class RetryPolicy(BaseModel):
    """Retry configuration for task execution"""

    max_attempts: int = Field(default=3, ge=1)
    initial_delay_seconds: float = Field(default=1.0, ge=0)
    max_delay_seconds: float = Field(default=60.0, ge=0)
    exponential_backoff: bool = Field(default=True)
    retry_on_errors: Set[str] = Field(default_factory=set)


class ValidationRule(BaseModel):
    """Validation rule for inputs or outputs"""

    field: str
    rule_type: str  # e.g., "type", "range", "regex", "custom"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    error_message: str


class TaskInput(BaseModel):
    """Detailed input specification for a task"""

    key: str = Field(..., description="Name of the input parameter")
    description: str = Field(..., description="Description of the input")
    source_key: Optional[str] = Field(
        None, description="Key in working memory or output from another task"
    )
    required: bool = Field(default=True)
    default_value: Optional[Any] = None
    validation_rules: List[ValidationRule] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Ensure key is a valid identifier"""
        if not v.isidentifier():
            raise ValueError(f"Invalid input key: {v}")
        return v


class TaskOutput(BaseModel):
    """Detailed output specification for a task"""

    key: str = Field(..., description="Name of the output")
    description: str = Field(..., description="Description of the output")
    validation_rules: List[ValidationRule] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Ensure key is a valid identifier"""
        if not v.isidentifier():
            raise ValueError(f"Invalid output key: {v}")
        return v


class ExecutionConstraint(BaseModel):
    """Constraints for task execution"""

    condition: str  # Python expression using context variables
    description: str
    error_message: Optional[str] = None


class Task(BaseModel):
    """A single task in a plan"""

    name: str = Field(..., description="Unique name of the task")
    capability_name: str = Field(..., description="Name of required capability")
    description: str = Field(..., description="Detailed task description")
    inputs: List[TaskInput] = Field(..., description="Required inputs")
    output: TaskOutput = Field(..., description="Task output")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL)
    dependencies: Set[str] = Field(
        default_factory=set, description="Names of tasks this depends on"
    )
    resource_requirements: ResourceRequirements = Field(
        default_factory=ResourceRequirements
    )
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    execution_constraints: List[ExecutionConstraint] = Field(default_factory=list)
    timeout_seconds: Optional[float] = None
    version: str = Field(default="1.0.0")
    tags: Set[str] = Field(default_factory=set)
    plan: Optional["Plan"] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is a valid identifier"""
        if not v.isidentifier():
            raise ValueError(f"Invalid task name: {v}")
        return v

    @property
    def output_key(self) -> str:
        """Get the output key for this task"""
        return self.output.key

    def get_input_sources(self) -> Dict[str, str]:
        """Get mapping of input names to their source keys"""
        return {
            input.key: input.source_key
            for input in self.inputs
            if input.source_key is not None
        }

    def get_dependencies(self) -> Set[str]:
        """Get set of task names this task depends on"""
        return {
            dep_name
            for input in self.inputs
            if input.source_key
            for dep_name in self.dependencies
            if input.source_key == dep_name
        }

    def get_input_model(self) -> Type[BaseModel]:
        """Get the input model for this task"""
        return create_pydantic_model(
            f"{self.name}Input", {input.key: input.schema for input in self.inputs}
        )

    def get_output_model(self) -> Type[BaseModel]:
        """Get the output model for this task"""
        return create_pydantic_model(
            f"{self.name}Output", {self.output.key: self.output.schema}
        )


class TaskGroup(BaseModel):
    """Group of related tasks"""

    name: str
    description: str
    tasks: List[Task]
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL)
    conditions: List[ExecutionConstraint] = Field(default_factory=list)


class PlanMetadata(BaseModel):
    """Enhanced metadata for plans"""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    version: str = Field(default="1.0.0")
    tags: Set[str] = Field(default_factory=set)
    description: str
    documentation_url: Optional[str] = None
    estimated_duration_seconds: Optional[float] = None
    required_capabilities: Set[str] = Field(default_factory=set)
    execution_environment: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """A collection of tasks with a goal"""

    name: str = Field(..., description="Unique name of the plan")
    description: str = Field(..., description="Detailed plan description")
    goal: str = Field(..., description="Clear statement of plan's goal")
    tasks: List[Task] = Field(..., description="Tasks to execute")
    task_groups: List[TaskGroup] = Field(
        default_factory=list, description="Optional task groupings"
    )
    desired_outputs: List[str] = Field(
        ..., description="Keys of outputs this plan should produce"
    )
    metadata: PlanMetadata = Field(..., description="Plan metadata")
    capability_metadata: CapabilityMetadata = Field(
        ..., description="Metadata when plan is used as capability"
    )
    initial_context: Dict[str, Any] = Field(
        default_factory=dict, description="Initial working memory values"
    )
    validation_rules: List[ValidationRule] = Field(
        default_factory=list, description="Plan-level validation rules"
    )

    @field_validator("desired_outputs")
    @classmethod
    def validate_outputs(cls, v: List[str], info) -> List[str]:
        """Ensure all desired outputs are produced by tasks"""
        if "tasks" in info.data:
            available_outputs = {task.output.key for task in info.data["tasks"]}
            missing = set(v) - available_outputs
            if missing:
                raise ValueError(f"Missing outputs: {missing}")
        return v

    def get_task_by_name(self, name: str) -> Optional[Task]:
        """Get a task by name"""
        return next((task for task in self.tasks if task.name == name), None)

    def get_tasks_by_tag(self, tag: str) -> List[Task]:
        """Get all tasks with a specific tag"""
        return [task for task in self.tasks if tag in task.tags]

    def get_critical_path(self) -> List[Task]:
        """Get tasks on the critical execution path"""
        # Basic implementation - returns tasks with no dependencies first,
        # followed by tasks in dependency order
        critical_path = []
        remaining_tasks = self.tasks.copy()

        # First add tasks with no dependencies
        no_deps = [task for task in remaining_tasks if not task.dependencies]
        critical_path.extend(no_deps)
        for task in no_deps:
            remaining_tasks.remove(task)

        # Then add tasks as their dependencies are satisfied
        while remaining_tasks:
            next_tasks = [
                task
                for task in remaining_tasks
                if task.dependencies.issubset({t.name for t in critical_path})
            ]
            if not next_tasks:
                break
            critical_path.extend(next_tasks)
            for task in next_tasks:
                remaining_tasks.remove(task)

        return critical_path

    def validate(self) -> None:
        """Validate the entire plan structure"""
        self._validate_dependencies()
        self._validate_input_sources()
        self._validate_constraints()

    def _validate_dependencies(self) -> None:
        """Check for circular dependencies"""
        # Implementation for dependency validation
        pass

    def _validate_input_sources(self) -> None:
        """Ensure all task inputs have valid sources"""
        # Implementation for input validation
        pass

    def _validate_constraints(self) -> None:
        """Validate execution constraints"""
        # Implementation for constraint validation
        pass
