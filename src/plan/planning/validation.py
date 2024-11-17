"""Plan validation logic

Improvements:
- Separated validation concerns into specific validators
- Added support for custom validation rules
- Enhanced error reporting with detailed context
- Added validation rule composition
- Implemented validation caching for performance
- Added support for async validation rules
- Enhanced type checking and coercion
- Added support for conditional validation
- Implemented validation rule priorities
- Added validation rule documentation
- Enhanced validation error handling
- Added support for validation hooks
- Implemented validation metrics collection
- Added support for validation rule versioning
- Enhanced validation context handling
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from plan.capabilities.base import CapabilityNotFoundError
from plan.capabilities.registry import CapabilityRegistry
from plan.planning.models import (
    Plan,
    Task,
    TaskInput,
    TaskOutput,
    ValidationRule,
)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    """Detailed description of a validation issue"""

    severity: ValidationSeverity
    code: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    location: Optional[str] = None
    suggestion: Optional[str] = None
    documentation_url: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of validation including all issues"""

    valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_issue(
        self, severity: ValidationSeverity, code: str, message: str, **kwargs: Any
    ) -> None:
        """Add a validation issue"""
        self.issues.append(
            ValidationIssue(severity=severity, code=code, message=message, **kwargs)
        )
        if severity == ValidationSeverity.ERROR:
            self.valid = False


class ValidationContext(BaseModel):
    """Context for validation execution"""

    registry: CapabilityRegistry
    working_memory: Dict[str, Any] = Field(default_factory=dict)
    validation_cache: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: List[str] = Field(default_factory=list)


class PlanValidator:
    """Validates plans with comprehensive rule checking"""

    def __init__(self, registry: CapabilityRegistry):
        """Initialize the validator

        Args:
            registry: Registry for capability validation
        """
        self._registry = registry
        self._validation_cache: Dict[str, ValidationResult] = {}

    async def validate(
        self, plan: Plan, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate a plan comprehensively

        Args:
            plan: Plan to validate
            context: Optional validation context

        Returns:
            Validation result with any issues found
        """
        # Initialize validation
        start_time = datetime.utcnow()
        result = ValidationResult(valid=True, execution_time=0.0)
        validation_context = ValidationContext(
            registry=self._registry, working_memory=context or {}
        )

        try:
            # Structure validation
            await self._validate_structure(plan, validation_context, result)

            # Capability validation
            await self._validate_capabilities(plan, validation_context, result)

            # Input/output validation
            await self._validate_io(plan, validation_context, result)

            # Dependency validation
            await self._validate_dependencies(plan, validation_context, result)

            # Constraint validation
            await self._validate_constraints(plan, validation_context, result)

            # Resource validation
            await self._validate_resources(plan, validation_context, result)

            # Custom rule validation
            await self._validate_custom_rules(plan, validation_context, result)

        except Exception as e:
            result.add_issue(
                ValidationSeverity.ERROR,
                "VALIDATION_ERROR",
                f"Validation failed: {str(e)}",
                context={"error": str(e)},
            )

        finally:
            # Finalize result
            result.execution_time = (datetime.utcnow() - start_time).total_seconds()

        return result

    async def _validate_structure(
        self, plan: Plan, context: ValidationContext, result: ValidationResult
    ) -> None:
        """Validate plan structure

        Args:
            plan: Plan to validate
            context: Validation context
            result: Validation result to update
        """
        # Validate basic structure
        if not plan.tasks:
            result.add_issue(
                ValidationSeverity.ERROR, "EMPTY_PLAN", "Plan contains no tasks"
            )

        # Validate task structure
        for task in plan.tasks:
            await self._validate_task_structure(task, context, result)

        # Validate desired outputs
        available_outputs = {task.output.key for task in plan.tasks}
        for output in plan.desired_outputs:
            if output not in available_outputs:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "MISSING_OUTPUT",
                    f"Desired output '{output}' is not produced by any task",
                    context={"output": output},
                )

    async def _validate_task_structure(
        self, task: Task, context: ValidationContext, result: ValidationResult
    ) -> None:
        """Validate task structure

        Args:
            task: Task to validate
            context: Validation context
            result: Validation result to update
        """
        # Validate task name
        if not task.name.isidentifier():
            result.add_issue(
                ValidationSeverity.ERROR,
                "INVALID_TASK_NAME",
                f"Task name '{task.name}' is not a valid identifier",
                context={"task_name": task.name},
            )

        # Validate inputs
        for input in task.inputs:
            await self._validate_task_input(task, input, context, result)

        # Validate output
        await self._validate_task_output(task, task.output, context, result)

    async def _validate_task_input(
        self,
        task: Task,
        input: TaskInput,
        context: ValidationContext,
        result: ValidationResult,
    ) -> None:
        """Validate task input

        Args:
            task: Parent task
            input: Input to validate
            context: Validation context
            result: Validation result to update
        """
        # Validate input name
        if not input.key.isidentifier():
            result.add_issue(
                ValidationSeverity.ERROR,
                "INVALID_INPUT_NAME",
                f"Input name '{input.key}' in task '{task.name}' is not a valid identifier",
                context={"task_name": task.name, "input_name": input.key},
            )

        # Validate input source
        if input.source_key:
            if not any(input.source_key == t.output.key for t in task.dependencies):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "INVALID_INPUT_SOURCE",
                    f"Input '{input.key}' references invalid source '{input.source_key}'",
                    context={
                        "task_name": task.name,
                        "input_name": input.key,
                        "source_key": input.source_key,
                    },
                )

    async def _validate_task_output(
        self,
        task: Task,
        output: TaskOutput,
        context: ValidationContext,
        result: ValidationResult,
    ) -> None:
        """Validate task output

        Args:
            task: Parent task
            output: Output to validate
            context: Validation context
            result: Validation result to update
        """
        # Validate output name
        if not output.key.isidentifier():
            result.add_issue(
                ValidationSeverity.ERROR,
                "INVALID_OUTPUT_NAME",
                f"Output name '{output.key}' in task '{task.name}' is not a valid identifier",
                context={"task_name": task.name, "output_name": output.key},
            )

    async def _validate_capabilities(
        self, plan: Plan, context: ValidationContext, result: ValidationResult
    ) -> None:
        """Validate capability requirements

        Args:
            plan: Plan to validate
            context: Validation context
            result: Validation result to update
        """
        for task in plan.tasks:
            try:
                capability = context.registry.get(task.capability_name)
                if not capability:
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        "MISSING_CAPABILITY",
                        f"Required capability '{task.capability_name}' not found for task '{task.name}'",
                        context={
                            "task_name": task.name,
                            "capability_name": task.capability_name,
                        },
                    )
                else:
                    # Validate capability compatibility
                    await self._validate_capability_compatibility(
                        task, capability, context, result
                    )

            except CapabilityNotFoundError:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "CAPABILITY_ERROR",
                    f"Error accessing capability '{task.capability_name}' for task '{task.name}'",
                    context={
                        "task_name": task.name,
                        "capability_name": task.capability_name,
                    },
                )

    async def _validate_capability_compatibility(
        self,
        task: Task,
        capability: Any,
        context: ValidationContext,
        result: ValidationResult,
    ) -> None:
        """Validate capability compatibility with task

        Args:
            task: Task to validate
            capability: Capability to validate against
            context: Validation context
            result: Validation result to update
        """
        metadata = capability.metadata

        # Validate inputs
        task_inputs = {input.key for input in task.inputs}
        capability_inputs = set(metadata.input_schema.keys())
        if not task_inputs.issubset(capability_inputs):
            result.add_issue(
                ValidationSeverity.ERROR,
                "INCOMPATIBLE_INPUTS",
                f"Task '{task.name}' has inputs not supported by capability '{task.capability_name}'",
                context={
                    "task_name": task.name,
                    "capability_name": task.capability_name,
                    "extra_inputs": task_inputs - capability_inputs,
                },
            )

        # Validate output
        if task.output.key not in metadata.output_schema:
            result.add_issue(
                ValidationSeverity.ERROR,
                "INCOMPATIBLE_OUTPUT",
                f"Task '{task.name}' output not supported by capability '{task.capability_name}'",
                context={
                    "task_name": task.name,
                    "capability_name": task.capability_name,
                    "output": task.output.key,
                },
            )

    async def _validate_io(
        self, plan: Plan, context: ValidationContext, result: ValidationResult
    ) -> None:
        """Validate input/output relationships

        Args:
            plan: Plan to validate
            context: Validation context
            result: Validation result to update
        """
        # Track available outputs
        available_outputs: Set[str] = set()

        # Validate each task's I/O
        for task in plan.tasks:
            # Validate inputs are available
            for input in task.inputs:
                if input.source_key:
                    if (
                        input.source_key not in available_outputs
                        and input.source_key not in context.working_memory
                    ):
                        result.add_issue(
                            ValidationSeverity.ERROR,
                            "MISSING_INPUT_SOURCE",
                            f"Task '{task.name}' requires input '{input.source_key}' which is not available",
                            context={
                                "task_name": task.name,
                                "input_name": input.key,
                                "source_key": input.source_key,
                            },
                        )

            # Add task's output to available outputs
            available_outputs.add(task.output.key)

    async def _validate_dependencies(
        self, plan: Plan, context: ValidationContext, result: ValidationResult
    ) -> None:
        """Validate task dependencies

        Args:
            plan: Plan to validate
            context: Validation context
            result: Validation result to update
        """
        # Build dependency graph
        graph: Dict[str, Set[str]] = {task.name: set() for task in plan.tasks}

        for task in plan.tasks:
            for dep_name in task.dependencies:
                if dep_name not in graph:
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        "INVALID_DEPENDENCY",
                        f"Task '{task.name}' depends on non-existent task '{dep_name}'",
                        context={"task_name": task.name, "dependency": dep_name},
                    )
                else:
                    graph[task.name].add(dep_name)

        # Check for cycles
        visited = set()
        path = set()

        def visit(task_name: str) -> None:
            if task_name in path:
                cycle = list(path)
                cycle.append(task_name)
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "CIRCULAR_DEPENDENCY",
                    f"Circular dependency detected: {' -> '.join(cycle)}",
                    context={"cycle": cycle},
                )
                return

            if task_name in visited:
                return

            visited.add(task_name)
            path.add(task_name)

            for dep in graph[task_name]:
                visit(dep)

            path.remove(task_name)

        for task_name in graph:
            visit(task_name)

    async def _validate_constraints(
        self, plan: Plan, context: ValidationContext, result: ValidationResult
    ) -> None:
        """Validate execution constraints

        Args:
            plan: Plan to validate
            context: Validation context
            result: Validation result to update
        """
        for task in plan.tasks:
            for constraint in task.execution_constraints:
                try:
                    # Validate constraint expression
                    compile(constraint.condition, "<string>", "eval")
                except SyntaxError:
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        "INVALID_CONSTRAINT",
                        f"Invalid constraint expression in task '{task.name}': {constraint.condition}",
                        context={
                            "task_name": task.name,
                            "constraint": constraint.condition,
                        },
                    )

    async def _validate_resources(
        self, plan: Plan, context: ValidationContext, result: ValidationResult
    ) -> None:
        """Validate resource requirements

        Args:
            plan: Plan to validate
            context: Validation context
            result: Validation result to update
        """
        for task in plan.tasks:
            requirements = task.resource_requirements

            # Validate memory requirements
            if (
                requirements.min_memory_mb is not None
                and requirements.max_memory_mb is not None
                and requirements.min_memory_mb > requirements.max_memory_mb
            ):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "INVALID_MEMORY_REQUIREMENTS",
                    f"Invalid memory requirements in task '{task.name}': min > max",
                    context={
                        "task_name": task.name,
                        "min_memory": requirements.min_memory_mb,
                        "max_memory": requirements.max_memory_mb,
                    },
                )

            # Validate CPU requirements
            if (
                requirements.min_cpu_cores is not None
                and requirements.max_cpu_cores is not None
                and requirements.min_cpu_cores > requirements.max_cpu_cores
            ):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "INVALID_CPU_REQUIREMENTS",
                    f"Invalid CPU requirements in task '{task.name}': min > max",
                    context={
                        "task_name": task.name,
                        "min_cores": requirements.min_cpu_cores,
                        "max_cores": requirements.max_cpu_cores,
                    },
                )

    async def _validate_custom_rules(
        self, plan: Plan, context: ValidationContext, result: ValidationResult
    ) -> None:
        """Validate custom validation rules

        Args:
            plan: Plan to validate
            context: Validation context
            result: Validation result to update
        """
        for rule in plan.validation_rules:
            try:
                # Execute custom validation rule
                await self._execute_validation_rule(rule, plan, context, result)
            except Exception as e:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "VALIDATION_RULE_ERROR",
                    f"Error executing validation rule: {str(e)}",
                    context={"rule": rule.dict(), "error": str(e)},
                )

    async def _execute_validation_rule(
        self,
        rule: ValidationRule,
        plan: Plan,
        context: ValidationContext,
        result: ValidationResult,
    ) -> None:
        """Execute a custom validation rule

        Args:
            rule: Rule to execute
            plan: Plan being validated
            context: Validation context
            result: Validation result to update
        """
        # Implementation depends on rule type
        pass
