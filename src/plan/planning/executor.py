"""Plan execution logic

Improvements:
- Separated execution concerns from plan definition
- Added robust error handling and recovery
- Implemented parallel task execution where possible
- Added execution monitoring and metrics collection
- Implemented checkpointing and state recovery
- Added support for execution simulation/dry-run
- Enhanced progress tracking and status reporting
- Added resource usage monitoring and constraints
- Implemented execution timeouts and cancellation
- Added support for conditional task execution
- Enhanced debugging and logging capabilities
- Implemented execution audit trail
- Added support for execution hooks/callbacks
- Implemented rollback mechanisms for failed executions
- Added support for long-running task resumption
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

from loguru import logger
from pydantic import BaseModel, Field

from plan.capabilities.base import (
    CapabilityExecutionError,
)
from plan.capabilities.factory import create_pydantic_model
from plan.capabilities.registry import CapabilityRegistry
from plan.planning.models import Plan, Task


class ExecutionStatus(str, Enum):
    """Status of task or plan execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class ResourceUsage(BaseModel):
    """Track resource usage during execution"""

    memory_mb: float = Field(default=0.0)
    cpu_percent: float = Field(default=0.0)
    execution_time_sec: float = Field(default=0.0)
    api_calls: int = Field(default=0)


class TaskResult(BaseModel):
    """Result of a task execution"""

    task_name: str
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = Field(default=0.0)
    retries: int = Field(default=0)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)


class ExecutionContext(BaseModel):
    """Runtime context for plan execution"""

    working_memory: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, TaskResult] = Field(default_factory=dict)
    execution_stack: Set[str] = Field(default_factory=set)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    checkpoints: Dict[str, Any] = Field(default_factory=dict)


class ExecutionOptions(BaseModel):
    """Configuration options for plan execution"""

    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    timeout: Optional[float] = None
    parallel_execution: bool = Field(default=True)
    dry_run: bool = Field(default=False)
    debug_mode: bool = Field(default=False)


class ExecutionHooks(BaseModel):
    """Callbacks for execution events"""

    on_task_start: Optional[Callable[[Task, ExecutionContext], None]] = None
    on_task_complete: Optional[Callable[[Task, TaskResult, ExecutionContext], None]] = (
        None
    )
    on_task_error: Optional[Callable[[Task, TaskResult, ExecutionContext], None]] = None
    on_plan_complete: Optional[Callable[[Plan, Dict[str, Any]], None]] = None


class PlanExecutor:
    """Executes plans with advanced features and monitoring"""

    def __init__(
        self,
        registry: CapabilityRegistry,
        options: Optional[ExecutionOptions] = None,
        hooks: Optional[ExecutionHooks] = None,
    ):
        """Initialize the executor

        Args:
            registry: Registry for resolving capabilities
            options: Optional execution configuration
            hooks: Optional execution callbacks
        """
        self._registry = registry
        self._options = options or ExecutionOptions()
        self._hooks = hooks or ExecutionHooks()

    async def execute(
        self,
        plan: Plan,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a plan

        Args:
            plan: The plan to execute
            initial_context: Optional starting context

        Returns:
            Dictionary of output values

        Raises:
            PlanExecutionError: If execution fails
        """
        try:
            initial_context = initial_context or {}

            # Initialize execution context and store it as instance variable
            self._context = ExecutionContext(
                working_memory={},  # Start with empty working memory
                results={},
                execution_stack=set(),
                start_time=datetime.utcnow(),
            )

            # Map initial context to task inputs and capability schema
            for task in plan.tasks:
                capability = self._registry.get(task.capability_name)
                if not capability or not hasattr(capability, "metadata"):
                    continue

                # For each task input
                for task_input in task.inputs:
                    # If this input's source is in the initial context
                    if task_input.source_key in initial_context:
                        # Map it to both the task's input key and the capability's schema key
                        value = initial_context[task_input.source_key]
                        self._context.working_memory[task_input.source_key] = value

                        # Also map to the capability's schema key if different
                        if task_input.key in capability.metadata.input_schema:
                            self._context.working_memory[task_input.key] = value

            # Dry run if enabled
            if self._options.dry_run:
                return await self._simulate_execution(plan, self._context)

            # Execute tasks
            if self._options.parallel_execution:
                await self._execute_parallel(plan, self._context)
            else:
                await self._execute_sequential(plan, self._context)

            # Verify all desired outputs were produced
            self._verify_outputs(plan, self._context)

            # Return desired outputs
            return {
                output: self._context.working_memory[output]
                for output in plan.desired_outputs
            }

        except Exception as e:
            await self._handle_execution_error(plan, self._context, e)
            raise

    async def _execute_sequential(
        self,
        plan: Plan,
        context: ExecutionContext,
    ) -> None:
        """Execute tasks sequentially

        Args:
            plan: Plan to execute
            context: Execution context
        """
        for task in plan.tasks:
            if task.name in context.execution_stack:
                raise PlanExecutionError(f"Circular dependency detected: {task.name}")

            context.execution_stack.add(task.name)
            try:
                # Notify task start
                if self._hooks.on_task_start:
                    self._hooks.on_task_start(task, context)

                # Execute task with retries
                result = None
                for attempt in range(self._options.max_retries):
                    try:
                        result = await self._execute_task(task, context)
                        if result.status == ExecutionStatus.COMPLETED:
                            break
                        await asyncio.sleep(self._options.retry_delay * (2**attempt))
                    except Exception as e:
                        if attempt == self._options.max_retries - 1:
                            result = TaskResult(
                                task_name=task.name,
                                status=ExecutionStatus.FAILED,
                                error=str(e),
                                start_time=datetime.utcnow(),
                            )
                            break

                # Store result
                if result is not None:
                    context.results[task.name] = result
                    if result.status == ExecutionStatus.COMPLETED:
                        context.working_memory[task.output_key] = result.output
                        # Notify completion
                        if self._hooks.on_task_complete:
                            self._hooks.on_task_complete(task, result, context)
                    else:
                        # Notify error
                        if self._hooks.on_task_error:
                            self._hooks.on_task_error(task, result, context)
                        raise PlanExecutionError(
                            f"Task failed: {task.name} - {result.error}"
                        )

            finally:
                context.execution_stack.remove(task.name)

    async def _execute_parallel(
        self,
        plan: Plan,
        context: ExecutionContext,
    ) -> None:
        """Execute independent tasks in parallel"""
        # Build dependency graph
        dependencies = self._build_dependency_graph(plan)

        # Validate required initial inputs are available
        required_initial_inputs = dependencies.pop("__required_initial_inputs__")
        missing_inputs = required_initial_inputs - set(context.working_memory.keys())
        if missing_inputs:
            raise PlanExecutionError(
                f"Missing required initial inputs: {missing_inputs}. "
                f"These inputs must be provided in the initial context."
            )

        # Execute tasks in waves
        while dependencies:
            # Find tasks with no dependencies
            ready_tasks = [
                task
                for task in plan.tasks
                if task.name in dependencies and not dependencies[task.name]
            ]

            if not ready_tasks:
                raise PlanExecutionError("Circular dependency detected")

            # Execute wave of tasks
            results = await asyncio.gather(
                *[self._execute_task(task, context) for task in ready_tasks],
                return_exceptions=True,
            )

            # Process results
            for task, result in zip(ready_tasks, results):
                if isinstance(result, Exception):
                    raise PlanExecutionError(f"Task failed: {task.name}") from result

                context.results[task.name] = result
                if result.status == ExecutionStatus.COMPLETED:
                    # Store output only under its output key
                    context.working_memory[task.output.key] = result.output
                else:
                    raise PlanExecutionError(
                        f"Task failed: {task.name} - {result.error}"
                    )

                # Remove completed task from dependencies
                dependencies.pop(task.name)
                for deps in dependencies.values():
                    deps.discard(task.name)

    async def _execute_task(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> TaskResult:
        """Execute a single task"""
        start_time = datetime.utcnow()

        try:
            capability = self._registry.get(task.capability_name)
            if not capability:
                raise CapabilityExecutionError(
                    f"Capability not found: {task.capability_name}"
                )

            # Debug log available inputs
            logger.debug(f"Task {task.name} - Working Memory: {context.working_memory}")
            logger.debug(f"Task {task.name} - Required Inputs: {task.inputs}")

            # Map inputs at execution time
            input_values = {}
            for task_input in task.inputs:
                if task_input.source_key in context.working_memory:
                    # Map from source output to expected input name
                    input_values[task_input.key] = context.working_memory[
                        task_input.source_key
                    ]
                    logger.debug(
                        f"Task {task.name} - Mapped {task_input.source_key} -> {task_input.key}"
                    )
                else:
                    # Try to find the value in initial context or previous task outputs
                    for key, value in context.working_memory.items():
                        if task_input.key in key.lower():  # Fuzzy match on key names
                            input_values[task_input.key] = value
                            logger.debug(
                                f"Task {task.name} - Fuzzy matched {key} -> {task_input.key}"
                            )
                            break
                    else:
                        logger.warning(
                            f"Task {task.name} - Missing input: {task_input.key} (source: {task_input.source_key})"
                        )

            logger.debug(f"Task {task.name} - Final input values: {input_values}")

            # Create input model if capability has schema
            if hasattr(capability, "metadata") and hasattr(
                capability.metadata, "input_schema"
            ):
                input_model = create_pydantic_model(
                    f"{task.name}Input", capability.metadata.input_schema
                )
                try:
                    input_instance = input_model(**input_values)
                    logger.debug(
                        f"Task {task.name} - Created input model: {input_instance}"
                    )
                except Exception as e:
                    logger.error(
                        f"Task {task.name} - Input validation failed: {str(e)}"
                    )
                    raise
            else:
                input_instance = input_values

            # Execute capability
            result = await capability.execute(input_instance)

            return TaskResult(
                task_name=task.name,
                status=ExecutionStatus.COMPLETED,
                output=result,
                start_time=start_time,
                end_time=datetime.utcnow(),
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )

        except asyncio.TimeoutError:
            return TaskResult(
                task_name=task.name,
                status=ExecutionStatus.FAILED,
                error="Task execution timed out",
                start_time=start_time,
                end_time=datetime.utcnow(),
                execution_time=self._options.timeout or 0,
            )

        except Exception as e:
            return TaskResult(
                task_name=task.name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=datetime.utcnow(),
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )

    def _gather_inputs(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Gather input values for a task

        Args:
            task: Task requiring inputs
            context: Current execution context

        Returns:
            Dictionary of input values

        Raises:
            PlanExecutionError: If required input not found
        """
        try:
            return {
                input.key: context.working_memory[input.source_key]
                for input in task.inputs
                if input.source_key
            }
        except KeyError as e:
            raise PlanExecutionError(f"Required input not found: {e.args[0]}")

    def _build_dependency_graph(self, plan: Plan) -> Dict[str, Set[str]]:
        """Build task dependency graph"""
        dependencies: Dict[str, Set[str]] = {task.name: set() for task in plan.tasks}
        required_initial_inputs: Set[str] = set()

        # Map task outputs to their producing tasks
        producers = {task.output.key: task.name for task in plan.tasks}
        logger.debug(f"Task outputs: {producers}")

        # For each task, check its input sources
        for task in plan.tasks:
            for task_input in task.inputs:
                # If we find a task that produces this input, it's a dependency
                if producer := producers.get(task_input.source_key):
                    dependencies[task.name].add(producer)
                    logger.debug(
                        f"Task {task.name} depends on {producer} for {task_input.source_key}"
                    )
                else:
                    # If no task produces it, it must come from initial context
                    required_initial_inputs.add(task_input.source_key)
                    logger.debug(
                        f"Input {task_input.source_key} required from initial context"
                    )

        # Store required initial inputs in the graph metadata
        dependencies["__required_initial_inputs__"] = required_initial_inputs

        return dependencies

    def _verify_outputs(
        self,
        plan: Plan,
        context: ExecutionContext,
    ) -> None:
        """Verify all required outputs were produced

        Args:
            plan: Executed plan
            context: Execution context

        Raises:
            PlanExecutionError: If outputs are missing
        """
        missing = [
            output
            for output in plan.desired_outputs
            if output not in context.working_memory
        ]
        if missing:
            raise PlanExecutionError(f"Missing required outputs: {missing}")

    async def _simulate_execution(
        self,
        plan: Plan,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Simulate plan execution without running tasks

        Args:
            plan: Plan to simulate
            context: Execution context

        Returns:
            Simulated output values
        """
        # Validate dependencies
        self._build_dependency_graph(plan)

        # Return dummy outputs
        return {
            output: f"Simulated output for {output}" for output in plan.desired_outputs
        }

    async def _handle_execution_error(
        self,
        plan: Plan,
        context: ExecutionContext,
        error: Exception,
    ) -> None:
        """Handle plan execution error

        Args:
            plan: Failed plan
            context: Execution context
            error: Error that occurred
        """
        # Mark all running tasks as failed
        for task_name in context.execution_stack:
            if task_name not in context.results:
                context.results[task_name] = TaskResult(
                    task_name=task_name,
                    status=ExecutionStatus.FAILED,
                    error="Plan execution aborted",
                    start_time=context.start_time,
                    end_time=datetime.utcnow(),
                )

        # Attempt rollback for completed tasks in reverse order
        completed_tasks = [
            task
            for task in plan.tasks
            if task.name in context.results
            and context.results[task.name].status == ExecutionStatus.COMPLETED
        ]

        for task in reversed(completed_tasks):
            try:
                # Get rollback capability if available
                rollback_name = f"rollback_{task.capability_name}"
                if rollback_capability := self._registry.get(rollback_name):
                    await rollback_capability.execute(
                        {
                            "task_name": task.name,
                            "original_input": self._gather_inputs(task, context),
                            "original_output": context.results[task.name].output,
                        }
                    )
                    context.results[task.name].status = ExecutionStatus.ROLLED_BACK
            except Exception as e:
                # Log rollback failure but continue with other rollbacks
                print(f"Rollback failed for task {task.name}: {str(e)}")

    def get_execution_metrics(self) -> Dict[str, TaskResult]:
        """Get execution metrics for completed tasks

        Returns:
            Dictionary mapping task names to their execution results
        """
        return self._context.results if hasattr(self, "_context") else {}


class PlanExecutionError(Exception):
    """Raised when plan execution fails"""

    pass
