"""Plan capability implementation"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from plan.capabilities.base import (
    Capability,
    CapabilityExecutionError,
    Input,
    Output,
)
from plan.capabilities.metadata import CapabilityMetadata
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.schema import Schema, SchemaField, SchemaType


class TaskResult(BaseModel):
    """Result of a task execution"""

    task_name: str
    output: Any
    execution_time: float
    success: bool
    error: Optional[str] = None


class ExecutionContext(BaseModel):
    """Runtime context for plan execution"""

    working_memory: Dict[str, Any] = Field(default_factory=dict)
    execution_stack: List[str] = Field(default_factory=list)
    results: Dict[str, TaskResult] = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=datetime.utcnow)


class PlanTask(BaseModel):
    """A single task in a plan"""

    name: str
    capability_name: str
    inputs: Dict[str, str]  # Maps task input names to source keys
    output_key: str
    description: str


class Plan(BaseModel):
    """A collection of tasks with a goal"""

    name: str
    description: str
    tasks: Dict[str, PlanTask]
    desired_outputs: List[str]
    initial_context: Dict[str, Any] = Field(default_factory=dict)


class PlanCapability(Capability[Input, Output]):
    """A capability implemented as a plan"""

    def __init__(
        self,
        plan: Plan,
        metadata: CapabilityMetadata,
        registry: CapabilityRegistry,
    ):
        """Initialize the plan capability"""
        super().__init__(metadata)
        self._validate_plan(plan)
        self._plan = plan
        self._registry = registry

    def _validate_plan(self, plan: Plan) -> None:
        """Validate plan structure and dependencies"""
        # Verify all desired outputs are produced
        available_outputs = {task.output_key for task in plan.tasks.values()}
        missing_outputs = set(plan.desired_outputs) - available_outputs
        if missing_outputs:
            raise ValueError(
                f"Plan does not produce required outputs: {missing_outputs}"
            )

        # Check for cycles in task dependencies
        self._check_cycles(plan)

    def _check_cycles(self, plan: Plan) -> None:
        """Check for circular dependencies in the plan"""
        # Build dependency graph
        graph: Dict[str, set[str]] = {task.name: set() for task in plan.tasks.values()}

        for task in plan.tasks.values():
            for input_source in task.inputs.values():
                producer = next(
                    (t for t in plan.tasks.values() if t.output_key == input_source),
                    None,
                )
                if producer:
                    graph[task.name].add(producer.name)

        # Check for cycles using DFS
        visited = set()
        path = set()

        def visit(task_name: str) -> None:
            if task_name in path:
                cycle = list(path)
                cycle.append(task_name)
                raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}")

            if task_name in visited:
                return

            visited.add(task_name)
            path.add(task_name)

            for dep in graph[task_name]:
                visit(dep)

            path.remove(task_name)

        for task_name in graph:
            visit(task_name)

    async def _execute_impl(self, input: Input) -> Output:
        """Execute the plan"""
        try:
            # Initialize execution context
            context = ExecutionContext(working_memory=dict(self._plan.initial_context))
            context.working_memory.update(input.__dict__)

            # Execute tasks in dependency order
            for task_name in self._get_execution_order():
                if task_name in context.execution_stack:
                    raise CapabilityExecutionError(
                        f"Circular dependency detected: {task_name}"
                    )

                context.execution_stack.append(task_name)
                try:
                    result = await self._execute_task(task_name, context)
                    context.results[task_name] = result
                    if result.success:
                        context.working_memory[result.task_name] = result.output
                    else:
                        raise CapabilityExecutionError(
                            f"Task failed: {task_name} - {result.error}"
                        )
                finally:
                    context.execution_stack.pop()

            # Create output model from working memory
            output_model = self._get_output_model()
            return output_model(
                **{
                    key: context.working_memory[key]
                    for key in self._plan.desired_outputs
                }
            )

        except Exception as e:
            raise CapabilityExecutionError(f"Plan execution failed: {str(e)}") from e

    def _get_execution_order(self) -> List[str]:
        """Determine optimal task execution order"""
        # Build dependency graph
        graph: Dict[str, set[str]] = {
            task.name: set() for task in self._plan.tasks.values()
        }

        for task in self._plan.tasks.values():
            for input_source in task.inputs.values():
                producer = next(
                    (
                        t
                        for t in self._plan.tasks.values()
                        if t.output_key == input_source
                    ),
                    None,
                )
                if producer:
                    graph[producer.name].add(task.name)

        # Topological sort
        order: List[str] = []
        visited = set()

        def visit(task_name: str) -> None:
            if task_name in visited:
                return
            visited.add(task_name)
            for dep in graph[task_name]:
                visit(dep)
            order.append(task_name)

        for task_name in graph:
            visit(task_name)

        return order

    async def _execute_task(
        self, task_name: str, context: ExecutionContext
    ) -> TaskResult:
        """Execute a single task"""
        task = self._plan.tasks[task_name]
        start_time = datetime.utcnow()

        try:
            # Resolve capability
            capability = self._registry.get(task.capability_name)
            if not capability:
                raise CapabilityExecutionError(
                    f"Capability not found: {task.capability_name}"
                )

            # Prepare inputs
            input_values = {
                input_name: context.working_memory[source_key]
                for input_name, source_key in task.inputs.items()
            }

            # Create input model
            input_model = self._get_input_model()
            validated_input = input_model(**input_values)

            # Execute capability
            result = await capability.execute(validated_input)

            return TaskResult(
                task_name=task_name,
                output=result,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                success=True,
            )

        except Exception as e:
            return TaskResult(
                task_name=task_name,
                output=None,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                success=False,
                error=str(e),
            )

    async def visualize(self) -> None:
        """Visualize the plan structure"""
        # Implementation using networkx and matplotlib
        pass

    async def simulate(self, input: Input) -> Dict[str, Any]:
        """Simulate plan execution without running tasks"""
        context = ExecutionContext(working_memory=dict(self._plan.initial_context))
        context.working_memory.update(input.__dict__)

        # Return simulated outputs
        return {
            output_key: f"Simulated output for {output_key}"
            for output_key in self._plan.desired_outputs
        }

    @property
    def plan(self) -> Plan:
        """Get the underlying plan"""
        return self._plan

    @classmethod
    def create(
        cls,
        name: str,
        plan: Plan,
        registry: CapabilityRegistry,
        description: str = "",
    ) -> "PlanCapability":
        """Create a new plan capability with schema validation"""
        # Create input schema from plan's initial context
        input_schema = Schema(
            fields={
                name: SchemaField(
                    type=SchemaType.ANY,
                    description=f"Input {name}",
                    required=True,
                )
                for name in plan.initial_context.keys()
            }
        )

        # Create output schema from plan's desired outputs
        output_schema = Schema(
            fields={
                name: SchemaField(
                    type=SchemaType.ANY,
                    description=f"Output {name}",
                    required=True,
                )
                for name in plan.desired_outputs
            }
        )

        metadata = CapabilityMetadata(
            name=name,
            type=CapabilityType.PLAN,
            description=description or plan.description,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        return cls(plan, metadata, registry)
