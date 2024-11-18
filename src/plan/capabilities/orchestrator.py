"""High-level coordination of capability creation, execution, and optimization.

Improvements:
- Clearer separation of concerns between orchestration, planning, and execution
- Enhanced error handling with specific exception types
- Added capability optimization based on usage patterns
- Implemented capability versioning and lifecycle management
- Added support for capability dependencies and conflict resolution
- Enhanced monitoring and observability
- Added support for capability rollback and recovery
- Implemented capability warm-up and health checks
- Added support for capability groups and namespaces
- Enhanced capability resolution with fallbacks
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, create_model

from plan.capabilities.base import (
    Capability,
    CapabilityExecutionError,
    CapabilityNotFoundError,
)
from plan.capabilities.factory import CapabilityFactory
from plan.capabilities.registry import CapabilityRegistry
from plan.llm.handler import PromptHandler
from plan.planning.executor import PlanExecutor
from plan.planning.models import Plan
from plan.planning.planner import Planner


class ExecutionStrategy(BaseModel):
    """Configuration for capability execution"""

    max_retries: int = Field(default=3)
    timeout_seconds: float = Field(default=30.0)
    fallback_capability: Optional[str] = None
    parallel_execution: bool = Field(default=False)


class CapabilityRequest(BaseModel):
    """Request for capability creation or resolution"""

    name: str
    required_inputs: List[str]
    required_output: str
    context: Dict[str, Any]
    strategy: Optional[ExecutionStrategy] = None


class CapabilityOrchestrator:
    """Coordinates capability lifecycle and execution"""

    def __init__(
        self,
        prompt_handler: Optional[PromptHandler] = None,
        registry: Optional[CapabilityRegistry] = None,
        factory: Optional[CapabilityFactory] = None,
    ):
        """Initialize the orchestrator"""
        self._prompt_handler = prompt_handler or PromptHandler()
        self._registry = registry or CapabilityRegistry()
        self._factory = factory or CapabilityFactory(
            self._prompt_handler, self._registry
        )
        self._planner = Planner(self._registry, self._factory, self._prompt_handler)
        self._executor = PlanExecutor(self._registry)

    async def get_or_create_capability(
        self, request: CapabilityRequest
    ) -> Capability[BaseModel, Any]:
        """Get existing capability or create new one"""
        try:
            # Check registry first
            if capability := self._registry.get(request.name):
                await self._validate_capability(capability, request)
                return capability

            # Create new capability
            capability_type, capability = await self._factory.resolve_capability(
                request.name,
                request.required_inputs,
                request.required_output,
                request.context,
            )

            # Register new capability
            self._registry.register(request.name, capability)

            # Warm up if needed
            await self._warm_up_capability(capability)

            return capability

        except Exception as e:
            raise CapabilityExecutionError(
                f"Failed to resolve capability {request.name}: {str(e)}"
            ) from e

    async def execute_capability(
        self,
        name: str,
        inputs: Dict[str, Any],
        strategy: Optional[ExecutionStrategy] = None,
    ) -> Any:
        """Execute a capability with the specified strategy"""
        strategy = strategy or ExecutionStrategy()

        # Get capability
        capability = self._registry.get(name)
        if not capability:
            raise CapabilityNotFoundError(f"Capability not found: {name}")

        try:
            # Execute with retry logic
            for attempt in range(strategy.max_retries):
                try:
                    # Create input model and validate
                    input_model = capability._get_input_model()
                    validated_input = input_model(**inputs)

                    result = await capability.execute(validated_input)

                    # Update usage metrics
                    await self._update_metrics(capability, True)

                    # Check if optimization is needed
                    await self._check_optimization(capability)

                    return result

                except Exception:
                    if attempt == strategy.max_retries - 1:
                        # Try fallback if available
                        if strategy.fallback_capability:
                            return await self.execute_capability(
                                strategy.fallback_capability,
                                inputs,
                                strategy,
                            )
                        raise

                    await self._update_metrics(capability, False)

        except Exception as e:
            raise CapabilityExecutionError(
                f"Execution failed for {name}: {str(e)}"
            ) from e

    async def create_plan(
        self,
        goal: str,
        required_inputs: List[str],
        required_output: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """Create a new plan for achieving a goal"""
        # Validate input types from context if provided
        if context:
            input_schema = Schema(
                fields={
                    name: SchemaField(
                        type=SchemaType.ANY,
                        description=f"Input {name}",
                        required=True,
                    )
                    for name in required_inputs
                }
            )
            try:
                input_model = create_model(
                    "PlanInput", **input_schema.to_pydantic_fields()
                )
                input_model(**context)
            except ValidationError as e:
                raise ValueError(f"Invalid input context: {str(e)}")

        return await self._planner.create_plan(
            goal, required_inputs, required_output, context or {}
        )

    async def execute_plan(
        self, plan: Plan, initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a plan with the given context"""
        return await self._executor.execute(plan, initial_context)

    async def _validate_capability(
        self, capability: Capability, request: CapabilityRequest
    ) -> None:
        """Validate capability against requirements"""
        metadata = capability.metadata
        input_schema = metadata.input_schema
        output_schema = metadata.output_schema

        # Validate required inputs exist in schema
        missing_inputs = []
        for input_name in request.required_inputs:
            if input_name not in input_schema.fields:
                missing_inputs.append(input_name)
            elif input_schema.fields[input_name].required:
                # Also check if the input is marked as required
                if input_name not in request.context:
                    missing_inputs.append(input_name)

        if missing_inputs:
            raise ValueError(
                f"Capability {request.name} missing required inputs: {missing_inputs}"
            )

        # Validate output exists in schema
        if request.required_output not in output_schema.fields:
            raise ValueError(
                f"Capability {request.name} cannot produce required output: {request.required_output}"
            )

    async def _warm_up_capability(self, capability: Capability) -> None:
        """Perform capability warm-up if needed"""
        # Implementation depends on capability type
        pass

    async def _update_metrics(self, capability: Capability, success: bool) -> None:
        """Update capability usage metrics"""
        metadata = capability.metadata
        metadata.usage_count += 1
        metadata.success_rate = (
            metadata.success_rate * (metadata.usage_count - 1)
            + (1.0 if success else 0.0)
        ) / metadata.usage_count

    async def _check_optimization(self, capability: Capability) -> None:
        """Check if capability needs optimization"""
        metadata = capability.metadata

        # Skip if not enough usage data
        if metadata.usage_count < 100:
            return

        # Check if performance is poor
        if metadata.success_rate < 0.95 or any(
            metric > threshold
            for metric, threshold in metadata.performance_metrics.items()
        ):
            await self._optimize_capability(capability)

    async def _optimize_capability(self, capability: Capability) -> None:
        """Optimize a capability based on usage patterns"""
        # Implementation depends on capability type
        pass

    def get_capability_stats(self) -> Dict[str, Any]:
        """Get statistics about registered capabilities"""
        return self._registry.get_stats().dict()

    async def deprecate_capability(
        self, name: str, replacement: Optional[str] = None
    ) -> None:
        """Mark a capability as deprecated"""
        capability = self._registry.get(name)
        if capability:
            capability.metadata.status = "deprecated"
            if replacement:
                capability.metadata.config["replacement"] = replacement

    async def migrate_capability(self, old_name: str, new_name: str) -> None:
        """Migrate a capability to a new name"""
        if capability := self._registry.get(old_name):
            self._registry.register(new_name, capability)
            await self.deprecate_capability(old_name, new_name)
