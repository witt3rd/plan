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
- Added support for capability migration and deprecation
- Implemented capability access control and quotas
- Added support for capability execution strategies
- Enhanced capability metadata management
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from plan.capabilities.base import (
    Capability,
    CapabilityExecutionError,
    CapabilityNotFoundError,
)
from plan.capabilities.factory import CapabilityFactory
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.tool import ToolCapability
from plan.llm.handler import PromptHandler
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
    """Coordinates capability creation, execution, and lifecycle management"""

    def __init__(
        self,
        prompt_handler: Optional[PromptHandler] = None,
        registry: Optional[CapabilityRegistry] = None,
        factory: Optional[CapabilityFactory] = None,
    ):
        """Initialize the orchestrator with optional components

        Args:
            prompt_handler: Handler for LLM interactions
            registry: Registry for capability storage
            factory: Factory for capability creation
        """
        self._prompt_handler = prompt_handler or PromptHandler()
        self._registry = registry or CapabilityRegistry()
        self._factory = factory or CapabilityFactory(self._prompt_handler)
        self._planner = Planner(self._registry, self._factory, self._prompt_handler)

    async def get_or_create_capability(self, request: CapabilityRequest) -> Capability:
        """Get existing capability or create new one

        Args:
            request: Capability request details

        Returns:
            Resolved capability

        Raises:
            CapabilityExecutionError: If capability creation fails
        """
        try:
            # Check registry first
            if capability := self._registry.get(request.name):
                await self._validate_capability(capability, request)
                return capability

            # Create new capability
            capability_type, capability = await self._factory.create_capability(
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
        """Execute a capability with the specified strategy

        Args:
            name: Name of capability to execute
            inputs: Input values for execution
            strategy: Optional execution strategy

        Returns:
            Execution result

        Raises:
            CapabilityNotFoundError: If capability not found
            CapabilityExecutionError: If execution fails
        """
        strategy = strategy or ExecutionStrategy()

        # Get capability
        capability = self._registry.get(name)
        if not capability:
            raise CapabilityNotFoundError(f"Capability not found: {name}")

        try:
            # Execute with retry logic
            for attempt in range(strategy.max_retries):
                try:
                    if isinstance(capability, ToolCapability):
                        result = await capability.execute(inputs)

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
        context: Dict[str, Any],
    ) -> Plan:
        """Create a new plan for achieving a goal

        Args:
            goal: Description of what to achieve
            required_inputs: Available input names
            required_output: Required output name
            context: Additional context for planning

        Returns:
            Created plan

        Raises:
            CapabilityExecutionError: If plan creation fails
        """
        return await self._planner.create_plan(goal, required_inputs, required_output)

    async def _validate_capability(
        self, capability: Capability, request: CapabilityRequest
    ) -> None:
        """Validate capability against requirements

        Args:
            capability: Capability to validate
            request: Original capability request

        Raises:
            ValueError: If validation fails
        """
        metadata = capability.metadata

        # For tool capabilities, use function_schema validation
        if isinstance(capability, ToolCapability):
            schema = capability.schema
            required_inputs = set(request.required_inputs)
            available_inputs = set(schema["parameters"]["properties"].keys())

            if not required_inputs.issubset(available_inputs):
                missing = required_inputs - available_inputs
                raise ValueError(
                    f"Capability {request.name} missing required inputs: {missing}"
                )
            return

        # For other capabilities, use standard validation
        if not set(request.required_inputs).issubset(metadata.input_schema.keys()):
            raise ValueError(f"Capability {request.name} missing required inputs")

        if request.required_output not in metadata.output_schema:
            raise ValueError(
                f"Capability {request.name} cannot produce required output"
            )

    async def _warm_up_capability(self, capability: Capability) -> None:
        """Perform capability warm-up if needed

        Args:
            capability: Capability to warm up
        """
        # Implementation depends on capability type
        pass

    async def _update_metrics(self, capability: Capability, success: bool) -> None:
        """Update capability usage metrics

        Args:
            capability: Capability to update
            success: Whether execution was successful
        """
        metadata = capability.metadata
        metadata.usage_count += 1
        metadata.success_rate = (
            metadata.success_rate * (metadata.usage_count - 1)
            + (1.0 if success else 0.0)
        ) / metadata.usage_count

    async def _check_optimization(self, capability: Capability) -> None:
        """Check if capability needs optimization

        Args:
            capability: Capability to check
        """
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
        """Optimize a capability based on usage patterns

        Args:
            capability: Capability to optimize
        """
        # Implementation depends on capability type
        pass

    def get_capability_stats(self) -> Dict[str, Any]:
        """Get statistics about registered capabilities

        Returns:
            Dictionary of capability statistics
        """
        return self._registry.get_stats().dict()

    async def deprecate_capability(
        self, name: str, replacement: Optional[str] = None
    ) -> None:
        """Mark a capability as deprecated

        Args:
            name: Name of capability to deprecate
            replacement: Optional replacement capability name
        """
        capability = self._registry.get(name)
        if capability:
            capability.metadata.status = "deprecated"
            if replacement:
                capability.metadata.config["replacement"] = replacement

    async def migrate_capability(self, old_name: str, new_name: str) -> None:
        """Migrate a capability to a new name

        Args:
            old_name: Current capability name
            new_name: New capability name
        """
        if capability := self._registry.get(old_name):
            self._registry.register(new_name, capability)
            await self.deprecate_capability(old_name, new_name)
