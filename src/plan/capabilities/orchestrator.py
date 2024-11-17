"""Capability orchestration and dynamic creation"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from plan.capabilities.base import (
    Capability,
    CapabilityExecutionError,
    CapabilityNotFoundError,
)
from plan.capabilities.factory import CapabilityFactory, create_pydantic_model
from plan.capabilities.metadata import CapabilityMetadata, CapabilityType
from plan.capabilities.registry import CapabilityRegistry


class CapabilityRequirements(BaseModel):
    """Requirements for a capability"""

    name: str = Field(..., description="Name of the capability")
    input_schema: Dict[str, Any] = Field(..., description="Required input schema")
    output_schema: Dict[str, Any] = Field(..., description="Required output schema")
    required_features: Set[str] = Field(
        default_factory=set, description="Required features"
    )
    min_success_rate: Optional[float] = Field(
        default=None, description="Minimum required success rate"
    )
    max_latency: Optional[float] = Field(
        default=None, description="Maximum allowed latency in seconds"
    )


class CapabilityResolutionError(Exception):
    """Raised when capability resolution fails"""

    pass


class CapabilityCreationError(Exception):
    """Raised when capability creation fails"""

    pass


class CapabilityOrchestrator:
    """Manages capability lifecycle and dynamic creation"""

    def __init__(
        self,
        registry: CapabilityRegistry,
        factory: CapabilityFactory,
    ):
        """Initialize the orchestrator

        Args:
            registry: Registry for capability storage
            factory: Factory for creating new capabilities
        """
        self._registry = registry
        self._factory = factory

    async def get_or_create_capability(
        self,
        requirements: CapabilityRequirements,
        context: Optional[Dict[str, Any]] = None,
    ) -> Capability[BaseModel, Any]:
        """Get existing capability or create new one"""
        try:
            # Check registry first
            if capability := self._registry.get(requirements.name):
                # Validate capability meets requirements
                if self._validate_capability_requirements(capability, requirements):
                    return capability

            # Create input/output models
            input_model = create_pydantic_model(
                f"{requirements.name}Input", requirements.input_schema
            )
            output_model = create_pydantic_model(
                f"{requirements.name}Output", requirements.output_schema
            )

            # Create new capability
            capability_type, capability = await self._factory.resolve_capability(
                requirements.name,
                list(requirements.input_schema.keys()),
                list(requirements.output_schema.keys())[0],
                context or {},
            )

            # Register new capability
            self._registry.register(requirements.name, capability)

            return capability

        except Exception as e:
            raise CapabilityResolutionError(
                f"Failed to resolve capability {requirements.name}: {str(e)}"
            ) from e

    def _validate_capability_requirements(
        self,
        capability: Capability,
        requirements: CapabilityRequirements,
    ) -> bool:
        """Validate capability meets requirements

        Args:
            capability: Capability to validate
            requirements: Required capabilities

        Returns:
            True if capability meets requirements
        """
        metadata = capability.metadata
        stats = capability.execution_stats

        # Verify inputs
        if not set(requirements.input_schema.keys()).issubset(
            metadata.input_schema.keys()
        ):
            return False

        # Verify outputs
        if not set(requirements.output_schema.keys()).issubset(
            metadata.output_schema.keys()
        ):
            return False

        # Verify features
        if not requirements.required_features.issubset(metadata.tags):
            return False

        # Verify performance requirements
        if (
            requirements.min_success_rate is not None
            and stats.success_rate < requirements.min_success_rate
        ):
            return False

        if (
            requirements.max_latency is not None
            and stats.average_execution_time > requirements.max_latency
        ):
            return False

        return True

    async def optimize_capability(
        self,
        name: str,
        target_success_rate: float = 0.95,
        target_latency: Optional[float] = None,
    ) -> None:
        """Optimize a capability to meet targets

        Args:
            name: Name of capability to optimize
            target_success_rate: Target success rate
            target_latency: Optional target latency in seconds

        Raises:
            CapabilityNotFoundError: If capability not found
            CapabilityExecutionError: If optimization fails
        """
        capability = self._registry.get(name)
        if not capability:
            raise CapabilityNotFoundError(f"Capability not found: {name}")

        stats = capability.execution_stats
        if stats.total_executions < 100:
            # Not enough data for optimization
            return

        try:
            if stats.success_rate < target_success_rate:
                # Implement success rate optimization
                pass

            if (
                target_latency is not None
                and stats.average_execution_time > target_latency
            ):
                # Implement latency optimization
                pass

        except Exception as e:
            raise CapabilityExecutionError(f"Optimization failed: {str(e)}") from e

    async def create_composite_capability(
        self,
        name: str,
        components: List[str],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
    ) -> Capability:
        """Create a new composite capability

        Args:
            name: Name for new capability
            components: List of component capability names
            input_schema: Input requirements
            output_schema: Output requirements

        Returns:
            Created composite capability

        Raises:
            CapabilityCreationError: If creation fails
        """
        try:
            # Verify all components exist
            for component in components:
                if not self._registry.get(component):
                    raise CapabilityNotFoundError(f"Component not found: {component}")

            # Create composite capability
            metadata = CapabilityMetadata(
                name=name,
                type=CapabilityType.PLAN,
                description=f"Composite capability using: {', '.join(components)}",
                input_schema=input_schema,
                output_schema=output_schema,
                created_at=datetime.now(),
                dependencies=components,
            )

            # Create and register capability
            capability_type, capability = await self._factory.create_plan(
                name=name,
                metadata=metadata,
                components=components,
            )

            self._registry.register(name, capability)
            return capability

        except Exception as e:
            raise CapabilityCreationError(
                f"Failed to create composite capability: {str(e)}"
            ) from e
