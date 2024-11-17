"""
Capability creation logic

Next steps would be to:

- Add validation for inputs/outputs
- Add capability testing before returning
- Enhance the decision making with more context
Add caching and optimization
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel, Field, create_model

from plan.capabilities.base import (
    Capability,
    CapabilityExecutionError,
    CapabilityResolutionError,
)
from plan.capabilities.instruction import InstructionCapability
from plan.capabilities.interfaces import CapabilityCreator
from plan.capabilities.metadata import (
    CapabilityMetadata,
    CapabilityType,
    DependencyRequirement,
)
from plan.capabilities.plan import PlanCapability
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.tool import ToolCapability
from plan.llm.handler import CompletionConfig, PromptHandler
from plan.llm.templates import TemplateManager
from plan.llm.tool import get_function_schema


class CapabilityTypeDecision(BaseModel):
    """Model for deciding capability implementation type"""

    capability_type: CapabilityType
    reasoning: str
    requirements: list[str]
    suggested_dependencies: list[str]
    performance_notes: str


class ToolSpecification(BaseModel):
    """Specification for a new tool implementation"""

    function_name: str
    description: str
    implementation: str
    test_cases: list[str]
    error_cases: list[str]
    dependencies: list[str] = Field(default_factory=list)


class InstructionSpecification(BaseModel):
    """Specification for a new instruction template"""

    template: str
    description: str
    required_inputs: list[str]
    example_outputs: list[str]
    validation_criteria: list[str]
    error_cases: list[str]


class CapabilityFactory(CapabilityCreator):
    """Creates and resolves capabilities"""

    def __init__(
        self,
        prompt_handler: PromptHandler,
        registry: CapabilityRegistry,
    ):
        """Initialize the factory"""
        self._prompt_handler = prompt_handler
        self._registry = registry
        self._template_manager = TemplateManager()

    async def resolve_capability(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
    ) -> Tuple[CapabilityType, Capability[BaseModel, Any]]:
        """Resolves a capability by finding or creating an appropriate implementation

        Args:
            name: Name of capability to resolve
            required_inputs: List of required input names
            required_output: Name of required output
            context: Additional context for capability creation

        Returns:
            Tuple of (capability_type, capability)

        Raises:
            CapabilityResolutionError: If capability cannot be resolved
        """
        try:
            # First check if capability exists
            if existing := self._registry.get(name):
                # Validate compatibility
                if self._is_capability_compatible(
                    existing, required_inputs, required_output
                ):
                    return existing.metadata.type, existing

            # Determine best implementation approach
            decision = await self._determine_capability_type(
                name, required_inputs, required_output, context
            )

            # Create appropriate implementation
            if decision.capability_type == CapabilityType.TOOL:
                return await self._create_tool(
                    name, required_inputs, required_output, context, decision
                )
            elif decision.capability_type == CapabilityType.INSTRUCTION:
                return await self._create_instruction(
                    name, required_inputs, required_output, context, decision
                )
            else:  # PLAN
                return await self._create_plan(
                    name, required_inputs, required_output, context, decision
                )

        except Exception as e:
            raise CapabilityResolutionError(
                f"Failed to resolve capability {name}: {str(e)}"
            ) from e

    def _is_capability_compatible(
        self,
        capability: Capability[BaseModel, Any],
        required_inputs: List[str],
        required_output: str,
    ) -> bool:
        """Check if existing capability meets requirements

        Args:
            capability: Existing capability to check
            required_inputs: Required input names
            required_output: Required output name

        Returns:
            True if capability is compatible
        """
        metadata = capability.metadata

        # Check inputs
        if not set(required_inputs).issubset(metadata.input_schema.keys()):
            return False

        # Check output
        if required_output not in metadata.output_schema:
            return False

        return True

    async def _determine_capability_type(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
    ) -> CapabilityTypeDecision:
        """Determines the most appropriate capability type"""
        template = self._template_manager.get_template("capability_type_decision")
        if not template:
            raise ValueError("Required template 'capability_type_decision' not found")

        prompt = template.render(
            name=name,
            required_inputs=required_inputs,
            required_output=required_output,
            context=context,
            available_capabilities=self._get_available_capabilities(),
        )

        return await self._prompt_handler.complete(
            prompt,
            config=CompletionConfig(
                response_format=CapabilityTypeDecision,
                system_message="You are an expert system architect who understands the tradeoffs between different implementation approaches.",
            ),
        )

    def _get_available_capabilities(self) -> str:
        """Get formatted string of available capabilities"""
        capabilities = []
        for name in self._registry._capabilities.keys():
            cap = self._registry.get(name)
            if cap:
                capabilities.append(
                    f"- {name} ({cap.metadata.type}): {cap.metadata.description}"
                )
        return "\n".join(capabilities)

    async def create_capability(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
    ) -> Tuple[CapabilityType, Capability[BaseModel, Any]]:
        """Create a new capability"""
        return await self.resolve_capability(
            name, required_inputs, required_output, context
        )

    async def _create_tool(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
        decision: CapabilityTypeDecision,
    ) -> Tuple[CapabilityType, ToolCapability]:
        """Creates a new tool capability"""
        template = self._template_manager.get_template("tool_implementation")
        if not template:
            raise ValueError("Required template 'tool_implementation' not found")

        spec = await self._prompt_handler.complete(
            template.render(
                name=name,
                required_inputs=required_inputs,
                required_output=required_output,
                context=context,
                requirements=decision.requirements,
            ),
            config=CompletionConfig(
                response_format=ToolSpecification,
                system_message="You are an expert Python developer focused on creating reliable, well-documented functions.",
            ),
        )

        # Create function from specification
        namespace: dict[str, Any] = {}
        exec(spec.implementation, namespace)
        func = namespace[spec.function_name]

        # Get schema automatically
        schema = get_function_schema(func)

        # Create metadata using schema information
        metadata = CapabilityMetadata(
            name=name,
            type=CapabilityType.TOOL,
            created_at=datetime.now(),
            description=spec.description,
            input_schema=schema.get("parameters", {}).get("properties", {}),
            output_schema={"result": schema.get("returns", {"type": "string"})},
            # Convert string dependencies to DependencyRequirement objects
            dependencies=[
                DependencyRequirement(capability_name=dep)
                for dep in decision.suggested_dependencies
            ]
            if decision.suggested_dependencies
            else [],
        )

        # Create input model dynamically
        input_model = create_pydantic_model(
            f"{name}Input",
            {
                name: schema_type
                for name, schema_type in schema.get("parameters", {})
                .get("properties", {})
                .items()
            },
        )

        # Create output model dynamically
        output_model = create_pydantic_model(
            f"{name}Output", {"result": schema.get("returns", {"type": "string"})}
        )

        return CapabilityType.TOOL, ToolCapability[input_model, output_model](
            func, metadata
        )

    async def _create_instruction(
        self,
        name: str,
        required_inputs: list[str],
        required_output: str,
        context: dict[str, Any],
        decision: CapabilityTypeDecision,
    ) -> Tuple[CapabilityType, InstructionCapability]:
        """Creates a new instruction capability"""
        instruction = f"""
        Create a prompt template for:

        Name: {name}
        Required Inputs: {required_inputs}
        Required Output: {required_output}
        Context: {context}
        Requirements: {decision.requirements}

        The template should:
        1. Clearly specify required inputs
        2. Provide clear guidance for the model
        3. Include validation criteria
        4. Handle potential error cases
        5. Produce well-structured output
        """

        spec = await self._prompt_handler.complete(
            instruction,
            config=CompletionConfig(
                response_format=InstructionSpecification,
                system_message="You are an expert at creating clear and effective prompts.",
            ),
        )

        metadata = CapabilityMetadata(
            name=name,
            type=CapabilityType.INSTRUCTION,
            description=spec.description,
            input_schema={name: "Any" for name in required_inputs},
            output_schema={required_output: "Any"},
            dependencies=[
                DependencyRequirement(capability_name=dep)
                for dep in decision.suggested_dependencies
            ],
        )

        return (
            CapabilityType.INSTRUCTION,
            InstructionCapability(
                spec.template, metadata, self._prompt_handler, instruction_metadata=None
            ),
        )

    async def _create_plan(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
        decision: CapabilityTypeDecision,
    ) -> Tuple[CapabilityType, PlanCapability]:
        """Creates a new plan capability"""
        # Create plan metadata
        metadata = CapabilityMetadata(
            name=name,
            type=CapabilityType.PLAN,
            description=f"Plan capability for {name}",
            input_schema={name: "Any" for name in required_inputs},
            output_schema={required_output: "Any"},
            dependencies=[
                DependencyRequirement(capability_name=dep)
                for dep in decision.suggested_dependencies
            ],
        )

        # Get planner from registry
        planner = self._registry.get_planner()
        if not planner:
            raise CapabilityExecutionError("No planner registered")

        # IMPORTANT: Check if we're already creating this capability to prevent recursion
        if name in context.get("creating_capabilities", set()):
            raise CapabilityExecutionError(
                f"Recursive capability creation detected for {name}"
            )

        # Track that we're creating this capability
        context.setdefault("creating_capabilities", set()).add(name)

        try:
            # Create plan from components
            plan = await planner.create_plan(
                goal=f"Implement {name} using {decision.suggested_dependencies}",
                required_inputs=required_inputs,
                required_output=required_output,
                context=context,
            )

            return CapabilityType.PLAN, PlanCapability(plan, metadata, self._registry)
        finally:
            # Remove capability from creation tracking
            context.get("creating_capabilities", set()).remove(name)

    def _create_input_model(self, name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Create input model for a capability"""
        return create_pydantic_model(f"{name}Input", schema)

    def _create_output_model(
        self, name: str, schema: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Create output model for a capability"""
        return create_pydantic_model(f"{name}Output", schema)

    def _validate_models(
        self,
        input_model: Type[BaseModel],
        output_model: Type[BaseModel],
        metadata: CapabilityMetadata,
    ) -> None:
        """Validate input/output models match metadata"""
        # Add validation logic


def create_pydantic_model(name: str, fields: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model dynamically

    Args:
        name: Name for the model class
        fields: Dictionary of field definitions

    Returns:
        New Pydantic model class
    """
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    model_fields = {}
    for field_name, schema in fields.items():
        if isinstance(schema, dict):
            field_type = type_mapping.get(schema.get("type", "string"), Any)
            is_required = schema.get("required", True)
        else:
            field_type = schema
            is_required = True

        model_fields[field_name] = (
            field_type,
            Field(default=... if is_required else None),
        )

    return create_model(name, **model_fields)
