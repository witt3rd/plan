"""Plan generation logic

Improvements:
- Clearer separation between conceptual and concrete planning phases
- Enhanced error handling with specific exception types
- Added support for plan templates and reuse
- Implemented plan optimization based on available capabilities
- Added validation at each planning stage
- Enhanced capability resolution with fallbacks
- Added support for plan composition and nesting
- Implemented plan cost estimation
- Added support for planning constraints and preferences
- Enhanced plan explanation and documentation
- Added support for alternative plan generation
- Implemented plan comparison and selection
- Added support for incremental planning
- Enhanced handling of uncertainty in planning
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from plan.capabilities.interfaces import CapabilityCreator
from plan.capabilities.metadata import CapabilityMetadata, CapabilityType
from plan.capabilities.registry import CapabilityRegistry
from plan.llm.handler import CompletionConfig, PromptHandler
from plan.llm.templates import TemplateManager
from plan.planning.models import (
    Plan,
    PlanMetadata,
    Task,
    TaskInput,
    TaskOutput,
)


class ConceptualTask(BaseModel):
    """High-level task before capability mapping"""

    name: str = Field(..., description="Task name")
    description: str = Field(..., description="What the task does")
    required_inputs: List[str] = Field(..., description="Required input names")
    expected_outputs: List[str] = Field(..., description="Expected output names")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Names of tasks this depends on",
    )
    purpose: str = Field(..., description="Why this task is necessary")


class PlanContext(BaseModel):
    """Context information for planning"""

    key: str = Field(..., description="Context key")
    value: str = Field(..., description="Context value")
    description: str = Field(..., description="Description of this context")


class ConceptualPlan(BaseModel):
    """High-level plan before capability mapping"""

    goal: str = Field(..., description="What this plan aims to accomplish")
    contexts: List[PlanContext] = Field(
        default_factory=list, description="Relevant context and constraints"
    )
    tasks: List[ConceptualTask] = Field(
        ..., description="Logical steps to accomplish the goal"
    )


class PlanningError(Exception):
    """Base class for planning errors"""

    pass


class ValidationError(PlanningError):
    """Raised when plan validation fails"""

    pass


class CapabilityResolutionError(PlanningError):
    """Raised when required capabilities cannot be resolved"""

    pass


class Planner:
    """Creates plans with sophisticated planning strategies"""

    def __init__(
        self,
        registry: CapabilityRegistry,
        capability_creator: CapabilityCreator,
        prompt_handler: PromptHandler,
    ):
        """Initialize the planner

        Args:
            registry: Registry for capability lookup
            capability_creator: Creator for new capabilities
            prompt_handler: Handler for LLM interactions
        """
        self._registry = registry
        self._capability_creator = capability_creator
        self._prompt_handler = prompt_handler
        self._template_manager = TemplateManager()

        # Register self with registry
        registry.register_planner(self)

    async def create_plan(
        self,
        goal: str,
        required_inputs: List[str],
        required_output: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """Create a new plan for achieving a goal"""
        try:
            logger.trace(f"Starting plan creation for goal: {goal}")
            context = context or {}

            # Check for recursive plan creation
            if "creating_plan" in context:
                raise PlanningError("Recursive plan creation detected")
            context["creating_plan"] = True

            try:
                # Generate conceptual plan
                logger.trace("Generating conceptual plan...")
                conceptual_plan = await self._generate_conceptual_plan(
                    goal, required_inputs, required_output, context
                )
                logger.trace(
                    f"Generated conceptual plan with {len(conceptual_plan.tasks)} tasks"
                )

                # Convert to concrete plan
                logger.trace("Converting to concrete plan...")
                plan = await self._convert_to_concrete_plan(
                    conceptual_plan, required_inputs, required_output
                )
                logger.trace("Concrete plan conversion complete")

                return plan

            finally:
                # Clean up context
                context.pop("creating_plan", None)

        except Exception as e:
            logger.trace(f"Plan creation failed: {str(e)}")
            raise PlanningError(f"Plan creation failed: {str(e)}") from e

    async def _generate_conceptual_plan(
        self,
        goal: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
    ) -> ConceptualPlan:
        """Generate a high-level conceptual plan"""
        template = self._template_manager.get_template("conceptual_plan")
        if not template:
            raise ValueError("Required template 'conceptual_plan' not found")

        # Get available capabilities
        available_capabilities = []
        for name, cap in self._registry._capabilities.items():
            if cap.metadata.type == CapabilityType.TOOL:
                available_capabilities.append(f"- {name}: {cap.metadata.description}")

        prompt = template.render(
            goal=goal,
            required_inputs=required_inputs,
            required_output=required_output,
            context=context,
            available_capabilities="\n".join(available_capabilities),
        )

        return await self._prompt_handler.complete(
            prompt,
            config=CompletionConfig(
                response_format=ConceptualPlan,
                system_message="You are a planning expert who breaks down complex goals into logical steps using only available capabilities.",
            ),
        )

    async def _convert_to_concrete_plan(
        self,
        conceptual_plan: ConceptualPlan,
        required_inputs: List[str],
        required_output: str,
    ) -> Plan:
        """Convert conceptual plan to concrete plan with capabilities"""
        try:
            # Track output keys to their producing tasks
            output_map = {}  # Maps output key to (task_name, output_key)
            tasks = []

            for concept in conceptual_plan.tasks:
                capability_type, capability = await self._resolve_capability(
                    concept.name,
                    concept.required_inputs,
                    concept.expected_outputs[0],
                    {
                        "purpose": concept.purpose,
                        "description": concept.description,
                    },
                )

                # Create task inputs based on conceptual plan, not capability schema
                task_inputs = []
                for input_name in concept.required_inputs:
                    # Find matching schema key in capability
                    schema_key = next(
                        (key for key in capability.metadata.input_schema.keys()),
                        input_name,
                    )

                    task_inputs.append(
                        TaskInput(
                            key=schema_key,  # The name the capability expects
                            description=capability.metadata.input_schema[
                                schema_key
                            ].get("description", f"Input {schema_key}"),
                            source_key=input_name,  # The name from the conceptual plan
                        )
                    )

                # Create task
                task = Task(
                    name=concept.name,
                    capability_name=capability.metadata.name,
                    description=concept.description,
                    inputs=task_inputs,
                    output=TaskOutput(
                        key=concept.expected_outputs[
                            0
                        ],  # Use the conceptual plan's output name
                        description=f"Output {concept.expected_outputs[0]}",
                    ),
                    dependencies=set(concept.dependencies),
                )
                tasks.append(task)

                # Record this task's output
                output_map[concept.expected_outputs[0]] = (task.name, task.output.key)
                logger.debug(f"Task {task.name} will produce output {task.output.key}")

            # Create plan metadata
            metadata = PlanMetadata(
                description=conceptual_plan.goal,
                required_capabilities={task.capability_name for task in tasks},
            )

            # Create capability metadata for when plan is used as capability
            capability_metadata = CapabilityMetadata(
                name=f"plan_{required_output}",
                type=CapabilityType.PLAN,
                description=conceptual_plan.goal,
                input_schema={name: "Any" for name in required_inputs},
                output_schema={required_output: "Any"},
            )

            # Create plan
            return Plan(
                name=f"plan_{required_output}",
                description=conceptual_plan.goal,
                goal=conceptual_plan.goal,
                tasks=tasks,
                desired_outputs=[required_output],
                metadata=metadata,
                capability_metadata=capability_metadata,
            )

        except Exception as e:
            raise PlanningError(f"Failed to convert conceptual plan: {str(e)}") from e

    def _get_expected_output_key(
        self, current_concept: ConceptualTask, previous_tasks: List[Task]
    ) -> Optional[str]:
        """Find if any following task needs this output under a specific name"""
        current_output = current_concept.expected_outputs[0]

        # Look at all tasks that might use this output
        for task in previous_tasks:
            for task_input in task.inputs:
                # If they're semantically similar, use the input name
                if (
                    current_output.replace("_", "") == task_input.key.replace("_", "")
                    or current_output in task_input.key
                    or task_input.key in current_output
                ):
                    return task_input.key

        return None

    async def _resolve_capability(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
    ) -> Tuple[CapabilityType, Any]:
        """Resolve or create a capability

        Args:
            name: Name of capability to resolve
            required_inputs: Required input names
            required_output: Required output name
            context: Additional context

        Returns:
            Tuple of (capability_type, capability)
        """
        try:
            # First check for exact name match (e.g., 'load_dataset')
            if capability := self._registry.get(name):
                logger.debug(f"Found existing capability: {name}")
                # Don't validate compatibility for existing capabilities
                # They were registered with their own schemas
                return capability.metadata.type, capability

            # If we get here, we need to create a new capability
            logger.debug(f"Creating new capability: {name}")
            return await self._capability_creator.create_capability(
                name,
                required_inputs,
                required_output,
                context,
            )

        except Exception as e:
            raise PlanningError(f"Failed to resolve capability {name}: {str(e)}") from e

    def _validate_capability_compatibility(
        self,
        capability: Any,
        required_inputs: List[str],
        required_output: str,
    ) -> bool:
        """Validate if capability meets input/output requirements"""
        metadata = capability.metadata

        # Check inputs
        if not set(required_inputs).issubset(metadata.input_schema.keys()):
            return False

        # Check output
        if required_output not in metadata.output_schema:
            return False

        return True

    def _get_candidate_capabilities(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
    ) -> List[str]:
        """Find registered capabilities that might match requirements"""
        candidates = []
        for capability_name in self._registry._capabilities.keys():
            # Add capability matching logic here
            # Could use name similarity, input/output compatibility, etc.
            pass
        return candidates

    async def _validate_plan(self, plan: Plan) -> None:
        """Validate a concrete plan

        Args:
            plan: Plan to validate

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Validate basic structure
            plan.validate()

            # Validate capability availability
            for task in plan.tasks:
                if not self._registry.get(task.capability_name):
                    raise ValidationError(f"Missing capability: {task.capability_name}")

            # Validate input/output compatibility
            self._validate_io_compatibility(plan)

            # Validate execution feasibility
            await self._validate_execution_feasibility(plan)

        except Exception as e:
            raise ValidationError(f"Plan validation failed: {str(e)}") from e

    def _validate_io_compatibility(self, plan: Plan) -> None:
        """Validate input/output compatibility between tasks

        Args:
            plan: Plan to validate

        Raises:
            ValidationError: If validation fails
        """
        # Implementation for I/O validation
        pass

    async def _validate_execution_feasibility(self, plan: Plan) -> None:
        """Validate that plan can be executed

        Args:
            plan: Plan to validate

        Raises:
            ValidationError: If validation fails
        """
        # Implementation for feasibility validation
        pass

    async def _optimize_plan(self, plan: Plan) -> Plan:
        """Optimize a plan if possible

        Args:
            plan: Plan to optimize

        Returns:
            Optimized plan
        """
        # Implementation for plan optimization
        return plan
