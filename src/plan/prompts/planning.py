"""Templates for planning-related prompts"""

from plan.llm.templates import (
    PromptTemplate,
    TemplateMetadata,
    TemplateParameter,
    TemplateType,
)

CONCEPTUAL_PLAN = PromptTemplate(
    name="conceptual_plan",
    type=TemplateType.SYSTEM,
    template="""
    Create a detailed, logical plan to accomplish this goal:

    GOAL:
    {{ goal }}

    AVAILABLE INPUTS:
    {{ required_inputs }}

    REQUIRED OUTPUT:
    {{ required_output }}

    AVAILABLE CAPABILITIES:
    {{ available_capabilities }}

    CONTEXT:
    {{ context }}

    Create a structured plan that:
    1. Breaks down the goal into logical steps using ONLY the available capabilities listed above
    2. Identifies dependencies between steps
    3. Specifies required information for each step
    4. Describes what each step produces
    5. Explains why each step is necessary

    The plan should:
    - Use only available inputs or outputs from previous steps
    - Use only the listed available capabilities
    - Ensure the final output is produced
    - Be logically complete and coherent
    - Handle potential error cases
    - Be efficient and avoid redundant steps

    IMPORTANT: Task names MUST exactly match the available capability names.

    Return the plan as a structured object with:
    - A clear goal statement
    - A list of relevant context items (each with a key, value, and description)
    - A list of conceptual tasks (each with name, description, inputs, outputs, dependencies, and purpose)
    """,
    parameters=[
        TemplateParameter(
            name="goal", description="Goal to accomplish", type="str", required=True
        ),
        TemplateParameter(
            name="required_inputs",
            description="Available input values",
            type="List[str]",
            required=True,
        ),
        TemplateParameter(
            name="required_output",
            description="Required output value",
            type="str",
            required=True,
        ),
        TemplateParameter(
            name="available_capabilities",
            description="List of available capabilities",
            type="str",
            required=True,
        ),
        TemplateParameter(
            name="context",
            description="Additional context information",
            type="Dict[str, Any]",
            required=True,
        ),
    ],
    metadata=TemplateMetadata(
        description="Template for generating conceptual plans",
        version="1.0.0",
        tags={"planning", "conceptual"},
    ),
)
