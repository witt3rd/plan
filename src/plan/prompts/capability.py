"""Templates for capability-related prompts"""

from plan.llm.templates import (
    PromptTemplate,
    TemplateMetadata,
    TemplateParameter,
    TemplateType,
)

CAPABILITY_TYPE_DECISION = PromptTemplate(
    name="capability_type_decision",
    type=TemplateType.SYSTEM,
    template="""
    Determine the most appropriate implementation type for this capability:

    Name: {{ name }}
    Required Inputs: {{ required_inputs }}
    Required Output: {{ required_output }}
    Context: {{ context }}

    Consider:
    1. Complexity and nature of the operation
    2. Need for external system interactions
    3. Data processing requirements
    4. Whether operation is primarily analytical or creative
    5. Performance requirements
    6. Potential for reuse and composition

    Available capability types:
    - TOOL: Python function for procedural/computational tasks
    - INSTRUCTION: Prompt template for creative/analytical tasks
    - PLAN: Composite workflow for complex multi-step operations

    Existing capabilities:
    {{ available_capabilities }}

    Provide:
    1. The most appropriate capability type
    2. Reasoning for this choice
    3. List of specific requirements
    4. List of suggested dependencies (if any)
    5. Any performance considerations
    """,
    parameters=[
        TemplateParameter(
            name="name", description="Name of the capability", type="str", required=True
        ),
        TemplateParameter(
            name="required_inputs",
            description="List of required input names",
            type="List[str]",
            required=True,
        ),
        TemplateParameter(
            name="required_output",
            description="Required output name",
            type="str",
            required=True,
        ),
        TemplateParameter(
            name="context",
            description="Additional context information",
            type="Dict[str, Any]",
            required=True,
        ),
        TemplateParameter(
            name="available_capabilities",
            description="List of existing capabilities",
            type="str",
            required=True,
        ),
    ],
    metadata=TemplateMetadata(
        description="Template for deciding capability implementation type",
        version="1.0.0",
        tags={"capability", "decision"},
    ),
)

TOOL_IMPLEMENTATION = PromptTemplate(
    name="tool_implementation",
    type=TemplateType.SYSTEM,
    template="""
    Create a Python function implementation for this capability:

    Name: {{ name }}
    Required Inputs: {{ required_inputs }}
    Required Output: {{ required_output }}
    Context: {{ context }}

    Requirements:
    {% for req in requirements %}
    - {{ req }}
    {% endfor %}

    The function should:
    1. Handle input validation
    2. Implement error handling
    3. Include type hints
    4. Be well-documented
    5. Follow Python best practices

    Provide:
    1. Function implementation
    2. Description of the implementation
    3. Test cases covering success and error scenarios
    4. Error cases that should be handled
    5. Any dependencies required
    """,
    parameters=[
        TemplateParameter(
            name="name", description="Name of the capability", type="str", required=True
        ),
        TemplateParameter(
            name="required_inputs",
            description="List of required input names",
            type="List[str]",
            required=True,
        ),
        TemplateParameter(
            name="required_output",
            description="Required output name",
            type="str",
            required=True,
        ),
        TemplateParameter(
            name="context",
            description="Additional context information",
            type="Dict[str, Any]",
            required=True,
        ),
        TemplateParameter(
            name="requirements",
            description="List of specific requirements",
            type="List[str]",
            required=True,
        ),
    ],
    metadata=TemplateMetadata(
        description="Template for generating tool implementations",
        version="1.0.0",
        tags={"capability", "tool", "implementation"},
    ),
)
