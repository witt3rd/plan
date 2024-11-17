"""Templates for instruction-related prompts"""

from plan.llm.templates import (
    PromptTemplate,
    TemplateMetadata,
    TemplateParameter,
    TemplateType,
)

INSTRUCTION_TEMPLATE = PromptTemplate(
    name="instruction_template",
    type=TemplateType.SYSTEM,
    template="""
    Create a prompt template for:

    Name: {{ name }}
    Required Inputs: {{ required_inputs }}
    Required Output: {{ required_output }}
    Context: {{ context }}
    Requirements: {{ requirements }}

    The template should:
    1. Clearly specify required inputs
    2. Provide clear guidance for the model
    3. Include validation criteria
    4. Handle potential error cases
    5. Produce well-structured output

    Include:
    1. The prompt template with input placeholders
    2. Description of the template's purpose
    3. Example inputs and expected outputs
    4. Validation criteria for outputs
    5. Potential error cases to handle

    The template should be:
    - Clear and unambiguous
    - Focused on the specific task
    - Reusable with different inputs
    - Robust to edge cases
    """,
    parameters=[
        TemplateParameter(
            name="name", description="Name of the capability", type="str", required=True
        ),
        TemplateParameter(
            name="required_inputs",
            description="Required input names",
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
            description="Additional context",
            type="Dict[str, Any]",
            required=True,
        ),
        TemplateParameter(
            name="requirements",
            description="Specific requirements",
            type="List[str]",
            required=True,
        ),
    ],
    metadata=TemplateMetadata(
        description="Template for creating instruction templates",
        version="1.0.0",
        tags={"instruction", "template"},
    ),
)

INSTRUCTION_EXECUTION = PromptTemplate(
    name="instruction_execution",
    type=TemplateType.SYSTEM,
    template="""
    Execute the following instruction template with the given inputs:

    TEMPLATE:
    {{ template }}

    INPUTS:
    {% for key, value in input.items() %}
    {{ key }}: {{ value }}
    {% endfor %}

    {% if examples %}
    EXAMPLES:
    {% for example in examples %}
    Input: {{ example.inputs }}
    Output: {{ example.output }}
    {% if example.explanation %}
    Explanation: {{ example.explanation }}
    {% endif %}
    {% endfor %}
    {% endif %}

    {% if validation_rules %}
    VALIDATION RULES:
    {% for rule in validation_rules %}
    - {{ rule }}
    {% endfor %}
    {% endif %}

    The output should:
    1. Follow the template format exactly
    2. Use only the provided inputs
    3. Meet all validation rules
    4. Be clear and well-structured
    """,
    parameters=[
        TemplateParameter(
            name="template",
            description="The instruction template to execute",
            type="str",
            required=True,
        ),
        TemplateParameter(
            name="input",
            description="Input values for execution",
            type="Dict[str, Any]",
            required=True,
        ),
        TemplateParameter(
            name="examples",
            description="Optional examples for few-shot learning",
            type="List[PromptExample]",
            required=False,
        ),
        TemplateParameter(
            name="validation_rules",
            description="Rules for validating the output",
            type="List[str]",
            required=False,
        ),
    ],
    metadata=TemplateMetadata(
        description="Template for executing instruction templates",
        version="1.0.0",
        tags={"instruction", "execution"},
    ),
)
