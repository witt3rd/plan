# File: examples/basic_workflow.py

"""
Basic workflow example demonstrating core functionality:
1. Creating basic capabilities
2. Generating a plan
3. Executing the plan
"""

import asyncio
from datetime import datetime

from plan.capabilities.metadata import CapabilityMetadata, CapabilityType
from plan.capabilities.orchestrator import CapabilityOrchestrator
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.schema import Schema, SchemaField, SchemaType
from plan.capabilities.tool import ToolCapability
from plan.llm.handler import PromptHandler


async def basic_workflow_example():
    """Demonstrates basic workflow using minimal example"""

    # 1. Create basic components
    prompt_handler = PromptHandler()
    registry = CapabilityRegistry()
    orchestrator = CapabilityOrchestrator(
        prompt_handler=prompt_handler, registry=registry
    )

    # 2. Define and register basic capabilities
    def get_project_request(project_id: str) -> str:
        """Retrieves project request details"""
        return f"Project request details for {project_id}"

    def analyze_sentiment(text: str) -> str:
        """Analyzes text sentiment"""
        return "positive"

    def generate_response(request_text: str, sentiment: str) -> str:
        """Generates a response based on request and sentiment"""
        return f"Thank you for your request. Based on your {sentiment} message: {request_text}"

    # Register capabilities with metadata using new schema system
    registry.register(
        "get_project_request",
        ToolCapability(
            get_project_request,
            CapabilityMetadata(
                name="get_project_request",
                type=CapabilityType.TOOL,
                created_at=datetime.now(),
                description="Retrieves project request details",
                input_schema=Schema(
                    fields={
                        "project_id": SchemaField(
                            type=SchemaType.STRING,
                            description="Project identifier",
                            required=True,
                        )
                    }
                ),
                output_schema=Schema(
                    fields={
                        "result": SchemaField(
                            type=SchemaType.STRING,
                            description="Project request details",
                            required=True,
                        )
                    }
                ),
            ),
        ),
    )

    registry.register(
        "analyze_sentiment",
        ToolCapability(
            analyze_sentiment,
            CapabilityMetadata(
                name="analyze_sentiment",
                type=CapabilityType.TOOL,
                created_at=datetime.now(),
                description="Analyzes text sentiment",
                input_schema=Schema(
                    fields={
                        "text": SchemaField(
                            type=SchemaType.STRING,
                            description="Text to analyze",
                            required=True,
                        )
                    }
                ),
                output_schema=Schema(
                    fields={
                        "result": SchemaField(
                            type=SchemaType.STRING,
                            description="Sentiment analysis result",
                            required=True,
                        )
                    }
                ),
            ),
        ),
    )

    registry.register(
        "generate_response",
        ToolCapability(
            generate_response,
            CapabilityMetadata(
                name="generate_response",
                type=CapabilityType.TOOL,
                created_at=datetime.now(),
                description="Generates a response based on request and sentiment",
                input_schema=Schema(
                    fields={
                        "request_text": SchemaField(
                            type=SchemaType.STRING,
                            description="Original request text",
                            required=True,
                        ),
                        "sentiment": SchemaField(
                            type=SchemaType.STRING,
                            description="Analyzed sentiment",
                            required=True,
                        ),
                    }
                ),
                output_schema=Schema(
                    fields={
                        "result": SchemaField(
                            type=SchemaType.STRING,
                            description="Generated response",
                            required=True,
                        )
                    }
                ),
            ),
        ),
    )

    # 3. Create a simple plan
    plan = await orchestrator.create_plan(
        goal="Process a project request and generate a response",
        required_inputs=["project_id"],
        required_output="response",
        context={
            "requirements": "Must retrieve request and generate appropriate response"
        },
    )

    # 4. Execute plan
    result = await orchestrator.execute_capability(
        name=plan.name, inputs={"project_id": "PROJ123"}
    )

    print("\nPlan Execution Results:")
    print(f"Response: {result}")

    return result


if __name__ == "__main__":
    asyncio.run(basic_workflow_example())
