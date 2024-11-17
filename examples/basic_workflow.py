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
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.tool import ToolCapability
from plan.llm.handler import PromptHandler
from plan.orchestration.orchestrator import CapabilityOrchestrator


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

    # Register capabilities with metadata
    registry.register(
        "get_project_request",
        ToolCapability(
            get_project_request,
            CapabilityMetadata(
                name="get_project_request",
                type=CapabilityType.TOOL,
                created_at=datetime.now().isoformat(),
                description="Retrieves project request details",
                input_schema={"project_id": "string"},
                output_schema={"result": "string"},
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
