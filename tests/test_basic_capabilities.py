from datetime import UTC, datetime

import pytest

from plan.capabilities.metadata import CapabilityMetadata, CapabilityType
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.tool import ToolCapability


@pytest.fixture
def registry():
    return CapabilityRegistry()


@pytest.fixture
def basic_tools():
    def get_project_request(project_id: str) -> str:
        """Retrieves project request details"""
        return f"Project request details for {project_id}"

    def analyze_sentiment(text: str) -> str:
        """Analyzes text sentiment"""
        return "positive"

    def generate_response(request_text: str, sentiment: str) -> str:
        """Generates a response based on request and sentiment"""
        return f"Thank you for your request. Based on your {sentiment} message: {request_text}"

    return {
        "get_project_request": get_project_request,
        "analyze_sentiment": analyze_sentiment,
        "generate_response": generate_response,
    }


class TestBasicCapabilities:
    def test_tool_registration(self, registry, basic_tools):
        # Create and register a tool capability
        metadata = CapabilityMetadata(
            name="get_project_request",
            type=CapabilityType.TOOL,
            created_at=datetime.now(UTC),
            description="Retrieves project request details",
            input_schema={"project_id": "string"},
            output_schema={"result": "string"},
        )

        tool = ToolCapability(basic_tools["get_project_request"], metadata)
        registry.register("get_project_request", tool)

        # Verify registration
        assert registry.get("get_project_request") is not None

    @pytest.mark.asyncio
    async def test_tool_execution(self, registry, basic_tools):
        # Create and register tool
        metadata = CapabilityMetadata(
            name="get_project_request",
            type=CapabilityType.TOOL,
            created_at=datetime.now(UTC),
            description="Retrieves project request details",
            input_schema={"project_id": "string"},
            output_schema={"result": "string"},
        )

        tool = ToolCapability(basic_tools["get_project_request"], metadata)
        registry.register("get_project_request", tool)

        # Execute tool
        result = await tool.execute({"project_id": "TEST123"})
        assert "TEST123" in result
