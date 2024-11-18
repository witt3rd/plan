from datetime import UTC, datetime

import pytest

from plan.capabilities.metadata import CapabilityMetadata, CapabilityType
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.schema import Schema, SchemaField, SchemaType
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
        )

        tool = ToolCapability(basic_tools["get_project_request"], metadata)
        registry.register("get_project_request", tool)

        # Execute tool
        result = await tool.execute({"project_id": "TEST123"})
        assert "TEST123" in result

    @pytest.mark.asyncio
    async def test_tool_validation(self, registry, basic_tools):
        # Create tool with validation rules
        metadata = CapabilityMetadata(
            name="analyze_sentiment",
            type=CapabilityType.TOOL,
            created_at=datetime.now(UTC),
            description="Analyzes text sentiment",
            input_schema=Schema(
                fields={
                    "text": SchemaField(
                        type=SchemaType.STRING,
                        description="Text to analyze",
                        required=True,
                        validation_rules=["len(value) > 0"],
                    )
                }
            ),
            output_schema=Schema(
                fields={
                    "result": SchemaField(
                        type=SchemaType.STRING,
                        description="Sentiment analysis result",
                        required=True,
                        validation_rules=[
                            "value in ['positive', 'negative', 'neutral']"
                        ],
                    )
                }
            ),
        )

        tool = ToolCapability(basic_tools["analyze_sentiment"], metadata)
        registry.register("analyze_sentiment", tool)

        # Test valid input
        result = await tool.execute({"text": "Great service!"})
        assert result in ["positive", "negative", "neutral"]

        # Test invalid input
        with pytest.raises(Exception):
            await tool.execute({"text": ""})

    @pytest.mark.asyncio
    async def test_tool_with_optional_params(self, registry, basic_tools):
        # Create tool with optional parameters
        metadata = CapabilityMetadata(
            name="generate_response",
            type=CapabilityType.TOOL,
            created_at=datetime.now(UTC),
            description="Generates a response",
            input_schema=Schema(
                fields={
                    "request_text": SchemaField(
                        type=SchemaType.STRING,
                        description="Request text",
                        required=True,
                    ),
                    "sentiment": SchemaField(
                        type=SchemaType.STRING,
                        description="Sentiment",
                        required=False,
                        default="neutral",
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
        )

        tool = ToolCapability(basic_tools["generate_response"], metadata)
        registry.register("generate_response", tool)

        # Test with all parameters
        result = await tool.execute(
            {"request_text": "Help needed", "sentiment": "positive"}
        )
        assert "positive" in result

        # Test with default parameter
        result = await tool.execute({"request_text": "Help needed"})
        assert "neutral" in result

    def test_schema_validation(self):
        # Test schema field validation
        with pytest.raises(ValueError):
            SchemaField(
                type="invalid_type",  # Should be SchemaType enum
                description="Test field",
                required=True,
            )

        # Test schema creation
        schema = Schema(
            fields={
                "test_field": SchemaField(
                    type=SchemaType.STRING,
                    description="Test field",
                    required=True,
                )
            }
        )
        assert "test_field" in schema.fields

        # Test schema conversion
        pydantic_fields = schema.to_pydantic_fields()
        assert "test_field" in pydantic_fields
