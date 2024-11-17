from datetime import datetime
from typing import List, Optional

import pytest
from function_schema import Annotated, Doc
from pydantic import BaseModel, Field

from plan.llm.handler import (
    CompletionConfig,
    CompletionError,
    CompletionMetrics,
    ConversationContext,
    PromptHandler,
)


# Test Models
class TestModel(BaseModel):
    """Simple model for basic structured output testing"""

    value: str


class Details(BaseModel):
    """Details model for nested structured output"""

    description: str
    tags: List[str]


class ComplexModel(BaseModel):
    """Complex model for testing nested structured output"""

    name: str
    age: int
    details: Details

    model_config = {"json_schema_extra": {"required": ["name", "age", "details"]}}


# Test Tools
class TestTools:
    """Collection of tools for testing"""

    @staticmethod
    async def fetch_weather(
        city: Annotated[str, Doc("Name of the city")],
        country: Annotated[str, Doc("Country code")],
    ) -> Annotated[str, Doc("Weather description")]:
        """Test tool: Weather fetching"""
        return f"Sunny, 22°C in {city}, {country}"

    @staticmethod
    def calculate_age(
        birth_year: Annotated[int, Doc("Year of birth")],
        reference_year: Annotated[Optional[int], Doc("Reference year")] = None,
    ) -> Annotated[int, Doc("Age in years")]:
        """Test tool: Age calculation"""
        ref_year = reference_year or datetime.now().year
        return ref_year - birth_year

    @staticmethod
    async def process_data(
        data: Annotated[str, Doc("Data to process (as JSON string)")],
        format: Annotated[str, Doc("Output format")] = "json",
    ) -> Annotated[dict, Doc("Processed data")]:
        """Test tool: Data processing"""
        import json

        try:
            parsed_data = json.loads(data) if isinstance(data, str) else data
            return {"processed": parsed_data, "format": format}
        except json.JSONDecodeError:
            return {"processed": str(data), "format": format}


@pytest.fixture
async def handler():
    """Provide configured PromptHandler"""
    handler = PromptHandler()
    yield handler
    # Cleanup conversations
    for conv_id in list(handler._conversations.keys()):
        handler.clear_conversation(conv_id)


@pytest.mark.unit
class TestPromptHandlerCore:
    """Unit tests for core PromptHandler functionality"""

    @pytest.mark.asyncio
    async def test_handler_initialization(self, handler):
        """Verify handler initializes correctly"""
        assert handler.client is not None
        assert isinstance(handler._conversations, dict)
        assert isinstance(handler._metrics, list)

    @pytest.mark.asyncio
    async def test_config_validation(self, handler):
        """Test configuration validation"""
        with pytest.raises(ValueError):
            CompletionConfig(temperature=3.0)  # Invalid temperature
        with pytest.raises(ValueError):
            CompletionConfig(max_tokens=-1)  # Invalid token count

    @pytest.mark.asyncio
    async def test_input_validation(self, handler):
        """Test input validation"""
        for invalid_input in [None, "", "   ", {"invalid": "type"}]:
            with pytest.raises(ValueError):
                await handler.complete(invalid_input)


@pytest.mark.unit
class TestPromptHandlerBasicCompletion:
    """Tests for basic text completion"""

    @pytest.mark.asyncio
    async def test_simple_completion(self, handler):
        """Test basic text completion"""
        result = await handler.complete("Test prompt")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_completion_with_system_message(self, handler):
        """Test completion with system message"""
        config = CompletionConfig(system_message="You are a helpful assistant.")
        result = await handler.complete("Test prompt", config=config)
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.unit
class TestPromptHandlerStructuredOutput:
    """Tests for structured output handling"""

    @pytest.mark.asyncio
    async def test_simple_structured_output(self, handler):
        """Test basic structured output"""
        config = CompletionConfig(response_format=TestModel)
        result = await handler.complete("Return a test value", config=config)
        assert isinstance(result, TestModel)
        assert isinstance(result.value, str)

    @pytest.mark.asyncio
    async def test_complex_structured_output(self, handler):
        """Test complex structured output"""
        config = CompletionConfig(response_format=ComplexModel)
        result = await handler.complete(
            "Return a complex object with name John, age 30", config=config
        )
        assert isinstance(result, ComplexModel)
        assert isinstance(result.name, str)
        assert isinstance(result.age, int)
        assert isinstance(result.details, Details)


@pytest.mark.unit
class TestPromptHandlerConversation:
    """Tests for conversation handling"""

    def test_conversation_creation(self, handler):
        """Test conversation context creation"""
        conv_id = "test_conv"
        handler.create_conversation(conv_id, "gpt-4o-mini")
        context = handler.get_conversation(conv_id)
        assert isinstance(context, ConversationContext)
        assert context.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_conversation_flow(self, handler):
        """Test conversation context maintenance"""
        conv_id = "test_conv"
        handler.create_conversation(conv_id, "gpt-4o-mini")

        # First message
        await handler.complete("My name is Alice", conversation_id=conv_id)

        # Follow-up
        result = await handler.complete("What's my name?", conversation_id=conv_id)
        assert "alice" in result.lower()


@pytest.mark.unit
class TestPromptHandlerMetrics:
    """Tests for metrics collection"""

    @pytest.mark.asyncio
    async def test_metrics_recording(self, handler):
        """Test basic metrics recording"""
        await handler.complete("Test prompt")
        metrics = handler.get_metrics()
        assert len(metrics) == 1
        assert isinstance(metrics[0], CompletionMetrics)
        assert metrics[0].total_tokens > 0

    @pytest.mark.asyncio
    async def test_metrics_accuracy(self, handler):
        """Test metrics calculation accuracy"""
        start_time = datetime.now()
        await handler.complete("Test prompt")
        metrics = handler.get_metrics()

        assert metrics[0].execution_time > 0
        assert metrics[0].prompt_tokens > 0
        assert metrics[0].completion_tokens > 0
        assert metrics[0].total_tokens == (
            metrics[0].prompt_tokens + metrics[0].completion_tokens
        )


@pytest.mark.integration
class TestPromptHandlerIntegration:
    """Integration tests for PromptHandler"""

    @pytest.mark.asyncio
    async def test_structured_conversation(self, handler):
        """Test structured output in conversation context"""
        conv_id = "test_conv"
        handler.create_conversation(conv_id, "gpt-4o-mini")

        config = CompletionConfig(response_format=TestModel)
        result = await handler.complete(
            "Return a test value", config=config, conversation_id=conv_id
        )
        assert isinstance(result, TestModel)

    @pytest.mark.asyncio
    async def test_error_recovery(self, handler):
        """Test error handling and recovery"""
        # Force an error
        with pytest.raises(CompletionError):
            await handler.complete(
                "Test prompt", config=CompletionConfig(model="invalid-model")
            )

        # Verify system recovers
        result = await handler.complete("Test prompt")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tool_with_structured_output(self, handler):
        """Test combining tool execution with structured output"""

        class WeatherReport(BaseModel):
            location: str = Field(..., description="City name")
            temperature: float = Field(..., description="Temperature in Celsius")
            conditions: str = Field(..., description="Weather conditions")

        config = CompletionConfig(response_format=WeatherReport)
        result = await handler.complete(
            "Get weather report for London, UK",
            config=config,
            tools=[TestTools.fetch_weather],
        )
        assert isinstance(result, WeatherReport)
        assert "London" in result.location
        assert result.temperature == 22.0
        assert "sunny" in result.conditions.lower()

    @pytest.mark.asyncio
    async def test_tool_conversation_memory(self, handler):
        """Test tool results in conversation memory"""
        conv_id = "test_tool_memory"
        handler.create_conversation(conv_id, "gpt-4o-mini")

        # Use tool and store result
        await handler.complete(
            "Check weather in Paris, France",
            tools=[TestTools.fetch_weather],
            conversation_id=conv_id,
        )

        # Verify tool result is in conversation memory
        context = handler.get_conversation(conv_id)
        assert any(
            "paris" in str(msg.get("content", "")).lower() for msg in context.messages
        )


@pytest.mark.performance
class TestPromptHandlerPerformance:
    """Performance tests for PromptHandler"""

    @pytest.mark.asyncio
    async def test_completion_timing(self, handler):
        """Test completion performance"""
        start_time = datetime.now()
        await handler.complete("Test prompt")
        execution_time = (datetime.now() - start_time).total_seconds()

        assert execution_time < 10  # Maximum acceptable time

    @pytest.mark.asyncio
    async def test_concurrent_completions(self, handler):
        """Test concurrent completion performance"""
        import asyncio

        start_time = datetime.now()
        tasks = [handler.complete("Test prompt") for _ in range(5)]
        results = await asyncio.gather(*tasks)
        execution_time = (datetime.now() - start_time).total_seconds()

        assert all(isinstance(r, str) for r in results)
        assert execution_time < 20  # Maximum acceptable time

    @pytest.mark.asyncio
    async def test_tool_execution_timing(self, handler):
        """Test tool execution performance"""
        start_time = datetime.now()
        await handler.complete(
            "Check weather in Tokyo, Japan", tools=[TestTools.fetch_weather]
        )
        execution_time = (datetime.now() - start_time).total_seconds()
        assert execution_time < 5  # Maximum acceptable time

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, handler):
        """Test concurrent tool execution performance"""
        import asyncio

        start_time = datetime.now()
        tasks = [
            handler.complete(
                f"Check weather in City{i}, Country{i}", tools=[TestTools.fetch_weather]
            )
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)
        execution_time = (datetime.now() - start_time).total_seconds()

        assert all(isinstance(r, str) for r in results)
        assert execution_time < 10  # Maximum acceptable time


@pytest.mark.regression
class TestPromptHandlerRegression:
    """Regression tests for known issues"""

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, handler):
        """Test handling of empty responses"""
        config = CompletionConfig(max_tokens=1)  # Force minimal response
        result = await handler.complete("Test prompt", config=config)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_tool_type_conversion(self, handler):
        """Test tool parameter type conversion"""
        prompt = "Calculate age for birth year '1990'"  # String instead of int
        result = await handler.complete(prompt, tools=[TestTools.calculate_age])
        assert isinstance(result, str)
        assert str(datetime.now().year - 1990) in result

    @pytest.mark.asyncio
    async def test_tool_null_handling(self, handler):
        """Test handling of null/None values in tool parameters"""
        prompt = "Calculate age for birth year 1990 with no reference year"
        result = await handler.complete(prompt, tools=[TestTools.calculate_age])
        assert isinstance(result, str)
        assert str(datetime.now().year - 1990) in result


@pytest.mark.unit
class TestPromptHandlerTools:
    """Tests for tool-based completion functionality"""

    @pytest.mark.asyncio
    async def test_single_tool_execution(self, handler):
        """Test execution of a single tool"""
        prompt = "What's the weather in London, UK?"
        result = await handler.complete(prompt, tools=[TestTools.fetch_weather])
        assert isinstance(result, str)
        assert all(term in result.lower() for term in ["london", "sunny"])

    @pytest.mark.asyncio
    async def test_multiple_tools(self, handler):
        """Test execution with multiple available tools"""
        prompt = """
        1. Check the weather in Paris, France
        2. Calculate the age of someone born in 1990
        """
        result = await handler.complete(
            prompt, tools=[TestTools.fetch_weather, TestTools.calculate_age]
        )
        assert isinstance(result, str)
        assert all(term in result.lower() for term in ["paris", "sunny"])
        assert any(str(age) in result for age in range(30, 35))

    @pytest.mark.asyncio
    async def test_tool_with_complex_types(self, handler):
        """Test tool handling complex input/output types"""
        prompt = "Process this data: [1, 2, 3] in JSON format"
        result = await handler.complete(prompt, tools=[TestTools.process_data])
        assert isinstance(result, str)
        assert "processed" in result.lower()
        assert "json" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, handler):
        """Test handling of tool execution errors"""

        async def failing_tool(
            input: Annotated[str, Doc("Input that will fail")],
        ) -> str:
            raise ValueError("Tool execution failed")

        with pytest.raises(CompletionError):
            await handler.complete("This should fail", tools=[failing_tool])

    @pytest.mark.asyncio
    async def test_tool_in_conversation(self, handler):
        """Test tool usage in conversation context"""
        conv_id = "test_tools_conv"
        handler.create_conversation(conv_id, "gpt-4o-mini")

        # First message with tool
        result1 = await handler.complete(
            "What's the weather in Tokyo, Japan?",
            tools=[TestTools.fetch_weather],
            conversation_id=conv_id,
        )
        assert "tokyo" in result1.lower()

        # Follow-up referring to previous result
        result2 = await handler.complete(
            "What was the temperature mentioned?", conversation_id=conv_id
        )
        assert "22" in result2

    @pytest.mark.asyncio
    async def test_tool_with_optional_params(self, handler):
        """Test tool with optional parameters"""
        prompt = "Calculate age for birth year 1990"
        result = await handler.complete(prompt, tools=[TestTools.calculate_age])
        assert isinstance(result, str)
        current_year = datetime.now().year
        expected_age = current_year - 1990
        assert str(expected_age) in result

    @pytest.mark.asyncio
    async def test_tool_schema_validation(self, handler):
        """Test tool schema validation"""

        # Tool with strict typing
        def strict_tool(numbers: Annotated[List[int], Doc("List of integers")]) -> int:
            return sum(numbers)

        # Should handle type conversion
        prompt = "Calculate sum of [1, 2, 3]"
        result = await handler.complete(prompt, tools=[strict_tool])
        assert isinstance(result, str)
        assert "6" in result

    @pytest.mark.asyncio
    async def test_tool_chaining(self, handler):
        """Test sequential tool execution"""
        prompt = """
        1. Get weather in Berlin, Germany
        2. Process that information as JSON
        """
        result = await handler.complete(
            prompt, tools=[TestTools.fetch_weather, TestTools.process_data]
        )
        assert isinstance(result, str)
        assert all(term in result.lower() for term in ["berlin", "processed"])
