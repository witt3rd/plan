import json
from datetime import datetime
from typing import List, Optional

import pytest
from pydantic import BaseModel
from pytest_asyncio import fixture as async_fixture

from plan.llm.handler import (
    CompletionConfig,
    CompletionError,
    CompletionMetrics,
    ConversationContext,
    PromptHandler,
)
from plan.llm.tool import Annotated, Doc


# Sample Models (renamed from Test* to Sample* to avoid pytest collection)
class SampleModel(BaseModel):
    """Simple model for basic structured output testing"""

    value: str


class SampleDetails(BaseModel):
    """Details model for nested structured output"""

    description: str
    tags: List[str]


class SampleComplexModel(BaseModel):
    """Complex model for testing nested structured output"""

    name: str
    age: int
    details: SampleDetails

    model_config = {"json_schema_extra": {"required": ["name", "age", "details"]}}


class WeatherData(BaseModel):
    """Weather data model"""

    city: str
    country: str
    temperature: str
    condition: str


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


@async_fixture(scope="function")
async def handler():
    """Provide configured PromptHandler"""
    _handler = PromptHandler()
    async with _handler:
        try:
            yield _handler
        finally:
            # Cleanup conversations
            for conv_id in list(_handler._conversations.keys()):
                _handler.clear_conversation(conv_id)


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
        config = CompletionConfig(response_format=SampleModel)
        result = await handler.complete("Return a test value", config=config)
        assert isinstance(result, SampleModel)
        assert isinstance(result.value, str)

    @pytest.mark.asyncio
    async def test_complex_structured_output(self, handler):
        """Test complex structured output"""
        config = CompletionConfig(response_format=SampleComplexModel)
        result = await handler.complete(
            "Return a complex object with name John, age 30", config=config
        )
        assert isinstance(result, SampleComplexModel)
        assert isinstance(result.name, str)
        assert isinstance(result.age, int)
        assert isinstance(result.details, SampleDetails)


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

        config = CompletionConfig(response_format=SampleModel)
        result = await handler.complete(
            "Return a test value", config=config, conversation_id=conv_id
        )
        assert isinstance(result, SampleModel)

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
        """Test tool selection performance"""
        start_time = datetime.now()
        result = await handler.complete(
            "Check weather in Tokyo, Japan", tools=[TestTools.fetch_weather]
        )

        execution_time = (datetime.now() - start_time).total_seconds()
        assert result is not None
        assert len(result) > 0
        assert result[0].function.name == "fetch_weather"
        assert execution_time < 5  # Maximum acceptable time

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, handler):
        """Test concurrent tool selection performance"""
        import asyncio

        start_time = datetime.now()
        tasks = [
            handler.complete(
                f"Check weather in City{i}, Country{i}", tools=[TestTools.fetch_weather]
            )
            for i in range(3)
        ]
        responses = await asyncio.gather(*tasks)

        execution_time = (datetime.now() - start_time).total_seconds()
        assert all(isinstance(r, list) for r in responses)
        assert all(len(r) > 0 for r in responses)
        assert all(r[0].function.name == "fetch_weather" for r in responses)
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
        """Test tool parameter parsing with string input"""
        prompt = "Calculate age for birth year '1990'"  # String instead of int
        result = await handler.complete(prompt, tools=[TestTools.calculate_age])

        assert result is not None
        assert len(result) > 0
        tool_call = result[0]
        args = json.loads(tool_call.function.arguments)
        assert "birth_year" in args
        assert isinstance(args["birth_year"], int)  # Should be parsed as integer
        assert args["birth_year"] == 1990

    @pytest.mark.asyncio
    async def test_tool_null_handling(self, handler):
        """Test handling of optional parameters"""
        prompt = "Calculate age for birth year 1990 with no reference year"
        result = await handler.complete(prompt, tools=[TestTools.calculate_age])

        assert result is not None
        assert len(result) > 0
        tool_call = result[0]
        args = json.loads(tool_call.function.arguments)
        assert "birth_year" in args
        assert args["birth_year"] == 1990
        assert "reference_year" not in args or args["reference_year"] is None


@pytest.mark.unit
class TestPromptHandlerTools:
    """Tests for tool-based completion functionality"""

    @pytest.mark.asyncio
    async def test_single_tool_completion(self, handler):
        """Test completion with a single tool"""
        prompt = "What's the weather in London, UK?"
        result = await handler.complete(prompt, tools=[TestTools.fetch_weather])

        assert result is not None
        assert len(result) == 1
        tool_call = result[0]
        assert tool_call.function.name == "fetch_weather"

        args = json.loads(tool_call.function.arguments)
        assert "city" in args and args["city"].lower() == "london"
        assert "country" in args and args["country"].upper() == "UK"

    @pytest.mark.asyncio
    async def test_multiple_tools_completion(self, handler):
        """Test completion with multiple available tools"""
        prompt = """
        1. Check the weather in Paris, France
        2. Calculate the age of someone born in 1990
        """
        result = await handler.complete(
            prompt, tools=[TestTools.fetch_weather, TestTools.calculate_age]
        )

        assert result is not None
        assert len(result) == 2
        tool_names = {call.function.name for call in result}
        assert tool_names == {"fetch_weather", "calculate_age"}

        # Verify arguments without executing
        for tool_call in result:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "fetch_weather":
                assert args["city"].lower() == "paris"
                assert args["country"].lower() in {"france", "fr"}
            elif tool_call.function.name == "calculate_age":
                assert args["birth_year"] == 1990

    @pytest.mark.asyncio
    async def test_tool_in_conversation(self, handler):
        """Test tool selection in conversation context"""
        conv_id = "test_tools_conv"
        handler.create_conversation(conv_id, "gpt-4o-mini")

        # First message with tool
        result = await handler.complete(
            "What's the weather in Tokyo, Japan?",
            tools=[TestTools.fetch_weather],
            conversation_id=conv_id,
        )

        assert result is not None
        assert len(result) == 1
        tool_call = result[0]
        assert tool_call.function.name == "fetch_weather"
        args = json.loads(tool_call.function.arguments)
        assert args["city"].lower() == "tokyo"
        assert args["country"].lower() in {"japan", "jp"}

        # Add a mock weather result to conversation
        mock_weather = "Sunny, 22°C in Tokyo, Japan"
        handler._conversations[conv_id].add_message(
            "assistant", mock_weather, len(mock_weather) // 4
        )

        # Test that conversation context is maintained
        result2 = await handler.complete(
            "What was the temperature mentioned?", conversation_id=conv_id
        )
        assert isinstance(result2, str)
        assert "22" in result2

    @pytest.mark.asyncio
    async def test_tool_with_complex_types(self, handler):
        """Test tool handling complex input/output types"""
        prompt = "Process this data: [1, 2, 3] in JSON format"
        result = await handler.complete(prompt, tools=[TestTools.process_data])
        if result is not None:
            assert len(result) > 0
            assert all(call.function.name == "process_data" for call in result)

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, handler):
        """Test handling of invalid tool selection"""

        # Test with a tool that has invalid schema/signature
        def invalid_tool(x):  # Missing return type and param annotation
            """This tool has an invalid signature"""
            return x

        # Test with no matching tools for the prompt
        result = await handler.complete(
            "Tell me a joke",  # Prompt unrelated to weather or age calculation
            tools=[TestTools.fetch_weather, TestTools.calculate_age],
        )

        # Should return None when no tools match the prompt
        assert result is None

        # Test with invalid tool separately
        result = await handler.complete("Test invalid tool", tools=[invalid_tool])
        # Should return None for invalid tool
        assert result is None

    @pytest.mark.asyncio
    async def test_tool_with_optional_params(self, handler):
        """Test tool with optional parameters"""
        prompt = "Calculate age for birth year 1990"
        result = await handler.complete(prompt, tools=[TestTools.calculate_age])
        if result is not None:
            assert len(result) > 0
            assert all(call.function.name == "calculate_age" for call in result)

    @pytest.mark.asyncio
    async def test_tool_schema_validation(self, handler):
        """Test tool schema validation"""

        def strict_tool(
            numbers: Annotated[str, Doc("JSON string containing list of integers")],
        ) -> Annotated[int, Doc("Sum of the numbers")]:
            """Adds up a list of numbers"""
            import json

            nums = json.loads(numbers)
            return sum(nums)

        # Should handle JSON string input
        prompt = "Calculate sum of [1, 2, 3]"
        result = await handler.complete(prompt, tools=[strict_tool])
        if result is not None:
            assert len(result) > 0
            assert all(call.function.name == "strict_tool" for call in result)

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
        if result is not None:
            assert len(result) > 0
            tool_names = {call.function.name for call in result}
            assert tool_names & {"fetch_weather", "process_data"}
