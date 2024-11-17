from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from plan.capabilities.base import CapabilityResolutionError
from plan.capabilities.factory import (
    CapabilityFactory,
    CapabilityTypeDecision,
    ToolSpecification,
)
from plan.capabilities.metadata import CapabilityMetadata, CapabilityType
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.tool import ToolCapability
from plan.llm.handler import PromptHandler


@pytest.fixture
def registry():
    """Create a test registry"""
    return CapabilityRegistry()


@pytest.fixture
def prompt_handler():
    """Create a mock prompt handler"""
    handler = Mock(spec=PromptHandler)
    handler.complete = AsyncMock()

    # Configure mock to return proper objects based on input
    async def mock_complete(instruction, config=None):
        # Basic input validation
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Invalid instruction")

        if "Determine the most appropriate implementation type" in instruction:
            # Parse instruction to extract values
            if (
                '"name": ""' in instruction.lower()
                or '"name":""' in instruction.lower()
            ):
                raise ValueError("Invalid capability name")

            if '"required_inputs": []' in instruction.lower() or "[]" in instruction:
                raise ValueError("No inputs specified")

            if '"required_output": ""' in instruction.lower():
                raise ValueError("Invalid output")

            return CapabilityTypeDecision(
                capability_type=CapabilityType.TOOL,
                reasoning="Test reasoning",
                requirements=["test requirement"],
                suggested_dependencies=[],
                performance_notes="Test performance notes",
            )
        elif "Create a Python function implementation" in instruction:
            # Validate required inputs exist
            if '"required_inputs": []' in instruction.lower():
                raise ValueError("No inputs specified")

            return ToolSpecification(
                function_name="test_function",
                description="Test function",
                implementation="""
def test_function(input1: str) -> str:
    '''Test function'''
    return input1
""",
                test_cases=["test case"],
                error_cases=["error case"],
            )

        # For any other instruction type, raise an error
        raise ValueError(f"Unsupported instruction type: {instruction}")

    handler.complete.side_effect = mock_complete
    return handler


@pytest.fixture
def factory(registry, prompt_handler):
    """Create a test factory"""
    return CapabilityFactory(prompt_handler, registry)


class TestCapabilityResolution:
    @pytest.mark.asyncio
    async def test_resolve_existing_capability(self, factory, registry):
        """Test resolving an existing compatible capability"""
        # Create test capability with matching function signature
        metadata = CapabilityMetadata(
            name="test_cap",
            type=CapabilityType.TOOL,
            description="Test capability",
            input_schema={"input1": {"type": "string"}},
            output_schema={"output1": {"type": "string"}},
            created_at=datetime.now(),
        )

        def test_func(input1: str) -> str:
            """Test function"""
            return input1

        capability = ToolCapability(test_func, metadata)
        registry.register("test_cap", capability)

        # Test resolution
        cap_type, resolved = await factory.resolve_capability(
            "test_cap", ["input1"], "output1", {}
        )

        # Verify same capability returned
        assert cap_type == CapabilityType.TOOL
        assert resolved == capability

    @pytest.mark.asyncio
    async def test_resolve_incompatible_capability(self, factory, registry):
        """Test handling of incompatible existing capability"""
        # Create test capability with different schema
        metadata = CapabilityMetadata(
            name="test_cap",
            type=CapabilityType.TOOL,
            description="Test capability",
            input_schema={"different_input": {"type": "string"}},
            output_schema={"different_output": {"type": "string"}},
            created_at=datetime.now(),
        )

        def test_func(different_input: str) -> str:
            """Test function"""
            return different_input

        capability = ToolCapability(test_func, metadata)
        registry.register("test_cap", capability)

        # Test resolution with incompatible requirements
        cap_type, resolved = await factory.resolve_capability(
            "test_cap", ["input1"], "output1", {}
        )

        # Verify new capability created
        assert resolved != capability

    @pytest.mark.asyncio
    async def test_resolve_new_capability(self, factory):
        """Test creating new capability when none exists"""
        # Test resolution of non-existent capability
        cap_type, capability = await factory.resolve_capability(
            "new_cap", ["input1"], "output1", {}
        )

        # Verify new capability created with correct type
        assert cap_type == CapabilityType.TOOL
        assert capability is not None
        assert capability.metadata.name == "new_cap"
        assert "input1" in capability.metadata.input_schema

    @pytest.mark.asyncio
    async def test_resolution_error_handling(self, factory):
        """Test handling of resolution errors"""
        with pytest.raises(CapabilityResolutionError) as exc_info:
            await factory.resolve_capability(
                "",  # Invalid name
                [],  # No inputs
                "",  # Invalid output
                {},
            )

        # Verify the error message contains one of the expected error strings
        error_msg = str(exc_info.value)
        assert any(
            msg in error_msg
            for msg in [
                "Invalid capability name",
                "No inputs specified",
                "Invalid output",
                "Invalid instruction",
            ]
        )
