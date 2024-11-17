"""Interfaces for capability creation and management"""

from typing import Any, Dict, List, Protocol, Tuple

from pydantic import BaseModel

from plan.capabilities.base import Capability
from plan.capabilities.metadata import CapabilityType


class CapabilityCreator(Protocol):
    """Interface for creating capabilities"""

    async def resolve_capability(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
    ) -> Tuple[CapabilityType, Capability[BaseModel, Any]]:
        """Resolve or create a capability"""
        ...

    async def create_capability(
        self,
        name: str,
        required_inputs: List[str],
        required_output: str,
        context: Dict[str, Any],
    ) -> Tuple[CapabilityType, Capability[BaseModel, Any]]:
        """Create a new capability"""
        ...
