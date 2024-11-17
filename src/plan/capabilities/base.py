"""Base capability interfaces and types"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from plan.capabilities.metadata import CapabilityMetadata

# Type variables for input/output typing
Input = TypeVar("Input", bound=BaseModel)
Output = TypeVar("Output")


@runtime_checkable
class CapabilityInterface(Protocol[Input, Output]):
    """Protocol defining the capability interface"""

    async def execute(self, input: Input) -> Output:
        """Execute the capability with given input"""
        ...

    @property
    def metadata(self) -> CapabilityMetadata:
        """Get capability metadata"""
        ...


class ExecutionStats(BaseModel):
    """Tracks execution statistics for a capability"""

    total_executions: int = Field(default=0)
    successful_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    total_execution_time: float = Field(default=0.0)
    average_execution_time: float = Field(default=0.0)
    last_execution_time: datetime | None = Field(default=None)
    success_rate: float = Field(default=0.0)

    def record_success(self, execution_time: float) -> None:
        """Record a successful execution"""
        self.total_executions += 1
        self.successful_executions += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.total_executions
        self.last_execution_time = datetime.now()
        self.success_rate = self.successful_executions / self.total_executions

    def record_failure(self) -> None:
        """Record a failed execution"""
        self.total_executions += 1
        self.failed_executions += 1
        self.last_execution_time = datetime.now()
        self.success_rate = self.successful_executions / self.total_executions


class Capability(Generic[Input, Output], ABC):
    """Base class for all capabilities"""

    def __init__(self, metadata: CapabilityMetadata):
        """Initialize the capability

        Args:
            metadata: Capability metadata
        """
        self._metadata = metadata
        self._execution_stats = ExecutionStats()

    async def execute(self, input: Input) -> Output:
        """Execute the capability with given input

        Args:
            input: Input values for execution

        Returns:
            Execution result

        Raises:
            InputValidationError: If input validation fails
            OutputValidationError: If output validation fails
            CapabilityExecutionError: If execution fails
        """
        try:
            # Validate input
            self._validate_input(input)

            # Record start time
            start_time = datetime.now()

            # Execute implementation
            result = await self._execute_impl(input)

            # Validate output
            self._validate_output(result)

            # Record success
            self._execution_stats.record_success(
                (datetime.now() - start_time).total_seconds()
            )

            return result

        except Exception as e:
            # Record failure
            self._execution_stats.record_failure()

            # Re-raise with appropriate error type
            if isinstance(e, (InputValidationError, OutputValidationError)):
                raise
            raise CapabilityExecutionError(
                f"Execution failed for {self.metadata.name}: {str(e)}"
            ) from e

    @abstractmethod
    async def _execute_impl(self, input: Input) -> Output:
        """Implementation of capability execution

        Args:
            input: Validated input values

        Returns:
            Execution result

        Raises:
            CapabilityExecutionError: If execution fails
        """
        pass

    @abstractmethod
    def _validate_input(self, input: Input) -> None:
        """Validate input against schema

        Args:
            input: Input to validate

        Raises:
            InputValidationError: If validation fails
        """
        pass

    @abstractmethod
    def _validate_output(self, output: Any) -> None:
        """Validate output against schema

        Args:
            output: Output to validate

        Raises:
            OutputValidationError: If validation fails
        """
        pass

    @property
    def metadata(self) -> CapabilityMetadata:
        """Get capability metadata"""
        return self._metadata

    @property
    def execution_stats(self) -> ExecutionStats:
        """Get execution statistics"""
        return self._execution_stats


class CapabilityValidationError(Exception):
    """Base class for capability validation errors"""

    pass


class InputValidationError(CapabilityValidationError):
    """Raised when input validation fails"""

    pass


class OutputValidationError(CapabilityValidationError):
    """Raised when output validation fails"""

    pass


class CapabilityExecutionError(Exception):
    """Raised when capability execution fails"""

    pass


class CapabilityNotFoundError(Exception):
    """Raised when a required capability is not found"""

    pass


class CapabilityResolutionError(Exception):
    """Raised when capability resolution fails"""

    pass


class CapabilityCompatibilityError(Exception):
    """Raised when capability is incompatible with requirements"""

    pass
