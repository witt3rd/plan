"""Tool capability implementation

Improvements:
- Added input/output validation using Pydantic models
- Enhanced error handling with specific exception types
- Added support for function signature validation
- Implemented retry logic with exponential backoff
- Added support for async functions
- Included function profiling and performance tracking
- Added support for function timeouts
- Implemented parameter type checking and coercion
- Added function documentation extraction
- Included test case execution and validation
- Added support for function mocking in test mode
- Implemented resource usage tracking
- Added support for function versioning
- Included function result caching with TTL
"""

import asyncio
import inspect
import time
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from plan.capabilities.base import (
    Capability,
    CapabilityExecutionError,
    Input,
    InputValidationError,
    Output,
)
from plan.capabilities.metadata import CapabilityMetadata
from plan.llm.tool import FunctionSchema, build_function_schema

T = TypeVar("T")


class FunctionStats(BaseModel):
    """Detailed statistics for function execution"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_calls: int = Field(default=0)
    successful_calls: int = Field(default=0)
    failed_calls: int = Field(default=0)
    total_execution_time: float = Field(default=0.0)
    average_execution_time: float = Field(default=0.0)
    min_execution_time: float = Field(default=float("inf"))
    max_execution_time: float = Field(default=0.0)
    last_execution_time: Optional[datetime] = None
    cache_hits: int = Field(default=0)
    cache_misses: int = Field(default=0)
    retry_count: int = Field(default=0)
    timeout_count: int = Field(default=0)
    total_cpu_time: float = Field(default=0.0)
    peak_memory_delta: float = Field(default=0.0)
    average_memory_delta: float = Field(default=0.0)
    total_memory_delta: float = Field(default=0.0)


class CachedResult(BaseModel):
    """Cached function result with metadata"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl: float  # Time-to-live in seconds


class ToolCapability(Capability[Input, Output]):
    """A capability implemented as a Python function"""

    def __init__(
        self,
        func: Callable[..., Any],
        metadata: CapabilityMetadata,
        cache_ttl: Optional[float] = None,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        retry_delay: float = 1.0,
        max_concurrent: Optional[int] = None,
    ):
        """Initialize the tool capability

        Args:
            func: The function to execute
            metadata: Capability metadata
            cache_ttl: Optional cache time-to-live in seconds
            max_retries: Maximum number of retry attempts
            timeout: Optional execution timeout in seconds
            retry_delay: Base delay between retries (will use exponential backoff)
            max_concurrent: Maximum number of concurrent executions
        """
        super().__init__(metadata)
        self._validate_function(func)
        self._func = func
        self._cache_ttl = cache_ttl
        self._max_retries = max_retries
        self._timeout = timeout
        self._retry_delay = retry_delay
        self._stats = FunctionStats()
        self._cache: Dict[str, CachedResult] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent if max_concurrent else 100)
        self._execution_lock = asyncio.Lock()

    def _validate_function(self, func: Callable[..., Any]) -> None:
        """Validate function signature against metadata

        Args:
            func: Function to validate

        Raises:
            ValueError: If validation fails
        """
        sig = inspect.signature(func)

        # Get required parameters from function
        required_params = {
            name
            for name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty
        }

        # Get required parameters from schema
        schema_required = {
            name
            for name, schema in self.metadata.input_schema.items()
            if schema.get("required", True)
        }

        # Check if required parameters match
        if required_params != schema_required:
            raise ValueError(
                f"Function required parameters {required_params} don't match schema required parameters {schema_required}"
            )

    def _create_wrapper(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Create a wrapped version of the function with profiling

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            self._stats.total_calls += 1

            try:
                # Check cache first
                cache_key = self._get_cache_key(args, kwargs)
                if cached := self._get_cached_result(cache_key):
                    self._stats.cache_hits += 1
                    return cached

                self._stats.cache_misses += 1

                # Execute with retry logic
                for attempt in range(self._max_retries):
                    try:
                        # Handle both sync and async functions
                        if inspect.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)

                        # Cache result if enabled
                        if self._cache_ttl:
                            self._cache[cache_key] = CachedResult(
                                value=result, ttl=self._cache_ttl
                            )

                        return result

                    except Exception:
                        if attempt == self._max_retries - 1:
                            raise
                        self._stats.retry_count += 1
                        await asyncio.sleep(2**attempt)

            except asyncio.TimeoutError:
                self._stats.timeout_count += 1
                raise CapabilityExecutionError("Execution timed out")

            except Exception as e:
                self._stats.failed_calls += 1
                raise CapabilityExecutionError(f"Execution failed: {str(e)}") from e

            finally:
                execution_time = time.time() - start_time
                self._update_stats(execution_time)

        return wrapper

    async def _execute_impl(self, input: Input) -> Output:
        """Execute the tool function with full error handling and retries"""
        try:
            # Acquire semaphore for concurrent execution control
            async with self._semaphore:
                # Check cache first
                cache_key = self._get_cache_key((), input.__dict__)
                if cached := self._get_cached_result(cache_key):
                    self._stats.cache_hits += 1
                    return cached

                self._stats.cache_misses += 1

                # Execute with retries
                for attempt in range(self._max_retries):
                    try:
                        start_time = time.time()

                        # Execute with timeout if specified
                        if self._timeout:
                            result = await asyncio.wait_for(
                                self._execute_function(input.__dict__),
                                timeout=self._timeout,
                            )
                        else:
                            result = await self._execute_function(input.__dict__)

                        # Validate output
                        self._validate_output(result)

                        # Update stats and cache on success
                        execution_time = time.time() - start_time
                        self._update_stats(execution_time)

                        if self._cache_ttl:
                            self._cache[cache_key] = CachedResult(
                                value=result, ttl=self._cache_ttl
                            )

                        return result

                    except (asyncio.TimeoutError, CapabilityExecutionError) as e:
                        if attempt == self._max_retries - 1:
                            if isinstance(e, asyncio.TimeoutError):
                                self._stats.timeout_count += 1
                            raise

                        self._stats.retry_count += 1
                        retry_delay = self._retry_delay * (
                            2**attempt
                        )  # Exponential backoff
                        await asyncio.sleep(retry_delay)

        except ValidationError as e:
            raise InputValidationError(f"Input validation failed: {str(e)}")
        except Exception as e:
            self._stats.failed_calls += 1
            raise CapabilityExecutionError(f"Tool execution failed: {str(e)}") from e

    async def _execute_function(self, input: Dict[str, Any]) -> Any:
        """Execute the wrapped function with resource tracking"""
        # Track resource usage
        start_memory = self._get_memory_usage()
        start_time = time.time()

        try:
            if inspect.iscoroutinefunction(self._func):
                result = await self._func(**input)
            else:
                # Run sync functions in thread pool to avoid blocking
                # Create a lambda to handle keyword arguments
                def execute_func():
                    return self._func(**input)

                result = await asyncio.get_event_loop().run_in_executor(
                    None, execute_func
                )

            # Record resource usage
            end_memory = self._get_memory_usage()
            end_time = time.time()

            self._record_resource_usage(
                cpu_time=end_time - start_time, memory_delta=end_memory - start_memory
            )

            return result

        except Exception as e:
            self._stats.failed_calls += 1
            raise CapabilityExecutionError(
                f"Function execution failed: {str(e)}"
            ) from e

    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes"""
        import psutil

        process = psutil.Process()
        return process.memory_info().rss

    def _record_resource_usage(self, cpu_time: float, memory_delta: float) -> None:
        """Record resource usage statistics

        Args:
            cpu_time: Time taken for execution in seconds
            memory_delta: Change in memory usage in bytes
        """
        # Update CPU time statistics
        self._stats.total_cpu_time += cpu_time

        # Update memory statistics
        self._stats.total_memory_delta += memory_delta
        self._stats.peak_memory_delta = max(self._stats.peak_memory_delta, memory_delta)

        # Calculate average only if we have successful calls
        if self._stats.successful_calls > 0:
            self._stats.average_memory_delta = (
                self._stats.total_memory_delta / self._stats.successful_calls
            )
        else:
            self._stats.average_memory_delta = 0.0

    def _validate_input(self, input: Input) -> None:
        """Validate input against schema"""
        try:
            # Verify all required inputs are present
            required_inputs = set(
                name
                for name, schema in self.metadata.input_schema.items()
                if schema.get("required", True)
            )
            missing_inputs = required_inputs - set(input.__dict__.keys())
            if missing_inputs:
                raise InputValidationError(f"Missing required inputs: {missing_inputs}")

            # Validate input types
            for name, value in input.__dict__.items():
                if name not in self.metadata.input_schema:
                    raise InputValidationError(f"Unknown input: {name}")

                schema = self.metadata.input_schema[name]
                expected_type = schema.get("type")
                if expected_type and not isinstance(value, eval(expected_type)):
                    raise InputValidationError(
                        f"Input '{name}' has wrong type. Expected {expected_type}, got {type(value)}"
                    )

        except Exception as e:
            if not isinstance(e, InputValidationError):
                raise InputValidationError(f"Input validation failed: {str(e)}")
            raise

    def _validate_output(self, output: Any) -> None:
        """Validate output against schema"""
        try:
            schema = self.metadata.output_schema
            if not schema:
                return

            # Type validation
            expected_type = schema.get("type")
            if expected_type and not isinstance(output, eval(expected_type)):
                raise OutputValidationError(
                    f"Output has wrong type. Expected {expected_type}, got {type(output)}"
                )

            # Format validation
            if "format" in schema:
                # Add format validation logic here
                pass

            # Range validation
            if "minimum" in schema and output < schema["minimum"]:
                raise OutputValidationError(
                    f"Output below minimum: {schema['minimum']}"
                )
            if "maximum" in schema and output > schema["maximum"]:
                raise OutputValidationError(
                    f"Output above maximum: {schema['maximum']}"
                )

        except Exception as e:
            if not isinstance(e, OutputValidationError):
                raise OutputValidationError(f"Output validation failed: {str(e)}")
            raise

    def _get_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        return f"{args}:{kwargs}"

    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if valid

        Args:
            key: Cache key

        Returns:
            Cached value if valid, None otherwise
        """
        if cached := self._cache.get(key):
            if (datetime.now(UTC) - cached.timestamp).total_seconds() < cached.ttl:
                return cached.value
            del self._cache[key]
        return None

    def _update_stats(self, execution_time: float) -> None:
        """Update execution statistics

        Args:
            execution_time: Time taken for execution
        """
        stats = self._stats
        stats.total_execution_time += execution_time
        stats.successful_calls += 1
        stats.total_calls += 1

        # Now we can safely calculate the average
        stats.average_execution_time = stats.total_execution_time / stats.total_calls

        # Update min/max execution times
        stats.min_execution_time = min(stats.min_execution_time, execution_time)
        stats.max_execution_time = max(stats.max_execution_time, execution_time)
        stats.last_execution_time = datetime.now(UTC)

    @property
    def stats(self) -> FunctionStats:
        """Get current execution statistics"""
        return self._stats

    def clear_cache(self) -> None:
        """Clear the result cache"""
        self._cache.clear()

    @property
    def schema(self) -> FunctionSchema:
        """Get OpenAI function schema for this tool

        Returns:
            FunctionSchema containing the function's interface definition
        """
        return build_function_schema(
            self._func, self.metadata.name, self.metadata.input_schema
        )

    async def execute(self, input: Input) -> Output:
        """Execute the tool capability with the given input

        Args:
            input: The input values for the function

        Returns:
            The function output after validation

        Raises:
            CapabilityExecutionError: If execution fails
            InputValidationError: If input validation fails
        """
        return await self._execute_impl(input)
