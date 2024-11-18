"""Tool capability implementation"""

import asyncio
import inspect
import time
from datetime import UTC, datetime
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
        """Initialize the tool capability"""
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

    def _validate_function(self, func: Callable[..., Any]) -> None:
        """Validate function signature against metadata schema"""
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
            for name, field in self.metadata.input_schema.fields.items()
            if field.required
        }

        # Check if required parameters match
        if required_params != schema_required:
            raise ValueError(
                f"Function required parameters {required_params} don't match schema required parameters {schema_required}"
            )

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
        """Record resource usage statistics"""
        # Update CPU time statistics
        self._stats.total_cpu_time += cpu_time

        # Update memory statistics
        self._stats.total_memory_delta += memory_delta
        self._stats.peak_memory_delta = max(self._stats.peak_memory_delta, memory_delta)

        if self._stats.successful_calls > 0:
            self._stats.average_memory_delta = (
                self._stats.total_memory_delta / self._stats.successful_calls
            )

    def _get_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments"""
        return f"{args}:{kwargs}"

    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if valid"""
        if cached := self._cache.get(key):
            if (datetime.now(UTC) - cached.timestamp).total_seconds() < cached.ttl:
                return cached.value
            del self._cache[key]
        return None

    def _update_stats(self, execution_time: float) -> None:
        """Update execution statistics"""
        stats = self._stats
        stats.total_execution_time += execution_time
        stats.successful_calls += 1
        stats.total_calls += 1
        stats.average_execution_time = stats.total_execution_time / stats.total_calls
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
        """Get OpenAI function schema for this tool"""
        return build_function_schema(
            self._func, self.metadata.name, self.metadata.input_schema.fields
        )
