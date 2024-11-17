# AI Planning System

A modular and extensible system for dynamic planning and execution using a capability-based architecture.

## Overview

This system provides a flexible framework for:

- Creating and executing complex plans
- Managing different types of capabilities (Tools, Instructions, Plans)
- Dynamic capability creation and resolution
- Robust plan validation and execution
- Sophisticated prompt handling and LLM interaction

## Core Components

### Capabilities

The capability system provides a unified interface for different types of operations:

- **Tools**: Python functions for procedural/computational tasks
- **Instructions**: LLM prompt templates for creative/analytical tasks
- **Plans**: Composite workflows combining multiple capabilities

Each capability:

- Has clear inputs and outputs
- Includes detailed metadata
- Tracks performance metrics
- Can be optimized based on usage

### Planning

The planning system handles:

- Plan generation from high-level goals
- Task decomposition and sequencing
- Capability resolution and creation
- Plan validation and optimization
- Execution monitoring

### Orchestration

The orchestrator manages:

- Capability lifecycle management
- Plan execution coordination
- Resource management
- Error handling and recovery
- Performance monitoring

### Prompt Handling

Robust LLM interaction with:

- Structured response parsing
- Conversation management
- Token tracking
- Retry logic
- Response validation

## Key Features

- **Dynamic Capability Creation**: Automatically creates new capabilities as needed
- **Type Safety**: Extensive use of Pydantic models for validation
- **Extensible**: Easy to add new capability types and planning strategies
- **Robust Validation**: Comprehensive validation at multiple levels
- **Performance Tracking**: Detailed metrics and optimization
- **Error Handling**: Sophisticated error recovery and fallbacks
- **Async Support**: Built for asynchronous operation

## Usage Example

```python
# Initialize components
registry = CapabilityRegistry()
factory = CapabilityFactory(prompt_handler)
orchestrator = CapabilityOrchestrator(
    prompt_handler=prompt_handler,
    registry=registry,
    factory=factory
)

# Create a plan
plan = await orchestrator.create_plan(
    goal="Process data and generate report",
    required_inputs=["data_source"],
    required_output="report",
    context={"format": "PDF"}
)

# Execute plan
result = await orchestrator.execute_capability(
    name=plan.name,
    inputs={"data_source": "data.csv"}
)
```

## Architecture

The system uses a layered architecture:

1. **Core Layer**: Base interfaces and common models
2. **Capability Layer**: Different capability implementations
3. **Planning Layer**: Plan creation and management
4. **Orchestration Layer**: High-level coordination
5. **Interface Layer**: External interaction handling

## Key Benefits

- **Flexibility**: Easily adapt to different use cases
- **Reliability**: Robust error handling and validation
- **Scalability**: Modular design for easy extension
- **Maintainability**: Clear separation of concerns
- **Observability**: Comprehensive metrics and logging

## Future Enhancements

- Enhanced parallel execution
- More sophisticated planning strategies
- Additional capability types
- Improved optimization algorithms
- Extended validation rules

## Requirements

- Python 3.9+
- Pydantic
- OpenAI API access
- NetworkX (for plan visualization)
- Additional dependencies in requirements.txt

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE for details
