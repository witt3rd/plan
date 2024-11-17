from typing import Callable, Dict, TypedDict

from function_schema import Annotated, get_function_schema
from function_schema.types import FunctionSchema, ParamSchema, RootProperty
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from typing_extensions import Doc


# Add new type definitions
class ParamInfo(TypedDict):
    type: str
    description: str
    enum: list[str] | None
    default: str | None


def build_function_schema(
    func: Callable, name: str, input_schema: Dict[str, ParamInfo]
) -> FunctionSchema:
    """Build OpenAI function schema from function and metadata

    Args:
        func: The function to build schema for
        name: Name of the function
        input_schema: Schema for function inputs

    Returns:
        Complete function schema
    """
    # Get function docstring
    doc = inspect.getdoc(func) or ""

    # Build parameter schemas from metadata
    properties = {}
    for param_name, param_info in input_schema.items():
        param_schema: ParamSchema = {
            "type": param_info["type"],
            "description": param_info["description"],
        }

        # Add optional fields if present
        if param_info.get("enum") is not None:
            param_schema["enum"] = param_info["enum"]
        if param_info.get("default") is not None:
            param_schema["default"] = str(param_info["default"])

        properties[param_name] = param_schema

    # Construct root property
    root_property: RootProperty = {"type": "object", "properties": properties}

    # Build complete schema
    schema: FunctionSchema = {
        "name": name,
        "description": doc,
        "parameters": root_property,
    }

    return schema


def get_function_definition(func: Callable) -> FunctionDefinition:
    """Convert a function to OpenAI's FunctionDefinition format"""
    schema = get_function_schema(func)

    # Extract description, defaulting to empty string if None
    description = func.__doc__ or ""

    # Convert parameters to dict explicitly
    parameters = dict(schema.get("parameters", {}))

    return FunctionDefinition(
        name=func.__name__,
        description=description,
        parameters=parameters,
    )


def get_chat_completion_tool_param(func: Callable) -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type="function", function=get_function_definition(func)
    )


__all__ = [
    "Annotated",
    "Doc",
    "FunctionSchema",
    "ParamSchema",
    "RootProperty",
    "build_function_schema",
    "get_function_schema",
    "get_function_definition",
    "get_chat_completion_tool_param",
]
