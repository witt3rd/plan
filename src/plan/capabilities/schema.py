"""Schema definitions for capability inputs and outputs"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field


class SchemaType(str, Enum):
    """Valid schema field types"""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


class SchemaField(BaseModel):
    """Definition of a schema field"""

    type: SchemaType
    description: str
    required: bool = True
    default: Optional[Any] = None
    validation_rules: List[str] = Field(default_factory=list)
    enum: Optional[List[Any]] = None
    items: Optional["SchemaField"] = None  # For array types
    properties: Optional[Dict[str, "SchemaField"]] = None  # For object types

    @property
    def python_type(self) -> Type:
        """Get the corresponding Python type"""
        type_mapping = {
            SchemaType.STRING: str,
            SchemaType.INTEGER: int,
            SchemaType.FLOAT: float,
            SchemaType.BOOLEAN: bool,
            SchemaType.ARRAY: list,
            SchemaType.OBJECT: dict,
            SchemaType.ANY: Any,
        }
        return type_mapping[self.type]


class Schema(BaseModel):
    """Complete schema definition"""

    fields: Dict[str, SchemaField]
    description: Optional[str] = None
    validation_rules: List[str] = Field(default_factory=list)

    def to_pydantic_fields(self) -> Dict[str, tuple[Type, Any]]:
        """Convert schema to Pydantic field definitions"""
        fields = {}
        for name, field in self.fields.items():
            field_type = field.python_type
            field_default = ... if field.required else field.default
            fields[name] = (
                field_type,
                Field(default=field_default, description=field.description),
            )
        return fields


def convert_dict_to_schema(schema_dict: Dict[str, Any]) -> Schema:
    """Convert a dictionary schema definition to Schema model"""
    fields = {}
    for name, field_def in schema_dict.items():
        if isinstance(field_def, str):
            # Simple type definition
            fields[name] = SchemaField(
                type=SchemaType(field_def.lower()),
                description=f"Field {name}",
                required=True,
            )
        elif isinstance(field_def, dict):
            # Detailed field definition
            field_type = field_def.get("type", "any")
            fields[name] = SchemaField(
                type=SchemaType(field_type.lower()),
                description=field_def.get("description", f"Field {name}"),
                required=field_def.get("required", True),
                default=field_def.get("default"),
                validation_rules=field_def.get("validation_rules", []),
                enum=field_def.get("enum"),
                items=convert_dict_to_schema({"item": field_def["items"]}).fields[
                    "item"
                ]
                if "items" in field_def
                else None,
                properties={
                    k: convert_dict_to_schema({"prop": v}).fields["prop"]
                    for k, v in field_def.get("properties", {}).items()
                },
            )

    return Schema(fields=fields)
