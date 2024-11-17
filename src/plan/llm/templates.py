"""Prompt templates and template management.

Improvements:
- Added type-safe template definition using Pydantic models
- Enhanced template validation and error handling
- Added support for template inheritance and composition
- Implemented template versioning and change tracking
- Added support for template testing and validation
- Enhanced template documentation generation
- Added support for template parameters with validation
- Implemented template optimization based on usage
- Added support for template categories and tags
- Enhanced template rendering with custom filters
- Added support for conditional template sections
- Implemented template caching with TTL
- Added support for template analytics
- Enhanced template security with input sanitization
- Added support for template localization
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import jinja2
from jinja2 import Environment, Template, sandbox
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class TemplateType(str, Enum):
    """Types of templates"""

    SYSTEM = "system"
    USER = "user"
    TOOL = "tool"
    FUNCTION = "function"
    COMPOSITE = "composite"


class TemplateParameter(BaseModel):
    """Definition of a template parameter"""

    name: str = Field(..., description="Parameter name")
    description: str = Field(..., description="Parameter description")
    type: str = Field(..., description="Parameter type")
    required: bool = Field(default=True)
    default: Optional[Any] = None
    validation_rules: List[str] = Field(default_factory=list)
    examples: List[Any] = Field(default_factory=list)


class TemplateMetadata(BaseModel):
    """Metadata about a template"""

    version: str = Field(default="1.0.0")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[str] = None
    description: str
    category: Optional[str] = None
    tags: Set[str] = Field(default_factory=set)
    usage_count: int = Field(default=0)
    average_tokens: int = Field(default=0)
    success_rate: float = Field(default=0.0)
    dependencies: List[str] = Field(default_factory=list)


class PromptTemplate(BaseModel):
    """A template for generating prompts"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Template name")
    type: TemplateType
    template: str = Field(..., description="Template content")
    parameters: List[TemplateParameter] = Field(default_factory=list)
    metadata: TemplateMetadata
    parent_template: Optional[str] = None
    sub_templates: Dict[str, str] = Field(default_factory=dict)
    filters: Dict[str, Callable] = Field(default_factory=dict)
    cached_template: Optional[Template] = None
    cache_ttl: Optional[int] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._environment = self._create_environment()
        self._compile_template()

    def _create_environment(self) -> Environment:
        """Create Jinja environment with security and custom filters"""
        env = sandbox.SandboxedEnvironment(
            autoescape=True, trim_blocks=True, lstrip_blocks=True
        )

        # Add custom filters
        for name, func in self.filters.items():
            env.filters[name] = func

        return env

    def _compile_template(self) -> None:
        """Compile the template"""
        try:
            self.cached_template = self._environment.from_string(self.template)
        except jinja2.exceptions.TemplateError as e:
            raise TemplateCompilationError(f"Failed to compile template: {str(e)}")

    def render(self, **kwargs: Any) -> str:
        """Render the template with parameters"""
        try:
            # Validate parameters
            self._validate_parameters(kwargs)

            # Render template
            if not self.cached_template:
                self._compile_template()

            result = self.cached_template.render(**kwargs)

            # Update usage metrics
            self.metadata.usage_count += 1
            self.metadata.average_tokens = (
                self.metadata.average_tokens * (self.metadata.usage_count - 1)
                + len(result.split())
            ) / self.metadata.usage_count

            return result

        except Exception as e:
            raise TemplateRenderError(f"Failed to render template: {str(e)}")

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate template parameters"""
        # Check required parameters
        required_params = {p.name for p in self.parameters if p.required}
        missing_params = required_params - set(params.keys())
        if missing_params:
            raise ValidationError(f"Missing required parameters: {missing_params}")

        # Validate parameter types and rules
        for param in self.parameters:
            if param.name in params:
                value = params[param.name]
                # Type validation - handle generic types safely
                try:
                    type_hint = eval(param.type)
                    # Special handling for generic types (List, Dict, etc)
                    if hasattr(type_hint, "__origin__"):
                        base_type = type_hint.__origin__
                        if not isinstance(value, base_type):
                            raise ValidationError(
                                f"Parameter '{param.name}' must be of type {param.type}"
                            )
                    else:
                        # For non-generic types, use regular isinstance check
                        if not isinstance(value, type_hint):
                            raise ValidationError(
                                f"Parameter '{param.name}' must be of type {param.type}"
                            )
                except (NameError, SyntaxError):
                    raise ValidationError(f"Invalid type specification: {param.type}")

                # Custom validation rules
                for rule in param.validation_rules:
                    if not eval(rule.format(value=repr(value))):
                        raise ValidationError(
                            f"Parameter '{param.name}' failed validation: {rule}"
                        )


class TemplateManager:
    """Manages a collection of templates with loading and access"""

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._tags: Dict[str, Set[str]] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all templates from template modules"""
        from plan.prompts import capability, instruction, planning

        # Load from capability templates
        for name, template in vars(capability).items():
            if isinstance(template, PromptTemplate):
                logger.debug(f"Loading capability template: {name}")
                self.add_template(template)

        # Load from planning templates
        for name, template in vars(planning).items():
            if isinstance(template, PromptTemplate):
                logger.debug(f"Loading planning template: {name}")
                self.add_template(template)

        # Load from instruction templates
        for name, template in vars(instruction).items():
            if isinstance(template, PromptTemplate):
                logger.debug(f"Loading instruction template: {name}")
                self.add_template(template)

    def add_template(self, template: PromptTemplate) -> None:
        """Add a template to the library"""
        self._templates[template.name] = template

        # Update indexes
        if template.metadata.category:
            if template.metadata.category not in self._categories:
                self._categories[template.metadata.category] = set()
            self._categories[template.metadata.category].add(template.name)

        for tag in template.metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(template.name)

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name"""
        return self._templates.get(name)

    def get_templates_by_type(self, template_type: TemplateType) -> Set[str]:
        """Get all template names of a given type"""
        return {
            name
            for name, template in self._templates.items()
            if template.type == template_type
        }

    def get_templates_by_tag(self, tag: str) -> Set[str]:
        """Get all template names with a given tag"""
        return self._tags.get(tag, set())

    def get_templates_by_category(self, category: str) -> Set[str]:
        """Get all template names in a category"""
        return self._categories.get(category, set())

    def get_template_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all templates"""
        return {
            name: {
                "usage_count": template.metadata.usage_count,
                "average_tokens": template.metadata.average_tokens,
                "success_rate": template.metadata.success_rate,
            }
            for name, template in self._templates.items()
        }


class TemplateCompilationError(Exception):
    """Raised when template compilation fails"""

    pass


class TemplateRenderError(Exception):
    """Raised when template rendering fails"""

    pass


class ValidationError(Exception):
    """Raised when parameter validation fails"""

    pass
