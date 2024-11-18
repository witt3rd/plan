"""Instruction capability implementation"""

import asyncio
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from plan.capabilities.base import (
    Capability,
    CapabilityExecutionError,
    Input,
    InputValidationError,
    Output,
)
from plan.capabilities.metadata import CapabilityMetadata
from plan.capabilities.schema import Schema, SchemaField, SchemaType
from plan.llm.handler import CompletionConfig, PromptHandler
from plan.llm.templates import TemplateManager


class PromptExample(BaseModel):
    """Example for few-shot learning"""

    inputs: Dict[str, Any]
    output: Any
    explanation: Optional[str] = None


class InstructionMetadata(BaseModel):
    """Additional metadata specific to instruction capabilities"""

    template_version: str = Field(default="1.0.0")
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    system_message: Optional[str] = None
    examples: List[PromptExample] = Field(default_factory=list)
    validation_rules: List[str] = Field(default_factory=list)
    retry_count: int = Field(default=3)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class InstructionCapability(Capability[Input, Output]):
    """A capability implemented as an LLM prompt template"""

    def __init__(
        self,
        template: str,
        metadata: CapabilityMetadata,
        prompt_handler: PromptHandler,
        instruction_metadata: Optional[InstructionMetadata] = None,
    ):
        """Initialize the instruction capability"""
        super().__init__(metadata)
        self._validate_template(template)
        self._template = template
        self._prompt_handler = prompt_handler
        self._instruction_metadata = instruction_metadata or InstructionMetadata()
        self._template_manager = TemplateManager()

    def _validate_template(self, template: str) -> None:
        """Validate the prompt template"""
        try:
            # Extract template variables
            template_vars = {
                name
                for _, name, _, _ in string.Formatter().parse(template)
                if name is not None
            }

            # Verify all required inputs are in template
            required_inputs = {
                name
                for name, field in self.metadata.input_schema.fields.items()
                if field.required
            }
            missing_inputs = required_inputs - template_vars
            if missing_inputs:
                raise ValueError(f"Template missing required inputs: {missing_inputs}")

            # Verify no extra variables
            extra_vars = template_vars - set(self.metadata.input_schema.fields.keys())
            if extra_vars:
                raise ValueError(f"Template contains unknown variables: {extra_vars}")

            # Basic syntax check
            template.format(**{var: "test" for var in template_vars})

        except Exception as e:
            raise ValueError(f"Template validation failed: {str(e)}")

    async def _execute_impl(self, input: Input) -> Output:
        """Implementation of capability execution"""
        try:
            # Get the instruction template
            template = self._template_manager.get_template("instruction_execution")
            if not template:
                raise ValueError("Required template 'instruction_execution' not found")

            # Format prompt with examples if available
            prompt = template.render(
                template=self._template,
                input=input.__dict__,
                examples=self._instruction_metadata.examples,
                validation_rules=self._instruction_metadata.validation_rules,
            )

            # Configure completion
            config = CompletionConfig(
                temperature=self._instruction_metadata.temperature,
                system_message=self._instruction_metadata.system_message,
            )

            # Execute with retry
            for attempt in range(self._instruction_metadata.retry_count):
                try:
                    result = await self._prompt_handler.complete(prompt, config=config)
                    # Create output model instance
                    output_model = self._get_output_model()
                    return output_model(result=result)

                except Exception as e:
                    if attempt == self._instruction_metadata.retry_count - 1:
                        raise CapabilityExecutionError(
                            f"All retry attempts failed: {str(e)}"
                        ) from e
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

        except ValidationError as e:
            raise InputValidationError(f"Input validation failed: {str(e)}")

        except Exception as e:
            raise CapabilityExecutionError(f"Execution failed: {str(e)}")

    async def optimize(self) -> None:
        """Optimize the instruction based on usage patterns"""
        if self.execution_stats.total_executions < 100:
            return

        if (
            self.execution_stats.success_rate < 0.95
            or self.execution_stats.average_execution_time > 1.0
        ):
            # Implement optimization logic here
            pass

    @property
    def template(self) -> str:
        """Get the current template"""
        return self._template

    @property
    def instruction_metadata(self) -> InstructionMetadata:
        """Get instruction-specific metadata"""
        return self._instruction_metadata

    @classmethod
    def create(
        cls,
        name: str,
        template: str,
        input_fields: Dict[str, Dict[str, Any]],
        output_fields: Dict[str, Dict[str, Any]],
        prompt_handler: PromptHandler,
        description: str = "",
    ) -> "InstructionCapability":
        """Create a new instruction capability with schema validation"""
        # Convert field definitions to Schema objects
        input_schema = Schema(
            fields={
                name: SchemaField(
                    type=SchemaType(field_def.get("type", "string").lower()),
                    description=field_def.get("description", f"Input {name}"),
                    required=field_def.get("required", True),
                    default=field_def.get("default"),
                    validation_rules=field_def.get("validation_rules", []),
                )
                for name, field_def in input_fields.items()
            }
        )

        output_schema = Schema(
            fields={
                name: SchemaField(
                    type=SchemaType(field_def.get("type", "string").lower()),
                    description=field_def.get("description", f"Output {name}"),
                    required=field_def.get("required", True),
                )
                for name, field_def in output_fields.items()
            }
        )

        metadata = CapabilityMetadata(
            name=name,
            type=CapabilityType.INSTRUCTION,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        return cls(template, metadata, prompt_handler)
