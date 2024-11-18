# File: examples/dynamic_analysis_workflow.py

"""
Dynamic analysis workflow example demonstrating:
1. Multiple capability types (tools, instructions)
2. Dynamic capability creation
3. Plan generation and optimization
4. Visualization of the execution
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from plan.capabilities.factory import CapabilityFactory
from plan.capabilities.metadata import CapabilityMetadata, CapabilityType
from plan.capabilities.registry import CapabilityRegistry
from plan.capabilities.schema import Schema, SchemaField, SchemaType
from plan.capabilities.tool import ToolCapability
from plan.llm.handler import PromptHandler
from plan.planning.executor import PlanExecutor
from plan.planning.planner import Planner
from plan.visualization import visualize_plan


async def dynamic_analysis_workflow():
    """Demonstrates a dynamic analysis workflow"""

    # 1. Initialize components
    prompt_handler = PromptHandler()
    registry = CapabilityRegistry()
    factory = CapabilityFactory(prompt_handler, registry)

    # 2. Register data processing capabilities
    def load_dataset(source: str) -> Dict[str, Any]:
        """Loads data from a source"""
        return {
            "data": f"Data loaded from {source}",
            "timestamp": datetime.now(),
        }

    registry.register(
        "load_dataset",
        ToolCapability(
            load_dataset,
            CapabilityMetadata(
                name="load_dataset",
                type=CapabilityType.TOOL,
                created_at=datetime.now(),
                description="Loads data from specified source",
                input_schema=Schema(
                    fields={
                        "source": SchemaField(
                            type=SchemaType.STRING,
                            description="Data source identifier",
                            required=True,
                        )
                    }
                ),
                output_schema=Schema(
                    fields={
                        "data": SchemaField(
                            type=SchemaType.OBJECT,
                            description="Loaded data with metadata",
                            required=True,
                        )
                    }
                ),
                tags=["data", "input", "loader"],
            ),
        ),
    )

    def validate_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validates data structure and content"""
        return {
            "valid": True,
            "data": data["data"],
            "validation_timestamp": datetime.now(),
        }

    registry.register(
        "validate_data",
        ToolCapability(
            validate_data,
            CapabilityMetadata(
                name="validate_data",
                type=CapabilityType.TOOL,
                created_at=datetime.now(),
                description="Validates data structure and content",
                input_schema=Schema(
                    fields={
                        "data": SchemaField(
                            type=SchemaType.OBJECT,
                            description="Data to validate",
                            required=True,
                        )
                    }
                ),
                output_schema=Schema(
                    fields={
                        "valid": SchemaField(
                            type=SchemaType.BOOLEAN,
                            description="Validation result",
                            required=True,
                        ),
                        "data": SchemaField(
                            type=SchemaType.OBJECT,
                            description="Validated data",
                            required=True,
                        ),
                    }
                ),
                tags=["data", "validation"],
            ),
        ),
    )

    def analyze_trends(
        data: Dict[str, Any], analysis_type: str = "basic"
    ) -> Dict[str, Any]:
        """Analyzes trends in the data"""
        return {
            "trends": f"Trends analyzed from {data['data']} using {analysis_type} analysis",
            "analysis_timestamp": datetime.now(),
        }

    registry.register(
        "analyze_trends",
        ToolCapability(
            analyze_trends,
            CapabilityMetadata(
                name="analyze_trends",
                type=CapabilityType.TOOL,
                created_at=datetime.now(),
                description="Analyzes trends in the validated data",
                input_schema=Schema(
                    fields={
                        "data": SchemaField(
                            type=SchemaType.OBJECT,
                            description="Validated data to analyze",
                            required=True,
                        ),
                        "analysis_type": SchemaField(
                            type=SchemaType.STRING,
                            description="Type of analysis to perform",
                            required=False,
                            default="basic",
                        ),
                    }
                ),
                output_schema=Schema(
                    fields={
                        "trends": SchemaField(
                            type=SchemaType.OBJECT,
                            description="Analysis results",
                            required=True,
                        )
                    }
                ),
                tags=["analysis", "trends"],
            ),
        ),
    )

    def generate_report(analysis_results: Dict[str, Any], format: str = "pdf") -> str:
        """Generates a report from analysis results"""
        return f"Report generated in {format} format: {analysis_results['trends']}"

    registry.register(
        "generate_report",
        ToolCapability(
            generate_report,
            CapabilityMetadata(
                name="generate_report",
                type=CapabilityType.TOOL,
                created_at=datetime.now(),
                description="Generates a report from analysis results",
                input_schema=Schema(
                    fields={
                        "analysis_results": SchemaField(
                            type=SchemaType.OBJECT,
                            description="Analysis results to include in report",
                            required=True,
                        ),
                        "format": SchemaField(
                            type=SchemaType.STRING,
                            description="Output format",
                            required=False,
                            default="pdf",
                        ),
                    }
                ),
                output_schema=Schema(
                    fields={
                        "report": SchemaField(
                            type=SchemaType.STRING,
                            description="Generated report",
                            required=True,
                        )
                    }
                ),
                tags=["report", "output"],
            ),
        ),
    )

    # 3. Create planner and executor
    planner = Planner(registry, factory, prompt_handler)
    executor = PlanExecutor(registry)

    # 4. Generate plan
    print("\nGenerating analysis plan...")
    plan = await planner.create_plan(
        goal="""
        Analyze data from a source and generate a comprehensive report by:
        1. Loading data from the source
        2. Validating data structure and content
        3. Analyzing trends in the validated data
        4. Generating a formatted report with validation and analysis results
        """,
        required_inputs=["data_source"],
        required_output="final_report",
        context={
            "requirements": """
            - Must validate data before analysis
            - Must include trend analysis
            - Must generate formatted report
            """,
            "preferences": {"analysis_type": "comprehensive", "report_format": "pdf"},
        },
    )

    # 5. Visualize plan
    print("\nVisualizing plan structure...")
    artifacts_dir = Path("artifacts/visualizations")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    visualize_plan(
        plan,
        output_path=artifacts_dir
        / f"analysis_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )

    # 6. Execute plan
    print("\nExecuting analysis plan...")
    result = await executor.execute(
        plan,
        initial_context={
            "data_source": "example_dataset.csv",
            "analysis_type": "comprehensive",
            "report_format": "pdf",
        },
    )

    print("\nExecution Results:")
    print(f"Final Report: {result['final_report']}")

    # 7. Display execution metrics
    print("\nExecution Metrics:")
    for task_name, task_result in executor.get_execution_metrics().items():
        print(f"\nTask: {task_name}")
        print(f"Status: {task_result.status}")
        print(f"Duration: {task_result.execution_time:.2f}s")
        print(f"Resource Usage: {task_result.resource_usage}")

    return result


if __name__ == "__main__":
    asyncio.run(dynamic_analysis_workflow())
