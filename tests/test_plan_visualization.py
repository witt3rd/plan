"""Tests for plan visualization"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from plan.capabilities.metadata import CapabilityType
from plan.planning.models import (
    CapabilityMetadata,
    Plan,
    PlanMetadata,
    Task,
    TaskInput,
    TaskOutput,
)
from plan.visualization import PlanVisualizer, VisualizationConfig, visualize_plan


@pytest.fixture
def artifacts_dir():
    """Create and return the artifacts directory"""
    path = Path("artifacts/visualizations")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def timestamp():
    """Get current timestamp string"""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


@pytest.fixture
def sample_plan():
    """Create a sample plan for testing"""
    return Plan(
        name="test_plan",
        description="Test plan",
        goal="Test goal",
        tasks=[
            Task(
                name="load_data",
                capability_name="data_loader",
                description="Load raw data from source",
                inputs=[
                    TaskInput(
                        key="source_path",
                        description="Path to data source",
                        source_key=None,
                    )
                ],
                output=TaskOutput(
                    key="raw_data", description="Raw data loaded from source"
                ),
            ),
            Task(
                name="clean_data",
                capability_name="data_cleaner",
                description="Clean and preprocess data",
                inputs=[
                    TaskInput(
                        key="data", source_key="raw_data", description="Data to clean"
                    )
                ],
                output=TaskOutput(key="clean_data", description="Cleaned dataset"),
            ),
            Task(
                name="analyze_data",
                capability_name="data_analyzer",
                description="Perform data analysis",
                inputs=[
                    TaskInput(
                        key="dataset",
                        source_key="clean_data",
                        description="Dataset to analyze",
                    )
                ],
                output=TaskOutput(
                    key="analysis_results", description="Analysis results"
                ),
            ),
            Task(
                name="generate_report",
                capability_name="report_generator",
                description="Generate final report",
                inputs=[
                    TaskInput(
                        key="results",
                        source_key="analysis_results",
                        description="Results to include in report",
                    )
                ],
                output=TaskOutput(key="final_report", description="Generated report"),
            ),
        ],
        desired_outputs=["final_report"],
        metadata=PlanMetadata(
            description="Data analysis pipeline",
            created_at=datetime.now(UTC),
            version="1.0.0",
        ),
        capability_metadata=CapabilityMetadata(
            name="data_analysis_pipeline",
            description="End-to-end data analysis workflow",
            version="1.0.0",
            type=CapabilityType.PLAN,
            input_schema={},
            output_schema={},
        ),
    )


def test_visualization_creation(sample_plan, artifacts_dir, timestamp):
    """Test basic visualization creation"""
    visualizer = PlanVisualizer()
    output_path = artifacts_dir / f"basic_visualization_{timestamp}.png"
    fig = visualizer.visualize(sample_plan, output_path=output_path)

    assert isinstance(fig, Figure)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_visualization_saving(sample_plan, artifacts_dir, timestamp):
    """Test saving visualization to temporary and permanent locations"""
    visualizer = PlanVisualizer()

    # Test with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "test_plan.png"
        visualizer.visualize(sample_plan, output_path=temp_path)
        assert temp_path.exists()

    # Test with artifacts directory
    output_path = artifacts_dir / f"saved_visualization_{timestamp}.png"
    visualizer.visualize(sample_plan, output_path=output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_custom_config(sample_plan, artifacts_dir, timestamp):
    """Test visualization with custom configuration"""
    config = VisualizationConfig(
        figure_size=(16, 12), dpi=150, show_task_details=False, layout_style="circular"
    )
    visualizer = PlanVisualizer(config)
    output_path = artifacts_dir / f"custom_config_{timestamp}.png"
    fig = visualizer.visualize(sample_plan, output_path=output_path)

    assert isinstance(fig, Figure)
    assert np.allclose(fig.get_size_inches(), (16, 12))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_nested_plan_visualization(sample_plan, artifacts_dir, timestamp):
    """Test visualization of nested plans"""
    # Create a nested plan
    nested_plan = Plan(
        name="nested_plan",
        description="Nested plan",
        goal="Nested goal",
        tasks=[
            Task(
                name="nested_task1",
                capability_name="cap3",
                description="Nested task",
                inputs=[
                    TaskInput(
                        key="nested_input", source_key=None, description="Nested input"
                    )
                ],
                output=TaskOutput(key="nested_output", description="Nested output"),
            )
        ],
        desired_outputs=["nested_output"],
        metadata=PlanMetadata(
            description="Nested plan metadata",
            created_at=datetime.now(UTC),
            version="1.0.0",
        ),
        capability_metadata=CapabilityMetadata(
            name="nested_capability",
            description="Nested capability",
            version="1.0.0",
            type=CapabilityType.PLAN,
            input_schema={},
            output_schema={},
        ),
    )

    # Add nested plan as a task
    sample_plan.tasks.append(
        Task(
            name="nested_plan_task",
            capability_name="plan",
            description="Nested plan task",
            inputs=[
                TaskInput(
                    key="plan_input", source_key="output2", description="Plan input"
                )
            ],
            output=TaskOutput(key="final_output", description="Final output"),
            plan=nested_plan,
        )
    )

    visualizer = PlanVisualizer()
    output_path = artifacts_dir / f"nested_plan_{timestamp}.png"
    fig = visualizer.visualize(sample_plan, output_path=output_path)

    assert isinstance(fig, Figure)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_convenience_function(sample_plan, artifacts_dir, timestamp):
    """Test the convenience function for visualization"""
    output_path = artifacts_dir / f"convenience_function_{timestamp}.png"
    fig = visualize_plan(sample_plan, output_path=output_path)

    assert isinstance(fig, Figure)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_different_layouts(sample_plan, artifacts_dir, timestamp):
    """Test different layout styles"""
    for layout in ["spring", "circular", "hierarchical", "spectral"]:
        config = VisualizationConfig(layout_style=layout)
        visualizer = PlanVisualizer(config)
        output_path = artifacts_dir / f"layout_{layout}_{timestamp}.png"
        fig = visualizer.visualize(
            sample_plan, output_path=output_path, title=f"Layout Style: {layout}"
        )

        assert isinstance(fig, Figure)
        assert output_path.exists()
        assert output_path.stat().st_size > 0
