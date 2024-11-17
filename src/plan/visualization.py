"""Plan visualization with persistence support."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from plan.planning.models import Plan, Task


class LayoutStyle(str, Enum):
    """Available layout algorithms"""

    SPRING = "spring"
    CIRCULAR = "circular"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"


class NodeStyle(BaseModel):
    """Style configuration for graph nodes"""

    shape: str = Field(default="rectangle")
    size: int = Field(default=2000)
    regular_color: str = Field(default="lightblue")
    plan_color: str = Field(default="lightgreen")
    font_size: int = Field(default=10)
    alpha: float = Field(default=0.7)


class EdgeStyle(BaseModel):
    """Style configuration for graph edges"""

    style: str = Field(default="solid")
    color: str = Field(default="gray")
    width: float = Field(default=1.0)
    alpha: float = Field(default=0.6)
    arrow_size: int = Field(default=10)


class VisualizationConfig(BaseModel):
    """Configuration for plan visualization"""

    layout_style: LayoutStyle = Field(default=LayoutStyle.SPRING)
    node_style: NodeStyle = Field(default_factory=NodeStyle)
    edge_style: EdgeStyle = Field(default_factory=EdgeStyle)
    figure_size: Tuple[int, int] = Field(default=(12, 8))
    dpi: int = Field(default=300)
    include_nested: bool = Field(default=True)
    show_task_details: bool = Field(default=True)


class PlanVisualizer:
    """Handles plan visualization and persistence"""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with configuration

        Args:
            config: Optional visualization configuration
        """
        self.config = config or VisualizationConfig()

    def visualize(
        self,
        plan: "Plan",
        output_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Figure:
        """Generate visualization of a plan

        Args:
            plan: Plan to visualize
            output_path: Optional path to save visualization
            title: Optional title for the visualization

        Returns:
            Generated matplotlib figure
        """
        # Create graph
        G = self._create_graph(plan)

        # Create figure
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Get layout
        pos = self._get_layout(G)

        # Draw nodes
        self._draw_nodes(G, pos)

        # Draw edges
        self._draw_edges(G, pos)

        # Add labels
        self._add_labels(G, pos)

        # Add title
        if title:
            plt.title(title)
        else:
            plt.title(f"Plan: {plan.name}")

        # Save if path provided
        if output_path:
            self._save_visualization(fig, output_path)

        return fig

    def _create_graph(self, plan: "Plan") -> nx.DiGraph:
        """Create NetworkX graph from plan

        Args:
            plan: Plan to convert

        Returns:
            Directed graph representation
        """
        if plan is None:
            return nx.DiGraph()

        G = nx.DiGraph()

        # Add all tasks as nodes
        for task in plan.tasks:
            node_attrs = {
                "output": task.output.key,
                "type": "plan_task" if self._is_plan_task(task) else "task",
                "description": task.description,
            }
            G.add_node(task.name, **node_attrs)

            # Add nested tasks if enabled
            if (
                self.config.include_nested
                and self._is_plan_task(task)
                and task.plan is not None
            ):
                nested_graph = self._create_graph(task.plan)
                # Add prefix to nested task names
                mapping = {x: f"{task.name}/{x}" for x in nested_graph.nodes()}
                nested_graph = nx.relabel_nodes(nested_graph, mapping)
                G = nx.compose(G, nested_graph)

        # Add edges for dependencies
        for task in plan.tasks:
            for input in task.inputs:
                if input.source_key:
                    producer = next(
                        (t for t in plan.tasks if t.output.key == input.source_key),
                        None,
                    )
                    if producer:
                        G.add_edge(producer.name, task.name)

        return G

    def _get_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Get node layout based on configuration

        Args:
            G: Graph to layout

        Returns:
            Dictionary mapping node names to positions
        """
        if self.config.layout_style == LayoutStyle.SPRING:
            return nx.spring_layout(G, k=1, iterations=50)
        elif self.config.layout_style == LayoutStyle.CIRCULAR:
            return nx.circular_layout(G)
        elif self.config.layout_style == LayoutStyle.HIERARCHICAL:
            return nx.kamada_kawai_layout(G)
        else:  # SPECTRAL
            return nx.spectral_layout(G)

    def _draw_nodes(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]]) -> None:
        """Draw graph nodes

        Args:
            G: Graph to draw
            pos: Node positions
        """
        # Separate plan tasks and regular tasks
        node_data = dict(G.nodes(data=True))
        plan_tasks = [n for n, d in node_data.items() if d.get("type") == "plan_task"]
        regular_tasks = [n for n, d in node_data.items() if d.get("type") == "task"]

        # Draw regular tasks
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=regular_tasks,
            node_color=self.config.node_style.regular_color,
            node_size=self.config.node_style.size,
            alpha=self.config.node_style.alpha,
        )

        # Draw plan tasks
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=plan_tasks,
            node_color=self.config.node_style.plan_color,
            node_size=self.config.node_style.size,
            alpha=self.config.node_style.alpha,
        )

    def _draw_edges(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]]) -> None:
        """Draw graph edges

        Args:
            G: Graph to draw
            pos: Node positions
        """
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=self.config.edge_style.color,
            width=self.config.edge_style.width,
            alpha=self.config.edge_style.alpha,
            arrowsize=self.config.edge_style.arrow_size,
            style=self.config.edge_style.style,
        )

    def _add_labels(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]]) -> None:
        """Add labels to graph

        Args:
            G: Graph to label
            pos: Node positions
        """
        labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            if self.config.show_task_details:
                label = f"{node}\n({node_data['output']})"
                if "description" in node_data:
                    label += f"\n{node_data['description'][:30]}..."
            else:
                label = f"{node}\n({node_data['output']})"
            labels[node] = label

        nx.draw_networkx_labels(
            G, pos, labels, font_size=self.config.node_style.font_size
        )

    def _save_visualization(self, fig: Figure, path: Path) -> None:
        """Save visualization to file

        Args:
            fig: Figure to save
            path: Output path
        """
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp to filename if it exists
        if path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_path = path.with_stem(f"{path.stem}_{timestamp}")
            path = new_path

        fig.savefig(path, bbox_inches="tight", dpi=self.config.dpi)

    @staticmethod
    def _is_plan_task(task: "Task") -> bool:
        """Check if task is a plan task

        Args:
            task: Task to check

        Returns:
            True if task is a plan task
        """
        return hasattr(task, "plan") and task.plan is not None


def visualize_plan(
    plan: "Plan",
    output_path: Optional[Path] = None,
    config: Optional[VisualizationConfig] = None,
) -> Figure:
    """Visualize a plan structure.

    This is a convenience function that creates a PlanVisualizer and visualizes the plan.

    Args:
        plan: Plan to visualize
        output_path: Optional path to save visualization
        config: Optional visualization configuration

    Returns:
        Generated matplotlib figure
    """
    visualizer = PlanVisualizer(config)
    return visualizer.visualize(plan, output_path)
