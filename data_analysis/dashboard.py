"""Interactive Dashboard for BDD100K Dataset Visualization.

Provides an interactive Streamlit dashboard for exploring dataset statistics,
class distributions, and sample visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_analysis_results(results_dir: str) -> Dict:
    """Load analysis results from JSON files.

    Args:
        results_dir: Directory containing analysis JSON files

    Returns:
        Dictionary containing all analysis results
    """
    results_path = Path(results_dir)
    results = {}

    files_to_load = [
        "class_distribution.json",
        "split_comparison.json",
        "image_statistics.json",
        "bbox_statistics.json",
        "occlusion_statistics.json",
        "scene_attributes.json",
        "anomalies.json",
    ]

    for filename in files_to_load:
        file_path = results_path / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                key = filename.replace(".json", "")
                results[key] = json.load(f)
        else:
            logger.warning(f"File not found: {file_path}")

    return results


def plot_class_distribution(data: Dict) -> go.Figure:
    """Create bar chart for class distribution comparison.

    Args:
        data: Class distribution data

    Returns:
        Plotly figure object
    """
    train_data = data["train"]
    val_data = data["val"]

    classes = list(train_data.keys())
    train_counts = list(train_data.values())
    val_counts = list(val_data.values())

    fig = go.Figure(
        data=[
            go.Bar(name="Train", x=classes, y=train_counts),
            go.Bar(name="Val", x=classes, y=val_counts),
        ]
    )

    fig.update_layout(
        title="Class Distribution: Train vs Validation",
        xaxis_title="Class",
        yaxis_title="Number of Instances",
        barmode="group",
        height=500,
    )

    return fig


def plot_class_percentage(data: Dict) -> go.Figure:
    """Create percentage distribution plot.

    Args:
        data: Split comparison data

    Returns:
        Plotly figure object
    """
    classes = []
    train_pcts = []
    val_pcts = []

    for cls, stats in data["class_distributions"].items():
        classes.append(cls)
        train_pcts.append(stats["train_percentage"])
        val_pcts.append(stats["val_percentage"])

    fig = go.Figure(
        data=[
            go.Bar(name="Train", x=classes, y=train_pcts),
            go.Bar(name="Val", x=classes, y=val_pcts),
        ]
    )

    fig.update_layout(
        title="Class Distribution Percentage",
        xaxis_title="Class",
        yaxis_title="Percentage (%)",
        barmode="group",
        height=500,
    )

    return fig


def plot_bbox_sizes(data: Dict, split: str = "train") -> go.Figure:
    """Create box plot for bounding box sizes per class.

    Args:
        data: Bbox statistics data
        split: Either 'train' or 'val'

    Returns:
        Plotly figure object
    """
    bbox_data = data[split]

    classes = []
    mean_areas = []

    for cls, stats in bbox_data.items():
        if stats is not None:
            classes.append(cls)
            mean_areas.append(stats["mean_area"])

    fig = go.Figure(
        data=[go.Bar(x=classes, y=mean_areas, marker_color="lightblue")]
    )

    fig.update_layout(
        title=f"Mean Bounding Box Area by Class ({split.capitalize()})",
        xaxis_title="Class",
        yaxis_title="Mean Area (pixels²)",
        height=500,
    )

    return fig


def plot_occlusion_stats(data: Dict, split: str = "train") -> go.Figure:
    """Create stacked bar chart for occlusion statistics.

    Args:
        data: Occlusion statistics data
        split: Either 'train' or 'val'

    Returns:
        Plotly figure object
    """
    occ_data = data[split]

    classes = []
    occluded_pcts = []
    truncated_pcts = []

    for cls, stats in occ_data.items():
        if stats["total"] > 0:
            classes.append(cls)
            occluded_pcts.append((stats["occluded"] / stats["total"]) * 100)
            truncated_pcts.append((stats["truncated"] / stats["total"]) * 100)

    fig = go.Figure(
        data=[
            go.Bar(name="Occluded", x=classes, y=occluded_pcts),
            go.Bar(name="Truncated", x=classes, y=truncated_pcts),
        ]
    )

    fig.update_layout(
        title=f"Occlusion and Truncation Rates ({split.capitalize()})",
        xaxis_title="Class",
        yaxis_title="Percentage (%)",
        barmode="group",
        height=500,
    )

    return fig


def plot_scene_attributes(data: Dict, split: str = "train") -> go.Figure:
    """Create pie chart for scene attributes.

    Args:
        data: Scene attributes data
        split: Either 'train' or 'val'

    Returns:
        Plotly figure object
    """
    scene_data = data[split]

    # Weather distribution
    weather_labels = list(scene_data["weather"].keys())
    weather_values = list(scene_data["weather"].values())

    fig = go.Figure(
        data=[go.Pie(labels=weather_labels, values=weather_values, hole=0.3)]
    )

    fig.update_layout(
        title=f"Weather Distribution ({split.capitalize()})", height=500
    )

    return fig


def plot_timeofday_distribution(data: Dict, split: str = "train") -> go.Figure:
    """Create pie chart for time of day distribution.

    Args:
        data: Scene attributes data
        split: Either 'train' or 'val'

    Returns:
        Plotly figure object
    """
    scene_data = data[split]

    # Time of day distribution
    time_labels = list(scene_data["timeofday"].keys())
    time_values = list(scene_data["timeofday"].values())

    fig = go.Figure(
        data=[go.Pie(labels=time_labels, values=time_values, hole=0.3)]
    )

    fig.update_layout(
        title=f"Time of Day Distribution ({split.capitalize()})", height=500
    )

    return fig


def main():
    """Main Streamlit dashboard application."""
    st.set_page_config(page_title="BDD100K Dataset Analysis", layout="wide")

    st.title("🚗 BDD100K Object Detection Dataset Analysis")
    st.markdown(
        "Interactive dashboard for exploring the BDD100K dataset statistics and distributions."
    )

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    results_dir = st.sidebar.text_input(
        "Results Directory", value="/app/results"
    )

    # Load data
    try:
        results = load_analysis_results(results_dir)
        st.sidebar.success(f"✅ Loaded {len(results)} analysis files")
    except Exception as e:
        st.error(f"Failed to load analysis results: {e}")
        st.stop()

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Class Distribution",
            "📦 Bounding Boxes",
            "🔍 Occlusion & Truncation",
            "🌤️ Scene Attributes",
            "⚠️ Anomalies",
        ]
    )

    # Tab 1: Class Distribution
    with tab1:
        st.header("Class Distribution Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Training Objects",
                results["class_distribution"]["total_train_objects"],
            )
        with col2:
            st.metric(
                "Total Validation Objects",
                results["class_distribution"]["total_val_objects"],
            )

        st.plotly_chart(
            plot_class_distribution(results["class_distribution"]),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_class_percentage(results["split_comparison"]),
            use_container_width=True,
        )

        # Display statistics
        st.subheader("Image Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Training Set**")
            train_stats = results["image_statistics"]["train"]
            st.write(f"- Total images: {train_stats['total_images']}")
            st.write(
                f"- Images with objects: {train_stats['images_with_objects']}"
            )
            st.write(
                f"- Avg objects/image: {train_stats['avg_objects_per_image']:.2f}"
            )
            st.write(
                f"- Max objects/image: {train_stats['max_objects_per_image']}"
            )

        with col2:
            st.write("**Validation Set**")
            val_stats = results["image_statistics"]["val"]
            st.write(f"- Total images: {val_stats['total_images']}")
            st.write(f"- Images with objects: {val_stats['images_with_objects']}")
            st.write(
                f"- Avg objects/image: {val_stats['avg_objects_per_image']:.2f}"
            )
            st.write(f"- Max objects/image: {val_stats['max_objects_per_image']}")

    # Tab 2: Bounding Boxes
    with tab2:
        st.header("Bounding Box Statistics")

        split_choice = st.radio("Select Split", ["train", "val"], horizontal=True)

        st.plotly_chart(
            plot_bbox_sizes(results["bbox_statistics"], split_choice),
            use_container_width=True,
        )

        st.subheader("Detailed Statistics")
        bbox_data = results["bbox_statistics"][split_choice]

        # Create dataframe for display
        bbox_table = []
        for cls, stats in bbox_data.items():
            if stats is not None:
                bbox_table.append(
                    {
                        "Class": cls,
                        "Mean Area": f"{stats['mean_area']:.0f}",
                        "Median Area": f"{stats['median_area']:.0f}",
                        "Mean Width": f"{stats['mean_width']:.0f}",
                        "Mean Height": f"{stats['mean_height']:.0f}",
                        "Aspect Ratio": f"{stats['mean_aspect_ratio']:.2f}",
                    }
                )

        st.dataframe(bbox_table, use_container_width=True)

    # Tab 3: Occlusion & Truncation
    with tab3:
        st.header("Occlusion and Truncation Analysis")

        split_choice = st.radio(
            "Select Split", ["train", "val"], horizontal=True, key="occ_split"
        )

        st.plotly_chart(
            plot_occlusion_stats(results["occlusion_statistics"], split_choice),
            use_container_width=True,
        )

        st.subheader("Detailed Statistics")
        occ_data = results["occlusion_statistics"][split_choice]

        occ_table = []
        for cls, stats in occ_data.items():
            if stats["total"] > 0:
                occ_table.append(
                    {
                        "Class": cls,
                        "Total Objects": stats["total"],
                        "Occluded": stats["occluded"],
                        "Occluded %": f"{(stats['occluded']/stats['total']*100):.1f}%",
                        "Truncated": stats["truncated"],
                        "Truncated %": f"{(stats['truncated']/stats['total']*100):.1f}%",
                    }
                )

        st.dataframe(occ_table, use_container_width=True)

    # Tab 4: Scene Attributes
    with tab4:
        st.header("Scene Attributes Distribution")

        split_choice = st.radio(
            "Select Split", ["train", "val"], horizontal=True, key="scene_split"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_scene_attributes(results["scene_attributes"], split_choice),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                plot_timeofday_distribution(
                    results["scene_attributes"], split_choice
                ),
                use_container_width=True,
            )

    # Tab 5: Anomalies
    with tab5:
        st.header("Dataset Anomalies")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training Set Anomalies")
            train_anom = results["anomalies"]["train"]
            for anom_type, images in train_anom.items():
                st.write(f"**{anom_type.replace('_', ' ').title()}**: {len(images)} images")
                if images:
                    with st.expander(f"Show examples"):
                        st.write(images[:10])

        with col2:
            st.subheader("Validation Set Anomalies")
            val_anom = results["anomalies"]["val"]
            for anom_type, images in val_anom.items():
                st.write(f"**{anom_type.replace('_', ' ').title()}**: {len(images)} images")
                if images:
                    with st.expander(f"Show examples"):
                        st.write(images[:10])


if __name__ == "__main__":
    main()
