import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

# Basic plot settings
PLOT_SETTINGS = {
    "style": "seaborn-v0_8",
    "palette": "husl",
    "figure.figsize": (12, 8),
    "figure.dpi": 100,
}

# Color schemes
COLOR_SCHEMES = {
    "correlation": "coolwarm",
    "risk": "YlOrRd",
    "success": "RdYlGn",
    "categorical": "Set2",
}

# Figure sizes for different plot types
FIGURE_SIZES: Dict[str, Tuple[int, int]] = {
    "default": (12, 8),
    "correlation": (12, 8),
    "distribution": (12, 6),
    "categorical": (12, 5),
    "performance": (10, 6),
    "semester": (15, 10),
}


def setup_plot_style() -> None:
    """Setup default plotting style and parameters."""
    # Set style
    plt.style.use(PLOT_SETTINGS["style"])

    # Set color palette
    sns.set_palette(PLOT_SETTINGS["palette"])

    # Set default figure size and DPI
    plt.rcParams["figure.figsize"] = PLOT_SETTINGS["figure.figsize"]
    plt.rcParams["figure.dpi"] = PLOT_SETTINGS["figure.dpi"]


def get_figure_size(plot_type: str) -> Tuple[int, int]:
    """
    Get recommended figure size for specific plot type.

    Args:
        plot_type: Type of plot

    Returns:
        Tuple containing width and height in inches
    """
    return FIGURE_SIZES.get(plot_type, FIGURE_SIZES["default"])


def get_color_scheme(plot_type: str) -> str:
    """
    Get recommended color scheme for specific plot type.

    Args:
        plot_type: Type of plot

    Returns:
        Color scheme name
    """
    return COLOR_SCHEMES.get(plot_type, COLOR_SCHEMES["categorical"])


# Common styling functions
def style_axis_labels(
    ax: plt.Axes, xlabel: str, ylabel: str, rotation: int = 0
) -> None:
    """
    Apply consistent styling to axis labels.

    Args:
        ax: Matplotlib axes object
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        rotation: Rotation angle for x-axis labels
    """
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if rotation != 0:
        plt.xticks(rotation=rotation)


def style_title(ax: plt.Axes, title: str) -> None:
    """
    Apply consistent styling to plot title.

    Args:
        ax: Matplotlib axes object
        title: Plot title
    """
    ax.set_title(title, fontsize=12, pad=20)
