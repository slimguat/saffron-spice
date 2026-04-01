"""
Model-structure visualization utilities.

This module provides a  implementation for drawing a
compact dependency diagram for spectral fitting models. The implementation is
organized into short helpers following NASA coding standards (this is a test of it).

Typical usage
-------------
>>> # Optional: enable LaTeX once for the session.
>>> # configure_model_visualization_latex(True)

>>> functions = {
...     "gaussian": {
...         0: {"I": 1.0, "x": 765.152, "s": 0.12},
...         1: {
...             "I": {
...                 "constraint": "lock",
...                 "operation": "mul",
...                 "value": 0.4,
...                 "reference": {
...                     "model_type": "gaussian",
...                     "element_index": 0,
...                     "parameter": "I",
...                 },
...             },
...             "x": {
...                 "constraint": "lock",
...                 "operation": "add",
...                 "value": 0.5,
...                 "reference": {
...                     "model_type": "gaussian",
...                     "element_index": 0,
...                     "parameter": "x",
...                 },
...             },
...             "s": {
...                 "constraint": "lock",
...                 "operation": "add",
...                 "value": 0.0,
...                 "reference": {
...                     "model_type": "gaussian",
...                     "element_index": 0,
...                     "parameter": "s",
...                 },
...             },
...         },
...     },
...     "polynome": {
...         0: {"B0": 0.0, "B1": 0.01, "lims": [754, 778]}
...     },
... }
>>> names = {
...     "gaussian": [["o_5", 765.152], ["o_5", 765.652]],
...     "polynome": [None],
... }
>>> fig, ax = visualize_model_structure(functions, functions_names=names, title="")
>>> ax is fig.axes[0]
True
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import roman


Coordinate = Tuple[float, float]
CardKey = Tuple[str, int]
ParamKey = Tuple[str, int, str]
RouteId = int


@dataclass(frozen=True)
class ElementSpec:
    """
    Immutable description of one model element.

    Attributes
    ----------
    model_type : str
        Model-family name, for example ``"gaussian"`` or ``"polynome"``.
    element_index : int
        Integer identifier of the element inside its model family.
    parameters : Mapping[str, Any]
        Dictionary of parameter values or constraint dictionaries.
    display_name : Optional[str]
        Human-readable label shown in the card header.

    Examples
    --------
    >>> elem = ElementSpec("gaussian", 0, {"I": 1.0, "x": 765.0}, "O V @ 765")
    >>> elem.element_index
    0
    >>> "I" in elem.parameters
    True
    """

    model_type: str
    element_index: int
    parameters: Mapping[str, Any]
    display_name: Optional[str]


@dataclass(frozen=True)
class ConstraintEdge:
    """
    Directed dependency edge between two parameters.

    The semantic direction is:
        constrained parameter -> reference parameter

    Attributes
    ----------
    source : ParamKey
        Reference parameter key ``(model_type, element_index, parameter_name)``.
    target : ParamKey
        Constrained parameter key.
    operation : Optional[str]
        Constraint operation, typically ``"add"`` or ``"mul"``.
    value : Any
        Constraint coefficient or offset.
    constraint : Optional[str]
        Constraint mode, for example ``"lock"``.

    Examples
    --------
    >>> edge = ConstraintEdge(
    ...     source=("gaussian", 0, "I"),
    ...     target=("gaussian", 1, "I"),
    ...     operation="mul",
    ...     value=0.4,
    ...     constraint="lock",
    ... )
    >>> edge.operation
    'mul'
    """

    source: ParamKey
    target: ParamKey
    operation: Optional[str]
    value: Any
    constraint: Optional[str]


@dataclass(frozen=True)
class CardRect:
    """
    Rectangle geometry for one card.

    Attributes
    ----------
    x : float
        Left coordinate of the card.
    y : float
        Top coordinate of the card.
    w : float
        Width of the card.
    h : float
        Height of the card.
    """

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class LayoutSpec:
    """
    Global layout parameters for the visualization canvas.

    Notes
    -----
    The layout is intentionally uniform: every card has the same size, based on
    the largest parameter count present in the model. This simplifies routing
    and keeps the figure predictable.

    Attributes
    ----------
    n_cards, n_cols, n_rows : int
        Number of cards, columns, and rows.
    scale : float
        Global size reduction factor.
    header_h, row_h, footer_h : float
        Internal card geometry.
    box_h, box_w : float
        Uniform card dimensions.
    inner_gap : float
        Gap between neighboring cards.
    x_step, y_step : float
        Horizontal and vertical spacing between card origins.
    left_pad, right_pad, bottom_pad, top_pad : float
        Outer figure padding.
    main_title_fs, sub_title_fs, row_text_fs : float
        Font sizes used for card contents.
    """

    n_cards: int
    n_cols: int
    n_rows: int
    scale: float
    header_h: float
    row_h: float
    footer_h: float
    box_h: float
    box_w: float
    inner_gap: float
    x_step: float
    y_step: float
    left_pad: float
    right_pad: float
    bottom_pad: float
    top_pad: float
    main_title_fs: float
    sub_title_fs: float
    row_text_fs: float


@dataclass(frozen=True)
class LaneSegment:
    """
    Segment occupying a routing lane.

    Attributes
    ----------
    segment_id : Tuple[int, str]
        Tuple ``(route_id, lane_part)`` where ``lane_part`` is one of
        ``"v1"``, ``"v2"``, or ``"h"``.
    start : float
        Lower coordinate on the lane axis.
    end : float
        Upper coordinate on the lane axis.

    Examples
    --------
    >>> seg = LaneSegment((3, "v1"), 1.0, 4.0)
    >>> seg.end > seg.start
    True
    """

    segment_id: Tuple[int, str]
    start: float
    end: float


def configure_model_visualization_latex(enable: bool = True) -> None:
    """
    Configure Matplotlib LaTeX rendering for ion labels.

    Parameters
    ----------
    enable : bool, default=True
        If ``True``, enables ``text.usetex`` and installs a compact text
        preamble using ``newtxtext``. If ``False``, disables TeX rendering.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `enable` is not a boolean.

    Notes
    -----
    This function changes global Matplotlib state. Call it once during
    initialization if you want LaTeX-rendered ion labels.

    Examples
    --------
    >>> old = bool(mpl.rcParams["text.usetex"])
    >>> configure_model_visualization_latex(False)
    >>> bool(mpl.rcParams["text.usetex"])
    False
    >>> configure_model_visualization_latex(old)
    """
    assert isinstance(enable, bool), "`enable` must be a boolean."
    assert "text.usetex" in mpl.rcParams, "Matplotlib rcParams must be available."

    if enable:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage[T1]{fontenc}" "\n" r"\usepackage{newtxtext}"
        )
    else:
        mpl.rcParams["text.usetex"] = False


def visualize_model_structure(
    model_or_functions: Any,
    functions_names: Optional[Mapping[str, Sequence[Any]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Model structure",
    show_values: bool = True,
    show_constraint_details: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Draw a compact dependency diagram for a model definition.

    Parameters
    ----------
    model_or_functions : Any
        Either a mapping with the structure of ``model.functions`` or an object
        exposing a ``functions`` attribute and optionally a
        ``functions_names`` attribute.
    functions_names : Optional[Mapping[str, Sequence[Any]]], default=None
        Optional mapping mirroring the structure of ``functions_names`` inside
        the model object. If not provided and `model_or_functions` has a
        ``functions_names`` attribute, that attribute is used.
    figsize : Optional[Tuple[float, float]], default=None
        Explicit Matplotlib figure size. If omitted, a deterministic compact
        size is derived from the number of cards.
    title : str, default="Model structure"
        Figure title shown above the grid.
    show_values : bool, default=True
        If ``True``, show free parameter values inside cards.
    show_constraint_details : bool, default=True
        If ``True``, show full constraint details for constrained parameters.
    ax : Optional[Axes], default=None
        Existing axes object to draw into. If omitted, a new figure and axes
        are created.

    Returns
    -------
    (Figure, Axes)
        The Matplotlib figure and axes used for the drawing.

    Raises
    ------
    TypeError
        If the input object does not provide model-function information.
    ValueError
        If no valid elements can be extracted from the input model.

    Notes
    -----
    The routing strategy is orthogonal and lane-based:
    1. Each dependency is routed from the constrained parameter to the
       referenced parameter.
    2. Routes leave cards horizontally, travel in shared vertical lanes, and
       cross between rows in shared horizontal lanes.
    3. When multiple routes share a lane, a simple greedy track assignment
       distributes them inside the available gap to reduce overlaps.

    This is not a full graph-routing engine, but it works well for compact
    model grids and remains deterministic.

    Examples
    --------
    >>> functions = {"gaussian": {0: {"I": 1.0, "x": 765.0, "s": 0.1}}}
    >>> fig, ax = visualize_model_structure(functions, title="")
    >>> isinstance(fig, Figure) and isinstance(ax, Axes)
    True

    Test Suggestions
    ----------------
    - Verify that a single unconstrained Gaussian produces one card and no
      routes.
    - Verify that a constrained parameter produces one route.
    - Verify that multiple overlapping routes are distributed across tracks.
    - Verify that both dict input and object input are accepted.
    """
    _validate_visualization_inputs(title, show_values, show_constraint_details)

    functions, extracted_names = _extract_functions_and_names(
        model_or_functions,
        functions_names,
    )
    elements = _normalize_elements(functions, extracted_names)
    _validate_extracted_elements(elements)

    edges = _build_constraint_edges(elements)
    layout = _compute_layout(elements)
    figure_size = _resolve_figsize(layout, figsize)
    fig, ax, created_fig = _resolve_figure_and_axes(figure_size, ax)

    bounds = _build_card_bounds(elements, layout)
    param_anchors = _draw_cards(
        ax,
        elements,
        bounds,
        layout,
        show_values,
        show_constraint_details,
    )

    routes, route_meta = _prepare_routes(
        elements,
        edges,
        bounds,
        param_anchors,
        layout,
    )
    vertical_usage, horizontal_usage = _build_corridor_occupancy(routes)
    vertical_offsets, horizontal_offsets = _build_corridor_offsets(
        vertical_usage,
        horizontal_usage,
        layout.inner_gap,
    )

    _draw_routes(
        ax,
        routes,
        route_meta,
        vertical_offsets,
        horizontal_offsets,
    )
    _finalize_axes(ax, layout, title, created_fig, fig)

    return fig, ax


def _validate_visualization_inputs(
    title: str,
    show_values: bool,
    show_constraint_details: bool,
) -> None:
    """
    Validate public `visualize_model_structure` inputs.

    Parameters
    ----------
    title : str
        Figure title.
    show_values : bool
        Free-value toggle.
    show_constraint_details : bool
        Constraint-detail toggle.

    Returns
    -------
    None
    """
    assert isinstance(show_values, bool), "`show_values` must be boolean."
    assert isinstance(show_constraint_details, bool), (
        "`show_constraint_details` must be boolean."
    )

    if not isinstance(title, str):
        raise TypeError("`title` must be a string.")
    if len(title) > 500:
        raise ValueError("`title` is unexpectedly long.")


def _extract_functions_and_names(
    model_or_functions: Any,
    functions_names: Optional[Mapping[str, Sequence[Any]]] = None,
) -> Tuple[Mapping[str, Any], Optional[Mapping[str, Sequence[Any]]]]:
    """
    Extract `functions` and `functions_names` from a dict or model-like object.

    Parameters
    ----------
    model_or_functions : Any
        Input mapping or model-like object.
    functions_names : Optional[Mapping[str, Sequence[Any]]]
        Optional explicit name mapping.

    Returns
    -------
    (functions, names)
        The extracted functions mapping and the resolved names mapping.

    Raises
    ------
    TypeError
        If no valid `functions` source exists.

    Examples
    --------
    >>> funcs, names = _extract_functions_and_names({"gaussian": {}}, None)
    >>> "gaussian" in funcs
    True
    """
    assert model_or_functions is not None, "Input object must not be None."
    assert functions_names is None or isinstance(functions_names, Mapping), (
        "`functions_names` must be a mapping or None."
    )

    if isinstance(model_or_functions, Mapping):
        return model_or_functions, functions_names

    if not hasattr(model_or_functions, "functions"):
        raise TypeError(
            "Input must be a mapping or an object exposing a `.functions` attribute."
        )

    functions = getattr(model_or_functions, "functions")
    names = functions_names
    if names is None and hasattr(model_or_functions, "functions_names"):
        names = getattr(model_or_functions, "functions_names")

    if not isinstance(functions, Mapping):
        raise TypeError("The `.functions` attribute must be a mapping.")

    return functions, names


def _normalize_elements(
    functions: Mapping[str, Any],
    functions_names: Optional[Mapping[str, Sequence[Any]]] = None,
) -> List[ElementSpec]:
    """
    Convert raw function dictionaries into normalized `ElementSpec` objects.

    Parameters
    ----------
    functions : Mapping[str, Any]
        Raw `functions` mapping from a model object.
    functions_names : Optional[Mapping[str, Sequence[Any]]]
        Optional display-name mapping.

    Returns
    -------
    List[ElementSpec]
        Normalized, sorted element descriptions.

    Notes
    -----
    Invalid or non-dictionary element payloads are skipped intentionally.
    The outer loop remains bounded by the size of `functions`, and the inner
    loop remains bounded by the size of each model-family mapping.
    """
    assert isinstance(functions, Mapping), "`functions` must be a mapping."
    assert functions_names is None or isinstance(functions_names, Mapping), (
        "`functions_names` must be a mapping or None."
    )

    elements: List[ElementSpec] = []

    for model_type, element_dict in functions.items():
        if not isinstance(element_dict, Mapping):
            continue

        sorted_items = sorted(element_dict.items(), key=lambda item: item[0])
        for element_index, params in sorted_items:
            if not isinstance(params, Mapping):
                continue

            display_name = _get_display_name(
                functions_names,
                str(model_type),
                int(element_index),
            )
            elements.append(
                ElementSpec(
                    model_type=str(model_type),
                    element_index=int(element_index),
                    parameters=params,
                    display_name=display_name,
                )
            )

    return elements


def _validate_extracted_elements(elements: Sequence[ElementSpec]) -> None:
    """
    Validate normalized elements before drawing.

    Parameters
    ----------
    elements : Sequence[ElementSpec]
        Normalized element list.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the element sequence is empty.
    """
    assert isinstance(elements, Sequence), "`elements` must be a sequence."
    assert all(isinstance(elem, ElementSpec) for elem in elements), (
        "All entries in `elements` must be ElementSpec instances."
    )

    if len(elements) == 0:
        raise ValueError("No model functions found to visualize.")


def _build_constraint_edges(elements: Sequence[ElementSpec]) -> List[ConstraintEdge]:
    """
    Extract dependency edges from constrained parameter definitions.

    Parameters
    ----------
    elements : Sequence[ElementSpec]
        Normalized model elements.

    Returns
    -------
    List[ConstraintEdge]
        Extracted dependency list.

    Examples
    --------
    >>> elems = [ElementSpec("g", 0, {"I": 1.0}, None)]
    >>> _build_constraint_edges(elems)
    []
    """
    assert isinstance(elements, Sequence), "`elements` must be a sequence."
    assert all(isinstance(elem, ElementSpec) for elem in elements), (
        "All `elements` must be ElementSpec instances."
    )

    edges: List[ConstraintEdge] = []

    for elem in elements:
        for param_name, param_info in elem.parameters.items():
            if not _is_constraint(param_info):
                continue

            ref = param_info["reference"]
            edges.append(
                ConstraintEdge(
                    source=(
                        str(ref["model_type"]),
                        int(ref["element_index"]),
                        str(ref["parameter"]),
                    ),
                    target=(elem.model_type, elem.element_index,
                            str(param_name)),
                    operation=_optional_str(param_info.get("operation")),
                    value=param_info.get("value"),
                    constraint=_optional_str(param_info.get("constraint")),
                )
            )

    return edges


def _compute_layout(elements: Sequence[ElementSpec]) -> LayoutSpec:
    """
    Compute uniform layout parameters for the full drawing.

    Parameters
    ----------
    elements : Sequence[ElementSpec]
        Normalized elements.

    Returns
    -------
    LayoutSpec
        Fully resolved layout specification.

    Notes
    -----
    Width is constrained to be exactly twice the height for every card.

    Examples
    --------
    >>> elems = [ElementSpec("g", 0, {"I": 1.0, "x": 1.0}, None)]
    >>> layout = _compute_layout(elems)
    >>> layout.box_w == 2.0 * layout.box_h
    True
    """
    assert isinstance(elements, Sequence), "`elements` must be a sequence."
    assert len(elements) > 0, "`elements` must not be empty."

    n_cards = len(elements)
    n_cols = min(5, n_cards)
    n_rows = math.ceil(n_cards / n_cols)
    scale = 0.60

    header_h = 1.35 * scale
    row_h = 0.72 * scale
    footer_h = 0.30 * scale

    max_param_count = max(len(elem.parameters) for elem in elements)
    box_h = header_h + max_param_count * row_h + footer_h
    box_w = 2.0 * box_h

    inner_gap = box_w / 5.0
    x_step = box_w + inner_gap
    y_step = box_h + inner_gap

    left_pad = box_h / 10.0
    right_pad = box_h / 10.0
    bottom_pad = box_h / 10.0
    top_pad = box_w / 5.0

    return LayoutSpec(
        n_cards=n_cards,
        n_cols=n_cols,
        n_rows=n_rows,
        scale=scale,
        header_h=header_h,
        row_h=row_h,
        footer_h=footer_h,
        box_h=box_h,
        box_w=box_w,
        inner_gap=inner_gap,
        x_step=x_step,
        y_step=y_step,
        left_pad=left_pad,
        right_pad=right_pad,
        bottom_pad=bottom_pad,
        top_pad=top_pad,
        main_title_fs=16.0 * scale,
        sub_title_fs=16.2 * scale,
        row_text_fs=12.8 * scale,
    )


def _resolve_figsize(
    layout: LayoutSpec,
    figsize: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Resolve the figure size to use for plotting.

    Parameters
    ----------
    layout : LayoutSpec
        Layout specification.
    figsize : Optional[Tuple[float, float]]
        Caller-provided figure size.

    Returns
    -------
    Tuple[float, float]
        Figure size in inches.
    """
    assert isinstance(layout, LayoutSpec), "`layout` must be a LayoutSpec."
    assert figsize is None or len(figsize) == 2, "`figsize` must be a 2-tuple."

    if figsize is not None:
        return figsize

    return (layout.n_cols * 2.0, layout.n_rows * 2.0)


def _resolve_figure_and_axes(
    figsize: Tuple[float, float],
    ax: Optional[Axes],
) -> Tuple[Figure, Axes, bool]:
    """
    Resolve figure and axes objects.

    Parameters
    ----------
    figsize : Tuple[float, float]
        Requested figure size.
    ax : Optional[Axes]
        Existing axes, if any.

    Returns
    -------
    (Figure, Axes, bool)
        Figure, axes, and a flag indicating whether the figure was created here.
    """
    assert len(figsize) == 2, "`figsize` must contain width and height."
    assert ax is None or isinstance(
        ax, Axes), "`ax` must be a Matplotlib Axes."

    if ax is not None:
        return ax.figure, ax, False

    fig, new_ax = plt.subplots(figsize=figsize)
    return fig, new_ax, True


def _build_card_bounds(
    elements: Sequence[ElementSpec],
    layout: LayoutSpec,
) -> Dict[CardKey, CardRect]:
    """
    Compute rectangle positions for all cards in the grid.

    Parameters
    ----------
    elements : Sequence[ElementSpec]
        Normalized elements.
    layout : LayoutSpec
        Layout parameters.

    Returns
    -------
    Dict[CardKey, CardRect]
        Map from card key to rectangle geometry.
    """
    assert isinstance(layout, LayoutSpec), "`layout` must be a LayoutSpec."
    assert len(elements) > 0, "`elements` must not be empty."

    bounds: Dict[CardKey, CardRect] = {}

    for index, elem in enumerate(elements):
        row = index // layout.n_cols
        col = index % layout.n_cols
        x = col * layout.x_step
        y = -row * layout.y_step
        bounds[(elem.model_type, elem.element_index)] = CardRect(
            x=x,
            y=y,
            w=layout.box_w,
            h=layout.box_h,
        )

    return bounds


def _draw_cards(
    ax: Axes,
    elements: Sequence[ElementSpec],
    bounds: Mapping[CardKey, CardRect],
    layout: LayoutSpec,
    show_values: bool,
    show_constraint_details: bool,
) -> Dict[ParamKey, Dict[str, Coordinate]]:
    """
    Draw all cards and return parameter-anchor coordinates for routing.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    elements : Sequence[ElementSpec]
        Normalized elements.
    bounds : Mapping[CardKey, CardRect]
        Card geometry.
    layout : LayoutSpec
        Layout parameters.
    show_values : bool
        Free-value toggle.
    show_constraint_details : bool
        Constraint-detail toggle.

    Returns
    -------
    Dict[ParamKey, Dict[str, Coordinate]]
        Per-parameter anchor points keyed by ``left``, ``right``, and ``center``.
    """
    assert isinstance(ax, Axes), "`ax` must be a Matplotlib Axes."
    assert isinstance(layout, LayoutSpec), "`layout` must be a LayoutSpec."

    param_anchors: Dict[ParamKey, Dict[str, Coordinate]] = {}

    for elem in elements:
        key = (elem.model_type, elem.element_index)
        rect = bounds[key]
        _draw_single_card(
            ax,
            elem,
            rect,
            layout,
            show_values,
            show_constraint_details,
            param_anchors,
        )

    return param_anchors


def _draw_single_card(
    ax: Axes,
    elem: ElementSpec,
    rect: CardRect,
    layout: LayoutSpec,
    show_values: bool,
    show_constraint_details: bool,
    param_anchors: MutableMapping[ParamKey, Dict[str, Coordinate]],
) -> None:
    """
    Draw one card and register its parameter anchors.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    elem : ElementSpec
        Element to draw.
    rect : CardRect
        Card geometry.
    layout : LayoutSpec
        Layout parameters.
    show_values : bool
        Free-value toggle.
    show_constraint_details : bool
        Constraint-detail toggle.
    param_anchors : MutableMapping[ParamKey, Dict[str, Coordinate]]
        Output anchor map that is updated in place.

    Returns
    -------
    None
    """
    assert isinstance(elem, ElementSpec), "`elem` must be an ElementSpec."
    assert isinstance(rect, CardRect), "`rect` must be a CardRect."

    facecolor, header_color, edgecolor = _card_colors(elem.model_type)
    _draw_card_background(ax, rect, facecolor, header_color, edgecolor, layout)
    _draw_card_header(ax, elem, rect, layout)
    _draw_card_rows(
        ax,
        elem,
        rect,
        layout,
        show_values,
        show_constraint_details,
        param_anchors,
    )


def _draw_card_background(
    ax: Axes,
    rect: CardRect,
    facecolor: str,
    header_color: str,
    edgecolor: str,
    layout: LayoutSpec,
) -> None:
    """
    Draw the background of a card, including the colored header band.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    rect : CardRect
        Card rectangle.
    facecolor : str
        Main card color.
    header_color : str
        Header-band color.
    edgecolor : str
        Card-edge color.
    layout : LayoutSpec
        Layout parameters.

    Returns
    -------
    None
    """
    assert rect.w > 0.0, "Card width must be positive."
    assert rect.h > 0.0, "Card height must be positive."

    outer = FancyBboxPatch(
        (rect.x, rect.y - rect.h),
        rect.w,
        rect.h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=1,
    )
    header = FancyBboxPatch(
        (rect.x + 0.04 * rect.w, rect.y - layout.header_h - 0.03 * rect.h),
        rect.w - 0.08 * rect.w,
        layout.header_h,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=0.0,
        edgecolor=header_color,
        facecolor=header_color,
        zorder=2,
    )

    ax.add_patch(outer)
    ax.add_patch(header)


def _draw_card_header(
    ax: Axes,
    elem: ElementSpec,
    rect: CardRect,
    layout: LayoutSpec,
) -> None:
    """
    Draw the two-line header content for one card.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    elem : ElementSpec
        Element to draw.
    rect : CardRect
        Card rectangle.
    layout : LayoutSpec
        Layout parameters.

    Returns
    -------
    None
    """
    assert isinstance(elem.model_type, str), "`model_type` must be a string."
    assert elem.element_index >= 0, "`element_index` must be non-negative."

    main_title = elem.display_name
    sub_title = f"{elem.model_type}[{elem.element_index}]"

    if main_title is not None and str(main_title).strip():
        ax.text(
            rect.x + 0.07 * rect.w,
            rect.y - 0.32 * layout.header_h,
            main_title,
            ha="left",
            va="center",
            fontsize=layout.main_title_fs,
            fontweight="bold",
            color="#0f172a",
            zorder=3,
        )
        ax.text(
            rect.x + 0.07 * rect.w,
            rect.y - 0.78 * layout.header_h,
            sub_title,
            ha="left",
            va="center",
            fontsize=layout.sub_title_fs,
            color="#334155",
            zorder=3,
        )
        return

    ax.text(
        rect.x + 0.07 * rect.w,
        rect.y - 0.55 * layout.header_h,
        sub_title,
        ha="left",
        va="center",
        fontsize=layout.main_title_fs,
        fontweight="bold",
        color="#0f172a",
        zorder=3,
    )


def _draw_card_rows(
    ax: Axes,
    elem: ElementSpec,
    rect: CardRect,
    layout: LayoutSpec,
    show_values: bool,
    show_constraint_details: bool,
    param_anchors: MutableMapping[ParamKey, Dict[str, Coordinate]],
) -> None:
    """
    Draw parameter rows for one card and record their routing anchors.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    elem : ElementSpec
        Element being drawn.
    rect : CardRect
        Card rectangle.
    layout : LayoutSpec
        Layout parameters.
    show_values : bool
        Free-value toggle.
    show_constraint_details : bool
        Constraint-detail toggle.
    param_anchors : MutableMapping[ParamKey, Dict[str, Coordinate]]
        Anchor map updated in place.

    Returns
    -------
    None
    """
    assert len(elem.parameters) >= 0, "Parameter count must be non-negative."
    assert isinstance(param_anchors, MutableMapping), (
        "`param_anchors` must support item assignment."
    )

    for row_index, (param_name, param_info) in enumerate(elem.parameters.items()):
        row_top = rect.y - layout.header_h - row_index * layout.row_h - 0.04 * rect.h
        row_y = row_top - layout.row_h + 0.02 * rect.h
        row_h = layout.row_h - 0.04 * rect.h

        _draw_parameter_row_background(ax, rect, row_y, row_h, param_name)
        _draw_parameter_row_text(
            ax,
            rect,
            row_y,
            row_h,
            layout,
            param_name,
            param_info,
            show_values,
            show_constraint_details,
        )
        _register_parameter_anchors(
            param_anchors,
            elem,
            str(param_name),
            rect,
            row_y,
            row_h,
        )


def _draw_parameter_row_background(
    ax: Axes,
    rect: CardRect,
    row_y: float,
    row_h: float,
    param_name: str,
) -> None:
    """
    Draw the colored rounded rectangle behind one parameter row.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    rect : CardRect
        Card rectangle.
    row_y : float
        Lower y coordinate of the row.
    row_h : float
        Row height.
    param_name : str
        Parameter name used for color selection.

    Returns
    -------
    None
    """
    assert row_h > 0.0, "Row height must be positive."
    assert rect.w > 0.0, "Card width must be positive."

    patch = FancyBboxPatch(
        (rect.x + 0.05 * rect.w, row_y),
        rect.w - 0.10 * rect.w,
        row_h,
        boxstyle="round,pad=0.008,rounding_size=0.04",
        linewidth=0.5,
        edgecolor="#cbd5e1",
        facecolor=_param_color(param_name),
        zorder=2,
    )
    ax.add_patch(patch)


def _draw_parameter_row_text(
    ax: Axes,
    rect: CardRect,
    row_y: float,
    row_h: float,
    layout: LayoutSpec,
    param_name: str,
    param_info: Any,
    show_values: bool,
    show_constraint_details: bool,
) -> None:
    """
    Draw the text label inside one parameter row.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    rect : CardRect
        Card rectangle.
    row_y : float
        Lower y coordinate of the row.
    row_h : float
        Row height.
    layout : LayoutSpec
        Layout parameters.
    param_name : str
        Parameter name.
    param_info : Any
        Parameter value or constraint dictionary.
    show_values : bool
        Free-value toggle.
    show_constraint_details : bool
        Constraint-detail toggle.

    Returns
    -------
    None
    """
    assert isinstance(layout, LayoutSpec), "`layout` must be a LayoutSpec."
    assert isinstance(param_name, str), "`param_name` must be a string."

    label = _format_param_label(
        param_name,
        param_info,
        show_values=show_values,
        show_constraint_details=show_constraint_details,
    )
    ax.text(
        rect.x + 0.08 * rect.w,
        row_y + row_h / 2.0,
        label,
        ha="left",
        va="center",
        fontsize=layout.row_text_fs,
        color="#0f172a",
        zorder=3,
    )


def _register_parameter_anchors(
    param_anchors: MutableMapping[ParamKey, Dict[str, Coordinate]],
    elem: ElementSpec,
    param_name: str,
    rect: CardRect,
    row_y: float,
    row_h: float,
) -> None:
    """
    Register the left, right, and center anchors for one parameter row.

    Parameters
    ----------
    param_anchors : MutableMapping[ParamKey, Dict[str, Coordinate]]
        Anchor map updated in place.
    elem : ElementSpec
        Parent element.
    param_name : str
        Parameter name.
    rect : CardRect
        Card rectangle.
    row_y : float
        Lower y coordinate of the row.
    row_h : float
        Row height.

    Returns
    -------
    None
    """
    assert isinstance(param_name, str), "`param_name` must be a string."
    assert row_h > 0.0, "Row height must be positive."

    y_center = row_y + row_h / 2.0
    param_anchors[(elem.model_type, elem.element_index, param_name)] = {
        "left": (rect.x + 0.05 * rect.w, y_center),
        "right": (rect.x + 0.95 * rect.w, y_center),
        "center": (rect.x + 0.50 * rect.w, y_center),
    }


def _prepare_routes(
    elements: Sequence[ElementSpec],
    edges: Sequence[ConstraintEdge],
    bounds: Mapping[CardKey, CardRect],
    param_anchors: Mapping[ParamKey, Dict[str, Coordinate]],
    layout: LayoutSpec,
) -> Tuple[Dict[RouteId, List[Coordinate]], Dict[RouteId, ConstraintEdge]]:
    """
    Build raw orthogonal routes before overlap offsets are applied.

    Parameters
    ----------
    elements : Sequence[ElementSpec]
        Normalized elements.
    edges : Sequence[ConstraintEdge]
        Dependency edges.
    bounds : Mapping[CardKey, CardRect]
        Card geometry map.
    param_anchors : Mapping[ParamKey, Dict[str, Coordinate]]
        Parameter anchor map.
    layout : LayoutSpec
        Layout parameters.

    Returns
    -------
    (routes, route_meta)
        Raw routes and a route-to-edge map.
    """
    assert len(elements) >= 1, "`elements` must not be empty."
    assert isinstance(layout, LayoutSpec), "`layout` must be a LayoutSpec."

    routes: Dict[RouteId, List[Coordinate]] = {}
    route_meta: Dict[RouteId, ConstraintEdge] = {}

    for route_id, edge in enumerate(edges):
        points = _route_single_edge(
            elements,
            edge,
            bounds,
            param_anchors,
            layout,
        )
        if points is None:
            continue

        routes[route_id] = points
        route_meta[route_id] = edge

    return routes, route_meta


def _route_single_edge(
    elements: Sequence[ElementSpec],
    edge: ConstraintEdge,
    bounds: Mapping[CardKey, CardRect],
    param_anchors: Mapping[ParamKey, Dict[str, Coordinate]],
    layout: LayoutSpec,
) -> Optional[List[Coordinate]]:
    """
    Build one raw orthogonal route for a dependency edge.

    Parameters
    ----------
    elements : Sequence[ElementSpec]
        Normalized elements.
    edge : ConstraintEdge
        Dependency edge.
    bounds : Mapping[CardKey, CardRect]
        Card geometry.
    param_anchors : Mapping[ParamKey, Dict[str, Coordinate]]
        Parameter anchors.
    layout : LayoutSpec
        Layout parameters.

    Returns
    -------
    Optional[List[Coordinate]]
        Route polyline or ``None`` if anchors cannot be resolved.
    """
    assert isinstance(edge, ConstraintEdge), "`edge` must be a ConstraintEdge."
    assert isinstance(layout, LayoutSpec), "`layout` must be a LayoutSpec."

    ref_key: CardKey = (edge.source[0], edge.source[1])
    tgt_key: CardKey = (edge.target[0], edge.target[1])

    if edge.source not in param_anchors or edge.target not in param_anchors:
        return None

    ref_index = _find_element_flat_index(elements, ref_key[0], ref_key[1])
    tgt_index = _find_element_flat_index(elements, tgt_key[0], tgt_key[1])
    ref_row, ref_col = divmod(ref_index, layout.n_cols)
    tgt_row, tgt_col = divmod(tgt_index, layout.n_cols)

    start_side, end_side = _choose_sides_for_orthogonal_route(
        start_col=tgt_col,
        end_col=ref_col,
        n_cols=layout.n_cols,
    )

    start_pt = param_anchors[edge.target][start_side]
    end_pt = param_anchors[edge.source][end_side]
    start_box = bounds[tgt_key]
    end_box = bounds[ref_key]

    x1 = _vertical_lane_x(start_box, start_side, layout.inner_gap)
    x2 = _vertical_lane_x(end_box, end_side, layout.inner_gap)
    y_lane = _choose_horizontal_lane(
        start_row=tgt_row,
        end_row=ref_row,
        box_h=layout.box_h,
        inner_gap=layout.inner_gap,
        y_step=layout.y_step,
        top_pad=layout.top_pad,
    )

    points = [
        start_pt,
        (x1, start_pt[1]),
        (x1, y_lane),
        (x2, y_lane),
        (x2, end_pt[1]),
        end_pt,
    ]
    return _compress_orthogonal_points(points)


def _draw_routes(
    ax: Axes,
    routes: Mapping[RouteId, List[Coordinate]],
    route_meta: Mapping[RouteId, ConstraintEdge],
    vertical_offsets: Mapping[float, Mapping[Tuple[int, str], float]],
    horizontal_offsets: Mapping[float, Mapping[Tuple[int, str], float]],
) -> None:
    """
    Draw all routes after offset assignment.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    routes : Mapping[RouteId, List[Coordinate]]
        Raw routes.
    route_meta : Mapping[RouteId, ConstraintEdge]
        Route metadata.
    vertical_offsets : Mapping[float, Mapping[Tuple[int, str], float]]
        Offsets for vertical lanes.
    horizontal_offsets : Mapping[float, Mapping[Tuple[int, str], float]]
        Offsets for horizontal lanes.

    Returns
    -------
    None
    """
    assert isinstance(ax, Axes), "`ax` must be a Matplotlib Axes."
    assert len(routes) == len(
        route_meta), "`routes` and `route_meta` must match."

    for route_id, points in routes.items():
        edge = route_meta[route_id]
        shifted = _apply_corridor_offsets_to_route(
            points,
            route_id,
            vertical_offsets,
            horizontal_offsets,
        )
        _draw_shifted_route(ax, shifted, edge)


def _draw_shifted_route(
    ax: Axes,
    shifted_points: Sequence[Coordinate],
    edge: ConstraintEdge,
) -> None:
    """
    Draw one final routed line, including its arrowhead.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    shifted_points : Sequence[Coordinate]
        Final route after overlap offsets.
    edge : ConstraintEdge
        Edge whose color is used.

    Returns
    -------
    None
    """
    assert isinstance(edge, ConstraintEdge), "`edge` must be a ConstraintEdge."
    assert len(shifted_points) >= 2, "A route must contain at least two points."

    color = _constraint_arrow_color(edge.operation)
    alpha = 0.75
    linewidth = 1.75

    if len(shifted_points) >= 3:
        xs = [pt[0] for pt in shifted_points[:-1]]
        ys = [pt[1] for pt in shifted_points[:-1]]
        ax.plot(
            xs,
            ys,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="round",
            solid_joinstyle="miter",
            zorder=4,
        )

    arrow = FancyArrowPatch(
        posA=shifted_points[-2],
        posB=shifted_points[-1],
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3,rad=0",
        zorder=5,
        alpha=alpha,
    )
    ax.add_patch(arrow)


def _finalize_axes(
    ax: Axes,
    layout: LayoutSpec,
    title: str,
    created_fig: bool,
    fig: Figure,
) -> None:
    """
    Apply axis limits, title placement, and optional figure padding cleanup.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    layout : LayoutSpec
        Layout parameters.
    title : str
        Figure title.
    created_fig : bool
        If ``True``, the figure was created inside the public entry point.
    fig : Figure
        Matplotlib figure.

    Returns
    -------
    None
    """
    assert isinstance(ax, Axes), "`ax` must be a Matplotlib Axes."
    assert isinstance(layout, LayoutSpec), "`layout` must be a LayoutSpec."

    total_width = layout.n_cols * layout.box_w + \
        (layout.n_cols - 1) * layout.inner_gap
    total_height = layout.n_rows * layout.box_h + \
        (layout.n_rows - 1) * layout.inner_gap

    ax.axis("off")
    ax.set_xlim(-layout.left_pad, total_width + layout.right_pad)
    ax.set_ylim(-(total_height + layout.bottom_pad), layout.top_pad)

    if title:
        ax.text(
            total_width / 2.0,
            layout.top_pad * 0.45,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#0f172a",
        )

    if created_fig:
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)


def _choose_sides_for_orthogonal_route(
    start_col: int,
    end_col: int,
    n_cols: int,
) -> Tuple[str, str]:
    """
    Choose route-exit and route-entry sides for one dependency.

    Parameters
    ----------
    start_col : int
        Column of the constrained card.
    end_col : int
        Column of the reference card.
    n_cols : int
        Number of columns in the grid.

    Returns
    -------
    Tuple[str, str]
        Start side and end side, each either ``"left"`` or ``"right"``.

    Examples
    --------
    >>> _choose_sides_for_orthogonal_route(0, 2, 5)
    ('right', 'left')
    >>> _choose_sides_for_orthogonal_route(3, 1, 5)
    ('left', 'right')
    """
    assert n_cols >= 1, "`n_cols` must be at least 1."
    assert start_col >= 0 and end_col >= 0, "Column indices must be non-negative."

    if end_col > start_col:
        return "right", "left"
    if end_col < start_col:
        return "left", "right"

    if start_col < n_cols / 2.0:
        return "left", "left"
    return "right", "right"


def _vertical_lane_x(box: CardRect, side: str, inner_gap: float) -> float:
    """
    Return the x coordinate of the vertical routing lane beside a card.

    Parameters
    ----------
    box : CardRect
        Card geometry.
    side : str
        Either ``"left"`` or ``"right"``.
    inner_gap : float
        Gap between neighboring cards.

    Returns
    -------
    float
        X coordinate of the lane.

    Examples
    --------
    >>> rect = CardRect(10.0, 0.0, 4.0, 2.0)
    >>> _vertical_lane_x(rect, "left", 2.0)
    9.0
    >>> _vertical_lane_x(rect, "right", 2.0)
    15.0
    """
    assert inner_gap > 0.0, "`inner_gap` must be positive."
    assert side in ("left", "right"), "`side` must be 'left' or 'right'."

    if side == "left":
        return box.x - inner_gap / 2.0
    return box.x + box.w + inner_gap / 2.0


def _choose_horizontal_lane(
    start_row: int,
    end_row: int,
    box_h: float,
    inner_gap: float,
    y_step: float,
    top_pad: float,
) -> float:
    """
    Choose a clear horizontal lane between rows for one route.

    Parameters
    ----------
    start_row : int
        Row of the constrained card.
    end_row : int
        Row of the reference card.
    box_h : float
        Card height.
    inner_gap : float
        Inter-card gap.
    y_step : float
        Vertical spacing between card origins.
    top_pad : float
        Outer top padding.

    Returns
    -------
    float
        Y coordinate of the chosen horizontal lane.
    """
    assert start_row >= 0 and end_row >= 0, "Row indices must be non-negative."
    assert box_h > 0.0 and inner_gap > 0.0 and y_step > 0.0, (
        "Layout distances must be positive."
    )

    if start_row == 0 and end_row == 0:
        return top_pad / 2.0
    if start_row == end_row:
        return -start_row * y_step + inner_gap / 2.0
    if start_row < end_row:
        return -start_row * y_step - box_h - inner_gap / 2.0
    return -start_row * y_step + inner_gap / 2.0


def _compress_orthogonal_points(
    points: Sequence[Coordinate],
    tol: float = 1.0e-12,
) -> List[Coordinate]:
    """
    Remove duplicate and collinear points from an orthogonal polyline.

    Parameters
    ----------
    points : Sequence[Coordinate]
        Raw polyline points.
    tol : float, default=1.0e-12
        Numerical tolerance used for collinearity checks.

    Returns
    -------
    List[Coordinate]
        Reduced point list.

    Examples
    --------
    >>> pts = [(0, 0), (1, 0), (2, 0), (2, 1)]
    >>> _compress_orthogonal_points(pts)
    [(0, 0), (2, 0), (2, 1)]
    """
    assert tol > 0.0, "`tol` must be positive."
    assert len(points) >= 1, "`points` must contain at least one coordinate."

    cleaned: List[Coordinate] = [points[0]]
    for point in points[1:]:
        if _points_differ(point, cleaned[-1], tol):
            cleaned.append(point)

    if len(cleaned) <= 2:
        return cleaned

    reduced: List[Coordinate] = [cleaned[0]]
    for index in range(1, len(cleaned) - 1):
        p0 = reduced[-1]
        p1 = cleaned[index]
        p2 = cleaned[index + 1]
        if _are_collinear_orthogonally(p0, p1, p2, tol):
            continue
        reduced.append(p1)

    reduced.append(cleaned[-1])
    return reduced


def _build_corridor_occupancy(
    routes: Mapping[RouteId, Sequence[Coordinate]],
) -> Tuple[
    Dict[float, List[LaneSegment]],
    Dict[float, List[LaneSegment]],
]:
    """
    Build lane-occupancy tables for vertical and horizontal route segments.

    Parameters
    ----------
    routes : Mapping[RouteId, Sequence[Coordinate]]
        Raw route polylines.

    Returns
    -------
    (vertical_usage, horizontal_usage)
        Vertical and horizontal lane occupancy maps.

    Notes
    -----
    Each route is expected to follow the six-point pattern:
        [start, p1, p2, p3, p4, end]
    after compression. Shorter routes are ignored defensively.
    """
    assert isinstance(routes, Mapping), "`routes` must be a mapping."
    assert len(routes) >= 0, "`routes` length must be valid."

    vertical_usage: Dict[float, List[LaneSegment]] = {}
    horizontal_usage: Dict[float, List[LaneSegment]] = {}

    for route_id, points in routes.items():
        if len(points) < 6:
            continue

        x1 = round(points[1][0], 9)
        x2 = round(points[4][0], 9)
        y_lane = round(points[2][1], 9)

        y1a, y1b = points[1][1], points[2][1]
        y2a, y2b = points[4][1], points[5][1]
        xh1, xh2 = points[2][0], points[3][0]

        _append_lane_segment(
            vertical_usage,
            x1,
            LaneSegment((route_id, "v1"), min(y1a, y1b), max(y1a, y1b)),
        )
        _append_lane_segment(
            vertical_usage,
            x2,
            LaneSegment((route_id, "v2"), min(y2a, y2b), max(y2a, y2b)),
        )
        _append_lane_segment(
            horizontal_usage,
            y_lane,
            LaneSegment((route_id, "h"), min(xh1, xh2), max(xh1, xh2)),
        )

    return vertical_usage, horizontal_usage


def _append_lane_segment(
    usage: MutableMapping[float, List[LaneSegment]],
    lane_key: float,
    segment: LaneSegment,
) -> None:
    """
    Append one segment to a lane-usage map.

    Parameters
    ----------
    usage : MutableMapping[float, List[LaneSegment]]
        Lane usage map updated in place.
    lane_key : float
        Lane coordinate.
    segment : LaneSegment
        Segment to add.

    Returns
    -------
    None
    """
    assert isinstance(segment, LaneSegment), "`segment` must be a LaneSegment."
    assert segment.end >= segment.start, "`segment.end` must be >= `segment.start`."

    usage.setdefault(lane_key, []).append(segment)


def _centered_offsets(n: int, span: float) -> List[float]:
    """
    Generate centered offsets across a finite span.

    Parameters
    ----------
    n : int
        Number of tracks.
    span : float
        Total width or height available for spreading.

    Returns
    -------
    List[float]
        Centered offsets.

    Examples
    --------
    >>> _centered_offsets(1, 1.0)
    [0.0]
    >>> len(_centered_offsets(3, 1.0))
    3
    """
    assert n >= 1, "`n` must be at least 1."
    assert span >= 0.0, "`span` must be non-negative."

    if n == 1:
        return [0.0]

    step = span / (n - 1)
    start = -span / 2.0
    return [start + i * step for i in range(n)]


def _assign_tracks_on_lane(
    segments: Sequence[LaneSegment],
) -> Tuple[Dict[Tuple[int, str], int], int]:
    """
    Greedy interval-track assignment for one routing lane.

    Parameters
    ----------
    segments : Sequence[LaneSegment]
        All segments that occupy the same lane.

    Returns
    -------
    (segment_tracks, n_tracks)
        Mapping from segment id to track index, and total number of tracks.

    Notes
    -----
    This is a simple bounded greedy algorithm. The loop count is limited by the
    number of segments on the lane.

    Examples
    --------
    >>> s1 = LaneSegment((0, "v1"), 0.0, 2.0)
    >>> s2 = LaneSegment((1, "v1"), 3.0, 4.0)
    >>> tracks, n = _assign_tracks_on_lane([s1, s2])
    >>> n >= 1
    True
    """
    assert isinstance(segments, Sequence), "`segments` must be a sequence."
    assert all(isinstance(seg, LaneSegment) for seg in segments), (
        "All `segments` must be LaneSegment instances."
    )

    ordered = sorted(segments, key=lambda seg: (seg.start, seg.end))
    active_track_ends: List[float] = []
    segment_tracks: Dict[Tuple[int, str], int] = {}

    for segment in ordered:
        placed = False
        for track_index, track_end in enumerate(active_track_ends):
            if segment.start > track_end:
                active_track_ends[track_index] = segment.end
                segment_tracks[segment.segment_id] = track_index
                placed = True
                break

        if placed:
            continue

        new_index = len(active_track_ends)
        active_track_ends.append(segment.end)
        segment_tracks[segment.segment_id] = new_index

    return segment_tracks, len(active_track_ends)


def _build_corridor_offsets(
    vertical_usage: Mapping[float, Sequence[LaneSegment]],
    horizontal_usage: Mapping[float, Sequence[LaneSegment]],
    inner_gap: float,
) -> Tuple[
    Dict[float, Dict[Tuple[int, str], float]],
    Dict[float, Dict[Tuple[int, str], float]],
]:
    """
    Assign per-segment offsets inside each shared lane.

    Parameters
    ----------
    vertical_usage : Mapping[float, Sequence[LaneSegment]]
        Vertical lane occupancy.
    horizontal_usage : Mapping[float, Sequence[LaneSegment]]
        Horizontal lane occupancy.
    inner_gap : float
        Gap between cards used as available routing width.

    Returns
    -------
    (vertical_offsets, horizontal_offsets)
        Offset maps for vertical and horizontal lanes.

    Examples
    --------
    >>> v_usage = {0.0: [LaneSegment((0, "v1"), 0.0, 1.0)]}
    >>> h_usage = {}
    >>> v_off, h_off = _build_corridor_offsets(v_usage, h_usage, 2.0)
    >>> 0.0 in v_off
    True
    """
    assert inner_gap > 0.0, "`inner_gap` must be positive."
    assert isinstance(
        vertical_usage, Mapping), "`vertical_usage` must be a mapping."
    assert isinstance(horizontal_usage,
                      Mapping), "`horizontal_usage` must be a mapping."

    vertical_offsets: Dict[float, Dict[Tuple[int, str], float]] = {}
    horizontal_offsets: Dict[float, Dict[Tuple[int, str], float]] = {}

    vertical_span = 0.55 * inner_gap
    horizontal_span = 0.55 * inner_gap

    for lane_x, segments in vertical_usage.items():
        segment_tracks, n_tracks = _assign_tracks_on_lane(list(segments))
        offsets = _centered_offsets(n_tracks, vertical_span)
        vertical_offsets[lane_x] = {
            seg_id: offsets[track_index]
            for seg_id, track_index in segment_tracks.items()
        }

    for lane_y, segments in horizontal_usage.items():
        segment_tracks, n_tracks = _assign_tracks_on_lane(list(segments))
        offsets = _centered_offsets(n_tracks, horizontal_span)
        horizontal_offsets[lane_y] = {
            seg_id: offsets[track_index]
            for seg_id, track_index in segment_tracks.items()
        }

    return vertical_offsets, horizontal_offsets


def _apply_corridor_offsets_to_route(
    points: Sequence[Coordinate],
    route_id: int,
    vertical_offsets: Mapping[float, Mapping[Tuple[int, str], float]],
    horizontal_offsets: Mapping[float, Mapping[Tuple[int, str], float]],
) -> List[Coordinate]:
    """
    Apply lane offsets to one route.

    Parameters
    ----------
    points : Sequence[Coordinate]
        Raw route points in the standard six-point form.
    route_id : int
        Route identifier.
    vertical_offsets : Mapping[float, Mapping[Tuple[int, str], float]]
        Vertical lane offsets.
    horizontal_offsets : Mapping[float, Mapping[Tuple[int, str], float]]
        Horizontal lane offsets.

    Returns
    -------
    List[Coordinate]
        Shifted, compressed route.

    Notes
    -----
    The function keeps the first and last points fixed and shifts only the lane
    segments. This preserves the parameter-row attachment points.
    """
    assert route_id >= 0, "`route_id` must be non-negative."
    assert len(points) >= 2, "`points` must contain at least two points."

    if len(points) < 6:
        return list(points)

    x1 = round(points[1][0], 9)
    x2 = round(points[4][0], 9)
    y_lane = round(points[2][1], 9)

    dx1 = vertical_offsets.get(x1, {}).get((route_id, "v1"), 0.0)
    dx2 = vertical_offsets.get(x2, {}).get((route_id, "v2"), 0.0)
    dy = horizontal_offsets.get(y_lane, {}).get((route_id, "h"), 0.0)

    shifted = list(points)
    shifted[1] = (shifted[1][0] + dx1, shifted[1][1])
    shifted[2] = (shifted[2][0] + dx1, shifted[2][1] + dy)
    shifted[3] = (shifted[3][0] + dx2, shifted[3][1] + dy)
    shifted[4] = (shifted[4][0] + dx2, shifted[4][1])

    return _compress_orthogonal_points(shifted)


def _get_display_name(
    functions_names: Optional[Mapping[str, Sequence[Any]]],
    model_type: str,
    element_index: int,
) -> Optional[str]:
    """
    Resolve the display name shown in the card header.

    Parameters
    ----------
    functions_names : Optional[Mapping[str, Sequence[Any]]]
        Optional mapping from model type to per-element names.
    model_type : str
        Model-family name.
    element_index : int
        Element index inside the family.

    Returns
    -------
    Optional[str]
        Header label, or ``None`` if no label is available.

    Examples
    --------
    >>> names = {"gaussian": [["o_5", 765.152]]}
    >>> _get_display_name(names, "gaussian", 0) is not None
    True
    """
    assert isinstance(model_type, str), "`model_type` must be a string."
    assert element_index >= 0, "`element_index` must be non-negative."

    if not isinstance(functions_names, Mapping):
        return None
    if model_type not in functions_names:
        return None

    names_for_type = functions_names[model_type]
    try:
        entry = names_for_type[element_index]
    except Exception:
        return None

    if entry is None:
        return None

    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        ion_name, value = entry
        pretty_ion = _format_ion_label(ion_name)
        return f"{pretty_ion} @ {_safe_scalar_str(value, precision=6)}"

    if isinstance(entry, (list, tuple)):
        return " | ".join(_safe_scalar_str(item, precision=6) for item in entry)

    return str(entry)


def _find_element_flat_index(
    elements: Sequence[ElementSpec],
    model_type: str,
    element_index: int,
) -> int:
    """
    Return the flattened grid index of one element.

    Parameters
    ----------
    elements : Sequence[ElementSpec]
        Normalized elements.
    model_type : str
        Model-family name.
    element_index : int
        Element index inside the family.

    Returns
    -------
    int
        Flat index in the `elements` list.

    Raises
    ------
    ValueError
        If the element cannot be found.
    """
    assert isinstance(model_type, str), "`model_type` must be a string."
    assert element_index >= 0, "`element_index` must be non-negative."

    for flat_index, elem in enumerate(elements):
        if elem.model_type == model_type and elem.element_index == element_index:
            return flat_index

    raise ValueError(
        f"Element ({model_type!r}, {element_index}) was not found in the element list."
    )


def _is_constraint(value: Any) -> bool:
    """
    Return `True` if a value matches the expected constraint dictionary shape.

    Parameters
    ----------
    value : Any
        Candidate value.

    Returns
    -------
    bool
        Constraint-detection result.

    Examples
    --------
    >>> _is_constraint({"reference": {}, "constraint": "lock"})
    True
    >>> _is_constraint(1.0)
    False
    """
    if not isinstance(value, Mapping):
        return False
    return "reference" in value and "constraint" in value


def _safe_scalar_str(value: Any, precision: int = 5) -> str:
    """
    Convert a scalar-like value into a compact display string.

    Parameters
    ----------
    value : Any
        Scalar, NumPy scalar, list, or tuple.
    precision : int, default=5
        Significant-digit precision used for floating-point values.

    Returns
    -------
    str
        Compact string representation.

    Examples
    --------
    >>> _safe_scalar_str(1.234567, precision=4)
    '1.235'
    >>> _safe_scalar_str([1, 2])
    '[1, 2]'
    """
    assert precision >= 1, "`precision` must be at least 1."
    assert precision <= 32, "`precision` is unexpectedly large."

    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.{precision}g}"
        return str(value)

    if isinstance(value, (list, tuple)):
        inner = ", ".join(_safe_scalar_str(item, precision=precision)
                          for item in value)
        return f"[{inner}]"

    return str(value)


def _format_param_label(
    param_name: str,
    param_info: Any,
    show_values: bool = True,
    show_constraint_details: bool = True,
) -> str:
    """
    Format one parameter label for card display.

    Parameters
    ----------
    param_name : str
        Parameter name.
    param_info : Any
        Free value or constraint dictionary.
    show_values : bool, default=True
        If ``True``, show free values.
    show_constraint_details : bool, default=True
        If ``True``, show full constraint details.

    Returns
    -------
    str
        Formatted parameter label.

    Examples
    --------
    >>> _format_param_label("I", 1.0)
    'I = 1'
    """
    assert isinstance(param_name, str), "`param_name` must be a string."
    assert isinstance(show_values, bool), "`show_values` must be boolean."

    if not _is_constraint(param_info):
        if show_values:
            return f"{param_name} = {_safe_scalar_str(param_info)}"
        return param_name

    if not show_constraint_details:
        return f"{param_name} | constrained"

    op = _optional_str(param_info.get("operation")) or "?"
    val = _safe_scalar_str(param_info.get("value"))
    ref = param_info["reference"]
    ref_txt = f"{ref['model_type']}[{ref['element_index']}].{ref['parameter']}"

    if mpl.rcParams.get("text.usetex", False):
        return (
            f"{_latex_escape_text(param_name)} | "
            f"{_latex_escape_text(ref_txt)} "
            r"$\rightarrow$ "
            f"{_latex_escape_text(op)} {_latex_escape_text(val)}"
        )

    return f"{param_name} | {ref_txt} -> {op} {val}"


def _format_ion_label(ion_text: Any) -> str:
    """
    Format an ion label such as ``'o_5'`` into a publication-style string.

    Parameters
    ----------
    ion_text : Any
        Ion label, typically in the form ``'<element>_<charge>'``.

    Returns
    -------
    str
        Formatted label. If LaTeX is enabled, returns a TeX string that uses
        bold small caps for the Roman charge state.

    Examples
    --------
    >>> label = _format_ion_label("o_5")
    >>> isinstance(label, str)
    True
    >>> "O" in label
    True
    """
    if ion_text is None:
        return ""

    text = str(ion_text).strip()
    if "_" not in text:
        return text[:1].upper() + text[1:].lower()

    element, charge = text.split("_", 1)
    element = element[:1].upper() + element[1:].lower()

    try:
        roman_charge = roman.toRoman(int(charge)).lower()
    except Exception:
        roman_charge = str(charge).lower()
    if mpl.rcParams.get("text.usetex", False):
        return rf"{element}\,\bfseries{{\textsc{{{roman_charge}}}}}"

    return f"{element} {roman_charge.upper()}"


def _latex_escape_text(text: str) -> str:
    """
    Escape a plain text fragment for safe use in LaTeX text mode.

    Parameters
    ----------
    text : str
        Raw text fragment.

    Returns
    -------
    str
        Escaped text.

    Examples
    --------
    >>> _latex_escape_text("a_b")
    'a\\_b'
    """
    assert isinstance(text, str), "`text` must be a string."
    assert len(text) >= 0, "`text` length must be valid."

    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _constraint_arrow_color(operation: Optional[str]) -> str:
    """
    Return the color used for one constraint operation.

    Parameters
    ----------
    operation : Optional[str]
        Constraint operation string.

    Returns
    -------
    str
        Hex color code.

    Examples
    --------
    >>> _constraint_arrow_color("add")
    '#2563eb'
    """
    op = str(operation).lower() if operation is not None else ""
    if op == "add":
        return "#2563eb"
    if op == "mul":
        return "#dc2626"
    return "#475569"


def _param_color(param_name: str) -> str:
    """
    Return the background color for one parameter row.

    Parameters
    ----------
    param_name : str
        Parameter name.

    Returns
    -------
    str
        Hex color code.

    Examples
    --------
    >>> _param_color("I")
    '#fee2e2'
    """
    assert isinstance(param_name, str), "`param_name` must be a string."
    assert len(param_name) >= 1, "`param_name` should not be empty."

    name = param_name.strip()
    if name == "I":
        return "#fee2e2"
    if name == "x":
        return "#dbeafe"
    if name == "s":
        return "#dcfce7"
    if re.fullmatch(r"B\d+", name):
        return "#fef3c7"
    if name.lower() in {"lims", "limits"}:
        return "#ede9fe"
    return "#f1f5f9"


def _card_colors(model_type: str) -> Tuple[str, str, str]:
    """
    Return the face, header, and edge colors for a model-family card.

    Parameters
    ----------
    model_type : str
        Model-family name.

    Returns
    -------
    (facecolor, header_color, edgecolor)
        Three hex color strings.

    Examples
    --------
    >>> len(_card_colors("gaussian"))
    3
    """
    assert isinstance(model_type, str), "`model_type` must be a string."
    assert len(model_type) >= 1, "`model_type` should not be empty."

    lowered = model_type.lower()
    if "gaussian" in lowered:
        return "#f8fbff", "#dbeafe", "#60a5fa"
    if "polynome" in lowered or "polynomial" in lowered:
        return "#fffdf7", "#fef3c7", "#f59e0b"
    return "#f8fafc", "#e2e8f0", "#94a3b8"


def _points_differ(p1: Coordinate, p2: Coordinate, tol: float) -> bool:
    """
    Return `True` if two points differ beyond the tolerance.

    Parameters
    ----------
    p1, p2 : Coordinate
        Points to compare.
    tol : float
        Tolerance.

    Returns
    -------
    bool
        Difference test result.
    """
    assert tol > 0.0, "`tol` must be positive."
    assert len(p1) == 2 and len(p2) == 2, "Points must be 2D coordinates."

    return abs(p1[0] - p2[0]) > tol or abs(p1[1] - p2[1]) > tol


def _are_collinear_orthogonally(
    p0: Coordinate,
    p1: Coordinate,
    p2: Coordinate,
    tol: float,
) -> bool:
    """
    Return `True` if three orthogonal-path points are collinear.

    Parameters
    ----------
    p0, p1, p2 : Coordinate
        Three candidate points.
    tol : float
        Tolerance.

    Returns
    -------
    bool
        Collinearity result.
    """
    assert tol > 0.0, "`tol` must be positive."
    assert len(p0) == 2 and len(p1) == 2 and len(p2) == 2, (
        "All points must be 2D coordinates."
    )

    same_x = abs(p0[0] - p1[0]) < tol and abs(p1[0] - p2[0]) < tol
    same_y = abs(p0[1] - p1[1]) < tol and abs(p1[1] - p2[1]) < tol
    return same_x or same_y


def _optional_str(value: Any) -> Optional[str]:
    """
    Convert a value into an optional string.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    Optional[str]
        ``None`` if the input is ``None``, otherwise ``str(value)``.

    Examples
    --------
    >>> _optional_str(None) is None
    True
    >>> _optional_str(5)
    '5'
    """
    if value is None:
        return None
    return str(value)
