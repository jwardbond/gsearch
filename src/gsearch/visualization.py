import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Annulus
from pyvis.network import Network


def vis_static(
    G: nx.Graph,
    ax: Axes,
    node_color: str = "lightblue",
    edge_color: str = "black",
    node_size: int = 200,
    alpha: float = 1.0,
    with_labels: bool = True,
    gridlines: bool = False,
) -> None:
    """Visualize the graph using matplotlib."""
    pos = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes}
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color=node_color,
        edge_color=edge_color,
        node_size=node_size,
        alpha=alpha,
        with_labels=with_labels,
    )

    if gridlines:
        # nx.draw turns the axis off; re-enable ticks so the grid is visible.
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.grid(True)


def vis_annulus(
    G: nx.Graph,
    ax: Axes,
    anchor: int,
    width: float,
    node_color: str = "lightblue",
    edge_color: str = "black",
    node_size: int = 200,
    alpha: float = 0.3,
    with_labels: bool = True,
) -> None:
    """Draw G with the annuli that ``_make_annulus_mask`` builds around an anchor node.

    Each non-anchor node sits at some distance ``r`` from the anchor. The mask accepts
    G nodes whose distance to the reference falls within ``width`` of ``r``; this draws
    that acceptance band as a ring of radius ``r`` and half-width ``width``, one per
    node, each in a distinct color, then draws G on top via ``vis_static``.

    Args:
        G: The graph to visualize.
        ax: The matplotlib axis to draw on.
        anchor: Node id of the anchor the annuli are centered on.
        width: The tolerance (half-width) of each annulus, in coordinate units.
        node_color: Color of the drawn nodes.
        edge_color: Color of the drawn edges.
        node_size: Size of the drawn nodes.
        alpha: Opacity of the annulus fills.
        with_labels: Whether to label nodes with their ids.
    """
    ax_x, ax_y = G.nodes[anchor]["x"], G.nodes[anchor]["y"]

    radii = [
        ((G.nodes[n]["x"] - ax_x) ** 2 + (G.nodes[n]["y"] - ax_y) ** 2) ** 0.5
        for n in G.nodes
        if n != anchor
    ]
    max_r = _draw_annuli(ax, (ax_x, ax_y), radii, width, alpha)

    vis_static(
        G,
        ax,
        node_color=node_color,
        edge_color=edge_color,
        node_size=node_size,
        with_labels=with_labels,
    )

    # Patches don't drive autoscale, so size the view to hold the largest ring.
    limit = max_r + width
    ax.set_xlim(ax_x - limit, ax_x + limit)
    ax.set_ylim(ax_y - limit, ax_y + limit)
    ax.set_aspect("equal")


def vis_result(
    Q: nx.Graph,
    G: nx.Graph,
    result: tuple[float, np.ndarray, np.ndarray, nx.Graph, dict[int, int]],
    ax: Axes,
    q_node_color: str = "lightgreen",
    g_node_color: str = "lightblue",
    q_edge_color: str = "black",
    node_size: int = 200,
    offset: tuple[float, float] | None = None,
    with_labels: bool = True,
    connector_color: str = "gray",
    mismatch_color: str = "red",
) -> None:
    """Draw a single alignment with Q floating "above" G, matches linked by connectors.

    G is drawn at its true coordinates with gray edges; the edges of ``subgraph``
    (the routed match returned by ``align_and_score``) are overdrawn in black. Q is
    transformed by the alignment's ``R``/``t`` into G's frame and then shifted by
    ``offset`` so it sits above the target, giving a layered, pseudo-3D look. A dashed
    connector runs from each Q node to its matched G node; connectors for matches
    whose ids differ (``match[q] != q``) are drawn in ``mismatch_color``.

    Args:
        Q: The query graph, with ``x``/``y`` node attributes in the query frame.
        G: The target graph to draw beneath Q (drawn as passed, so pass something
            renderable, e.g. the crop ``align_and_score`` used).
        result: An ``align_and_score`` tuple ``(score, R, t, subgraph, match)``.
        ax: The matplotlib axis to draw on.
        q_node_color: Color of Q's nodes.
        g_node_color: Color of G's nodes.
        q_edge_color: Color of Q's edges.
        node_size: Size of the drawn nodes.
        offset: Vector added to Q's aligned coords to lift it above G. Defaults to
            straight up by ~1.4x Q's own span, offset slightly right.
        with_labels: Whether to label nodes with their ids.
        connector_color: Color of connectors whose match ids agree.
        mismatch_color: Color of connectors whose match ids differ.
    """
    score, R, t, subgraph, match = result

    G_pos = {n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]]) for n in G.nodes}

    # Transform Q into G's frame with the alignment, then lift it above G.
    Q_nodes = list(Q.nodes)
    Q_arr = np.array([[Q.nodes[n]["x"], Q.nodes[n]["y"]] for n in Q_nodes])
    Q_aligned = Q_arr @ R + t

    if offset is None:
        span = (Q_aligned.max(axis=0) - Q_aligned.min(axis=0)).max()
        offset = (0.3 * span, 1.4 * span)
    Q_pos = {n: Q_aligned[i] + np.asarray(offset) for i, n in enumerate(Q_nodes)}

    # G: all edges gray, then subgraph edges overdrawn black.
    nx.draw_networkx_edges(G, G_pos, ax=ax, edge_color="lightgray", width=1.0)
    nx.draw_networkx_edges(
        subgraph, G_pos, ax=ax, edge_color="black", width=1.5
    )
    nx.draw_networkx_nodes(
        G, G_pos, ax=ax, node_color=g_node_color, node_size=node_size
    )

    # Dashed connectors from each Q node down to its matched G node.
    for q, g in match.items():
        color = connector_color if q == g else mismatch_color
        (qx, qy), (gx, gy) = Q_pos[q], G_pos[g]
        ax.plot(
            [qx, gx],
            [qy, gy],
            linestyle="--",
            color=color,
            linewidth=1.0,
            zorder=1.5,
        )

    # Q on top, in its own frame.
    nx.draw_networkx_edges(Q, Q_pos, ax=ax, edge_color=q_edge_color, width=1.5)
    nx.draw_networkx_nodes(
        Q, Q_pos, ax=ax, node_color=q_node_color, node_size=node_size
    )

    if with_labels:
        nx.draw_networkx_labels(G, G_pos, ax=ax, font_size=8)
        nx.draw_networkx_labels(Q, Q_pos, ax=ax, font_size=8)

    ax.set_title(f"Score: {score:.4f}")
    ax.set_aspect("equal")
    ax.autoscale_view()


def vis_gif(
    Qs: list[nx.Graph],
    scores: list[float],
    G: nx.Graph,
    filename: str = "matches.gif",
    k: int | None = None,
    node_size: int = 100,
    interval: int = 1000,
    fps: int = 1,
) -> FuncAnimation:
    """Animate the top matches over a static background of G, one frame per match.

    Each frame draws G faintly, overlays one match graph in red, and titles it with
    its rank and score. Matches are shown worst-to-best, so the animation ends on the
    best (rank 1). Each match graph is expected to carry ``x``/``y`` node attributes in
    G's coordinate frame (e.g. the ``subgraph`` returned by ``align_and_score``).

    Args:
        Qs: The match graphs to animate, best first (as ``run_gsearch`` returns them).
        scores: The score for each match, parallel to ``Qs``.
        G: The target graph, drawn as a faint static background every frame.
        filename: Path to write the GIF to (via the pillow writer).
        k: Number of top matches to animate. Defaults to all of ``Qs``.
        node_size: Size of the overlaid match nodes.
        interval: Delay between frames in milliseconds (display only).
        fps: Frames per second of the saved GIF.

    Returns:
        The animation, so a notebook can display it inline.
    """
    # jshtml lets a returned animation render inline in a notebook.
    plt.rcParams["animation.html"] = "jshtml"

    k = len(Qs) if k is None else min(k, len(Qs))
    # Reverse so frame 0 is the worst and the animation ends on rank 1.
    frames = list(zip(Qs[:k], scores[:k]))[::-1]

    # ioff so building the figure doesn't also display a static copy.
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 10))

    def animate(i: int):
        ax.clear()

        vis_static(G, ax, alpha=0.3, with_labels=False)

        match, score = frames[i]
        vis_static(
            match,
            ax,
            node_color="red",
            edge_color="red",
            alpha=1.0,
            node_size=node_size,
            with_labels=False,
        )

        ax.set_title(f"Rank {len(frames) - i} | Score: {score:.4f}")
        return (ax,)

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(frames),
        interval=interval,
        blit=False,  # blit=False is required when clearing the axes.
    )
    ani.save(filename, writer="pillow", fps=fps)

    return ani


def _draw_annuli(
    ax: Axes,
    center: tuple[float, float],
    radii: list[float],
    width: float,
    alpha: float,
) -> float:
    """Draw one ``[r - width, r + width]`` ring per radius, each a distinct color.

    Returns the largest radius drawn, so callers can size the axis view.
    """
    cx, cy = center
    cmap = plt.get_cmap("hsv", max(len(radii), 1))

    max_r = 0.0
    for i, r in enumerate(radii):
        max_r = max(max_r, r)
        outer = r + width
        # Annulus requires 0 < ring_width <= outer, so clamp when r < width.
        ring_width = min(2 * width, outer)
        ax.add_patch(
            Annulus(
                (cx, cy),
                outer,
                ring_width,
                color=cmap(i),
                alpha=alpha,
                zorder=0,
            )
        )

    return max_r


def vis(
    G: nx.Graph,
    node_size: int = 6,
    with_labels: bool = True,
    height: str = "600px",
    filename: str = "nx.html",
):
    nt = Network(height=height, width="100%")
    nt.toggle_physics(status=False)
    nt.options.interaction.dragNodes = False
    nt.options.interaction.zoomView = True
    nt.options.edges.smooth.enabled = False

    for n in G.nodes:
        n = int(n)
        nt.add_node(
            n,
            label=str(n) if with_labels else None,
            x=G.nodes[n]["x"],
            y=-1 * G.nodes[n]["y"],
            size=node_size,
            shape="dot",
        )

    for source, target, data in G.edges(data=True):
        source = int(source)
        target = int(target)
        is_aug = bool(data.get("augmented", False))
        nt.add_edge(
            source,
            target,
            hidden=is_aug,
            group="aug" if is_aug else "base",
            color="red" if is_aug else None,
        )

    in_notebook = _in_notebook()
    nt.write_html(filename, notebook=False, open_browser=not in_notebook)

    # Post-process the generated HTML to auto-fit the view on load
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()

    fit_script = """
    <script type="text/javascript">
      network.once("afterDrawing", function () {
        network.fit({ animation: false });
      });
    </script>
    """

    extra = """
    <div style="position: absolute; top: 10px; left: 10px; z-index: 1000;">
      <label style="font-family: sans-serif; font-size: 14px;">
        <input type="checkbox" id="augToggle" onclick="toggleAugEdges()"> Show aug edges
      </label>
    </div>

    <script type="text/javascript">
      network.once("afterDrawing", function () {
        network.fit({ animation: false });
      });

      function toggleAugEdges() {
        var show = document.getElementById("augToggle").checked;
        var updates = [];
        edges.forEach(function (edge) {
          if (edge.group === "aug") {
            updates.push({ id: edge.id, hidden: !show });
          }
        });
        edges.update(updates);
      }
    </script>
    """
    html = html.replace("</body>", fit_script + extra + "</body>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    if in_notebook:
        from IPython.display import IFrame

        return IFrame(filename, width="100%", height=height)
    return None


def _in_notebook() -> bool:
    """True if running inside a Jupyter/IPython notebook kernel."""
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False
