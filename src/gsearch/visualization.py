import networkx as nx
from matplotlib.axes import Axes
from pyvis.network import Network


def vis_static(
    G: nx.Graph,
    ax: Axes,
    node_color: str = "lightblue",
    edge_color: str = "black",
    node_size: int = 200,
    alpha: float = 1.0,
    with_labels: bool = True,
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
