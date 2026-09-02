from pathlib import Path

import click

from gsearch.core import run_gsearch
from gsearch.io import load_graph


@click.command()
@click.argument("query", type=click.Path(exists=True, path_type=Path))
@click.argument("graph", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-e",
    "--tol",
    type=float,
    default=10.0,
    show_default=True,
    help="Alignment tolerance (radius around nodes).",
)
@click.option(
    "-k",
    type=int,
    default=1,
    show_default=True,
    help="Number of results to return.",
)
def gsearch(query, graph, tol, k):
    Q = load_graph(query)
    G = load_graph(graph)

    results = run_gsearch(Q, G, tol, k)
