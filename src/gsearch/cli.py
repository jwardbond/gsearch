import logging
from pathlib import Path

import click

from gsearch.core import run_gsearch
from gsearch.io import load_graph
from gsearch.log import configure_logging


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
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write logs to this file instead of stderr.",
)
def gsearch(query, graph, tol, k, verbose, log_file):
    if verbose or log_file:
        configure_logging(logging.DEBUG if verbose else logging.INFO, logfile=log_file)

    Q = load_graph(query)
    G = load_graph(graph)

    results = run_gsearch(Q, G, tol, k)
