# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`gsearch` — searching for shapes (small geometric patterns) inside large geometric graphs. A *query* graph `Q` (a handful of nodes with 2D coordinates) is matched against a large *target* graph `G` (up to ~10^6 nodes), returning the top-`k` locations in `G` whose local geometry matches `Q` within a tolerance.

The project is early-stage: the matching pipeline in `core.py` is a sketch with several unimplemented functions (`_similarity_score`, `_crop_graph`, and the body/return of `run_gsearch`).

## Commands

Managed with `uv` (Python 3.14, `uv_build` backend, src layout).

```bash
uv sync                                  # install deps + dev group
uv run pytest                            # run tests
uv run pytest tests/test_x.py::test_name # run a single test
uv run jupyter lab                       # notebooks/ prototyping (gitignored)
```

No linter or formatter is configured.

### Test conventions

- One test file per module (`tests/test_spatial.py`). Tests for a function are grouped in a class named after it; when the function is already a method, the class covers the owning class instead (`TestSpatialGraphIndex` covers all of its methods).
- Each test body follows a single **plan / do / test** flow, but without `# plan` / `# do` / `# test` marker comments. Exactly one "do" — the call under test — per test function.
- If the intent isn't obvious from the test name, open with a short one-line docstring. Precede each assert, or each group of related asserts, with a comment under ten words saying what it checks.

### Running the CLI

`pyproject.toml` declares `gsearch = "gsearch:main"`, which resolves to the hello-world stub in `__init__.py` — **not** the real command. The actual Click command is `gsearch.cli:gsearch`:

```bash
uv run python -c "from gsearch.cli import gsearch; gsearch()" QUERY GRAPH [-e TOL] [-k K]
```

Fixing the entry point to `gsearch.cli:gsearch` is a reasonable change if you touch packaging.

## Architecture

Four modules under `src/gsearch/`:

- `cli.py` — Click command; loads `QUERY` and `GRAPH` paths, delegates to `run_gsearch`.
- `file_utils.py` — `load_graph(path)` reads the JSON graph format into an `nx.Graph` with `x`/`y` node attributes. (Replaces the removed `load.py`.)
- `spatial.py` — `SpatialGraphIndex` wraps a graph in a scipy `KDTree` over node coordinates; `query_radius` returns nearby nodes nearest-first, `make_crop` returns the induced subgraph within a radius of a node. Module-level `distance_from_node` gives Euclidean distances from one node to all others.
- `core.py` — the matching pipeline.

### The intended pipeline (`run_gsearch`)

1. Index `G` spatially (`SpatialGraphIndex`).
2. Pick an anchor node in `Q` (`_get_anchor_node`; currently just the first node).
3. Compute `Q`'s anchor eccentricity (`Q_radius`) — the farthest any query node sits from the anchor.
4. Rank every node of `G` as a candidate anchor by `_similarity_score`.
5. For each candidate, crop `G` to a disc of `Q_radius + tol` and prune by comparing anchor-to-node distance profiles between `Q` and the crop.

### Key invariants

- **Distances are Euclidean over `x`/`y`, never graph-hop distances.** Every distance in `spatial.py` and `core.py` is geometric; edges only define connectivity/structure.
- **Query and target live in different coordinate frames.** Query files are centred near the origin with arbitrary node ids, while target coordinates are large positive values. Absolute positions are meaningless across the two — matching must rely on relative geometry only (and, in principle, be invariant to translation and probably rotation).
- `tol` is a spatial radius, in the same units as the target graph's coordinates (CLI default `10.0`).

## Data

`data/` is gitignored and not distributed with the repo; there is no generation script in-tree, so the corpus is assumed to already exist locally.

- `data/db/graphs/<id>.json` — `{"graph_id", "nodes": [{"id","x","y"}], "edges": [{"source","target"}]}`. Three families of 50 each: `gd*`, `gg*`, `go*`, ranging from ~10^5 to ~10^6 nodes (`go*` graphs are far denser, ~10^7 edges).
- `data/db/graphs/manifest.jsonl` — one JSON object per graph with precomputed stats (`n_nodes`, `n_edges`, `is_planar`, `density`, `avg_clustering`, `n_components`, …). Use this to pick test graphs instead of loading multi-hundred-MB JSON files.
- `data/db/queries/<graph_id>_q.json` — same schema; small patterns (3–20 nodes) paired by name with a target graph.

When testing interactively, prefer the smallest graphs found via `manifest.jsonl`; loading a `go*` graph is expensive.
