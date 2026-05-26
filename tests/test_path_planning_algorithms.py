from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rrt_demo_seed_finds_a_collision_free_path() -> None:
    rrt = load_module("rrt_module", "04_robotics/path_planning/rrt.py")

    sequence = rrt.generate_sample_sequence(rrt.DEMO_SEED, rrt.HIGH_BUDGET)
    result = rrt.run_rrt(rrt.HIGH_BUDGET, rrt.DEMO_SEED, sequence)

    assert result.success
    assert result.path_indices[0] == 0
    assert result.nodes[result.path_indices[-1]] == rrt.GOAL
    assert result.path_length > 0
    assert result.iterations_used <= rrt.HIGH_BUDGET


def test_bidirectional_rrt_demo_seed_connects_both_trees() -> None:
    bidirectional_rrt = load_module(
        "bidirectional_rrt_module",
        "04_robotics/path_planning/bidirectional_rrt.py",
    )

    result = bidirectional_rrt.run_bidirectional_search(bidirectional_rrt.DEMO_SEED)

    assert result.connected
    assert result.path_points[0] == bidirectional_rrt.START
    assert result.path_points[-1] == bidirectional_rrt.GOAL
    assert result.path_length > 0
    assert result.meeting_point in result.path_points


def test_prm_search_finds_path_on_simple_clear_roadmap() -> None:
    prm = load_module("prm_module", "04_robotics/path_planning/prm.py")

    points = [
        prm.START,
        prm.GOAL,
        (220, 460),
        (360, 380),
        (500, 300),
        (640, 225),
        (780, 150),
    ]
    edges, adjacency = prm.build_roadmap(points, obstacles=[])
    path, visited_order, path_cost = prm.search_roadmap(points, adjacency)

    assert edges
    assert visited_order[0] == 0
    assert path[0] == 0
    assert path[-1] == 1
    assert math.isfinite(path_cost)
