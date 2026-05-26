from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import dataclass

Point = tuple[float, float]

WORLD_WIDTH = 420
WORLD_HEIGHT = 360
MARGIN = 18

START: Point = (42, WORLD_HEIGHT - 42)
GOAL: Point = (WORLD_WIDTH - 42, 42)

LOW_BUDGET = 100
HIGH_BUDGET = 10_000
GOAL_BIAS = 0.08
STEP_SIZE = 15.0
COLLISION_STEP = 4.0
MERGE_RADIUS = 4.0
OBSTACLE_CLEARANCE = 2.0

DEMO_SEED = 27

BG = "#f2eee6"
PANEL_BG = "#fff9ee"
CANVAS_BG = "#fbf7ef"
GRID = "#ede3d2"
TEXT = "#24313a"
MUTED = "#596774"
OBSTACLE_FILL = "#243745"
OBSTACLE_OUTLINE = "#162129"
TREE_COLOR = "#89a3b2"
PATH_COLOR = "#ffd469"
START_COLOR = "#2a9d8f"
GOAL_COLOR = "#e76f51"
BUTTON_BG = "#e5d0b0"
BUTTON_ACTIVE = "#d7bc94"


@dataclass(frozen=True)
class RectObstacle:
    x: float
    y: float
    w: float
    h: float


@dataclass
class RRTResult:
    budget: int
    seed: int
    nodes: list[Point]
    parents: list[int]
    edge_order: list[tuple[int, int]]
    path_indices: list[int]
    iterations_used: int
    success: bool
    goal_samples: int

    @property
    def path_length(self) -> float:
        if len(self.path_indices) < 2:
            return 0.0
        total = 0.0
        for left, right in zip(self.path_indices, self.path_indices[1:]):
            total += distance(self.nodes[left], self.nodes[right])
        return total


OBSTACLES = [
    RectObstacle(108, 0, 34, 165),
    RectObstacle(108, 220, 34, 140),
    RectObstacle(198, 0, 34, 100),
    RectObstacle(198, 155, 34, 205),
    RectObstacle(288, 0, 34, 165),
    RectObstacle(288, 220, 34, 140),
    RectObstacle(335, 112, 46, 30),
]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_sq(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def point_in_rect(point: Point, rect: RectObstacle, padding: float = 0.0) -> bool:
    x, y = point
    return (
        rect.x - padding <= x <= rect.x + rect.w + padding
        and rect.y - padding <= y <= rect.y + rect.h + padding
    )


def point_is_free(point: Point) -> bool:
    x, y = point
    if x < MARGIN or x > WORLD_WIDTH - MARGIN:
        return False
    if y < MARGIN or y > WORLD_HEIGHT - MARGIN:
        return False
    return not any(
        point_in_rect(point, obstacle, padding=OBSTACLE_CLEARANCE) for obstacle in OBSTACLES
    )


def segment_is_free(start: Point, end: Point) -> bool:
    total = distance(start, end)
    steps = max(2, int(total / COLLISION_STEP))
    for step in range(steps + 1):
        t = step / steps
        point = (
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        )
        if not point_is_free(point):
            return False
    return True


def steer_toward(start: Point, target: Point) -> Point | None:
    total = distance(start, target)
    if total < 1e-9:
        return None

    if target == GOAL and segment_is_free(start, GOAL):
        return GOAL

    max_travel = min(STEP_SIZE, total)
    dx = (target[0] - start[0]) / total
    dy = (target[1] - start[1]) / total

    last_free: Point | None = None
    travel = COLLISION_STEP
    while travel < max_travel:
        candidate = (start[0] + dx * travel, start[1] + dy * travel)
        if not point_is_free(candidate):
            break
        last_free = candidate
        travel += COLLISION_STEP

    candidate = (start[0] + dx * max_travel, start[1] + dy * max_travel)
    if point_is_free(candidate):
        last_free = candidate

    return last_free


def nearest_node(nodes: list[Point], target: Point) -> int:
    best_index = 0
    best_distance = distance_sq(nodes[0], target)
    for index in range(1, len(nodes)):
        candidate = distance_sq(nodes[index], target)
        if candidate < best_distance:
            best_distance = candidate
            best_index = index
    return best_index


def generate_sample_sequence(seed: int, count: int) -> list[tuple[Point, bool]]:
    rng = random.Random(seed)
    sequence: list[tuple[Point, bool]] = []
    for _ in range(count):
        if rng.random() < GOAL_BIAS:
            sequence.append((GOAL, True))
        else:
            sample = (
                rng.uniform(MARGIN, WORLD_WIDTH - MARGIN),
                rng.uniform(MARGIN, WORLD_HEIGHT - MARGIN),
            )
            sequence.append((sample, False))
    return sequence


def build_path(parents: list[int], goal_index: int) -> list[int]:
    path = [goal_index]
    current = goal_index
    while current != 0:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


def run_rrt(budget: int, seed: int, sequence: list[tuple[Point, bool]]) -> RRTResult:
    nodes = [START]
    parents = [-1]
    edge_order: list[tuple[int, int]] = []
    path_indices: list[int] = []
    goal_samples = 0
    success = False
    iterations_used = 0

    for iteration in range(budget):
        target, sampled_goal = sequence[iteration]
        if sampled_goal:
            goal_samples += 1

        nearest_index = nearest_node(nodes, target)
        candidate = steer_toward(nodes[nearest_index], target)
        iterations_used = iteration + 1

        if candidate is None:
            continue

        if not segment_is_free(nodes[nearest_index], candidate):
            continue

        if any(distance(candidate, node) < MERGE_RADIUS for node in nodes):
            continue

        nodes.append(candidate)
        parents.append(nearest_index)
        new_index = len(nodes) - 1
        edge_order.append((nearest_index, new_index))

        if candidate == GOAL:
            success = True
            path_indices = build_path(parents, new_index)
            break

        if distance(candidate, GOAL) <= STEP_SIZE and segment_is_free(candidate, GOAL):
            nodes.append(GOAL)
            parents.append(new_index)
            goal_index = len(nodes) - 1
            edge_order.append((new_index, goal_index))
            success = True
            path_indices = build_path(parents, goal_index)
            break

    return RRTResult(
        budget=budget,
        seed=seed,
        nodes=nodes,
        parents=parents,
        edge_order=edge_order,
        path_indices=path_indices,
        iterations_used=iterations_used,
        success=success,
        goal_samples=goal_samples,
    )


def make_demo_results(seed: int) -> tuple[RRTResult, RRTResult]:
    sequence = generate_sample_sequence(seed, HIGH_BUDGET)
    low = run_rrt(LOW_BUDGET, seed, sequence)
    high = run_rrt(HIGH_BUDGET, seed, sequence)
    return low, high


class ResultWindow:
    def __init__(
        self,
        app: "RRTComparisonApp",
        result: RRTResult,
        title: str,
        geometry: str,
    ) -> None:
        self.app = app
        self.result = result
        self.window = tk.Toplevel(app.root)
        self.window.title(title)
        self.window.geometry(geometry)
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.app.quit_all)

        self.edge_cursor = 0
        self.path_cursor = 0
        self.after_done = False
        self.edge_batch = max(1, math.ceil(max(1, len(result.edge_order)) / 220))

        self.stats_var = tk.StringVar()
        self.stage_var = tk.StringVar()

        self.dynamic_items: list[int] = []
        self.path_items: list[int] = []
        self.anchor_items: list[int] = []
        self.obstacle_items: list[int] = []

        self.build_ui()
        self.reset()

    def build_ui(self) -> None:
        outer = tk.Frame(self.window, bg=BG, padx=14, pady=14)
        outer.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            outer,
            text=f"RRT Comparison: {self.result.budget:,} Samples",
            font=("Menlo", 18, "bold"),
            bg=BG,
            fg=TEXT,
        ).pack(anchor="w")

        tk.Label(
            outer,
            text=(
                "Sample q_r from free space or the goal with probability p_f, "
                "find nearest q_n, extend a collision-free local planner, and add q_e."
            ),
            font=("Menlo", 9),
            bg=BG,
            fg=MUTED,
            justify=tk.LEFT,
            wraplength=720,
        ).pack(anchor="w", pady=(4, 10))

        controls = tk.Frame(outer, bg=BG)
        controls.pack(anchor="w", pady=(0, 10))

        for label, command in [
            ("Run", self.app.run_animation),
            ("Replay", self.app.replay),
            ("Quit", self.app.quit_all),
        ]:
            tk.Button(
                controls,
                text=label,
                command=command,
                font=("Menlo", 10, "bold"),
                bg=BUTTON_BG,
                fg=TEXT,
                padx=12,
                pady=6,
                relief=tk.FLAT,
                activebackground=BUTTON_ACTIVE,
            ).pack(side=tk.LEFT, padx=(0, 10))

        row = tk.Frame(outer, bg=BG)
        row.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            row,
            width=WORLD_WIDTH,
            height=WORLD_HEIGHT,
            bg=CANVAS_BG,
            highlightthickness=1,
            highlightbackground="#d7ccb8",
        )
        self.canvas.pack(side=tk.LEFT)

        side = tk.Frame(row, bg=PANEL_BG, padx=14, pady=14, width=250, height=WORLD_HEIGHT)
        side.pack(side=tk.LEFT, fill=tk.BOTH, padx=(12, 0))
        side.pack_propagate(False)

        tk.Label(
            side,
            text="Stage",
            font=("Menlo", 13, "bold"),
            bg=PANEL_BG,
            fg=TEXT,
        ).pack(anchor="w")

        tk.Label(
            side,
            textvariable=self.stage_var,
            font=("Menlo", 10),
            bg=PANEL_BG,
            fg=TEXT,
            justify=tk.LEFT,
            wraplength=220,
        ).pack(anchor="w", pady=(5, 12))

        tk.Label(
            side,
            text="Stats",
            font=("Menlo", 13, "bold"),
            bg=PANEL_BG,
            fg=TEXT,
        ).pack(anchor="w")

        tk.Label(
            side,
            textvariable=self.stats_var,
            font=("Menlo", 10),
            bg=PANEL_BG,
            fg=TEXT,
            justify=tk.LEFT,
            wraplength=220,
        ).pack(anchor="w", pady=(5, 12))

        tk.Label(
            side,
            text="Legend",
            font=("Menlo", 13, "bold"),
            bg=PANEL_BG,
            fg=TEXT,
        ).pack(anchor="w")

        legend = "\n".join(
            [
                "Green / Red: start and goal",
                "Dark blocks: obstacles",
                "Blue lines: RRT tree edges",
                "Yellow line: extracted path",
            ]
        )
        tk.Label(
            side,
            text=legend,
            font=("Menlo", 10),
            bg=PANEL_BG,
            fg=MUTED,
            justify=tk.LEFT,
            wraplength=220,
        ).pack(anchor="w", pady=(5, 12))

        tk.Label(
            side,
            text=(
                "Both windows use the same obstacle map and the same sample sequence prefix. "
                "Only the sample budget differs."
            ),
            font=("Menlo", 9),
            bg=PANEL_BG,
            fg=MUTED,
            justify=tk.LEFT,
            wraplength=220,
        ).pack(anchor="w")

    def draw_static_scene(self) -> None:
        self.canvas.delete("all")
        for x in range(0, WORLD_WIDTH, 30):
            self.canvas.create_line(x, 0, x, WORLD_HEIGHT, fill=GRID, width=1)
        for y in range(0, WORLD_HEIGHT, 30):
            self.canvas.create_line(0, y, WORLD_WIDTH, y, fill=GRID, width=1)

        self.obstacle_items.clear()
        for obstacle in OBSTACLES:
            item = self.canvas.create_rectangle(
                obstacle.x,
                obstacle.y,
                obstacle.x + obstacle.w,
                obstacle.y + obstacle.h,
                fill=OBSTACLE_FILL,
                outline=OBSTACLE_OUTLINE,
                width=2,
            )
            self.obstacle_items.append(item)

        self.anchor_items.clear()
        self.anchor_items.extend(self._draw_anchor(START, START_COLOR, "S"))
        self.anchor_items.extend(self._draw_anchor(GOAL, GOAL_COLOR, "G"))

    def _draw_anchor(self, point: Point, color: str, label: str) -> tuple[int, int]:
        x, y = point
        circle = self.canvas.create_oval(
            x - 11,
            y - 11,
            x + 11,
            y + 11,
            fill=color,
            outline="#ffffff",
            width=2,
        )
        text = self.canvas.create_text(
            x,
            y,
            text=label,
            fill="#ffffff",
            font=("Menlo", 10, "bold"),
        )
        return circle, text

    def reset(self) -> None:
        self.edge_cursor = 0
        self.path_cursor = 0
        self.after_done = False
        self.dynamic_items.clear()
        self.path_items.clear()
        self.draw_static_scene()
        self.update_stats("Ready. Press Run to animate this RRT tree.")

    def update_stats(self, stage: str) -> None:
        status = "Reached goal" if self.result.success else "Budget exhausted"
        self.stage_var.set(stage)
        self.stats_var.set(
            "\n".join(
                [
                    f"Seed: {self.result.seed}",
                    f"Budget: {self.result.budget:,}",
                    f"Goal bias p_f: {GOAL_BIAS:.2f}",
                    f"Step size: {STEP_SIZE:.0f}",
                    f"Samples used: {self.result.iterations_used:,}",
                    f"Goal samples: {self.result.goal_samples}",
                    f"Tree nodes: {len(self.result.nodes):,}",
                    f"Tree edges: {self.edge_cursor:,}/{len(self.result.edge_order):,}",
                    f"Status: {status}",
                    f"Path length: {self.result.path_length:.1f}",
                ]
            )
        )

    def step(self) -> bool:
        if self.edge_cursor < len(self.result.edge_order):
            for _ in range(self.edge_batch):
                if self.edge_cursor >= len(self.result.edge_order):
                    break
                parent, child = self.result.edge_order[self.edge_cursor]
                px, py = self.result.nodes[parent]
                cx, cy = self.result.nodes[child]
                item = self.canvas.create_line(px, py, cx, cy, fill=TREE_COLOR, width=1)
                if self.obstacle_items:
                    self.canvas.tag_lower(item, self.obstacle_items[0])
                self.dynamic_items.append(item)
                self.edge_cursor += 1

            for item in self.anchor_items:
                self.canvas.tag_raise(item)

            self.update_stats("Growing the tree with nearest-neighbor expansion...")
            return False

        if self.result.success and self.path_cursor < len(self.result.path_indices) - 1:
            left = self.result.path_indices[self.path_cursor]
            right = self.result.path_indices[self.path_cursor + 1]
            lx, ly = self.result.nodes[left]
            rx, ry = self.result.nodes[right]
            item = self.canvas.create_line(
                lx,
                ly,
                rx,
                ry,
                fill=PATH_COLOR,
                width=5,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )
            if self.obstacle_items:
                self.canvas.tag_lower(item, self.obstacle_items[0])
            self.path_items.append(item)
            self.path_cursor += 1

            for path_item in self.path_items:
                self.canvas.tag_raise(path_item)
            for item in self.anchor_items:
                self.canvas.tag_raise(item)

            self.update_stats("Tracing the unique path back through parent links...")
            return False

        if not self.after_done:
            if self.result.success:
                self.update_stats("Done. The yellow polyline is the RRT path to the goal.")
            else:
                self.update_stats("Done. This budget was not enough to connect to the goal.")
            self.after_done = True

        return True


class RRTComparisonApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.after_id: str | None = None

        low_result, high_result = make_demo_results(DEMO_SEED)
        self.windows = [
            ResultWindow(self, low_result, "RRT vs Samples: 100", "760x460+24+50"),
            ResultWindow(self, high_result, "RRT vs Samples: 10000", "760x460+812+50"),
        ]

        self.run_animation()

    def replay(self) -> None:
        self.cancel_animation()
        for window in self.windows:
            window.reset()
        self.run_animation()

    def run_animation(self) -> None:
        self.cancel_animation()
        self.animate()

    def animate(self) -> None:
        done = True
        for window in self.windows:
            if not window.step():
                done = False

        if done:
            self.after_id = None
            return

        self.after_id = self.root.after(32, self.animate)

    def cancel_animation(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def quit_all(self) -> None:
        self.cancel_animation()
        for window in self.windows:
            if window.window.winfo_exists():
                window.window.destroy()
        self.root.quit()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    RRTComparisonApp().run()
