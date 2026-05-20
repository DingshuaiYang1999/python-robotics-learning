from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import dataclass

Point = tuple[float, float]

WIDTH = 760
HEIGHT = 500
MARGIN = 28

START: Point = (70, HEIGHT - 70)
GOAL: Point = (WIDTH - 70, 70)

MAX_ITERATIONS = 1200
STEP_SIZE = 18.0
COLLISION_STEP = 4.0
OBSTACLE_CLEARANCE = 2.0
MERGE_RADIUS = 4.0
DEMO_SEED = 14

BG = "#f2eee6"
PANEL_BG = "#fff9ef"
CANVAS_BG = "#fbf7ef"
GRID = "#ede3d2"
TEXT = "#24313a"
MUTED = "#5e6a76"
BUTTON_BG = "#e5d0b0"
BUTTON_ACTIVE = "#d8bc94"

OBSTACLE_FILL = "#243745"
OBSTACLE_OUTLINE = "#17222b"
START_TREE_COLOR = "#5f8fb7"
GOAL_TREE_COLOR = "#e29063"
PATH_COLOR = "#ffd869"
START_COLOR = "#2a9d8f"
GOAL_COLOR = "#e76f51"
SAMPLE_COLOR = "#7f8a94"
ACTIVE_OUTLINE = "#17222b"


@dataclass(frozen=True)
class RectObstacle:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Tree:
    name: str
    root: Point
    nodes: list[Point]
    parents: list[int]


@dataclass(frozen=True)
class EdgeEvent:
    tree_name: str
    parent_index: int
    child_index: int
    new_point: Point
    sample_point: Point
    nearest_point: Point
    iteration: int


@dataclass
class RunResult:
    start_tree: Tree
    goal_tree: Tree
    events: list[EdgeEvent]
    path_points: list[Point]
    connected: bool
    iterations_used: int
    seed: int
    meeting_point: Point | None

    @property
    def path_length(self) -> float:
        if len(self.path_points) < 2:
            return 0.0
        total = 0.0
        for left, right in zip(self.path_points, self.path_points[1:]):
            total += distance(left, right)
        return total


OBSTACLES = [
    RectObstacle(180, 0, 46, 180),
    RectObstacle(180, 250, 46, HEIGHT - 250),
    RectObstacle(355, 0, 46, 120),
    RectObstacle(355, 190, 46, HEIGHT - 190),
    RectObstacle(530, 0, 46, 175),
    RectObstacle(530, 245, 46, HEIGHT - 245),
    RectObstacle(620, 140, 66, 42),
]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_rect(point: Point, rect: RectObstacle, padding: float = 0.0) -> bool:
    x, y = point
    return (
        rect.x - padding <= x <= rect.x + rect.w + padding
        and rect.y - padding <= y <= rect.y + rect.h + padding
    )


def point_is_free(point: Point) -> bool:
    x, y = point
    if x < MARGIN or x > WIDTH - MARGIN:
        return False
    if y < MARGIN or y > HEIGHT - MARGIN:
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


def nearest_index(nodes: list[Point], target: Point) -> int:
    best = 0
    best_distance = distance(nodes[0], target)
    for index in range(1, len(nodes)):
        candidate = distance(nodes[index], target)
        if candidate < best_distance:
            best_distance = candidate
            best = index
    return best


def extend_once(start: Point, target: Point) -> Point | None:
    total = distance(start, target)
    if total < 1e-9:
        return None

    travel = min(STEP_SIZE, total)
    dx = (target[0] - start[0]) / total
    dy = (target[1] - start[1]) / total
    candidate = (start[0] + dx * travel, start[1] + dy * travel)

    if segment_is_free(start, candidate):
        if total <= STEP_SIZE:
            return target
        return candidate
    return None


def append_unique_node(tree: Tree, parent_index: int, point: Point) -> int | None:
    if any(distance(point, existing) < MERGE_RADIUS for existing in tree.nodes):
        return None
    tree.nodes.append(point)
    tree.parents.append(parent_index)
    return len(tree.nodes) - 1


def build_branch_to_root(tree: Tree, node_index: int) -> list[Point]:
    branch = [tree.nodes[node_index]]
    current = node_index
    while current != 0:
        current = tree.parents[current]
        branch.append(tree.nodes[current])
    branch.reverse()
    return branch


def find_meeting_index(tree: Tree, point: Point) -> int:
    for index, node in enumerate(tree.nodes):
        if distance(node, point) < 1e-6:
            return index
    raise ValueError("Meeting point not found in tree.")


def try_connect_tree(
    tree: Tree,
    target_point: Point,
    iteration: int,
    events: list[EdgeEvent],
) -> Point | None:
    current_target = target_point
    while True:
        nearest = nearest_index(tree.nodes, current_target)
        nearest_point = tree.nodes[nearest]
        new_point = extend_once(nearest_point, current_target)
        if new_point is None:
            return None

        new_index = append_unique_node(tree, nearest, new_point)
        if new_index is None:
            return None

        events.append(
            EdgeEvent(
                tree_name=tree.name,
                parent_index=nearest,
                child_index=new_index,
                new_point=new_point,
                sample_point=target_point,
                nearest_point=nearest_point,
                iteration=iteration,
            )
        )

        if distance(new_point, current_target) < 1e-6:
            return new_point


def run_bidirectional_search(seed: int) -> RunResult:
    rng = random.Random(seed)
    start_tree = Tree("start", START, [START], [-1])
    goal_tree = Tree("goal", GOAL, [GOAL], [-1])
    events: list[EdgeEvent] = []

    active_tree = start_tree
    passive_tree = goal_tree
    connected = False
    meeting_point: Point | None = None
    iterations_used = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        iterations_used = iteration
        sample = (
            rng.uniform(MARGIN, WIDTH - MARGIN),
            rng.uniform(MARGIN, HEIGHT - MARGIN),
        )

        nearest = nearest_index(active_tree.nodes, sample)
        nearest_point = active_tree.nodes[nearest]
        new_point = extend_once(nearest_point, sample)
        if new_point is None:
            continue

        new_index = append_unique_node(active_tree, nearest, new_point)
        if new_index is None:
            continue

        events.append(
            EdgeEvent(
                tree_name=active_tree.name,
                parent_index=nearest,
                child_index=new_index,
                new_point=new_point,
                sample_point=sample,
                nearest_point=nearest_point,
                iteration=iteration,
            )
        )

        connected_point = try_connect_tree(passive_tree, new_point, iteration, events)
        if connected_point is not None and distance(connected_point, new_point) < 1e-6:
            connected = True
            meeting_point = new_point
            break

        if len(active_tree.nodes) > len(passive_tree.nodes):
            active_tree, passive_tree = passive_tree, active_tree

    path_points: list[Point] = []
    if connected and meeting_point is not None:
        start_meeting = find_meeting_index(start_tree, meeting_point)
        goal_meeting = find_meeting_index(goal_tree, meeting_point)
        path_from_start = build_branch_to_root(start_tree, start_meeting)
        path_from_goal = build_branch_to_root(goal_tree, goal_meeting)
        path_points = path_from_start + list(reversed(path_from_goal[:-1]))

    return RunResult(
        start_tree=start_tree,
        goal_tree=goal_tree,
        events=events,
        path_points=path_points,
        connected=connected,
        iterations_used=iterations_used,
        seed=seed,
        meeting_point=meeting_point,
    )


class BidirectionalSearchApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Bidirectional Search")
        self.root.configure(bg=BG)

        self.result = run_bidirectional_search(DEMO_SEED)

        self.event_index = 0
        self.path_index = 0
        self.after_id: str | None = None
        self.edge_items: list[int] = []
        self.path_items: list[int] = []
        self.obstacle_items: list[int] = []
        self.sample_item: int | None = None
        self.nearest_item: int | None = None
        self.new_item: int | None = None

        self.stage_var = tk.StringVar()
        self.stats_var = tk.StringVar()

        self.build_ui()
        self.reset_scene()

    def build_ui(self) -> None:
        outer = tk.Frame(self.root, bg=BG, padx=16, pady=16)
        outer.pack()

        tk.Label(
            outer,
            text="Bidirectional Search",
            font=("Menlo", 22, "bold"),
            bg=BG,
            fg=TEXT,
        ).pack(anchor="w")

        tk.Label(
            outer,
            text=(
                "Two trees are grown in parallel: one from the start and one from the goal. "
                "A random sample expands the active tree, then the other tree tries to connect to the new node."
            ),
            font=("Menlo", 10),
            bg=BG,
            fg=MUTED,
            justify=tk.LEFT,
            wraplength=1100,
        ).pack(anchor="w", pady=(4, 12))

        controls = tk.Frame(outer, bg=BG)
        controls.pack(anchor="w", pady=(0, 12))

        for label, command in [
            ("Run", self.run_animation),
            ("Replay", self.replay),
            ("Quit", self.root.destroy),
        ]:
            tk.Button(
                controls,
                text=label,
                command=command,
                font=("Menlo", 11, "bold"),
                bg=BUTTON_BG,
                fg=TEXT,
                padx=12,
                pady=7,
                relief=tk.FLAT,
                activebackground=BUTTON_ACTIVE,
            ).pack(side=tk.LEFT, padx=(0, 10))

        row = tk.Frame(outer, bg=BG)
        row.pack()

        self.canvas = tk.Canvas(
            row,
            width=WIDTH,
            height=HEIGHT,
            bg=CANVAS_BG,
            highlightthickness=1,
            highlightbackground="#d7ccb8",
        )
        self.canvas.pack(side=tk.LEFT)

        side = tk.Frame(row, bg=PANEL_BG, padx=16, pady=16, width=320, height=HEIGHT)
        side.pack(side=tk.LEFT, padx=(14, 0), fill=tk.BOTH)
        side.pack_propagate(False)

        tk.Label(
            side,
            text="Stage",
            font=("Menlo", 14, "bold"),
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
            wraplength=285,
        ).pack(anchor="w", pady=(6, 14))

        tk.Label(
            side,
            text="Stats",
            font=("Menlo", 14, "bold"),
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
            wraplength=285,
        ).pack(anchor="w", pady=(6, 14))

        tk.Label(
            side,
            text="Legend",
            font=("Menlo", 14, "bold"),
            bg=PANEL_BG,
            fg=TEXT,
        ).pack(anchor="w")

        legend = "\n".join(
            [
                "Green / Red: start and goal",
                "Blue tree: start-side growth",
                "Orange tree: goal-side growth",
                "Gray ring: sampled q_r",
                "Dark ring: nearest q_n",
                "Yellow line: final connected path",
            ]
        )
        tk.Label(
            side,
            text=legend,
            font=("Menlo", 10),
            bg=PANEL_BG,
            fg=MUTED,
            justify=tk.LEFT,
            wraplength=285,
        ).pack(anchor="w", pady=(6, 14))

        tk.Label(
            side,
            text=(
                "This demo follows the bidirectional tree idea from the slide: "
                "expand one tree toward a random sample and let the other tree grow toward the new node."
            ),
            font=("Menlo", 10),
            bg=PANEL_BG,
            fg=MUTED,
            justify=tk.LEFT,
            wraplength=285,
        ).pack(anchor="w")

    def draw_anchor(self, point: Point, color: str, label: str) -> None:
        x, y = point
        self.canvas.create_oval(
            x - 12,
            y - 12,
            x + 12,
            y + 12,
            fill=color,
            outline="#ffffff",
            width=2,
        )
        self.canvas.create_text(
            x,
            y,
            text=label,
            fill="#ffffff",
            font=("Menlo", 10, "bold"),
        )

    def draw_scene(self) -> None:
        self.canvas.delete("all")
        self.obstacle_items.clear()

        for x in range(0, WIDTH, 35):
            self.canvas.create_line(x, 0, x, HEIGHT, fill=GRID, width=1)
        for y in range(0, HEIGHT, 35):
            self.canvas.create_line(0, y, WIDTH, y, fill=GRID, width=1)

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

        self.draw_anchor(START, START_COLOR, "S")
        self.draw_anchor(GOAL, GOAL_COLOR, "G")

    def clear_markers(self) -> None:
        for item in (self.sample_item, self.nearest_item, self.new_item):
            if item is not None:
                self.canvas.delete(item)
        self.sample_item = None
        self.nearest_item = None
        self.new_item = None

    def reset_scene(self) -> None:
        self.cancel_animation()
        self.event_index = 0
        self.path_index = 0
        self.edge_items.clear()
        self.path_items.clear()
        self.clear_markers()
        self.draw_scene()
        self.update_status("Ready. Press Run to animate both trees.")

    def replay(self) -> None:
        self.reset_scene()
        self.run_animation()

    def run_animation(self) -> None:
        self.cancel_animation()
        self.animate()

    def cancel_animation(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def update_status(self, stage: str) -> None:
        self.stage_var.set(stage)
        self.stats_var.set(
            "\n".join(
                [
                    f"Seed: {self.result.seed}",
                    f"Max iterations: {MAX_ITERATIONS}",
                    f"Step size: {STEP_SIZE:.0f}",
                    f"Iterations used: {self.result.iterations_used}",
                    f"Animated edges: {self.event_index}/{len(self.result.events)}",
                    f"Start-tree nodes: {len(self.result.start_tree.nodes)}",
                    f"Goal-tree nodes: {len(self.result.goal_tree.nodes)}",
                    f"Connected: {'yes' if self.result.connected else 'no'}",
                    f"Path length: {self.result.path_length:.1f}",
                ]
            )
        )

    def color_for_tree(self, tree_name: str) -> str:
        return START_TREE_COLOR if tree_name == "start" else GOAL_TREE_COLOR

    def draw_marker(self, point: Point, radius: float, outline: str, width: int) -> int:
        x, y = point
        return self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline=outline,
            width=width,
        )

    def step(self) -> bool:
        if self.event_index < len(self.result.events):
            event = self.result.events[self.event_index]
            tree = self.result.start_tree if event.tree_name == "start" else self.result.goal_tree
            parent = tree.nodes[event.parent_index]
            child = tree.nodes[event.child_index]
            color = self.color_for_tree(event.tree_name)

            edge = self.canvas.create_line(
                parent[0],
                parent[1],
                child[0],
                child[1],
                fill=color,
                width=2,
            )
            if self.obstacle_items:
                self.canvas.tag_lower(edge, self.obstacle_items[0])
            self.edge_items.append(edge)

            self.clear_markers()
            self.sample_item = self.draw_marker(event.sample_point, 6, SAMPLE_COLOR, 2)
            self.nearest_item = self.draw_marker(event.nearest_point, 7, ACTIVE_OUTLINE, 2)
            self.new_item = self.draw_marker(event.new_point, 5, color, 2)

            self.event_index += 1

            if event.tree_name == "start":
                stage = (
                    "Start-side tree expands toward sampled q_r. "
                    "Then the goal-side tree tries to connect to the new node."
                )
            else:
                stage = (
                    "Goal-side tree expands toward sampled q_r or the opponent tree's newest node. "
                    "The algorithm keeps both trees balanced."
                )

            self.update_status(stage)
            return False

        if self.result.connected and self.path_index < len(self.result.path_points) - 1:
            left = self.result.path_points[self.path_index]
            right = self.result.path_points[self.path_index + 1]
            item = self.canvas.create_line(
                left[0],
                left[1],
                right[0],
                right[1],
                fill=PATH_COLOR,
                width=5,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )
            if self.obstacle_items:
                self.canvas.tag_lower(item, self.obstacle_items[0])
            self.path_items.append(item)
            self.path_index += 1
            self.update_status("The two trees are connected. Rendering the final path.")
            return False

        if self.result.connected:
            self.update_status("Done. The yellow polyline is the path after the two trees meet.")
        else:
            self.update_status("Done. This run did not connect both trees within the iteration budget.")

        return True

    def animate(self) -> None:
        if self.step():
            self.after_id = None
            return
        self.after_id = self.root.after(26, self.animate)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    BidirectionalSearchApp().run()
