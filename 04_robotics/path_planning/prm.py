from __future__ import annotations

import heapq
import math
import random
import tkinter as tk
from dataclasses import dataclass

Point = tuple[float, float]

CANVAS_WIDTH = 980
CANVAS_HEIGHT = 620
WORLD_MARGIN = 32

START: Point = (85, CANVAS_HEIGHT - 85)
GOAL: Point = (CANVAS_WIDTH - 85, 85)

OBSTACLE_COUNT = 10
SAMPLE_COUNT = 145
K_NEIGHBORS = 12
MAX_EDGE_LENGTH = 220
MIN_POINT_SPACING = 16
POINT_CLEARANCE = 10
EDGE_CLEARANCE = 7

POINTS_PER_TICK = 7
EDGES_PER_TICK = 18
VISITS_PER_TICK = 4
PATH_SEGMENTS_PER_TICK = 1

MAX_SCENE_ATTEMPTS = 30
MAX_ROADMAP_ATTEMPTS = 12

BG = "#f2efe8"
PANEL_BG = "#fffaf1"
CANVAS_BG = "#fbf7ef"
GRID_LINE = "#efe6d8"
TEXT = "#26313c"
MUTED = "#5f6c78"
OBSTACLE_FILL = "#263745"
OBSTACLE_OUTLINE = "#16212a"
EDGE_COLOR = "#9db0b8"
POINT_COLOR = "#4c5966"
VISITED_COLOR = "#e9a03b"
PATH_COLOR = "#ffda6a"
START_COLOR = "#2a9d8f"
GOAL_COLOR = "#e76f51"
BUTTON_BG = "#e6d2b4"
BUTTON_ACTIVE = "#d9be98"


@dataclass(frozen=True)
class RectObstacle:
    x: float
    y: float
    w: float
    h: float


@dataclass
class PRMScene:
    obstacles: list[RectObstacle]
    points: list[Point]
    edges: list[tuple[int, int]]
    visited_order: list[int]
    path: list[int]
    path_cost: float
    roadmap_attempts: int

    @property
    def sampled_points(self) -> list[Point]:
        return self.points[2:]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_rect(point: Point, rect: RectObstacle, padding: float = 0.0) -> bool:
    x, y = point
    return (
        rect.x - padding <= x <= rect.x + rect.w + padding
        and rect.y - padding <= y <= rect.y + rect.h + padding
    )


def rects_overlap(a: RectObstacle, b: RectObstacle, padding: float = 0.0) -> bool:
    return not (
        a.x + a.w + padding < b.x
        or b.x + b.w + padding < a.x
        or a.y + a.h + padding < b.y
        or b.y + b.h + padding < a.y
    )


def point_is_free(point: Point, obstacles: list[RectObstacle], padding: float) -> bool:
    x, y = point
    if not (WORLD_MARGIN <= x <= CANVAS_WIDTH - WORLD_MARGIN):
        return False
    if not (WORLD_MARGIN <= y <= CANVAS_HEIGHT - WORLD_MARGIN):
        return False
    for obstacle in obstacles:
        if point_in_rect(point, obstacle, padding):
            return False
    return True


def segment_is_free(
    start: Point, end: Point, obstacles: list[RectObstacle], padding: float
) -> bool:
    length = distance(start, end)
    steps = max(2, int(length / 5))
    for step in range(steps + 1):
        t = step / steps
        point = (
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        )
        if not point_is_free(point, obstacles, padding):
            return False
    return True


def generate_obstacles() -> list[RectObstacle]:
    protected_points = [START, GOAL]
    for _ in range(150):
        obstacles: list[RectObstacle] = []
        attempts = 0
        while len(obstacles) < OBSTACLE_COUNT and attempts < 800:
            attempts += 1
            width = random.randint(75, 150)
            height = random.randint(55, 130)
            x = random.randint(WORLD_MARGIN + 20, CANVAS_WIDTH - WORLD_MARGIN - width - 20)
            y = random.randint(WORLD_MARGIN + 20, CANVAS_HEIGHT - WORLD_MARGIN - height - 20)
            candidate = RectObstacle(x, y, width, height)

            if any(point_in_rect(point, candidate, 70) for point in protected_points):
                continue
            if any(rects_overlap(candidate, obstacle, padding=22) for obstacle in obstacles):
                continue
            obstacles.append(candidate)

        if len(obstacles) == OBSTACLE_COUNT:
            return obstacles

    raise RuntimeError("Could not generate obstacles.")


def sample_free_points(obstacles: list[RectObstacle], count: int) -> list[Point]:
    points: list[Point] = []
    attempts = 0
    max_attempts = count * 500

    while len(points) < count and attempts < max_attempts:
        attempts += 1
        point = (
            random.uniform(WORLD_MARGIN, CANVAS_WIDTH - WORLD_MARGIN),
            random.uniform(WORLD_MARGIN, CANVAS_HEIGHT - WORLD_MARGIN),
        )
        if not point_is_free(point, obstacles, POINT_CLEARANCE):
            continue
        if any(distance(point, other) < MIN_POINT_SPACING for other in points):
            continue
        points.append(point)

    if len(points) != count:
        raise RuntimeError("Could not sample enough collision-free points.")
    return points


def build_roadmap(
    points: list[Point], obstacles: list[RectObstacle]
) -> tuple[list[tuple[int, int]], dict[int, dict[int, float]]]:
    adjacency: dict[int, dict[int, float]] = {index: {} for index in range(len(points))}
    edges: list[tuple[int, int]] = []

    for index, point in enumerate(points):
        candidates = sorted(
            (
                (distance(point, points[other]), other)
                for other in range(len(points))
                if other != index
            ),
            key=lambda item: item[0],
        )

        connected = 0
        for dist, other in candidates:
            if dist > MAX_EDGE_LENGTH:
                break
            if other in adjacency[index]:
                continue
            if not segment_is_free(point, points[other], obstacles, EDGE_CLEARANCE):
                continue

            adjacency[index][other] = dist
            adjacency[other][index] = dist
            edges.append((index, other))
            connected += 1

            if connected >= K_NEIGHBORS:
                break

    return edges, adjacency


def search_roadmap(
    points: list[Point], adjacency: dict[int, dict[int, float]]
) -> tuple[list[int], list[int], float]:
    start_index = 0
    goal_index = 1
    frontier: list[tuple[float, float, int]] = [
        (distance(points[start_index], points[goal_index]), 0.0, start_index)
    ]
    came_from: dict[int, int] = {}
    costs: dict[int, float] = {start_index: 0.0}
    visited: set[int] = set()
    visited_order: list[int] = []

    while frontier:
        _, current_cost, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        visited_order.append(node)

        if node == goal_index:
            break

        for neighbor, edge_cost in adjacency[node].items():
            candidate_cost = current_cost + edge_cost
            if candidate_cost >= costs.get(neighbor, float("inf")):
                continue
            costs[neighbor] = candidate_cost
            came_from[neighbor] = node
            heuristic = distance(points[neighbor], points[goal_index])
            heapq.heappush(frontier, (candidate_cost + heuristic, candidate_cost, neighbor))

    if goal_index not in visited:
        return [], visited_order, float("inf")

    path = [goal_index]
    current = goal_index
    while current != start_index:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path, visited_order, costs[goal_index]


def build_scene_for_obstacles(obstacles: list[RectObstacle]) -> PRMScene | None:
    for roadmap_attempt in range(1, MAX_ROADMAP_ATTEMPTS + 1):
        points = [START, GOAL] + sample_free_points(obstacles, SAMPLE_COUNT)
        edges, adjacency = build_roadmap(points, obstacles)
        path, visited_order, path_cost = search_roadmap(points, adjacency)
        if path:
            return PRMScene(
                obstacles=obstacles,
                points=points,
                edges=edges,
                visited_order=visited_order,
                path=path,
                path_cost=path_cost,
                roadmap_attempts=roadmap_attempt,
            )
    return None


def solve_random_scene() -> PRMScene:
    for _ in range(MAX_SCENE_ATTEMPTS):
        obstacles = generate_obstacles()
        scene = build_scene_for_obstacles(obstacles)
        if scene is not None:
            return scene
    raise RuntimeError("Could not generate a solvable PRM scene.")


class PRMVisualizer:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Probabilistic Roadmap Path Planning")
        self.root.configure(bg=BG)

        self.scene: PRMScene | None = None
        self.after_id: str | None = None
        self.node_items: dict[int, int] = {}
        self.dynamic_items: list[int] = []
        self.robot_item: int | None = None

        self.point_index = 0
        self.edge_index = 0
        self.visit_index = 0
        self.path_index = 0

        self.stage_var = tk.StringVar(value="Preparing scene...")
        self.stats_var = tk.StringVar(value="")

        self.build_layout()
        self.new_world()

    def build_layout(self) -> None:
        outer = tk.Frame(self.root, bg=BG, padx=18, pady=18)
        outer.pack()

        header = tk.Label(
            outer,
            text="Probabilistic Roadmap (PRM)",
            font=("Menlo", 23, "bold"),
            bg=BG,
            fg=TEXT,
        )
        header.pack(anchor="w")

        subtitle = tk.Label(
            outer,
            text=(
                "Random rectangular obstacles + random milestone sampling + "
                "collision-checked roadmap edges + graph search query."
            ),
            font=("Menlo", 10),
            bg=BG,
            fg=MUTED,
            justify=tk.LEFT,
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        controls = tk.Frame(outer, bg=BG)
        controls.pack(anchor="w", pady=(0, 12))

        buttons = [
            ("Build PRM", self.build_prm),
            ("Replay", self.replay),
            ("New World", self.new_world),
            ("Quit", self.root.destroy),
        ]
        for text, command in buttons:
            button = tk.Button(
                controls,
                text=text,
                command=command,
                font=("Menlo", 11, "bold"),
                bg=BUTTON_BG,
                fg=TEXT,
                padx=12,
                pady=7,
                relief=tk.FLAT,
                activebackground=BUTTON_ACTIVE,
            )
            button.pack(side=tk.LEFT, padx=(0, 10))

        content = tk.Frame(outer, bg=BG)
        content.pack()

        self.canvas = tk.Canvas(
            content,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg=CANVAS_BG,
            highlightthickness=1,
            highlightbackground="#d7ccb8",
        )
        self.canvas.pack(side=tk.LEFT)

        side = tk.Frame(content, bg=PANEL_BG, padx=16, pady=16, width=290, height=CANVAS_HEIGHT)
        side.pack(side=tk.LEFT, padx=(14, 0), fill=tk.BOTH)
        side.pack_propagate(False)

        tk.Label(
            side,
            text="Stage",
            font=("Menlo", 14, "bold"),
            bg=PANEL_BG,
            fg=TEXT,
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            side,
            textvariable=self.stage_var,
            font=("Menlo", 11),
            bg=PANEL_BG,
            fg=TEXT,
            justify=tk.LEFT,
            wraplength=250,
        ).pack(anchor="w", pady=(6, 14))

        tk.Label(
            side,
            text="Stats",
            font=("Menlo", 14, "bold"),
            bg=PANEL_BG,
            fg=TEXT,
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            side,
            textvariable=self.stats_var,
            font=("Menlo", 10),
            bg=PANEL_BG,
            fg=TEXT,
            justify=tk.LEFT,
            wraplength=250,
        ).pack(anchor="w", pady=(6, 16))

        tk.Label(
            side,
            text="Legend",
            font=("Menlo", 14, "bold"),
            bg=PANEL_BG,
            fg=TEXT,
            anchor="w",
        ).pack(anchor="w")

        legend_lines = [
            "Green / Red: start and goal",
            "Dark blocks: random obstacles",
            "Gray dots: sampled milestones",
            "Gray lines: collision-free roadmap edges",
            "Orange dots: nodes explored by the graph search",
            "Yellow line: final planned path",
        ]
        tk.Label(
            side,
            text="\n".join(legend_lines),
            font=("Menlo", 10),
            bg=PANEL_BG,
            fg=MUTED,
            justify=tk.LEFT,
            wraplength=250,
        ).pack(anchor="w", pady=(6, 16))

        explanation = (
            "PRM first samples many random collision-free points, then connects "
            "nearby pairs whose straight-line edge does not cross any obstacle. "
            "The resulting graph acts like a reusable roadmap for path queries."
        )
        tk.Label(
            side,
            text=explanation,
            font=("Menlo", 10),
            bg=PANEL_BG,
            fg=MUTED,
            justify=tk.LEFT,
            wraplength=250,
        ).pack(anchor="w")

    def cancel_animation(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def reset_animation_state(self) -> None:
        self.point_index = 0
        self.edge_index = 0
        self.visit_index = 0
        self.path_index = 0

    def clear_dynamic(self) -> None:
        for item in self.dynamic_items:
            self.canvas.delete(item)
        self.dynamic_items.clear()
        self.node_items.clear()
        self.robot_item = None

    def draw_background_grid(self) -> None:
        for x in range(0, CANVAS_WIDTH, 40):
            self.canvas.create_line(x, 0, x, CANVAS_HEIGHT, fill=GRID_LINE, width=1)
        for y in range(0, CANVAS_HEIGHT, 40):
            self.canvas.create_line(0, y, CANVAS_WIDTH, y, fill=GRID_LINE, width=1)

    def draw_marker(self, point: Point, color: str, label: str) -> None:
        radius = 12
        x, y = point
        circle = self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
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
        self.dynamic_items.extend([circle, text])

    def draw_base_scene(self) -> None:
        self.canvas.delete("all")
        self.draw_background_grid()

        if self.scene is None:
            return

        for obstacle in self.scene.obstacles:
            self.canvas.create_rectangle(
                obstacle.x,
                obstacle.y,
                obstacle.x + obstacle.w,
                obstacle.y + obstacle.h,
                fill=OBSTACLE_FILL,
                outline=OBSTACLE_OUTLINE,
                width=2,
            )

        self.clear_dynamic()
        self.draw_marker(START, START_COLOR, "S")
        self.draw_marker(GOAL, GOAL_COLOR, "G")

    def update_stats(self, stage: str) -> None:
        if self.scene is None:
            self.stats_var.set("")
            return

        stats = [
            f"Obstacles: {len(self.scene.obstacles)}",
            f"Sampled milestones: {self.point_index}/{len(self.scene.sampled_points)}",
            f"Roadmap edges: {self.edge_index}/{len(self.scene.edges)}",
            f"Visited nodes: {self.visit_index}/{len(self.scene.visited_order)}",
            f"Path cost: {self.scene.path_cost:.1f} px",
            f"Roadmap attempts: {self.scene.roadmap_attempts}",
        ]
        self.stage_var.set(stage)
        self.stats_var.set("\n".join(stats))

    def new_world(self) -> None:
        self.cancel_animation()
        self.stage_var.set("Generating a fresh random obstacle field...")
        self.root.update_idletasks()

        self.scene = solve_random_scene()
        self.reset_animation_state()
        self.draw_base_scene()
        self.update_stats("World ready. Press Build PRM to animate the roadmap.")

    def build_prm(self) -> None:
        if self.scene is None:
            self.new_world()
            return

        self.cancel_animation()
        self.reset_animation_state()
        self.draw_base_scene()
        self.update_stats("Sampling collision-free milestones...")
        self.animate()

    def replay(self) -> None:
        self.build_prm()

    def draw_sample_point(self, point: Point, index: int) -> None:
        radius = 4
        x, y = point
        item = self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=POINT_COLOR,
            outline="",
        )
        self.node_items[index] = item
        self.dynamic_items.append(item)

    def draw_edge(self, a_index: int, b_index: int) -> None:
        if self.scene is None:
            return
        ax, ay = self.scene.points[a_index]
        bx, by = self.scene.points[b_index]
        item = self.canvas.create_line(ax, ay, bx, by, fill=EDGE_COLOR, width=1)
        self.canvas.tag_lower(item)
        self.dynamic_items.append(item)

    def mark_visited(self, node_index: int) -> None:
        if self.scene is None or node_index in (0, 1):
            return
        item = self.node_items.get(node_index)
        if item is not None:
            self.canvas.itemconfig(item, fill=VISITED_COLOR)

    def draw_path_segment(self, start_index: int, end_index: int) -> None:
        if self.scene is None:
            return
        ax, ay = self.scene.points[start_index]
        bx, by = self.scene.points[end_index]
        line = self.canvas.create_line(
            ax,
            ay,
            bx,
            by,
            fill=PATH_COLOR,
            width=5,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )
        self.dynamic_items.append(line)

        if self.robot_item is not None:
            self.canvas.delete(self.robot_item)
        self.robot_item = self.canvas.create_oval(
            bx - 7,
            by - 7,
            bx + 7,
            by + 7,
            fill=PATH_COLOR,
            outline="#ffffff",
            width=2,
        )
        self.dynamic_items.append(self.robot_item)
        self.canvas.tag_raise(self.robot_item)

    def animate(self) -> None:
        if self.scene is None:
            return

        if self.point_index < len(self.scene.sampled_points):
            for _ in range(POINTS_PER_TICK):
                if self.point_index >= len(self.scene.sampled_points):
                    break
                point = self.scene.sampled_points[self.point_index]
                absolute_index = self.point_index + 2
                self.draw_sample_point(point, absolute_index)
                self.point_index += 1
            self.update_stats("Sampling collision-free milestones...")
            self.after_id = self.root.after(22, self.animate)
            return

        if self.edge_index < len(self.scene.edges):
            for _ in range(EDGES_PER_TICK):
                if self.edge_index >= len(self.scene.edges):
                    break
                a_index, b_index = self.scene.edges[self.edge_index]
                self.draw_edge(a_index, b_index)
                self.edge_index += 1
            self.update_stats("Connecting neighbors into a collision-free roadmap...")
            self.after_id = self.root.after(18, self.animate)
            return

        if self.visit_index < len(self.scene.visited_order):
            for _ in range(VISITS_PER_TICK):
                if self.visit_index >= len(self.scene.visited_order):
                    break
                node_index = self.scene.visited_order[self.visit_index]
                self.mark_visited(node_index)
                self.visit_index += 1
            self.update_stats("Querying the roadmap graph from start to goal...")
            self.after_id = self.root.after(30, self.animate)
            return

        if self.path_index < len(self.scene.path) - 1:
            for _ in range(PATH_SEGMENTS_PER_TICK):
                if self.path_index >= len(self.scene.path) - 1:
                    break
                start_index = self.scene.path[self.path_index]
                end_index = self.scene.path[self.path_index + 1]
                self.draw_path_segment(start_index, end_index)
                self.path_index += 1
            self.update_stats("Rendering the final path found on the roadmap...")
            self.after_id = self.root.after(90, self.animate)
            return

        self.stage_var.set("Done. Yellow path is the planned route through the roadmap.")
        self.after_id = None

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    PRMVisualizer().run()
