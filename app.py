import math
import heapq
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pygame


WORLD_WIDTH = 800
WORLD_HEIGHT = 600

WINDOW_WIDTH = 1240
WINDOW_HEIGHT = 760
FPS = 60

GRID_CELL_SIZE = 20
GRID_COLS = WORLD_WIDTH // GRID_CELL_SIZE
GRID_ROWS = WORLD_HEIGHT // GRID_CELL_SIZE

CAMERA_WIDTH = 384
CAMERA_HEIGHT = 216
CAMERA_FOV_DEGREES = 70
CAMERA_RANGE = 260

ROBOT_RADIUS = 12
ROBOT_SPEED = 95
ROBOT_TURN_SPEED = 4.0

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

BLACK = (10, 12, 18)
PANEL = (24, 30, 44)
WHITE = (235, 240, 245)
MUTED = (155, 165, 180)

GREEN = (60, 220, 145)
BLUE = (75, 155, 255)
YELLOW = (255, 215, 90)
RED = (240, 80, 85)
CYAN = (80, 230, 235)

GRID_LINE = (42, 50, 68)
UNKNOWN_COLOR = (42, 46, 58)
FREE_COLOR = (62, 74, 95)
OCCUPIED_COLOR = (235, 90, 95)


@dataclass
class Detection:
    bbox: tuple
    angle: float
    distance: float
    world_point: tuple


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def distance(a, b):
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)


def world_to_cell(pos):
    x, y = pos

    col = int(x // GRID_CELL_SIZE)
    row = int(y // GRID_CELL_SIZE)

    col = clamp(col, 0, GRID_COLS - 1)
    row = clamp(row, 0, GRID_ROWS - 1)

    return row, col


def cell_to_world(cell):
    row, col = cell

    x = col * GRID_CELL_SIZE + GRID_CELL_SIZE / 2
    y = row * GRID_CELL_SIZE + GRID_CELL_SIZE / 2

    return np.array([x, y], dtype=float)


def heuristic(a, b):
    ar, ac = a
    br, bc = b

    return math.hypot(ar - br, ac - bc)


def astar(grid, start, goal):
    rows, cols = grid.shape

    if start == goal:
        return [start]

    def in_bounds(cell):
        r, c = cell
        return 0 <= r < rows and 0 <= c < cols

    def blocked(cell):
        r, c = cell
        return grid[r, c] >= OCCUPIED

    directions = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (1, 1, 1.414),
    ]

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()
            return path

        for dr, dc, move_cost in directions:
            next_cell = (current[0] + dr, current[1] + dc)

            if not in_bounds(next_cell):
                continue

            if blocked(next_cell):
                continue

            new_g = g_score[current] + move_cost

            if new_g < g_score.get(next_cell, float("inf")):
                came_from[next_cell] = current
                g_score[next_cell] = new_g

                f_score = new_g + heuristic(next_cell, goal)
                heapq.heappush(open_set, (f_score, next_cell))

    return []


def inflate_obstacles(grid, amount=1):
    obstacle_mask = (grid >= OCCUPIED).astype(np.uint8)

    kernel_size = amount * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    inflated = cv2.dilate(obstacle_mask, kernel, iterations=1)

    new_grid = grid.copy()
    new_grid[inflated == 1] = OCCUPIED

    return new_grid


class NeuroNavSim:
    def __init__(self):
        pygame.init()

        pygame.display.set_caption("NeuroNav-Sim")

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont("consolas", 15)
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_big = pygame.font.SysFont("consolas", 26, bold=True)

        self.world_offset = (20, 20)

        self.robot_pos = np.array([90.0, 90.0])
        self.robot_heading = 0.0

        self.goal = np.array([720.0, 520.0])

        self.grid = np.full((GRID_ROWS, GRID_COLS), UNKNOWN, dtype=np.int16)

        self.path = []
        self.path_world = []

        self.detections = []
        self.camera_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        self.status = "Starting..."
        self.explanation = "The robot is scanning, mapping, planning, and moving."

        self.last_plan_time = 0
        self.replan_delay = 0.35

        self.goal_reached = False
        self.obstacles = self.make_obstacles()

        self.mark_robot_area_free()

    def make_obstacles(self):
        return [
            pygame.Rect(180, 100, 90, 210),
            pygame.Rect(360, 70, 70, 160),
            pygame.Rect(520, 150, 160, 70),
            pygame.Rect(120, 420, 190, 65),
            pygame.Rect(410, 350, 90, 180),
            pygame.Rect(590, 390, 80, 120),
        ]

    def reset(self):
        self.robot_pos = np.array([90.0, 90.0])
        self.robot_heading = 0.0
        self.goal = np.array([720.0, 520.0])

        self.grid = np.full((GRID_ROWS, GRID_COLS), UNKNOWN, dtype=np.int16)

        self.path = []
        self.path_world = []
        self.detections = []

        self.status = "Reset"
        self.explanation = "The robot cleared its map and started over."
        self.goal_reached = False

        self.mark_robot_area_free()

    def point_hits_obstacle(self, point):
        x, y = point

        if x < 0 or x >= WORLD_WIDTH or y < 0 or y >= WORLD_HEIGHT:
            return True

        for obstacle in self.obstacles:
            if obstacle.collidepoint(int(x), int(y)):
                return True

        return False

    def robot_hits_obstacle(self, pos):
        for angle in np.linspace(0, 2 * math.pi, 16):
            point = np.array([
                pos[0] + math.cos(angle) * ROBOT_RADIUS,
                pos[1] + math.sin(angle) * ROBOT_RADIUS,
            ])

            if self.point_hits_obstacle(point):
                return True

        return False

    def ray_cast(self, origin, angle, max_range=CAMERA_RANGE):
        for d in range(0, max_range, 4):
            x = origin[0] + math.cos(angle) * d
            y = origin[1] + math.sin(angle) * d

            if self.point_hits_obstacle((x, y)):
                return d, (x, y)

        x = origin[0] + math.cos(angle) * max_range
        y = origin[1] + math.sin(angle) * max_range

        return None, (x, y)

    def build_camera_frame(self):
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        frame[: CAMERA_HEIGHT // 2, :] = (22, 28, 42)
        frame[CAMERA_HEIGHT // 2 :, :] = (34, 42, 58)

        for y in range(CAMERA_HEIGHT // 2, CAMERA_HEIGHT, 18):
            cv2.line(frame, (0, y), (CAMERA_WIDTH, y), (45, 55, 75), 1)

        fov = math.radians(CAMERA_FOV_DEGREES)
        horizon = CAMERA_HEIGHT // 2

        for x in range(CAMERA_WIDTH):
            percent = x / max(1, CAMERA_WIDTH - 1)
            ray_angle = self.robot_heading - fov / 2 + percent * fov

            hit_distance, _ = self.ray_cast(self.robot_pos, ray_angle)

            if hit_distance is None:
                continue

            height = int(clamp(16000 / (hit_distance + 1), 12, CAMERA_HEIGHT))
            y1 = int(clamp(horizon - height // 2, 0, CAMERA_HEIGHT - 1))
            y2 = int(clamp(horizon + height // 2, 0, CAMERA_HEIGHT - 1))

            brightness = int(clamp(255 - hit_distance * 0.45, 90, 255))
            color = (brightness, 40, 45)

            cv2.line(frame, (x, y1), (x, y2), color, 2)

        cv2.putText(
            frame,
            "SIM CAMERA",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 235, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    def detect_obstacles(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        lower_red_1 = np.array([0, 70, 70])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 70, 70])
        upper_red_2 = np.array([180, 255, 255])

        mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        mask = cv2.bitwise_or(mask_1, mask_2)

        kernel = np.ones((5, 5), dtype=np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        fov = math.radians(CAMERA_FOV_DEGREES)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 45:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            center_x = x + w / 2
            image_percent = center_x / CAMERA_WIDTH

            relative_angle = -fov / 2 + image_percent * fov
            estimated_distance = clamp(15000 / max(h, 1), 25, CAMERA_RANGE)

            world_angle = self.robot_heading + relative_angle

            wx = self.robot_pos[0] + math.cos(world_angle) * estimated_distance
            wy = self.robot_pos[1] + math.sin(world_angle) * estimated_distance

            detections.append(
                Detection(
                    bbox=(x, y, w, h),
                    angle=relative_angle,
                    distance=estimated_distance,
                    world_point=(wx, wy),
                )
            )

            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 120), 2)

            cv2.putText(
                frame,
                "obstacle",
                (x, max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 255, 120),
                1,
                cv2.LINE_AA,
            )

        return detections, frame

    def mark_robot_area_free(self):
        robot_cell = world_to_cell(self.robot_pos)

        for dr in range(-2, 3):
            for dc in range(-2, 3):
                row = robot_cell[0] + dr
                col = robot_cell[1] + dc

                if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                    self.grid[row, col] = FREE

    def mark_ray_free(self, angle, max_distance):
        d = 0
        step = GRID_CELL_SIZE / 2

        while d < max_distance:
            x = self.robot_pos[0] + math.cos(angle) * d
            y = self.robot_pos[1] + math.sin(angle) * d

            row, col = world_to_cell((x, y))

            if self.grid[row, col] != OCCUPIED:
                self.grid[row, col] = FREE

            d += step

    def update_map(self, detections):
        fov = math.radians(CAMERA_FOV_DEGREES)

        for i in range(31):
            percent = i / 30
            relative_angle = -fov / 2 + percent * fov
            world_angle = self.robot_heading + relative_angle

            self.mark_ray_free(world_angle, CAMERA_RANGE * 0.75)

        for detection in detections:
            world_angle = self.robot_heading + detection.angle

            self.mark_ray_free(world_angle, max(0, detection.distance - GRID_CELL_SIZE))

            row, col = world_to_cell(detection.world_point)

            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr = row + dr
                    cc = col + dc

                    if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                        self.grid[rr, cc] = OCCUPIED

        self.mark_robot_area_free()

    def plan_path(self):
        start = world_to_cell(self.robot_pos)
        goal = world_to_cell(self.goal)

        planning_grid = inflate_obstacles(self.grid, amount=1)

        planning_grid[start] = FREE
        planning_grid[goal] = FREE

        self.path = astar(planning_grid, start, goal)
        self.path_world = [cell_to_world(cell) for cell in self.path]

        if self.path:
            self.status = "Path found"
            self.explanation = "A* found a route through the map."
        else:
            self.status = "No path found"
            self.explanation = "The robot needs a safer route or more mapped space."

    def move_robot(self, dt):
        if self.goal_reached:
            return

        if distance(self.robot_pos, self.goal) < 18:
            self.goal_reached = True
            self.status = "Goal reached"
            self.explanation = "The robot reached the target."
            return

        if not self.path_world:
            return

        while self.path_world and distance(self.robot_pos, self.path_world[0]) < 15:
            self.path_world.pop(0)

        if not self.path_world:
            return

        target = self.path_world[0]
        vector = target - self.robot_pos

        desired_heading = math.atan2(vector[1], vector[0])
        heading_error = angle_wrap(desired_heading - self.robot_heading)

        max_turn = ROBOT_TURN_SPEED * dt
        turn = clamp(heading_error, -max_turn, max_turn)

        self.robot_heading = angle_wrap(self.robot_heading + turn)

        alignment = max(0.0, math.cos(heading_error))
        speed = ROBOT_SPEED * alignment

        next_pos = self.robot_pos + np.array([
            math.cos(self.robot_heading) * speed * dt,
            math.sin(self.robot_heading) * speed * dt,
        ])

        if not self.robot_hits_obstacle(next_pos):
            self.robot_pos = next_pos
            self.explanation = "The robot is following the next waypoint."
        else:
            hit_cell = world_to_cell(next_pos)
            self.grid[hit_cell] = OCCUPIED

            self.status = "Blocked, replanning"
            self.explanation = "The robot found a blocked move and recalculated the path."

            self.plan_path()

    def randomize_goal(self):
        for _ in range(200):
            x = random.randint(40, WORLD_WIDTH - 40)
            y = random.randint(40, WORLD_HEIGHT - 40)

            if not self.point_hits_obstacle((x, y)):
                self.goal = np.array([float(x), float(y)])
                self.goal_reached = False
                self.status = "Random goal"
                self.explanation = "A new goal was placed."
                self.plan_path()
                return

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

                if event.key == pygame.K_r:
                    self.reset()

                if event.key == pygame.K_SPACE:
                    self.plan_path()

                if event.key == pygame.K_g:
                    self.randomize_goal()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                wx = mx - self.world_offset[0]
                wy = my - self.world_offset[1]

                if 0 <= wx < WORLD_WIDTH and 0 <= wy < WORLD_HEIGHT:
                    if not self.point_hits_obstacle((wx, wy)):
                        self.goal = np.array([float(wx), float(wy)])
                        self.goal_reached = False
                        self.status = "New goal"
                        self.explanation = "The robot is planning to the clicked goal."
                        self.plan_path()

        return True

    def update(self, dt):
        self.camera_frame = self.build_camera_frame()

        detections, camera_frame = self.detect_obstacles(self.camera_frame)

        self.camera_frame = camera_frame
        self.detections = detections

        self.update_map(detections)

        current_time = pygame.time.get_ticks() / 1000

        if current_time - self.last_plan_time > self.replan_delay:
            self.plan_path()
            self.last_plan_time = current_time

        self.move_robot(dt)

    def draw_text(self, text, x, y, font=None, color=WHITE):
        if font is None:
            font = self.font

        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def draw_wrapped_text(self, text, x, y, max_width, font=None, color=WHITE):
        if font is None:
            font = self.font_small

        words = text.split()
        line = ""

        for word in words:
            test_line = line + word + " "

            if font.size(test_line)[0] > max_width and line:
                surface = font.render(line, True, color)
                self.screen.blit(surface, (x, y))

                y += font.get_height() + 4
                line = word + " "
            else:
                line = test_line

        if line:
            surface = font.render(line, True, color)
            self.screen.blit(surface, (x, y))

        return y

    def draw_panel(self, rect, title=None):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=14)
        pygame.draw.rect(self.screen, (58, 70, 96), rect, 1, border_radius=14)

        if title:
            self.draw_text(title, rect.x + 14, rect.y + 10, self.font, WHITE)

    def draw_world(self):
        ox, oy = self.world_offset

        world_rect = pygame.Rect(ox, oy, WORLD_WIDTH, WORLD_HEIGHT)

        pygame.draw.rect(self.screen, (16, 20, 30), world_rect, border_radius=10)
        pygame.draw.rect(self.screen, (70, 84, 110), world_rect, 2, border_radius=10)

        for x in range(0, WORLD_WIDTH + 1, GRID_CELL_SIZE):
            pygame.draw.line(self.screen, GRID_LINE, (ox + x, oy), (ox + x, oy + WORLD_HEIGHT), 1)

        for y in range(0, WORLD_HEIGHT + 1, GRID_CELL_SIZE):
            pygame.draw.line(self.screen, GRID_LINE, (ox, oy + y), (ox + WORLD_WIDTH, oy + y), 1)

        for obstacle in self.obstacles:
            rect = pygame.Rect(ox + obstacle.x, oy + obstacle.y, obstacle.w, obstacle.h)

            pygame.draw.rect(self.screen, RED, rect, border_radius=8)
            pygame.draw.rect(self.screen, (255, 160, 160), rect, 2, border_radius=8)

        fov = math.radians(CAMERA_FOV_DEGREES)

        left_angle = self.robot_heading - fov / 2
        right_angle = self.robot_heading + fov / 2

        robot_screen = (ox + int(self.robot_pos[0]), oy + int(self.robot_pos[1]))

        left_end = (
            ox + int(self.robot_pos[0] + math.cos(left_angle) * CAMERA_RANGE),
            oy + int(self.robot_pos[1] + math.sin(left_angle) * CAMERA_RANGE),
        )

        right_end = (
            ox + int(self.robot_pos[0] + math.cos(right_angle) * CAMERA_RANGE),
            oy + int(self.robot_pos[1] + math.sin(right_angle) * CAMERA_RANGE),
        )

        pygame.draw.line(self.screen, (70, 120, 160), robot_screen, left_end, 2)
        pygame.draw.line(self.screen, (70, 120, 160), robot_screen, right_end, 2)

        if len(self.path_world) > 1:
            points = [(ox + int(p[0]), oy + int(p[1])) for p in self.path_world]

            pygame.draw.lines(self.screen, CYAN, False, points, 4)

            for point in points[::2]:
                pygame.draw.circle(self.screen, CYAN, point, 3)

        goal_screen = (ox + int(self.goal[0]), oy + int(self.goal[1]))

        pygame.draw.circle(self.screen, GREEN, goal_screen, 12)
        pygame.draw.circle(self.screen, WHITE, goal_screen, 12, 2)

        self.draw_text("GOAL", goal_screen[0] + 14, goal_screen[1] - 10, self.font_small, GREEN)

        pygame.draw.circle(self.screen, BLUE, robot_screen, ROBOT_RADIUS)
        pygame.draw.circle(self.screen, WHITE, robot_screen, ROBOT_RADIUS, 2)

        nose = (
            robot_screen[0] + int(math.cos(self.robot_heading) * 22),
            robot_screen[1] + int(math.sin(self.robot_heading) * 22),
        )

        pygame.draw.line(self.screen, WHITE, robot_screen, nose, 3)

        self.draw_text("ROBOT", robot_screen[0] + 16, robot_screen[1] - 22, self.font_small, BLUE)

        for detection in self.detections:
            wx, wy = detection.world_point
            pygame.draw.circle(self.screen, YELLOW, (ox + int(wx), oy + int(wy)), 5)

        self.draw_text(
            "World View: robot, obstacles, camera FOV, A* path, and goal",
            ox + 14,
            oy + WORLD_HEIGHT + 10,
            self.font_small,
            MUTED,
        )

    def draw_camera_panel(self):
        rect = pygame.Rect(840, 20, 370, 245)

        self.draw_panel(rect, "Simulated Camera + OpenCV")

        frame = self.camera_frame.copy()

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (340, 191))

        self.screen.blit(surface, (rect.x + 15, rect.y + 42))

        color = GREEN if self.detections else MUTED

        self.draw_text(
            f"CV detections: {len(self.detections)}",
            rect.x + 15,
            rect.y + 218,
            self.font_small,
            color,
        )

    def draw_map_panel(self):
        rect = pygame.Rect(840, 280, 370, 275)

        self.draw_panel(rect, "Occupancy Grid Map")

        map_x = rect.x + 15
        map_y = rect.y + 42

        cell_w = 340 / GRID_COLS
        cell_h = 200 / GRID_ROWS

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                value = self.grid[row, col]

                if value == UNKNOWN:
                    color = UNKNOWN_COLOR
                elif value == FREE:
                    color = FREE_COLOR
                else:
                    color = OCCUPIED_COLOR

                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(
                        int(map_x + col * cell_w),
                        int(map_y + row * cell_h),
                        math.ceil(cell_w),
                        math.ceil(cell_h),
                    ),
                )

        for row, col in self.path:
            pygame.draw.rect(
                self.screen,
                CYAN,
                pygame.Rect(
                    int(map_x + col * cell_w),
                    int(map_y + row * cell_h),
                    math.ceil(cell_w),
                    math.ceil(cell_h),
                ),
            )

        robot_cell = world_to_cell(self.robot_pos)
        goal_cell = world_to_cell(self.goal)

        for cell, color in [(robot_cell, BLUE), (goal_cell, GREEN)]:
            row, col = cell

            pygame.draw.circle(
                self.screen,
                color,
                (
                    int(map_x + col * cell_w + cell_w / 2),
                    int(map_y + row * cell_h + cell_h / 2),
                ),
                5,
            )

        self.draw_text("gray = unknown", rect.x + 15, rect.y + 248, self.font_small, MUTED)
        self.draw_text("blue = path", rect.x + 135, rect.y + 248, self.font_small, CYAN)
        self.draw_text("red = obstacle", rect.x + 245, rect.y + 248, self.font_small, RED)

    def draw_dashboard(self):
        rect = pygame.Rect(840, 570, 370, 165)

        self.draw_panel(rect, "Navigation Dashboard")

        x = rect.x + 15
        y = rect.y + 42

        self.draw_text(f"Status: {self.status}", x, y, self.font_small, WHITE)
        y += 24

        self.draw_text(
            f"Heading: {math.degrees(self.robot_heading):.1f} deg",
            x,
            y,
            self.font_small,
            MUTED,
        )
        y += 22

        self.draw_text(
            f"Path cells: {len(self.path)}",
            x,
            y,
            self.font_small,
            MUTED,
        )
        y += 22

        known = np.count_nonzero(self.grid != UNKNOWN)
        total = self.grid.size
        coverage = known / total * 100

        self.draw_text(
            f"Map coverage: {coverage:.1f}%",
            x,
            y,
            self.font_small,
            MUTED,
        )
        y += 27

        self.draw_wrapped_text(
            self.explanation,
            x,
            y,
            335,
            self.font_small,
            YELLOW,
        )

    def draw_controls(self):
        rect = pygame.Rect(20, 650, 800, 85)

        self.draw_panel(rect, "Controls")

        self.draw_text(
            "Mouse click inside world: set goal",
            rect.x + 15,
            rect.y + 42,
            self.font_small,
            WHITE,
        )

        self.draw_text(
            "SPACE: force replan",
            rect.x + 300,
            rect.y + 42,
            self.font_small,
            WHITE,
        )

        self.draw_text(
            "G: random goal",
            rect.x + 500,
            rect.y + 42,
            self.font_small,
            WHITE,
        )

        self.draw_text(
            "R: reset",
            rect.x + 650,
            rect.y + 42,
            self.font_small,
            WHITE,
        )

    def draw_title(self):
        self.draw_text("NeuroNav-Sim", 20, 725, self.font_big, WHITE)

        self.draw_text(
            "camera -> CV -> map -> A* -> movement",
            235,
            732,
            self.font_small,
            MUTED,
        )

    def draw(self):
        self.screen.fill(BLACK)

        self.draw_world()
        self.draw_camera_panel()
        self.draw_map_panel()
        self.draw_dashboard()
        self.draw_controls()
        self.draw_title()

        pygame.display.flip()

    def run(self):
        running = True

        while running:
            dt = self.clock.tick(FPS) / 1000.0

            running = self.handle_events()

            self.update(dt)
            self.draw()

        pygame.quit()


if __name__ == "__main__":
    app = NeuroNavSim()
    app.run()
