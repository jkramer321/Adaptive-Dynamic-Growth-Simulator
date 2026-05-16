"""
Mycelium Growth Engine — Python port of growthEngine.js

Simulates fungal network growth with branching, environmental interaction,
and network fusion (anastomosis). All spatial units are pixels, angles in radians.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_SIZE = 2.0          # px per normalized step (at 1x speed, 60fps)
WANDER_ANGLE = 0.15      # radians max random deviation per step

# Branching
BRANCH_ANGLE_MIN = math.pi / 12   # ~15°
BRANCH_ANGLE_MAX = math.pi / 3    # ~60°
BRANCH_RATE_MIN = 0.001
BRANCH_RATE_MAX = 0.005
MAX_TIPS = 200

# Nutrient source / organic matter
NUTRIENT_RADIUS = 160.0
NUTRIENT_MAX_ACCEL = 2.5
NUTRIENT_ATTRACTION = 0.5

# Symbiotic plant
PLANT_RADIUS = 300.0
PLANT_ATTRACTION = 0.6

# Water source
WATER_RADIUS = 130.0
WATER_MAX_ACCEL = 2.0
WATER_ATTRACTION = 0.5

# Obstacle
OBSTACLE_WIDTH = 60.0
OBSTACLE_HEIGHT = 30.0
OBSTACLE_SENSE_DIST = 50.0
OBSTACLE_STEER_STRENGTH = 0.4

# Suppression zone
SUPPRESSION_RADIUS = 50.0
SUPPRESSION_SPEED_FACTOR = 0.35   # 65% speed reduction

# Anastomosis (network fusion)
ANASTOMOSIS_PROXIMITY_RADIUS = 6.0
ANASTOMOSIS_COOLDOWN_STEPS = 80   # new tips immune for this many steps

# Dead organic matter
DEAD_ORG_ZONE_RADIUS = 80.0
DEAD_ORG_ATTRACTION = 0.25
DEAD_ORG_COLONIZATION_INCREMENT = 0.0003

# Toxin
TOXIN_REPULSION_RADIUS = 120.0
TOXIN_LETHAL_RADIUS = 30.0
TOXIN_REPULSION_STRENGTH = 0.4

# Universal steering dead zone — no steering closer than this
ELEMENT_STEER_DEAD_ZONE = 20.0

# Forward cone constraint for attraction steering: ±120° (2π/3 radians)
FORWARD_CONE_HALF = 2 * math.pi / 3

# Speed presets
SPEED_SLOW = 0.15
SPEED_NORMAL = 1.0
SPEED_FAST = 2.0

# ---------------------------------------------------------------------------
# Element types
# ---------------------------------------------------------------------------

class ElementType(str, Enum):
    STARTING_POINT     = "starting-point"
    NUTRIENT_SOURCE    = "nutrient-source"
    ORGANIC_MATTER     = "organic-matter"
    SYMBIOTIC_PLANT    = "symbiotic-plant"
    OBSTACLE           = "obstacle"
    SUPPRESSION_ZONE   = "suppression-zone"
    COMPETING_ORGANISM = "competing-organism"
    TOXIN              = "toxin"
    WATER_SOURCE       = "water-source"
    DEAD_ORGANIC_MATTER = "dead-organic-matter"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Element:
    """An environmental element placed on the canvas."""
    id: str
    type: ElementType
    x: float
    y: float
    # Obstacles also carry width/height; defaults match JS constants
    width: float = OBSTACLE_WIDTH
    height: float = OBSTACLE_HEIGHT


@dataclass
class Tip:
    """A live growing end of the mycelium network."""
    id: str
    x: float
    y: float
    angle: float           # radians
    branch_rate: float     # probability per step
    network_id: str        # 'primary', 'competitor', or 'primary-{ts}'
    points: list[float] = field(default_factory=list)  # flattened [x1,y1,x2,y2,...]
    dead: bool = False

    def step_count(self) -> int:
        """Number of steps taken (points / 2 coordinates each)."""
        return len(self.points) // 2


@dataclass
class FusedPair:
    """Record of two tips that fused via anastomosis."""
    id: str
    tip_a_points: list[float]
    tip_b_points: list[float]
    network_id: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _in_forward_cone(tip_angle: float, target_x: float, target_y: float,
                     tip_x: float, tip_y: float) -> bool:
    """Return True if the target is within ±120° of the tip's heading."""
    target_angle = math.atan2(target_y - tip_y, target_x - tip_x)
    diff = abs(_wrap_angle(target_angle - tip_angle))
    return diff < FORWARD_CONE_HALF


def _steer_toward(tip: Tip, tx: float, ty: float, t: float, strength: float) -> None:
    """Interpolate tip angle toward (tx, ty) if within forward cone."""
    dist = _dist(tip.x, tip.y, tx, ty)
    if dist <= ELEMENT_STEER_DEAD_ZONE:
        return
    if not _in_forward_cone(tip.angle, tx, ty, tip.x, tip.y):
        return
    target_angle = math.atan2(ty - tip.y, tx - tip.x)
    tip.angle += _wrap_angle(target_angle - tip.angle) * t * strength


def _steer_away(tip: Tip, fx: float, fy: float, strength: float) -> None:
    """Steer tip directly away from point (fx, fy)."""
    repel_angle = math.atan2(tip.y - fy, tip.x - fx)
    tip.angle += _wrap_angle(repel_angle - tip.angle) * strength


# ---------------------------------------------------------------------------
# Obstacle geometry helpers
# ---------------------------------------------------------------------------

def _tip_inside_obstacle(tip: Tip, el: Element) -> bool:
    """Return True if tip is inside the obstacle rectangle."""
    half_w = el.width / 2
    half_h = el.height / 2
    return (el.x - half_w <= tip.x <= el.x + half_w and
            el.y - half_h <= tip.y <= el.y + half_h)


def _obstacle_edge_dist(tip: Tip, el: Element) -> tuple[float, float, float]:
    """
    Return (edge_dist, nearest_edge_x, nearest_edge_y).
    edge_dist = 0 if inside obstacle.
    """
    half_w = el.width / 2
    half_h = el.height / 2
    # Clamp tip to rectangle to find nearest point on rect
    cx = max(el.x - half_w, min(tip.x, el.x + half_w))
    cy = max(el.y - half_h, min(tip.y, el.y + half_h))
    d = _dist(tip.x, tip.y, cx, cy)
    return d, cx, cy

# ---------------------------------------------------------------------------
# Event system (simple pub-sub)
# ---------------------------------------------------------------------------

class EventEmitter:
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, callback: Callable) -> Callable:
        """Register a listener. Returns an unsubscribe function."""
        self._listeners.setdefault(event, []).append(callback)
        def unsub():
            self._listeners[event].remove(callback)
        return unsub

    def emit(self, event: str, data: dict | None = None) -> None:
        for cb in self._listeners.get(event, []):
            cb(data or {})


growth_events = EventEmitter()

# ---------------------------------------------------------------------------
# Core growth logic
# ---------------------------------------------------------------------------

def _new_tip(x: float, y: float, angle: float, network_id: str,
             branch_rate: Optional[float] = None) -> Tip:
    if branch_rate is None:
        branch_rate = random.uniform(BRANCH_RATE_MIN, BRANCH_RATE_MAX)
    return Tip(
        id=str(uuid.uuid4()),
        x=x,
        y=y,
        angle=angle,
        branch_rate=branch_rate,
        network_id=network_id,
        points=[x, y],
    )


def _spawn_branch(parent: Tip) -> Tip:
    """Create a child branch tip from a parent."""
    magnitude = random.uniform(BRANCH_ANGLE_MIN, BRANCH_ANGLE_MAX)
    sign = random.choice([-1, 1])
    new_angle = _wrap_angle(parent.angle + sign * magnitude)

    # Inherit branch rate with ±20% variance
    variance = parent.branch_rate * 0.2
    new_rate = parent.branch_rate + random.uniform(-variance, variance)
    new_rate = max(BRANCH_RATE_MIN, min(BRANCH_RATE_MAX, new_rate))

    child = _new_tip(parent.x, parent.y, new_angle, parent.network_id, new_rate)
    growth_events.emit("tip_branched", {"parent_id": parent.id, "branch_id": child.id})
    return child


def _step_tip(tip: Tip, delta_ms: float, speed_multiplier: float) -> Optional[Tip]:
    """
    Advance tip one engine tick.

    Returns a new branch Tip if branching occurred, else None.
    Mutates tip in-place (position, angle, points).
    """
    steps = (delta_ms / 16.67) * speed_multiplier  # normalize to 60fps

    # Random wander
    tip.angle += random.uniform(-WANDER_ANGLE, WANDER_ANGLE)
    tip.angle = _wrap_angle(tip.angle)

    # Move forward
    tip.x += math.cos(tip.angle) * STEP_SIZE * steps
    tip.y += math.sin(tip.angle) * STEP_SIZE * steps
    tip.points.extend([tip.x, tip.y])

    # Branching check
    if random.random() < tip.branch_rate * steps:
        return _spawn_branch(tip)
    return None


# ---------------------------------------------------------------------------
# Environmental influence
# ---------------------------------------------------------------------------

def _apply_environment(tip: Tip, elements: list[Element],
                        emitted: dict[tuple, bool]) -> float:
    """
    Apply all element influences to a tip.

    Mutates tip.angle and tip.dead.
    Returns local speed multiplier (1.0 = no change).
    """
    local_speed = 1.0

    def emit_once(event: str, element_id: str) -> None:
        key = (tip.id, element_id, event)
        if key not in emitted:
            emitted[key] = True
            growth_events.emit(event, {"tip_id": tip.id, "element_id": element_id})

    for el in elements:
        if tip.dead:
            break

        etype = el.type

        # ------------------------------------------------------------------
        # NUTRIENT SOURCE / ORGANIC MATTER — accelerate + attract
        # ------------------------------------------------------------------
        if etype in (ElementType.NUTRIENT_SOURCE, ElementType.ORGANIC_MATTER):
            d = _dist(tip.x, tip.y, el.x, el.y)
            if d < NUTRIENT_RADIUS:
                t = 1.0 - d / NUTRIENT_RADIUS
                local_speed = max(local_speed, 1.0 + t * (NUTRIENT_MAX_ACCEL - 1.0))
                _steer_toward(tip, el.x, el.y, t, NUTRIENT_ATTRACTION)
                event = "tip_influenced" if etype == ElementType.NUTRIENT_SOURCE else "tip_near_organic"
                emit_once(event, el.id)

        # ------------------------------------------------------------------
        # SYMBIOTIC PLANT — attract only (no speed boost)
        # ------------------------------------------------------------------
        elif etype == ElementType.SYMBIOTIC_PLANT:
            d = _dist(tip.x, tip.y, el.x, el.y)
            if d < PLANT_RADIUS and d > ELEMENT_STEER_DEAD_ZONE:
                t = 1.0 - d / PLANT_RADIUS
                _steer_toward(tip, el.x, el.y, t, PLANT_ATTRACTION)
                emit_once("tip_attracted", el.id)

        # ------------------------------------------------------------------
        # WATER SOURCE — accelerate + attract
        # ------------------------------------------------------------------
        elif etype == ElementType.WATER_SOURCE:
            d = _dist(tip.x, tip.y, el.x, el.y)
            if d < WATER_RADIUS:
                t = 1.0 - d / WATER_RADIUS
                local_speed = max(local_speed, 1.0 + t * (WATER_MAX_ACCEL - 1.0))
                _steer_toward(tip, el.x, el.y, t, WATER_ATTRACTION)
                emit_once("water_attraction", el.id)

        # ------------------------------------------------------------------
        # OBSTACLE — deflect or kill
        # ------------------------------------------------------------------
        elif etype == ElementType.OBSTACLE:
            if _tip_inside_obstacle(tip, el):
                tip.dead = True
                break
            edge_dist, ex, ey = _obstacle_edge_dist(tip, el)
            if edge_dist < OBSTACLE_SENSE_DIST:
                strength = (1.0 - edge_dist / OBSTACLE_SENSE_DIST) * OBSTACLE_STEER_STRENGTH
                _steer_away(tip, ex, ey, strength)
                emit_once("tip_rerouting", el.id)

        # ------------------------------------------------------------------
        # SUPPRESSION ZONE — slow tip down
        # ------------------------------------------------------------------
        elif etype == ElementType.SUPPRESSION_ZONE:
            d = _dist(tip.x, tip.y, el.x, el.y)
            if d < SUPPRESSION_RADIUS:
                local_speed = min(local_speed, SUPPRESSION_SPEED_FACTOR)
                emit_once("tip_suppressed", el.id)

        # ------------------------------------------------------------------
        # DEAD ORGANIC MATTER — weak attraction (colonization handled in main loop)
        # ------------------------------------------------------------------
        elif etype == ElementType.DEAD_ORGANIC_MATTER:
            d = _dist(tip.x, tip.y, el.x, el.y)
            if d < DEAD_ORG_ZONE_RADIUS and d > ELEMENT_STEER_DEAD_ZONE:
                t = 1.0 - d / DEAD_ORG_ZONE_RADIUS
                _steer_toward(tip, el.x, el.y, t, DEAD_ORG_ATTRACTION)

        # ------------------------------------------------------------------
        # TOXIN — repel or kill
        # ------------------------------------------------------------------
        elif etype == ElementType.TOXIN:
            d = _dist(tip.x, tip.y, el.x, el.y)
            if d < TOXIN_LETHAL_RADIUS:
                tip.dead = True
                emit_once("toxin_kill", el.id)
                break
            elif d < TOXIN_REPULSION_RADIUS:
                if d > ELEMENT_STEER_DEAD_ZONE:
                    strength = (1.0 - d / TOXIN_REPULSION_RADIUS) * TOXIN_REPULSION_STRENGTH
                    _steer_away(tip, el.x, el.y, strength)
                emit_once("toxin_repulsion", el.id)

    return local_speed


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

class MyceliumSimulation:
    """
    Self-contained mycelium growth simulation.

    Usage:
        sim = MyceliumSimulation(width=800, height=600)
        sim.start_growth(400, 300)          # plant a starting tip
        for _ in range(500):
            sim.step(16.67)                 # advance ~one 60fps frame
        print(len(sim.tips), "live tips")
    """

    def __init__(self, width: float = 800.0, height: float = 600.0,
                 speed_multiplier: float = SPEED_NORMAL):
        self.width = width
        self.height = height
        self.speed_multiplier = speed_multiplier
        self.running = False

        self.tips: list[Tip] = []
        self.segments: list[Tip] = []        # archived / dead tips
        self.elements: list[Element] = []
        self.fused_pairs: list[FusedPair] = []

        # colonization progress: element_id → float [0, 1]
        self.colonization_progress: dict[str, float] = {}
        # decomposition event flags: element_id → {"start": bool, "advanced": bool}
        self.decomposition_fired: dict[str, dict] = {}

        # emit-once tracker: (tip_id, element_id, event_name) → True
        self._emitted: dict[tuple, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed_growth(self, x: float, y: float) -> None:
        """Plant a primary network tip without starting the simulation."""
        self._reset()
        tip = _new_tip(x, y, random.uniform(0, 2 * math.pi), "primary")
        self.tips.append(tip)
        self.running = False
        growth_events.emit("growth_started", {"origin_x": x, "origin_y": y})

    def start_growth(self, x: float, y: float) -> None:
        """Plant a primary network tip and immediately start running."""
        self.seed_growth(x, y)
        self.running = True

    def seed_competitor(self, x: float, y: float) -> None:
        """Add a competing organism tip (isolated — cannot fuse with primary)."""
        tip = _new_tip(x, y, -math.pi / 2, "competitor")
        self.tips.append(tip)

    def add_starting_point(self, x: float, y: float) -> None:
        """Add an additional primary-network starting point (multi-start)."""
        ts = str(int(random.random() * 1e9))
        tip = _new_tip(x, y, -math.pi / 2, f"primary-{ts}")
        self.tips.append(tip)
        growth_events.emit("growth_started", {"origin_x": x, "origin_y": y})

    def add_element(self, etype: ElementType, x: float, y: float,
                    width: float = OBSTACLE_WIDTH,
                    height: float = OBSTACLE_HEIGHT) -> Element:
        """Place an environmental element on the canvas."""
        el = Element(id=str(uuid.uuid4()), type=etype, x=x, y=y,
                     width=width, height=height)
        self.elements.append(el)
        return el

    def remove_element(self, element_id: str) -> None:
        self.elements = [e for e in self.elements if e.id != element_id]

    def remove_competitor_tips(self) -> None:
        """Remove all competitor tips and segments from the simulation."""
        self.tips = [t for t in self.tips if t.network_id != "competitor"]
        self.segments = [s for s in self.segments if s.network_id != "competitor"]

    def set_running(self, running: bool) -> None:
        self.running = running

    def set_speed(self, multiplier: float) -> None:
        self.speed_multiplier = multiplier

    def reset(self) -> None:
        self._reset()
        growth_events.emit("growth_reset", {})

    # ------------------------------------------------------------------
    # Main step — call once per frame (pass elapsed ms)
    # ------------------------------------------------------------------

    def step(self, delta_ms: float = 16.67) -> None:
        """Advance the simulation by delta_ms milliseconds."""
        if not self.running or not self.tips:
            return

        new_branches: list[Tip] = []
        newly_dead: list[Tip] = []
        new_fused_pairs: list[FusedPair] = []

        # --- 1. Advance each live tip ---
        for tip in self.tips:
            if tip.dead:
                continue

            # Environment modifies angle + dead flag; returns speed multiplier
            local_speed = _apply_environment(tip, self.elements, self._emitted)
            effective_speed = self.speed_multiplier * local_speed

            # Move tip forward; may return a branch
            branch = _step_tip(tip, delta_ms, effective_speed)
            if branch is not None:
                new_branches.append(branch)

        # --- 2. Boundary reflection ---
        for tip in self.tips:
            if tip.dead:
                continue
            if tip.x < 0:
                tip.x = 0.0
                tip.angle = _wrap_angle(math.pi - tip.angle)
            elif tip.x > self.width:
                tip.x = self.width
                tip.angle = _wrap_angle(math.pi - tip.angle)
            if tip.y < 0:
                tip.y = 0.0
                tip.angle = _wrap_angle(-tip.angle)
            elif tip.y > self.height:
                tip.y = self.height
                tip.angle = _wrap_angle(-tip.angle)

        # --- 3. Collect dead tips ---
        for tip in self.tips:
            if tip.dead:
                newly_dead.append(tip)
                # Clear emitted cache for this tip
                keys_to_delete = [k for k in self._emitted if k[0] == tip.id]
                for k in keys_to_delete:
                    del self._emitted[k]
                growth_events.emit("tip_died", {"tip_id": tip.id})

        live_tips = [t for t in self.tips if not t.dead]

        # --- 4. Anastomosis (same-network fusion) ---
        for i in range(len(live_tips)):
            for j in range(i + 1, len(live_tips)):
                a = live_tips[i]
                b = live_tips[j]
                if a.dead or b.dead:
                    continue
                # Competitor tips never fuse with any other network
                if a.network_id == "competitor" or b.network_id == "competitor":
                    continue
                # Both must meet the cooldown threshold
                if (a.step_count() < ANASTOMOSIS_COOLDOWN_STEPS or
                        b.step_count() < ANASTOMOSIS_COOLDOWN_STEPS):
                    continue
                # Proximity check
                if _dist(a.x, a.y, b.x, b.y) < ANASTOMOSIS_PROXIMITY_RADIUS:
                    # Guard: don't eliminate all live tips
                    surviving = [t for t in live_tips if not t.dead and t is not a and t is not b]
                    if not surviving and not new_branches:
                        continue
                    a.dead = True
                    b.dead = True
                    pair = FusedPair(
                        id=str(uuid.uuid4()),
                        tip_a_points=list(a.points),
                        tip_b_points=list(b.points),
                        network_id=a.network_id,
                    )
                    new_fused_pairs.append(pair)
                    growth_events.emit("anastomosis_occurred",
                                       {"tip_a_id": a.id, "tip_b_id": b.id})

        # Re-filter after anastomosis kills
        dead_after_anast = [t for t in live_tips if t.dead]
        live_tips = [t for t in live_tips if not t.dead]

        # --- 5. Dead organic matter colonization ---
        dead_org_elements = [e for e in self.elements
                             if e.type == ElementType.DEAD_ORGANIC_MATTER]
        for el in dead_org_elements:
            if el.id not in self.colonization_progress:
                self.colonization_progress[el.id] = 0.0
                self.decomposition_fired[el.id] = {"start": False, "advanced": False}

            for tip in live_tips:
                if _dist(tip.x, tip.y, el.x, el.y) < DEAD_ORG_ZONE_RADIUS:
                    self.colonization_progress[el.id] = min(
                        1.0,
                        self.colonization_progress[el.id] + DEAD_ORG_COLONIZATION_INCREMENT
                    )

            progress = self.colonization_progress[el.id]
            fired = self.decomposition_fired[el.id]
            if progress >= 0.25 and not fired["start"]:
                fired["start"] = True
                growth_events.emit("decomp_start", {"element_id": el.id})
            if progress >= 0.75 and not fired["advanced"]:
                fired["advanced"] = True
                growth_events.emit("decomp_advanced", {"element_id": el.id})

        # --- 6. Commit updated state ---
        self.segments.extend(newly_dead)
        self.segments.extend(dead_after_anast)
        self.fused_pairs.extend(new_fused_pairs)

        # Enforce MAX_TIPS soft cap (remove oldest live tips first)
        next_tips = live_tips + new_branches
        if len(next_tips) > MAX_TIPS:
            excess = len(next_tips) - MAX_TIPS
            pruned = next_tips[:excess]
            self.segments.extend(pruned)
            next_tips = next_tips[excess:]

        self.tips = next_tips

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self.tips = []
        self.segments = []
        self.elements = []
        self.fused_pairs = []
        self.colonization_progress = {}
        self.decomposition_fired = {}
        self._emitted = {}
        self.running = False

    @property
    def total_strand_count(self) -> int:
        """Total number of strands (live tips + archived segments)."""
        return len(self.tips) + len(self.segments)

    def snapshot(self) -> dict:
        """Return a plain-dict snapshot of the current simulation state."""
        return {
            "running": self.running,
            "speed_multiplier": self.speed_multiplier,
            "tips": [
                {
                    "id": t.id,
                    "x": t.x,
                    "y": t.y,
                    "angle": t.angle,
                    "network_id": t.network_id,
                    "step_count": t.step_count(),
                }
                for t in self.tips
            ],
            "segment_count": len(self.segments),
            "fused_pair_count": len(self.fused_pairs),
            "colonization_progress": dict(self.colonization_progress),
        }


# ---------------------------------------------------------------------------
# Quick demo (run directly: python mycelium_engine.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Wire up a few event listeners
    def on_branch(data):
        pass  # could print or collect

    def on_anastomosis(data):
        print(f"  [anastomosis] tips fused: {data['tip_a_id'][:8]}… ↔ {data['tip_b_id'][:8]}…")

    def on_decomp(data):
        print(f"  [decomp] element {data['element_id'][:8]}… colonization milestone reached")

    growth_events.on("anastomosis_occurred", on_anastomosis)
    growth_events.on("decomp_start", on_decomp)
    growth_events.on("decomp_advanced", on_decomp)

    print("=== Mycelium Growth Engine — Python Demo ===\n")

    sim = MyceliumSimulation(width=800, height=600, speed_multiplier=SPEED_NORMAL)

    # Place some environmental elements
    sim.add_element(ElementType.NUTRIENT_SOURCE, 600, 300)
    sim.add_element(ElementType.WATER_SOURCE, 200, 450)
    sim.add_element(ElementType.OBSTACLE, 400, 200)
    sim.add_element(ElementType.TOXIN, 150, 150)
    sim.add_element(ElementType.DEAD_ORGANIC_MATTER, 650, 500)
    sim.add_element(ElementType.SYMBIOTIC_PLANT, 700, 100)

    # Start the primary network at center
    sim.start_growth(400, 300)

    # Also add a competitor and a second starting point
    sim.seed_competitor(100, 100)
    sim.add_starting_point(700, 500)

    # Run 1000 frames (~16.67 seconds at 60fps)
    FRAMES = 1000
    print(f"Running {FRAMES} frames…\n")
    for frame in range(FRAMES):
        sim.step(16.67)
        if frame % 200 == 199:
            snap = sim.snapshot()
            print(f"  Frame {frame + 1:4d}: {len(snap['tips'])} live tips, "
                  f"{snap['segment_count']} archived, "
                  f"{snap['fused_pair_count']} fusions")

    print("\n=== Final snapshot ===")
    print(json.dumps(sim.snapshot(), indent=2))
